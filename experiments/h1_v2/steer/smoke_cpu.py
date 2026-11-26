"""CPU-only checks of the method code (no model weights, no GPU): run before any GPU run.

    CUDA_VISIBLE_DEVICES= python smoke_cpu.py

1. pilot split by image (train / val disjoint), every instruction's {X} slot recovered
2. real Qwen2.5-Omni processor on real T / A / V inputs: text span decodes to the instruction, audio positions,
   image positions, box positions (subset of the image tokens), CFG pair encoding
3. Steerer hook on a toy layer stack: adds alpha*v only at masked positions, only on the prompt pass
4. CFGProcessor: both rows of a pair get the same token; w=0 = plain greedy of the 'with' row; plausibility cut
5. box_token_mask geometry; config ids distinct; analyze_steer on a synthetic scored file
"""
import json
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

from steerlib import (DATA, RESULTS, CFGProcessor, Steerer, SteerModel, box_token_mask, cond_inputs,  # noqa: E402
                      config_id, contrast_pair, load_pilot, position_mask, slot_x)

ok = lambda c, msg: print(("PASS " if c else "FAIL ") + msg) or (c or fails.append(msg))
fails = []

# 1 ------------------------------------------------------------------
train, val = load_pilot()
ok(len(train) + len(val) == 1000, f"pilot split: {len(train)} train, {len(val)} val")
ok(not ({s['image_id'] for s in train} & {s['image_id'] for s in val}), "no image in both train and val")
bad = []
for s in train + val:
    try:
        slot_x(s)
    except ValueError as e:
        bad.append(str(e))
ok(not bad, f"slot X recovered for all 1000 instructions ({len(bad)} failures{': ' + bad[0] if bad else ''})")
print("   e.g.", [contrast_pair(s) for s in val[:2]])
ok(all(s.get("text_box") for s in val), "every val sample has a text_box")

# 2 ------------------------------------------------------------------
from omnilib import Omni  # noqa: E402
from transformers import Qwen2_5OmniProcessor  # noqa: E402
m = SteerModel.__new__(SteerModel)
m.omni = Omni.__new__(Omni)
m.omni.processor = m.proc = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
m.proc.tokenizer.padding_side = "left"
m.omni.torch = m.torch = torch
m.tok, m.device = m.proc.tokenizer, "cpu"
from run_steer import abs_job  # noqa: E402
from PIL import Image  # noqa: E402
ss = [s for s in val if s["branch"] in ("attribute", "count", "relation")][:3]
for cond, pos in (("T", "text"), ("A", "audio"), ("V", "box"), ("V", "image")):
    jobs = [abs_job(*cond_inputs(s, cond)) for s in ss]
    enc, grids = m.encode(jobs)
    sizes = [Image.open(DATA / s["negation_image"]).size for s in ss]
    mask = position_mask(enc["input_ids"], pos, grids, sizes, [s["text_box"] for s in ss], [s["negation"] for s in ss], m.tok)
    n = mask.sum(1).tolist()
    ok(all(k > 0 for k in n), f"{cond}/{pos}: positions per row {n}")
    if pos == "text":
        for b, s in enumerate(ss):
            got = m.tok.decode(enc["input_ids"][b][mask[b]])
            ok(got.strip() == s["negation"], f"   text span = {got!r}")
    if pos == "box":
        n_img = [(g[0] // 2) * (g[1] // 2) for g in grids]
        ok(all(0 < a < b for a, b in zip(n, n_img)), f"   box tokens {n} < image tokens {n_img}")
pairs = [(abs_job(*cond_inputs(s, "A")), abs_job(*cond_inputs(s, "C0"))) for s in ss[:2]]
enc, _ = m.encode([j for p in pairs for j in p])
na = (enc["input_ids"] == 151646).sum(1).tolist()
ok(na[0] > 0 and na[1] == 0 and na[2] > 0 and na[3] == 0, f"CFG pair batch: audio tokens per row {na}")

# 3 ------------------------------------------------------------------
class Toy(torch.nn.Module):
    def forward(self, h):
        return h
layers = torch.nn.ModuleList([Toy() for _ in range(4)])
mask = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=torch.bool)
v = torch.arange(3, dtype=torch.float32) + 1
with Steerer(layers, {2: v}, 2.0, mask):
    h = layers[2](torch.zeros(2, 5, 3))
    h1 = layers[2](torch.zeros(2, 1, 3))
    h0 = layers[1](torch.zeros(2, 5, 3))
ok(torch.equal(h[0, 1], 2 * v) and torch.equal(h[1, 3], 2 * v) and h[0, 0].abs().sum() == 0 and h[1, 4].abs().sum() == 0,
   "Steerer adds alpha*v exactly at the masked positions")
ok(h1.abs().sum() == 0 and h0.abs().sum() == 0, "no steering on decode steps or other layers")
h2 = layers[2](torch.zeros(2, 5, 3))
ok(h2.abs().sum() == 0, "hooks removed after the context")

# 4 ------------------------------------------------------------------
sc = torch.tensor([[5.0, 4.0, 1.0, 0.0], [0.0, 4.5, 1.0, 0.0],    # pair 1: with prefers 0, without strongly prefers 1
                   [1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0]])   # pair 2: identical rows
for w, beta, exp in ((0.0, 0.1, [0, 2]), (1.0, 1e-6, [0, 2]), (3.0, 1e-6, [0, 2])):
    out = CFGProcessor(w, beta)(None, sc.clone())
    got = out.argmax(-1).tolist()
    ok(got == [exp[0], exp[0], exp[1], exp[1]], f"CFG w={w}: tokens {got}")
sc2 = torch.tensor([[5.0, 0.0, -9.0], [5.0, 0.0, -20.0]])
ok(CFGProcessor(5.0, 1e-6)(None, sc2).argmax(-1).tolist() == [0, 0], "CFG plausibility cut keeps implausible token 2 out")
ok(CFGProcessor(5.0, 1e-9)(None, sc2).argmax(-1).tolist() == [2, 2], "   (without the cut it would pick token 2)")

# 5 ------------------------------------------------------------------
bm = box_token_mask((28, 28), (392, 392), [0, 0, 28, 28])     # 14x14 merged grid, box = top-left 28 px
ok(sum(bm) == 1 and bm[0], f"box mask: one token for a 28-px corner box ({sum(bm)})")
cfgs = [{"kind": "steer", "layers": [16], "alpha": 4.0, "pos": "audio", "dir_tag": "mention"},
        {"kind": "rand", "layers": [16], "alpha": 4.0, "pos": "audio", "dir_tag": "mention"},
        {"kind": "steer", "layers": [12, 13], "alpha": 4.0, "pos": "box", "dir_tag": "mention"},
        {"kind": "cfg", "w": 1.0, "beta": 0.1}, {"kind": "transcribe"}]
ids = [config_id(c) for c in cfgs]
ok(len(set(ids)) == len(ids), f"config ids distinct: {ids}")
import random  # noqa: E402
from analyze_steer import analyze  # noqa: E402
d = RESULTS / "smoke-cpu"
d.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
with open(d / "scored_val.jsonl", "w") as f:
    for s in val[:120]:
        for cfg, c, pv in (("none", "C0", .85), ("none", "A", .5), ("none", "V", .9), (ids[0], "A", .3), (ids[2], "V", .85)):
            f.write(json.dumps({"config": cfg, "sample_id": s["id"], "image_id": s["image_id"], "condition": c,
                                "violation": rng.random() < pv, "required_recall": rng.random()}) + "\n")
analyze("smoke-cpu", "val")
md = (RESULTS.parent / "analysis_smoke-cpu_val.md").read_text()
ok(ids[0] in md and ids[2] in md, "analyze_steer writes a table with both configs")
shutil.rmtree(d)
(RESULTS.parent / "analysis_smoke-cpu_val.md").unlink()

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + '; '.join(fails)}")
sys.exit(1 if fails else 0)
