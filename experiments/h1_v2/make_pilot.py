"""Fixed, balanced pilot list (Vy, 2026-11-25): 200 questions per branch = 1,000, saved as ids so that H1 / H2 / H3 / H4
and the method experiments all use the SAME questions.

Pool = one negative sample per unit (same draw as run_h1v2.one_per_unit, seed 42), excluded units dropped
(load_dataset()), asr_ok only, distinct images. Within each branch the picks are balanced (greedy, seeded) over
source (equal share), voice (50/50), split train/val (50/50) and template N1-N7 (equal share).

    python make_pilot.py            -> pilot1000.json (ids + metadata), pilot1000_summary.md
CPU only; does not touch run_h1v2.py or any result file.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from omnilib import BRANCHES, load_dataset, split_group  # noqa: E402
import run_h1v2  # noqa: E402

N_PER_BRANCH, SEED = 200, 42


def pick_branch(cands, n, rng, used):
    keys = {"source": lambda s: s["source"], "voice": lambda s: s["voice"],
            "split": lambda s: split_group(s["split"]), "template": lambda s: s["template_id"]}
    share = {k: {v: 1 / len({f(s) for s in cands}) for v in {f(s) for s in cands}} for k, f in keys.items()}
    cnt = {k: Counter() for k in keys}
    pool, picked = rng.sample(cands, len(cands)), []
    while len(picked) < n:
        best, best_score = None, None
        for s in pool:
            if s["image"] in used:
                continue
            sc = sum(cnt[k][f(s)] / share[k][f(s)] for k, f in keys.items())
            if best is None or sc < best_score:
                best, best_score = s, sc
        if best is None:
            break
        picked.append(best)
        used.add(best["image"])
        for k, f in keys.items():
            cnt[k][f(best)] += 1
    return picked


def main():
    data = load_dataset()
    negs = [s for s in run_h1v2.one_per_unit(data, SEED) if s["asr_ok"]]
    forms = defaultdict(list)
    for s in data.values():
        if s["form"] != "negative":
            forms[s["pair_of"]].append(s["form"])
    # H4: keep only units that have every form their branch supports (existence has no affirmative form by design),
    # so the same 1,000 questions can be used for H1 and H4 without holes
    per_branch = defaultdict(Counter)
    for s in negs:
        per_branch[s["branch"]].update(set(forms.get(s["id"], [])))
    need = {b: {f for f, n in c.items() if n > 0.5 * sum(1 for s in negs if s["branch"] == b)} for b, c in per_branch.items()}
    negs = [s for s in negs if need[s["branch"]] <= set(forms.get(s["id"], []))]
    rng = random.Random(SEED)
    chosen, used = [], set()   # images distinct over the whole pilot, not only inside a branch
    for b in BRANCHES:
        chosen += pick_branch([s for s in negs if s["branch"] == b], N_PER_BRANCH, rng, used)
    rows = [{"sample_id": s["id"], "unit_id": s["unit_id"], "image_id": s["image_id"], "branch": s["branch"],
             "source": s["source"], "split": split_group(s["split"]), "voice": s["voice"],
             "template_id": s["template_id"], "occludes_subject": s["occludes_subject"],
             "forms": sorted(forms.get(s["id"], []))} for s in chosen]
    assert len({r["sample_id"] for r in rows}) == len({r["image_id"] for r in rows}) == len(rows)
    (HERE / "pilot1000.json").write_text(json.dumps({"seed": SEED, "n_per_branch": N_PER_BRANCH, "samples": rows}, indent=1))
    lines = [f"# pilot1000 ({len(rows)} questions, seed {SEED})", "",
             "Fixed list in `pilot1000.json` (sample_ids). One negative sample per unit, asr_ok, distinct images over the whole list, excluded units dropped.", ""]
    for name in ("branch", "source", "voice", "split", "template_id", "occludes_subject"):
        c = Counter(r[name] for r in rows)
        lines += [f"**{name}**: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda x: str(x[0]))), ""]
    lines += ["**branch x source**: " + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(Counter((r['branch'], r['source']) for r in rows).items())), "",
              "**branch x voice**: " + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(Counter((r['branch'], r['voice']) for r in rows).items())), "",
              "**branch x split**: " + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(Counter((r['branch'], r['split']) for r in rows).items())), "",
              f"**H4 forms present per branch**: " + ", ".join(f"{k[0]}/{'+'.join(k[1])}={v}" for k, v in sorted(Counter((r['branch'], tuple(r['forms'])) for r in rows).items()))]
    (HERE / "pilot1000_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


main()
