"""Prohibition direction from the TEXT channel, TRAIN images of pilot1000 only.

For every TRAIN question s: two T-like inputs, same image and question, instruction as its own text item:
    neg  "Do not mention X."      pos  "Mention X."          (X = the slot of the sample's own instruction)
Per decoder layer L: mean hidden state over the instruction tokens (layer output), then
    d_L = mean_s h_neg(s, L) - mean_s h_pos(s, L)
Saved to steer/results/<model>/direction_<contrast>.pt with: dirs {L: (H,)}, per-layer mean token norm, n, the
train image ids. --check_val: how well the projection on d_L separates neg from pos on the VAL questions (accuracy
of sign(proj - midpoint)); a layer-choice diagnostic that does not look at violations.

    python direction.py --model qwen2.5-omni-3b [--contrast mention|own] [--batch_size 8] [--check_val]
"""
import argparse
import json

from steerlib import DATA, RESULTS, SteerModel, contrast_pair, load_pilot


def means(m, samples, which, contrast, layers, bs):
    """-> {L: tensor (N, H)} mean over the instruction span for neg (which=0) or pos (which=1) inputs."""
    import torch
    acc = {L: [] for L in layers}
    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        texts = [contrast_pair(s, contrast)[which] for s in chunk]
        jobs = [(DATA / s["image"], [("text", s["question"]), ("text", t)]) for s, t in zip(chunk, texts)]
        got = m.span_means(jobs, texts, layers)
        for L in layers:
            acc[L].append(got[L])
        if (i // bs) % 20 == 0:
            print(f"  {'neg' if which == 0 else 'pos'} {i + len(chunk)}/{len(samples)}", flush=True)
    return {L: torch.cat(v) for L, v in acc.items()}


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen2.5-omni-3b", "qwen2.5-omni-7b"])
    ap.add_argument("--contrast", default="mention", choices=["mention", "own"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--check_val", action="store_true")
    ap.add_argument("--limit", type=int, help="first N train questions only (smoke)")
    args = ap.parse_args()
    train, val = load_pilot()
    if args.limit:
        train, val = train[:args.limit], val[:args.limit]
    out = RESULTS / args.model / f"direction_{args.contrast}{'_lim' + str(args.limit) if args.limit else ''}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    m = SteerModel(args.model)
    layers = list(range(len(m.layers)))
    print(f"{args.model}: {len(train)} train questions, {len(layers)} layers, contrast {args.contrast}", flush=True)
    neg, pos = means(m, train, 0, args.contrast, layers, args.batch_size), means(m, train, 1, args.contrast, layers, args.batch_size)
    dirs = {L: neg[L].mean(0) - pos[L].mean(0) for L in layers}
    info = {"model": args.model, "contrast": args.contrast, "n_train": len(train),
            "train_image_ids": sorted({s["image_id"] for s in train}),
            "dir_norm": {L: float(dirs[L].norm()) for L in layers},
            "token_norm": {L: float(torch.cat([neg[L], pos[L]]).norm(dim=1).mean()) for L in layers}}
    if args.check_val:
        vn, vp = means(m, val, 0, args.contrast, layers, args.batch_size), means(m, val, 1, args.contrast, layers, args.batch_size)
        acc = {}
        for L in layers:
            mid = (neg[L].mean(0) + pos[L].mean(0)) / 2
            pn, pp = (vn[L] - mid) @ dirs[L], (vp[L] - mid) @ dirs[L]
            acc[L] = float(((pn > 0).float().mean() + (pp < 0).float().mean()) / 2)
        info["val_sep_acc"] = acc
        info["n_val"] = len(val)
    m.close()
    torch.save({"dirs": dirs, "info": info}, out)
    print(f"-> {out}")
    for L in layers:
        print(f"L{L:>2}  |d| {info['dir_norm'][L]:8.2f}  |h| {info['token_norm'][L]:8.2f}"
              + (f"  val sep acc {info['val_sep_acc'][L]:.3f}" if args.check_val else ""))
    (out.with_suffix(".json")).write_text(json.dumps({k: v for k, v in info.items() if k != "train_image_ids"}, indent=1))


if __name__ == "__main__":
    main()
