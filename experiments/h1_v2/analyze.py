"""Analysis of H1 v2 (+ H4) from results/<model>/scored_<tag>.jsonl.

    python analyze.py --tag pilot40            # every model with that tag
    python analyze.py --tag all --models qwen2.5-omni-7b qwen2.5-omni-3b

Definitions (per model):
  primary sample    one negative sample per unit: the non-sweep sample, or N1 for template-sweep units (the same
                    sample the exceptive / affirmative forms are paired with). Tests use primary samples only;
                    template tables use every sample.
  C0-mentioned      primary samples whose C0 answer states the target (there is something to suppress).
  violation rate    P(target stated | condition), among C0-mentioned.
  F_form,m          1 - violation rate of that form in channel m (T_sep = text, A = audio, V = visual).
  Delta_m (H4)      F_aff,m - F_neg,m on units that have an affirmative form (all but open existence).
  McNemar           exact two-sided, on paired outcomes (target stated yes/no) of the same samples.
  Bootstrap CI      95% percentile interval, 2000 resamples of IMAGES (all samples of an image move together).

Writes analysis_<tag>.md next to this script and figures into experiments/figures/.
"""
import argparse
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE.parent / "figures"
BRANCHES = ["existence", "existence_closed", "count", "attribute", "relation"]
NEG_CONDS = ["T_join", "T_sep", "A", "V"]
CHANNELS = {"T_sep": "text", "A": "audio", "V": "visual"}
PAIRS = [("T_join", "T_sep"), ("T_sep", "A"), ("T_sep", "V"), ("A", "V")]
N_BOOT = 2000


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


class ModelData:
    def __init__(self, rows):
        self.rows = rows
        self.by = {(r["sample_id"], r["condition"]): r for r in rows}
        neg = [r for r in rows if r["form"] == "negative" and r["condition"] == "C0"]
        self.neg = {r["sample_id"]: r for r in neg}
        paired = {r["pair_of"] for r in rows if r["form"] != "negative"}
        self.primary = [sid for sid, r in self.neg.items()
                        if not r["template_sweep"] or r["template_id"] == "N1" or sid in paired]
        # one primary per unit
        seen, prim = set(), []
        for sid in sorted(self.primary, key=lambda s: (s not in paired, s)):
            u = self.neg[sid]["unit_id"]
            if u not in seen:
                seen.add(u)
                prim.append(sid)
        self.primary = prim
        self.c0m = [sid for sid in self.primary if self.neg[sid]["target_mentioned"]]
        self.forms = defaultdict(dict)  # pair_of -> form -> sample_id
        for r in rows:
            if r["form"] != "negative":
                self.forms[r["pair_of"]][r["form"]] = r["sample_id"]
        self.image = {sid: r.get("image", r["unit_id"]) for sid, r in self.neg.items()}

    def outcome(self, sid, cond, form="negative", field="target_mentioned"):
        if form != "negative":
            sid = self.forms.get(sid, {}).get(form)
            if sid is None:
                return None
        r = self.by.get((sid, cond))
        return None if r is None else r[field]

    def rate(self, sids, cond, form="negative", field="target_mentioned"):
        vals = [self.outcome(s, cond, form, field) for s in sids]
        vals = [v for v in vals if v is not None]
        return (mean(vals), len(vals)) if vals else (None, 0)


def bootstrap(sids, group, stat, rng):
    """95% CI of stat(list of sids) resampling groups."""
    groups = defaultdict(list)
    for s in sids:
        groups[group[s]].append(s)
    keys = list(groups)
    vals = []
    for _ in range(N_BOOT):
        pick = [s for k in rng.choices(keys, k=len(keys)) for s in groups[k]]
        v = stat(pick)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def diff_stat(md, a, b, form_a="negative", form_b="negative"):
    def f(sids):
        pa = [(md.outcome(s, a, form_a), md.outcome(s, b, form_b)) for s in sids]
        pa = [(x, y) for x, y in pa if x is not None and y is not None]
        return mean(x - y for x, y in pa) if pa else None
    return f


def section_model(name, md, out, rng, figs):
    w = out.append
    w(f"\n## {name}\n")
    n_units = len(md.primary)
    w(f"Primary negative samples (one per unit): {n_units}; C0 states the target in {len(md.c0m)} "
      f"({100 * len(md.c0m) / max(1, n_units):.1f}%).\n")
    # C0 vs C0_box: does the empty box change what is said?
    a = sum(1 for s in md.primary if md.outcome(s, "C0") and md.outcome(s, "C0_box") is False)
    b = sum(1 for s in md.primary if md.outcome(s, "C0") is False and md.outcome(s, "C0_box"))
    r0, _ = md.rate(md.primary, "C0")
    rb, nb = md.rate(md.primary, "C0_box")
    w(f"Target stated without instruction: C0 {pct(r0)}%, C0_box {pct(rb)}% (n={nb}; discordant {a} vs {b}, "
      f"McNemar p={mcnemar(a, b):.3g}).\n")

    w("\n### Violation rate (%) among C0-mentioned, negative form\n")
    w("| branch | n | " + " | ".join(NEG_CONDS) + " | C0_box |")
    w("|---|---|" + "---|" * (len(NEG_CONDS) + 1))
    for br in BRANCHES + ["ALL"]:
        sids = [s for s in md.c0m if br == "ALL" or md.neg[s]["branch"] == br]
        if not sids:
            continue
        cells = [pct(md.rate(sids, c)[0]) for c in NEG_CONDS + ["C0_box"]]
        w(f"| {br} | {len(sids)} | " + " | ".join(cells) + " |")

    w("\n### Paired comparisons (C0-mentioned, negative form)\n")
    w("| pair | violation A | violation B | A-B (pp) | 95% CI (image bootstrap) | only A violates | only B "
      "violates | McNemar p |")
    w("|---|---|---|---|---|---|---|---|")
    for x, y in PAIRS:
        sids = [s for s in md.c0m if md.outcome(s, x) is not None and md.outcome(s, y) is not None]
        if not sids:
            continue
        ox = sum(1 for s in sids if md.outcome(s, x) and not md.outcome(s, y))
        oy = sum(1 for s in sids if md.outcome(s, y) and not md.outcome(s, x))
        d = diff_stat(md, x, y)(sids)
        lo, hi = bootstrap(sids, md.image, diff_stat(md, x, y), rng)
        w(f"| {x} vs {y} | {pct(md.rate(sids, x)[0])} | {pct(md.rate(sids, y)[0])} | {100 * d:+.1f} | "
          f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {ox} | {oy} | {mcnemar(ox, oy):.3g} |")

    w("\n### Required recall (over-suppression), C0-mentioned primary samples\n")
    w("| branch | C0 | " + " | ".join(NEG_CONDS) + " |")
    w("|---|---|" + "---|" * len(NEG_CONDS))
    for br in BRANCHES + ["ALL"]:
        sids = [s for s in md.c0m if br == "ALL" or md.neg[s]["branch"] == br]
        if sids:
            w(f"| {br} | " + " | ".join(pct(md.rate(sids, c, field="required_recall")[0])
                                        for c in ["C0"] + NEG_CONDS) + " |")

    def by_field(title, field, sids_all, conds=NEG_CONDS):
        w(f"\n### Violation rate (%) by {title}\n")
        w(f"| {title} | n | " + " | ".join(conds) + " |")
        w("|---|---|" + "---|" * len(conds))
        groups = defaultdict(list)
        for s in sids_all:
            groups[str(md.neg[s][field])].append(s)
        for g in sorted(groups):
            w(f"| {g} | {len(groups[g])} | " + " | ".join(pct(md.rate(groups[g], c)[0]) for c in conds) + " |")

    by_field("split (contamination check)", "split_group", md.c0m)
    by_field("source", "source", md.c0m)
    allc0 = [s for s, r in md.neg.items() if r["target_mentioned"]]
    by_field("template (every sample, sweep copies included)", "template_id", allc0)
    by_field("voice", "voice", md.c0m, ["A"])
    by_field("asr_ok", "asr_ok", md.c0m, ["A"])
    by_field("occludes_subject", "occludes_subject", md.c0m, ["V", "C0_box"])

    # H4
    w("\n### H4: forms (C0-mentioned units that have the form)\n")
    w("| form | channel | n | violation % | F = 1 - violation |")
    w("|---|---|---|---|---|")
    for form in ("negative", "exceptive", "affirmative"):
        sids = [s for s in md.c0m if form == "negative" or form in md.forms.get(s, {})]
        for c, ch in CHANNELS.items():
            r, n = md.rate(sids, c, form)
            if n:
                w(f"| {form} | {ch} | {n} | {pct(r)} | {pct(1 - r)} |")
    w("\nDelta_m = F_aff,m - F_neg,m (and the exceptive analogue), same units; 95% image-bootstrap CI.\n")
    w("| contrast | text | audio | visual | audio - text | visual - text |")
    w("|---|---|---|---|---|---|")
    for form in ("affirmative", "exceptive"):
        sids = [s for s in md.c0m if form in md.forms.get(s, {})]
        if not sids:
            continue

        def delta(ch_cond):
            # F_form - F_neg = violation_neg - violation_form
            return diff_stat(md, ch_cond, ch_cond, "negative", form)

        def dd(c1, c2):
            def f(ss):
                a1, a2 = delta(c1)(ss), delta(c2)(ss)
                return None if a1 is None or a2 is None else a1 - a2
            return f
        cells = []
        for c in CHANNELS:
            v = delta(c)(sids)
            lo, hi = bootstrap(sids, md.image, delta(c), rng)
            cells.append(f"{100 * v:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" if v is not None else "-")
        for c in ("A", "V"):
            v = dd(c, "T_sep")(sids)
            lo, hi = bootstrap(sids, md.image, dd(c, "T_sep"), rng)
            cells.append(f"{100 * v:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" if v is not None else "-")
        w(f"| Delta ({form[:3]} - neg), n={len(sids)} | " + " | ".join(cells) + " |")
    figs[name] = md


def plots(figs, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    names = list(figs)
    conds = NEG_CONDS
    fig, ax = plt.subplots(figsize=(1.6 + 1.9 * len(names), 3.4))
    width = 0.8 / len(conds)
    colors = ["#9aa5b1", "#3b6ea8", "#d1853a", "#4f9a6a"]
    for j, c in enumerate(conds):
        ys = [100 * (figs[n].rate(figs[n].c0m, c)[0] or 0) for n in names]
        ax.bar([i + (j - (len(conds) - 1) / 2) * width for i in range(len(names))], ys, width, label=c,
               color=colors[j])
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("violation rate (%)\namong C0-mentioned")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p1 = FIGS / f"h1v2_violation_{tag}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    # forms x channel
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.2), squeeze=False)
    for ax, n in zip(axes[0], names):
        md = figs[n]
        for k, form in enumerate(("negative", "exceptive", "affirmative")):
            sids = [s for s in md.c0m if form == "negative" or form in md.forms.get(s, {})]
            ys = [100 * (md.rate(sids, c, form)[0] or 0) for c in CHANNELS]
            ax.plot(list(CHANNELS.values()), ys, marker="o", label=form, color=colors[k + 1])
        ax.set_title(n, fontsize=9)
        ax.set_ylabel("violation rate (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p2 = FIGS / f"h1v2_forms_{tag}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return [p1, p2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    files = sorted(RESULTS.glob(f"*/scored_{args.tag}.jsonl"))
    if args.models:
        files = [f for f in files if f.parent.name in args.models]
    rng = random.Random(args.seed)
    out = [f"# H1 v2 analysis ({args.tag})", "", "Generated by experiments/h1_v2/analyze.py; definitions in its "
           "docstring. Violation = the forbidden component is stated (LLM judge, regex second opinion)."]
    figs = {}
    for f in files:
        section_model(f.parent.name, ModelData(read_jsonl(f)), out, rng, figs)
    if figs:
        for p in plots(figs, args.tag):
            out.append(f"\n![{p.stem}](../figures/{p.name})")
    path = HERE / f"analysis_{args.tag}.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
