"""Analysis of H1 v2 (+ H4) from results/<model>/scored_<tag>.jsonl.

    python analyze.py --tag pilot40            # every model with that tag
    python analyze.py --tag all --models qwen2.5-omni-7b qwen2.5-omni-3b

Definitions (per model):
  primary sample    one negative sample per unit: the non-sweep sample, or N1 for template-sweep units (the same
                    sample the exceptive / affirmative forms are paired with). Tests use primary samples only;
                    template tables use every sample.
  C0-mentioned      primary samples whose C0 answer states the target (there is something to suppress).
  violation rate    P(target stated | condition), among C0-mentioned.
  F_form,m          1 - violation rate of that form in channel m (T = text, A = audio, V = visual).
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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from omnilib import removal_set, removed   # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE.parent / "figures"
BRANCHES = ["existence", "existence_closed", "count", "attribute", "relation"]
ALL_CONDS = ["T", "A", "V", "T_join"]   # those present in the scored file are used
NEG_CONDS = ALL_CONDS
CHANNELS = {"T": "text", "A": "audio", "V": "visual"}
PAIRS = [("T", "A"), ("T", "V"), ("A", "V"), ("T_join", "T")]
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
        # PRIMARY metric (Vy, 2026-11-23): violation = ANY mention of the forbidden thing (STATE / META / DENY).
        # Rows scored before that have only the judge's verdict on the correct fact (= "leak") and are used as is.
        for r in rows:
            if "violation" in r:
                r["leak"] = r.get("leak", r["target_mentioned"])
                r["target_mentioned"] = r["violation"]
        self.primary_metric = "any mention (STATE/META/DENY)" if any("violation" in r for r in rows) \
            else "judge: correct fact stated"
        self.rows = rows
        self.by = {(r["sample_id"], r["condition"]): r for r in rows}
        neg = [r for r in rows if r["form"] == "negative" and r["condition"] == "C0"]
        self.neg = {r["sample_id"]: r for r in neg}
        paired = {r["pair_of"] for r in rows if r["form"] != "negative"}
        # one primary per unit: the sample the forms are paired with if any, else the (single) negative run
        seen, prim = set(), []
        for sid in sorted(self.neg, key=lambda s: (s not in paired, s)):
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


def diff_stat(md, a, b, form_a="negative", form_b="negative", field="target_mentioned"):
    def f(sids):
        pa = [(md.outcome(s, a, form_a, field), md.outcome(s, b, form_b, field)) for s in sids]
        pa = [(x, y) for x, y in pa if x is not None and y is not None]
        return mean(x - y for x, y in pa) if pa else None
    return f


def section_model(name, md, out, rng, figs):
    global NEG_CONDS
    present = {r["condition"] for r in md.rows}
    NEG_CONDS = [c for c in ALL_CONDS if c in present]
    box = ["C0_box"] if "C0_box" in present else []
    w = out.append
    w(f"\n## {name}\n")
    w(f"Violation = {md.primary_metric}.\n")
    n_units = len(md.primary)
    w(f"Primary negative samples (one per unit): {n_units}; C0 states the target in {len(md.c0m)} "
      f"({100 * len(md.c0m) / max(1, n_units):.1f}%).\n")
    # C0 vs C0_box: does the empty box change what is said?
    if not box:
        w(f"Target stated without instruction (C0): {pct(md.rate(md.primary, 'C0')[0])}%.\n")
    a = sum(1 for s in md.primary if md.outcome(s, "C0") and md.outcome(s, "C0_box") is False)
    b = sum(1 for s in md.primary if md.outcome(s, "C0") is False and md.outcome(s, "C0_box"))
    r0, _ = md.rate(md.primary, "C0")
    rb, nb = md.rate(md.primary, "C0_box")
    if box:
        w(f"Target stated without instruction: C0 {pct(r0)}%, C0_box {pct(rb)}% (n={nb}; discordant {a} vs {b}, "
      f"McNemar p={mcnemar(a, b):.3g}).\n")

    w("\n### Violation rate (%) among C0-mentioned, negative form\n")
    w("| branch | n | " + " | ".join(NEG_CONDS + box) + " |")
    w("|---|---|" + "---|" * (len(NEG_CONDS) + len(box)))
    for br in BRANCHES + ["ALL"]:
        sids = [s for s in md.c0m if br == "ALL" or md.neg[s]["branch"] == br]
        if not sids:
            continue
        cells = [pct(md.rate(sids, c)[0]) for c in NEG_CONDS + box]
        w(f"| {br} | {len(sids)} | " + " | ".join(cells) + " |")

    w("\n### Paired comparisons (C0-mentioned, negative form)\n")
    w("| pair | violation A | violation B | A-B (pp) | 95% CI (image bootstrap) | only A violates | only B "
      "violates | McNemar p |")
    w("|---|---|---|---|---|---|---|---|")
    for x, y in [p for p in PAIRS if p[0] in NEG_CONDS and p[1] in NEG_CONDS]:
        sids = [s for s in md.c0m if md.outcome(s, x) is not None and md.outcome(s, y) is not None]
        if not sids:
            continue
        ox = sum(1 for s in sids if md.outcome(s, x) and not md.outcome(s, y))
        oy = sum(1 for s in sids if md.outcome(s, y) and not md.outcome(s, x))
        d = diff_stat(md, x, y)(sids)
        lo, hi = bootstrap(sids, md.image, diff_stat(md, x, y), rng)
        w(f"| {x} vs {y} | {pct(md.rate(sids, x)[0])} | {pct(md.rate(sids, y)[0])} | {100 * d:+.1f} | "
          f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {ox} | {oy} | {mcnemar(ox, oy):.3g} |")

    # robustness (Vy, 2026-11-23): N3 "Never refer to ..." is kept in every main number and reported here:
    # the same paired comparisons with and without N3 questions, plus N3 alone
    tid = lambda sid: md.neg[sid].get("template_id")
    w("\n### Template N3 (\"Never refer to ...\"): paired comparisons with / without it\n")
    w("| pair | subset | n | A-B (pp) | 95% CI (image bootstrap) |")
    w("|---|---|---|---|---|")
    for x, y in [p for p in PAIRS if p[0] in NEG_CONDS and p[1] in NEG_CONDS]:
        for label, keep in (("all", lambda sid: True), ("without N3", lambda sid: tid(sid) != "N3"),
                            ("N3 only", lambda sid: tid(sid) == "N3")):
            sids = [s for s in md.c0m if keep(s) and md.outcome(s, x) is not None and md.outcome(s, y) is not None]
            if len(sids) >= 10:
                st = diff_stat(md, x, y)
                lo, hi = bootstrap(sids, md.image, st, rng)
                w(f"| {x} vs {y} | {label} | {len(sids)} | {100 * st(sids):+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] |")

    if any("leak" in r for r in md.rows):
        # breakdown of the primary violation by kind (Vy, 2026-11-23); the four kinds add up to the violation rate
        kinds = ["STATE", "META", "DENY", "WRONG"]
        share = lambda rs, lab: pct(sum(r.get("target_label") == lab for r in rs) / len(rs)) if rs else "-"
        w("\n### Violation by kind (% of C0-mentioned; the kinds add up to the violation), negative form\n")
        w("STATE = says the forbidden fact · META = names it to exclude it · DENY = says it is absent / false · "
          "WRONG = gives a different value.\n")
        w("| condition | violation | " + " | ".join(kinds) + " | leak (= STATE) |")
        w("|---|---|" + "---|" * (len(kinds) + 1))
        for c in NEG_CONDS:
            rs = [r for r in (md.by.get((s, c)) for s in md.c0m) if r is not None and "target_label" in r]
            if rs:
                w(f"| {c} | {pct(md.rate(md.c0m, c)[0])} | " + " | ".join(share(rs, k) for k in kinds)
                  + f" | {pct(md.rate(md.c0m, c, field='leak')[0])} |")
        w("\nBy branch (STATE / META / DENY / WRONG, %):\n")
        w("| branch | n | " + " | ".join(NEG_CONDS) + " |")
        w("|---|---|" + "---|" * len(NEG_CONDS))
        for br in BRANCHES:
            sids = [s for s in md.c0m if md.neg[s]["branch"] == br]
            if sids:
                cells = []
                for c in NEG_CONDS:
                    rs = [r for r in (md.by.get((s, c)) for s in sids) if r is not None and "target_label" in r]
                    cells.append(" / ".join(share(rs, k) for k in kinds))
                w(f"| {br} | {len(sids)} | " + " | ".join(cells) + " |")
        w("\nPaired differences by kind (A-B, pp, 95% CI image bootstrap):\n")
        w("| pair | " + " | ".join(kinds + ["leak"]) + " |")
        w("|---|" + "---|" * (len(kinds) + 1))
        for r0 in md.rows:
            if "target_label" in r0:
                for k in kinds:
                    r0[f"is_{k}"] = r0["target_label"] == k
        for x, y in [p for p in PAIRS if p[0] in NEG_CONDS and p[1] in NEG_CONDS]:
            sids = [s for s in md.c0m if md.outcome(s, x) is not None and md.outcome(s, y) is not None]
            cells = []
            for f in [f"is_{k}" for k in kinds] + ["leak"]:
                st = diff_stat(md, x, y, field=f)
                lo, hi = bootstrap(sids, md.image, st, rng)
                cells.append(f"{100 * st(sids):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
            w(f"| {x} vs {y} | " + " | ".join(cells) + " |")
        flagged = sum(r.get("judge_source") == "think_error" for r in md.rows)
        amb = sum(r.get("count_rule") == "AMBIG" for r in md.rows)
        w(f"\nFlagged for a human check: thinking judge gave no JSON {flagged}; ambiguous bare-number counts {amb}.\n")

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
    by_field("occludes_subject", "occludes_subject", md.c0m, ["V"] + box)

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
            v = dd(c, "T")(sids)
            lo, hi = bootstrap(sids, md.image, dd(c, "T"), rng)
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
    ap.add_argument("--asr_ok_only", action="store_true",
                    help="main analysis (Vy, 2026-11-24): leave out units whose spoken instruction failed the Whisper "
                         "check (asr_ok False); run once without it as the sensitivity check; output gets _asrok")
    ap.add_argument("--remove", nargs="*", default=[],
                    help="optional removal lists (image_id / unit_id / sample_id; jsonl or one id per line): rows of "
                         "those images/units/samples are left out at analysis time, nothing regenerated; output gets _rm")
    args = ap.parse_args()
    rm = removal_set(args.remove)
    files = sorted(RESULTS.glob(f"*/scored_{args.tag}.jsonl"))
    if args.models:
        files = [f for f in files if f.parent.name in args.models]
    rng = random.Random(args.seed)
    out = [f"# H1 v2 analysis ({args.tag})", "", "Generated by experiments/h1_v2/analyze.py; definitions in its "
           "docstring. Violation = the forbidden component is stated (LLM judge, regex second opinion)."]
    figs = {}
    excl = HERE.parents[1] / "create_data" / "dataset" / "out" / "excluded_units.jsonl"
    dropped = {r["unit_id"] for r in read_jsonl(excl)} if excl.exists() else set()   # flag_excluded_units.py
    if dropped:
        out.append(f"\nExcluded units left out: {len(dropped)} (double label / part-whole / attribute overlap, create_data/src/flag_excluded_units.py).")
    if args.asr_ok_only:
        out.append("\nONLY units with asr_ok = True (every audio instruction of the unit passed the Whisper check).")
    if args.remove:
        out.append(f"\nRemoval lists {args.remove}: {len(rm[0])} images, {len(rm[1])} unit/sample ids left out.")
    for f in files:
        rows = read_jsonl(f)
        # merge extra runs of the same tag (H4 forms, C0_box controls): only (sample_id, condition) pairs that the
        # main file does not have, so the main H1 numbers never change. sepv1 = old history, skipped.
        have = {(r["sample_id"], r["condition"]) for r in rows}
        for g in sorted(f.parent.glob(f"scored_{args.tag}_*.jsonl")):
            if "sepv1" in g.name:
                continue
            extra = [r for r in read_jsonl(g) if (r["sample_id"], r["condition"]) not in have]
            have |= {(r["sample_id"], r["condition"]) for r in extra}
            rows += extra
            if extra:
                print(f"{f.parent.name}: +{len(extra)} rows from {g.name}")
        rows = [r for r in rows if r["unit_id"] not in dropped and not removed(r, rm)]
        if args.asr_ok_only:
            bad = {r["unit_id"] for r in rows if not r["asr_ok"]}
            out.append(f"\n{f.parent.name}: {len(bad)} units left out because of asr_ok = False")
            rows = [r for r in rows if r["unit_id"] not in bad]
        section_model(f.parent.name, ModelData(rows), out, rng, figs)
    if figs:
        for p in plots(figs, args.tag + ("_asrok" if args.asr_ok_only else "") + ("_rm" if args.remove else "")):
            out.append(f"\n![{p.stem}](../figures/{p.name})")
    path = HERE / f"analysis_{args.tag}{'_asrok' if args.asr_ok_only else ''}{'_rm' if args.remove else ''}.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
