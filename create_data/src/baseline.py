"""NOT PART OF THE PIPELINE since 2026-11-19 (kept for reference / optional use). Formerly step 4: baseline run of every question WITHOUT negation on the reference models.

Runs per unit, before B3, so B3 can pick its target only among components the reference models really state
(v2; v1 ran after B3 and dropped samples whose random target nobody mentioned).

- Every reference model answers the original question; answers are cached per model.
- Which components each answer states: the LLM judge (judge.py) decides, the regex matcher (mention.py) is kept
  as a second opinion and is the fallback when the judge reply cannot be parsed.
- A component is an eligible target when EVERY reference model states it and, for every model, at least
  BASELINE_MIN_REQUIRED of the unit's other components are stated too (so the model clearly sees the image and
  there is something left to say after the negation). Units with no eligible target are dropped.
- An alias report lets missing / misleading aliases be fixed by hand.

Output: dataset/b2/units_baseline.jsonl (B2 units + "baseline" and "eligible_targets"), dataset/out/alias_report.csv
"""
import csv
import json
from collections import Counter

from config import B2_DIR, BASELINE_MIN_REQUIRED, BASELINE_MODELS, DATA, OUT_DIR, ROOT
from judge import Judge
from mention import mentions
from utils import read_jsonl, write_jsonl

CACHE = B2_DIR / "baseline_cache"


def run_models(units):
    """units: [unit] -> {model: {unit_id: answer}}; cached on disk per model."""
    from vlm import LocalVLM
    CACHE.mkdir(parents=True, exist_ok=True)
    answers = {}
    for name in BASELINE_MODELS:
        path = CACHE / (name.replace("/", "__") + ".json")
        cached = json.loads(path.read_text()) if path.exists() else {}
        key = {u["unit_id"]: f"{u['image']}||{u['question']}" for u in units}
        todo = [u for u in units if cached.get(u["unit_id"], {}).get("key") != key[u["unit_id"]]]
        if todo:
            vlm = LocalVLM(name)
            for k, u in enumerate(todo, 1):
                cached[u["unit_id"]] = {"key": key[u["unit_id"]], "answer": vlm.answer(DATA / u["image"], u["question"])}
                if k % 50 == 0:
                    print(f"{name}: {k}/{len(todo)}", flush=True)
                    path.write_text(json.dumps(cached, ensure_ascii=False, indent=1))
            vlm.close()
            path.write_text(json.dumps(cached, ensure_ascii=False, indent=1))
        answers[name] = {uid: v["answer"] for uid, v in cached.items()}
    return answers


def main():
    units = read_jsonl(B2_DIR / "units.jsonl")
    answers = run_models(units)
    jobs = [(u, m) for u in units for m in BASELINE_MODELS]
    regex = [{c["id"]: mentions(answers[m][u["unit_id"]], c, [o for o in u["components"] if o is not c])
              for c in u["components"]} for u, m in jobs]
    judge = Judge(CACHE / "judge.jsonl")
    verdicts = judge.judge([(u["components"], answers[m][u["unit_id"]]) for u, m in jobs], regex=regex)
    print("judge passes:", dict(Counter(judge.last_source)))
    judge.close()

    kept, stats, alias_rows = [], Counter(), []
    per = {}
    for (u, m), rg, lm in zip(jobs, regex, verdicts):
        per.setdefault(u["unit_id"], {})[m] = {"answer": answers[m][u["unit_id"]], "mentioned": lm or rg,
                                               "mentioned_llm": lm, "mentioned_regex": rg}
        stats["judge_parse_error"] += lm is None
    for u in units:
        base = per[u["unit_id"]]
        ids = [c["id"] for c in u["components"]]
        eligible = []
        for t in ids:
            others = [i for i in ids if i != t]
            if all(b["mentioned"][t] and sum(b["mentioned"][i] for i in others) / len(others) >= BASELINE_MIN_REQUIRED
                   for b in base.values()):
                eligible.append(t)
        for c in u["components"]:
            for m, b in base.items():
                rg = int(b["mentioned_regex"][c["id"]])
                alias_rows.append([u["unit_id"], u["source"]["dataset"], u["branch"], c["id"], c["text"],
                                   "; ".join(c["aliases"]), m, int(b["mentioned"][c["id"]]), rg,
                                   b["answer"].replace("\n", " ")])
                stats["regex_disagrees_with_llm"] += int(b["mentioned"][c["id"]]) != rg
        if not eligible:
            stats[f"dropped_no_eligible_target/{u['source']['dataset']}-{u['branch']}"] += 1
            continue
        kept.append(dict(u, baseline=base, eligible_targets=eligible))

    write_jsonl(B2_DIR / "units_baseline.jsonl", kept)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "alias_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "dataset", "branch", "component", "text", "aliases", "model", "mentioned",
                    "mentioned_regex", "answer"])
        w.writerows(alias_rows)
    print(f"kept {len(kept)}/{len(units)} units; {dict(stats)}")
    print("kept by branch:", dict(Counter(f"{u['source']['dataset']}-{u['branch']}" for u in kept)))
    print("files:", (B2_DIR / "units_baseline.jsonl").relative_to(ROOT), (OUT_DIR / "alias_report.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
