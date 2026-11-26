"""Share of answers regenerated at 256 tokens, per model x condition tag (answers.jsonl cache; tag = condition)."""
import json, collections
from pathlib import Path
RES = Path(__file__).resolve().parent.parent / "results"
CONDS = ["C0", "T", "A", "V", "C0_box"]
print("| model | " + " | ".join(CONDS) + " |\n|---|" + "---|" * len(CONDS))
for m in ["qwen2.5-omni-7b", "qwen2.5-omni-3b", "minicpm-o-2.6", "phi-4-mm", "internomni"]:
    tot, reg = collections.Counter(), collections.Counter()
    for l in open(RES / m / "answers.jsonl"):
        r = json.loads(l); t = r.get("tag")
        if str(r.get("max_new_tokens")) == "100": tot[t] += 1
        else: reg[t] += 1
    print(f"| {m} | " + " | ".join(f"{100*reg[c]/max(tot[c],1):.1f}% ({reg[c]}/{tot[c]})" for c in CONDS) + " |")
