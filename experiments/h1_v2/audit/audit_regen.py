"""Count answers regenerated at 256 tokens because the 100-token answer was cut (answers.jsonl cache, per model)."""
import json, collections
from pathlib import Path
RES = Path(__file__).resolve().parent.parent / "results"
print("| model | cached answers (100 tok) | regenerated at 256 tok | % |\n|---|---|---|---|")
for m in ["qwen2.5-omni-7b", "qwen2.5-omni-3b", "minicpm-o-2.6", "phi-4-mm", "internomni"]:
    c = collections.Counter()
    for l in open(RES / m / "answers.jsonl"):
        c[str(json.loads(l).get("max_new_tokens"))] += 1
    a, b = c.get("100", 0), sum(v for k, v in c.items() if k != "100")
    print(f"| {m} | {a} | {b} {dict(c) if len(c) > 2 else ''} | {100*b/max(a,1):.1f} |")
