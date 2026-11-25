"""run_h1v2.py on the FIXED pilot list (experiments/h1_v2/pilot1000.json) without editing run_h1v2.py.

Same arguments as run_h1v2.py (--model, --forms, --conds, --generate_only, --score_only, --batch_size ...). It only
replaces run_h1v2.select() by the list of sample ids, so the answer cache (answers.jsonl), the judge cache, the scoring
and the output files (tag pilot1000[_forms][_<conds>]) are exactly those of run_h1v2. A pilot already run here and a
later --all run share every answer, so extending to full only runs the missing questions.

    python run_pilot.py --model qwen2.5-omni-7b --forms                       # H4 on the 1,000 questions
    python run_pilot.py --model qwen2.5-omni-7b --conds COMP_T COMP_A COMP_V  # H3-B
    python run_pilot.py --model qwen2.5-omni-7b --conds C0_box OCR_V          # controls
    python run_pilot.py --model minicpm-o-2.6 --forms --generate_only         # then --score_only in env `negation`
`--pilot N` / `--all` are not needed (and `--all` is refused); the list file can be changed with PILOT_LIST=...
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_h1v2 as R  # noqa: E402

LIST = Path(os.environ.get("PILOT_LIST", HERE / "pilot1000.json"))
IDS = [r["sample_id"] for r in json.loads(LIST.read_text())["samples"]]


def select_list(data, pilot, seed):
    if pilot is None:
        raise SystemExit("run_pilot.py runs the fixed list only; use run_h1v2.py --all for the full set")
    missing = [i for i in IDS if i not in data]
    if missing:
        raise SystemExit(f"{len(missing)} ids of {LIST} are not in the dataset (first: {missing[0]})")
    return [data[i] for i in IDS]


if __name__ == "__main__":
    if "--all" in sys.argv:
        raise SystemExit("run_pilot.py: --all is not allowed")
    if "--pilot" not in sys.argv:
        sys.argv += ["--pilot", str(len(IDS))]   # only names the output tag: pilot1000
    R.select = select_list
    R.main()
