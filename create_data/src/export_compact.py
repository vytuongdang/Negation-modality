"""Write a slim per-sample file for running models; `id` links back to the full record in samples.jsonl."""
from b1_ground_truth import plural
from config import OUT_DIR
from utils import read_jsonl, write_jsonl


def label(c):
    """Short readable key for a component."""
    if c["type"] == "existence":
        return c["subject"]
    if c["type"] == "count":
        return f"{c['value']} {c['subject'] if c['value'] == 1 else plural(c['subject'])}"
    if c["type"] == "attribute":
        return f"{c['slot']}: {c['value']}"
    return f"{c['relation']}: {c['value']}"


def surface_forms(c, max_forms=8):
    """Phrases that count as mentioning the component; count = number next to the object name."""
    if c["type"] != "count":
        return c["aliases"]
    # keep singular aliases only (a word is plural if it is the plural of another alias), then inflect to the count
    names = [c["subject"]] + c.get("subject_aliases", [])
    plurals = {plural(a) for a in names}
    singulars = [a for a in dict.fromkeys(names) if a not in plurals or a == c["subject"]]
    nouns = [a if c["value"] == 1 else plural(a) for a in singulars]
    nums = [a for a in c["aliases"] if a not in ("both", "single")]
    return [f"{n} {s}" for n in nums for s in nouns[:2]][:max_forms]


def compact(s):
    comps = {c["id"]: c for c in s["components"]}
    return {
        "id": s["sample_id"],
        "image": s["image"],
        "question": s["prompt"],
        "negation": s["negation_text"],
        "forbidden": {label(comps[i]): surface_forms(comps[i]) for i in s["expected"]["forbidden"]},
        "required": {label(comps[i]): surface_forms(comps[i]) for i in s["expected"]["required"]},
        "meta": {"branch": s["branch"], "source": s["source"]["dataset"], "template_id": s["negation"]["template_id"]},
    }


def main(src="samples.jsonl", dst="samples_compact.jsonl"):
    rows = [compact(s) for s in read_jsonl(OUT_DIR / src)]
    write_jsonl(OUT_DIR / dst, rows)
    print(f"{len(rows)} compact samples -> {OUT_DIR / dst}")


if __name__ == "__main__":
    main()
