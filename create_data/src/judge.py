"""LLM judge: which of a sample's components does an answer state? Primary scorer for the baseline filter and
for experiments; the regex matcher in mention.py is kept as a second opinion.

    from judge import Judge
    j = Judge(cache_path)                       # model loads lazily, only if something is not cached
    res = j.judge([(components, answer), ...], regex=[{"C1": True, ...}, ...])
    # -> [{"C1": True, "C2": False, ...} or None if the reply cannot be parsed]
    j.last_source                               # per job: "fast", "think" (adjudicated) or "error"
    j.close()

Closed set: the judge only says which of the listed facts the answer states (no free-form grading).
Two passes: a fast pass (no thinking) on every job; where `regex` is given and disagrees with the fast pass on
any component, a second pass with Qwen3 thinking enabled decides (the fast pass misses e.g. "matte" = rubber,
the regex misses paraphrases and hits subject words). Results are cached by (pass, prompt version, facts,
answer) in a JSONL file, so re-running is free.
"""
import hashlib
import json
from pathlib import Path

JUDGE_MODEL = "Qwen/Qwen3-8B"
PROMPT_VERSION = "v2"

PROMPT = """Below is an answer about an image, and a numbered list of facts.
Which facts does the answer state? Rules:
- A fact counts only if the answer itself asserts it. Synonyms and paraphrases are fine
  ("sofa" = "couch", "table" = "dining table", "monitor" = "tv", "big" = "large", "made of wood" = "wooden",
  "matte" = "rubber", "shiny" or "metallic" = "metal").
- Object present: the answer says that object is in the image.
- Count: the answer gives that number for that object; "a fork" or "one fork" states 1 fork.
  A different number for that object does NOT state the fact.
- Color, size, material, shape, texture or other attribute: the answer gives that same value for that object.
- Relation (e.g. "woman is standing on grass"): the answer says that subject has that relation to that object.
  Just naming the object elsewhere, or using its word for the subject, does not count.
- Saying only that it will not talk about something (e.g. "I won't mention the size") states nothing.
- Ignore anything in the answer that is not in the list.

Answer:
\"\"\"{answer}\"\"\"

Facts:
{facts}

Reply with JSON only, e.g. {{"stated": [1, 3]}} or {{"stated": []}}."""


def fact_text(c):
    return f"there is a {c['subject']} in the image" if c["type"] == "existence" else c["text"]


def build_prompt(comps, answer):
    facts = "\n".join(f"{i}. {fact_text(c)}" for i, c in enumerate(comps, 1))
    return PROMPT.format(answer=answer, facts=facts)


def key(comps, answer, thinking=False):
    payload = json.dumps([PROMPT_VERSION, JUDGE_MODEL, [fact_text(c) for c in comps], answer]
                         + (["think"] if thinking else []), ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()


def parse(reply, comps):
    from llm import parse_json
    obj = parse_json(reply)
    try:
        idx = {int(k) for k in obj["stated"]}
    except (TypeError, KeyError, ValueError):
        return None
    return {c["id"]: (i in idx) for i, c in enumerate(comps, 1)}


class Judge:
    def __init__(self, cache_path, batch_size=8):
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.cache = {}
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.cache[r["key"]] = r
        self.llm = None

    def _run(self, jobs, thinking, progress):
        keys = [key(c, a, thinking) for c, a in jobs]
        todo = list({k: (c, a) for k, (c, a) in zip(keys, jobs) if k not in self.cache}.items())
        if todo:
            if self.llm is None:
                from llm import LocalLLM
                self.llm = LocalLLM(JUDGE_MODEL)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            step = self.batch_size * 8
            for i in range(0, len(todo), step):
                chunk = todo[i:i + step]
                replies = self.llm.chat([build_prompt(c, a) for _, (c, a) in chunk],
                                        max_new_tokens=1500 if thinking else 40, batch_size=self.batch_size,
                                        thinking=thinking)
                with open(self.cache_path, "a") as f:
                    for (k, (c, a)), reply in zip(chunk, replies):
                        r = {"key": k, "thinking": thinking, "answer": a, "facts": [fact_text(x) for x in c],
                             "reply": reply.strip()}
                        self.cache[k] = r
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if progress:
                    print(f"judge{' (thinking)' if thinking else ''} {min(i + step, len(todo))}/{len(todo)}",
                          flush=True)
        return [parse(self.cache[k]["reply"], c) for k, (c, _) in zip(keys, jobs)]

    def judge(self, jobs, regex=None, progress=True):
        """jobs: [(components, answer)]; regex: optional [{component_id: bool}] aligned with jobs.
        -> [{component_id: bool} or None (unparseable reply)]"""
        fast = self._run(jobs, False, progress)
        out, self.last_source = list(fast), ["fast" if v is not None else "error" for v in fast]
        if regex is not None:
            idx = [i for i, (v, r) in enumerate(zip(fast, regex)) if v is None or v != r]
            if idx:
                slow = self._run([jobs[i] for i in idx], True, progress)
                for i, v in zip(idx, slow):
                    if v is not None:
                        out[i], self.last_source[i] = v, "think"
        return out

    def close(self):
        if self.llm is not None:
            import torch
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
