"""LLM judge: which of a sample's components does an answer state? Primary scorer for the baseline filter and
for experiments; the regex matcher in mention.py is kept as a second opinion.

    from judge import Judge
    j = Judge(cache_path)                       # model loads lazily, only if something is not cached
    res = j.judge([(components, answer), ...], regex=[{"C1": True, ...}, ...])
    res = j.judge([(components, answer, question), ...], ...)   # v3: the judge also reads the question
    # -> [{"C1": True, "C2": False, ...} or None if the reply cannot be parsed]
    j.last_source                               # per job: "fast", "think" (adjudicated), "think_error"
                                                #   (thinking never gave JSON: fast value kept, FLAGGED) or "error"
    labels = j.label_target([(component, answer), ...])   # -> "STATE" / "META" / "DENY" / "NONE" per job
    j.close()

Closed set: the judge only says which of the listed facts the answer states (no free-form grading).
Two passes: a fast pass (no thinking) on every job; where `regex` is given and disagrees with the fast pass on
any component, a second pass with Qwen3 thinking enabled decides (the fast pass misses e.g. "matte" = rubber,
the regex misses paraphrases and hits subject words). Results are cached by (pass, prompt version, facts,
answer) in a JSONL file, so re-running is free.
Jobs with a question (3-tuples) use PROMPT_V3: the judge reads the question too, so bare answers ("Cart" to
"say what it is pulling") are understood (Vy's human check, 2026-11-23). Jobs without one keep the v2 prompt and
its cache (dataset baseline).
If a thinking reply has no parseable JSON (it ran out of tokens: 152/2269 with the old 1500-token budget), it is
retried once with THINK_RETRY_TOKENS; if that fails too the job is marked "think_error" instead of silently
passing as "fast" (fix 2026-11-23).
"""
import hashlib
import json
import re
from pathlib import Path

JUDGE_MODEL = "Qwen/Qwen3-8B"
PROMPT_VERSION = "v2"       # prompt without the question (dataset baseline)
PROMPT_VERSION_Q = "v3"     # prompt with the question (experiments, since 2026-11-23)
THINK_TOKENS = 1500         # first thinking try (cache key unchanged, so old replies are reused)
THINK_RETRY_TOKENS = 4096   # retry for replies that did not reach the JSON

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


LABEL_VERSION = "label-v2"
LABEL_PROMPT = """Below is a question about an image, the answer a model gave, and one fact X.
The question is only there so you can read short answers ("Cart" to "what is the horse pulling?").
X: {fact}

How does the answer treat X? Reply with exactly one word:
STATE - the answer says X is true / present, even if it also says it will not talk about it
        ("there's a horse too, but I won't mention it").
META  - the answer names X only to exclude it or to say it will not talk about it, without saying X is true
        ("besides the bed, there is a chair", "excluding the suitcase", "I won't mention the color").
DENY  - the answer says X is false / absent, or gives a different value ("there is no knife").
NONE  - the answer does not refer to X at all (words that appear only in the question do not count).

Question:
\"\"\"{question}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

Reply with one word: STATE, META, DENY or NONE."""


PROMPT_V3 = """Below is a question about an image, the answer a model gave, and a numbered list of facts.
Which facts does the ANSWER state? The question is only there so you can read short answers: if the question
asks "what is the horse pulling?" and the answer is "Cart", the answer states "horse is pulling cart".
Rules:
- A fact counts only if the answer itself asserts it, read in the context of the question. Words that appear
  only in the question do not count. Synonyms and paraphrases are fine
  ("sofa" = "couch", "table" = "dining table", "monitor" = "tv", "big" = "large", "made of wood" = "wooden",
  "matte" = "rubber", "shiny" or "metallic" = "metal", "holding" = "carrying").
- Object present: the answer says that object is in the image.
- Count: the answer gives that number for that object; "a fork" or "one fork" states 1 fork.
  A different number for that object does NOT state the fact.
- Color, size, material, shape, texture or other attribute: the answer gives that same value for that object.
- Relation (e.g. "woman is standing on grass"): the answer says that subject has that relation to that object.
  Just naming the object elsewhere, or using its word for the subject, does not count.
- Saying only that it will not talk about something (e.g. "I won't mention the size") states nothing.
- Ignore anything in the answer that is not in the list.

Question:
\"\"\"{question}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

Facts:
{facts}

Reply with JSON only, e.g. {{"stated": [1, 3]}} or {{"stated": []}}."""


def fact_text(c):
    return f"there is a {c['subject']} in the image" if c["type"] == "existence" else c["text"]


def build_prompt(comps, answer, question=None):
    facts = "\n".join(f"{i}. {fact_text(c)}" for i, c in enumerate(comps, 1))
    if question is None:
        return PROMPT.format(answer=answer, facts=facts)
    return PROMPT_V3.format(question=question, answer=answer, facts=facts)


def key(comps, answer, thinking=False, budget=THINK_TOKENS, question=None):
    head = [PROMPT_VERSION, JUDGE_MODEL] if question is None else [PROMPT_VERSION_Q, JUDGE_MODEL, question]
    payload = json.dumps(head + [[fact_text(c) for c in comps], answer]
                         + (["think"] if thinking else [])
                         + ([f"think-{budget}"] if thinking and budget != THINK_TOKENS else []), ensure_ascii=False)
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

    def _run(self, jobs, thinking, progress, budget=THINK_TOKENS):
        jobs = [(j[0], j[1], j[2] if len(j) > 2 else None) for j in jobs]   # (components, answer, question)
        keys = [key(c, a, thinking, budget, q) for c, a, q in jobs]
        todo = list({k: (c, a, q) for k, (c, a, q) in zip(keys, jobs) if k not in self.cache}.items())
        if todo:
            if self.llm is None:
                from llm import LocalLLM
                self.llm = LocalLLM(JUDGE_MODEL)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            step = self.batch_size * 8
            for i in range(0, len(todo), step):
                chunk = todo[i:i + step]
                replies = self.llm.chat([build_prompt(c, a, q) for _, (c, a, q) in chunk],
                                        max_new_tokens=budget if thinking else 40, batch_size=self.batch_size,
                                        thinking=thinking)
                with open(self.cache_path, "a") as f:
                    for (k, (c, a, q)), reply in zip(chunk, replies):
                        r = {"key": k, "thinking": thinking, "budget": budget if thinking else 40, "answer": a,
                             "question": q, "facts": [fact_text(x) for x in c],
                             "reply": reply.strip()}
                        self.cache[k] = r
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if progress:
                    print(f"judge{' (thinking)' if thinking else ''} {min(i + step, len(todo))}/{len(todo)}",
                          flush=True)
        return [parse(self.cache[k]["reply"], c) for k, (c, _, _) in zip(keys, jobs)]

    def judge(self, jobs, regex=None, progress=True):
        """jobs: [(components, answer)] or [(components, answer, question)] (v3 prompt);
        regex: optional [{component_id: bool}] aligned with jobs.
        -> [{component_id: bool} or None (unparseable reply)]"""
        fast = self._run(jobs, False, progress)
        out, self.last_source = list(fast), ["fast" if v is not None else "error" for v in fast]
        if regex is not None:
            idx = [i for i, (v, r) in enumerate(zip(fast, regex)) if v is None or v != r]
            if idx:
                slow = self._run([jobs[i] for i in idx], True, progress)
                retry = [i for i, v in zip(idx, slow) if v is None]
                if retry:   # thinking ran out of tokens before the JSON: once more with a larger budget
                    again = dict(zip(retry, self._run([jobs[i] for i in retry], True, progress, THINK_RETRY_TOKENS)))
                    slow = [again.get(i, v) for i, v in zip(idx, slow)]
                for i, v in zip(idx, slow):
                    if v is not None:
                        out[i], self.last_source[i] = v, "think"
                    elif out[i] is not None:
                        self.last_source[i] = "think_error"   # fast value kept but flagged for human review
        return out

    def label_target(self, jobs, progress=True):
        """jobs: [(component, answer, question)] -> "STATE" / "META" / "DENY" / "NONE" per job (None if
        unparseable).
        For answers that name the forbidden component (strict-violation robustness check, 2026-11-23)."""
        keys = [hashlib.sha1(json.dumps([LABEL_VERSION, JUDGE_MODEL, fact_text(c), a, q], ensure_ascii=False)
                             .encode()).hexdigest() for c, a, q in jobs]
        todo = list({k: (c, a, q) for k, (c, a, q) in zip(keys, jobs) if k not in self.cache}.items())
        if todo:
            if self.llm is None:
                from llm import LocalLLM
                self.llm = LocalLLM(JUDGE_MODEL)
            step = self.batch_size * 8
            for i in range(0, len(todo), step):
                chunk = todo[i:i + step]
                replies = self.llm.chat([LABEL_PROMPT.format(fact=fact_text(c), answer=a, question=q)
                                         for _, (c, a, q) in chunk],
                                        max_new_tokens=20, batch_size=self.batch_size, thinking=False)
                with open(self.cache_path, "a") as f:
                    for (k, (c, a, q)), reply in zip(chunk, replies):
                        r = {"key": k, "label_pass": LABEL_VERSION, "answer": a, "question": q, "fact": fact_text(c),
                             "reply": reply.strip()}
                        self.cache[k] = r
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if progress:
                    print(f"label {min(i + step, len(todo))}/{len(todo)}", flush=True)
        out = []
        for k in keys:
            m = re.search(r"\b(STATE|META|DENY|NONE)\b", self.cache[k]["reply"].upper())
            out.append(m.group(1) if m else None)
        return out

    def close(self):
        if self.llm is not None:
            import torch
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
