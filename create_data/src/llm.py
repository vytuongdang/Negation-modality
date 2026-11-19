"""Minimal local chat LLM wrapper (transformers, batched greedy decoding, JSON extraction)."""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    def __init__(self, name, dtype=torch.float16):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to("cuda").eval()

    @torch.no_grad()
    def chat(self, prompts, max_new_tokens=512, batch_size=8, thinking=False):
        outs = []
        for i in range(0, len(prompts), batch_size):
            texts = [
                self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=thinking)
                for p in prompts[i:i + batch_size]
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            outs += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs


def parse_json(text):
    """Pull the first JSON object out of a model reply; None if there is none."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
