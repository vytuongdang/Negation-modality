"""Minimal local VLM wrapper (transformers image-text-to-text, greedy decoding, one image per prompt)."""
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MAX_SIDE = 1024  # downscale large images so both reference models fit on one 24GB GPU


def load_image(path):
    img = Image.open(path).convert("RGB")
    scale = MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return img


class LocalVLM:
    def __init__(self, name, dtype=torch.float16):
        self.name = name
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype).to("cuda").eval()

    @torch.no_grad()
    def answer(self, image_path, text, max_new_tokens=160):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.processor(text=[prompt], images=[load_image(image_path)], return_tensors="pt").to("cuda")
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def close(self):
        del self.model
        torch.cuda.empty_cache()
