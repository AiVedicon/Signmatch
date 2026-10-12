"""Optional Qwen2.5-VL helper: transcribes text in a signature/handwriting crop.

Advisory only - it is NOT part of the numeric score, because VLM outputs are not
reproducible enough to threshold. Use it to give reviewers a word/letter diff.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class QwenReader:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)

    @staticmethod
    def to_pil(mask: np.ndarray) -> Image.Image:
        """Black ink on white background, with a margin."""
        img = Image.fromarray(255 - mask).convert("RGB")
        canvas = Image.new("RGB", (img.width + 40, img.height + 40), "white")
        canvas.paste(img, (20, 20))
        return canvas

    def _ask(self, image: Image.Image, prompt: str, max_new_tokens: int = 64) -> str:
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()

    def transcribe(self, mask: np.ndarray) -> str:
        return self._ask(
            self.to_pil(mask),
            "Transcribe the handwritten text in this image exactly, letter by letter. "
            "Output only the text. If it is an illegible scribble, output ILLEGIBLE.")
