"""YOLO signature locator. Falls back to the whole image if no weights/detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class SignatureDetector:
    def __init__(self, weights=None, conf: float = 0.25, device: str = "cuda"):
        self.conf, self.device, self.model = conf, device, None
        if weights and Path(weights).exists():
            from ultralytics import YOLO
            self.model = YOLO(str(weights))

    def crop(self, img_bgr: np.ndarray, pad: float = 0.03) -> np.ndarray:
        if self.model is None:
            return img_bgr
        res = self.model.predict(img_bgr, conf=self.conf, verbose=False, device=self.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return img_bgr
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
        h, w = img_bgr.shape[:2]
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1 = int(max(x1 - px, 0)), int(max(y1 - py, 0))
        x2, y2 = int(min(x2 + px, w)), int(min(y2 + py, h))
        return img_bgr[y1:y2, x1:x2]
