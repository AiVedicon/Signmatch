"""Image cleaning + normalisation shared by enrolment, search and training."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CANVAS_H, CANVAS_W = 128, 256   # CNN input size
FEAT_H = 256                    # height used for forensic features


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def read_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _limit_size(gray: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = gray.shape
    s = max(h, w)
    if s <= max_side:
        return gray
    f = max_side / s
    return cv2.resize(gray, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)


def extract_ink(img: np.ndarray) -> np.ndarray:
    """Return uint8 mask: ink = 255, background = 0 (handles uneven lighting)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = _limit_size(gray)
    # estimate paper background, divide it out to flatten shadows
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    bg = cv2.GaussianBlur(bg, (0, 0), 10)
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    _, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # remove isolated specks (but keep real dots: threshold is tiny)
    h, w = mask.shape
    min_blob = max(6, int(1.5e-5 * h * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob
    return (keep[labels].astype(np.uint8)) * 255


def crop_to_ink(mask: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No ink found in image")
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    return mask[y0:y1, x0:x1]


def resize_to_height(mask: np.ndarray, height: int = FEAT_H) -> np.ndarray:
    h, w = mask.shape
    f = height / h
    out = cv2.resize(mask, (max(1, int(round(w * f))), height), interpolation=cv2.INTER_AREA)
    return ((out > 127).astype(np.uint8)) * 255


def to_canvas(mask: np.ndarray, h: int = CANVAS_H, w: int = CANVAS_W, margin: int = 8) -> np.ndarray:
    """Aspect-preserving fit into a fixed canvas. float32, ink = 1."""
    mh, mw = mask.shape
    s = min((h - 2 * margin) / mh, (w - 2 * margin) / mw)
    nh, nw = max(1, int(round(mh * s))), max(1, int(round(mw * s)))
    small = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w), np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas.astype(np.float32) / 255.0


@dataclass
class Prepared:
    mask: np.ndarray        # cropped ink mask, original scale, 0/255
    norm_mask: np.ndarray   # cropped ink mask, height-normalised, 0/255
    canvas: np.ndarray      # float32 (CANVAS_H, CANVAS_W), ink = 1  -> CNN input


def prepare(img_bgr: np.ndarray) -> Prepared:
    mask = crop_to_ink(extract_ink(img_bgr))
    return Prepared(mask=mask, norm_mask=resize_to_height(mask), canvas=to_canvas(mask))
