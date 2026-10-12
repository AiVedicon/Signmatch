"""Central configuration. Override any key in settings.py via SIGMATCH = {...}."""
from __future__ import annotations

DEFAULTS = {
    "YOLO_WEIGHTS": None,               # path to your trained signature-detector .pt (None = use whole image)
    "YOLO_CONF": 0.25,
    "EMBEDDER_WEIGHTS": {               # fine-tuned checkpoints from training/train_embedder.py
        "signature": None,
        "handwriting": None,
    },
    "INDEX_DIR": "faiss_indexes",
    "EMBED_DIM": 256,
    "DEVICE": "cuda",
    "TOP_K": 20,

    # score fusion (all scores are in 0..1)
    "WEIGHT_EMBEDDING": 0.5,
    "WEIGHT_FEATURES": 0.5,
    "WEIGHT_TEXT": 0.15,               # share of the Qwen word/letter score in the final score
    # CALIBRATE THESE ON YOUR OWN GENUINE/FORGED PAIRS (see README step 8)
    "ACCEPT": 0.80,
    "REVIEW": 0.65,

    # weights of the forensic (stroke / loop / dot / slant ...) sub-scores
    "FEATURE_WEIGHTS": {
        "stroke_overlay": 0.20,
        "parts": 0.15,
        "loops": 0.12,
        "orientation": 0.10,
        "slant": 0.10,
        "dots": 0.08,
        "underline": 0.07,
        "aspect": 0.05,
        "stroke_width": 0.05,
        "endpoints": 0.05,
        "junctions": 0.05,
        "ink_layout": 0.05,
    },

    "USE_VLM": False,
    "VLM_MODEL": "Qwen/Qwen2.5-VL-7B-Instruct",
}


def load() -> dict:
    """Merge settings.SIGMATCH over DEFAULTS (Django imported lazily)."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    try:
        from django.conf import settings
        user = getattr(settings, "SIGMATCH", {}) or {}
    except Exception:
        user = {}
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg
