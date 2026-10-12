"""Fuse CNN-embedding similarity with the forensic feature scores."""
from __future__ import annotations

import difflib
import re

import numpy as np

from .config import DEFAULTS
from .features import compare_features, extract_features
from .preprocess import Prepared


def text_diff(a: str, b: str) -> dict:
    """Word-by-word and letter-by-letter comparison of two transcriptions."""
    a_n, b_n = " ".join(a.lower().split()), " ".join(b.lower().split())
    letter = difflib.SequenceMatcher(None, a_n, b_n)
    words = difflib.SequenceMatcher(None, a_n.split(), b_n.split())
    changes = [
        {"op": op, "a": a_n[i1:i2], "b": b_n[j1:j2]}
        for op, i1, i2, j1, j2 in letter.get_opcodes() if op != "equal"
    ]
    return {
        "text_a": a, "text_b": b,
        "letter_similarity": round(letter.ratio(), 3),
        "word_similarity": round(words.ratio(), 3),
        "letter_changes": changes,
    }


def verdict_for(score: float, cfg: dict) -> str:
    return "match" if score >= cfg["ACCEPT"] else "review" if score >= cfg["REVIEW"] else "mismatch"


def is_legible(text) -> bool:
    """True if the transcription contains real text (not empty / ILLEGIBLE)."""
    if not text or text.strip().lower() in ("illegible", "unreadable"):
        return False
    return len(re.sub(r"\W", "", text)) >= 2


def add_text_score(res: dict, text_a: str, text_b: str, cfg: dict | None = None) -> dict:
    """Blend the Qwen word/letter similarity into an existing compare() result."""
    cfg = cfg or DEFAULTS
    res["text_comparison"] = text_diff(text_a or "", text_b or "")
    if is_legible(text_a) and is_legible(text_b):
        d = res["text_comparison"]
        ts = 0.5 * d["letter_similarity"] + 0.5 * d["word_similarity"]
        wt = cfg["WEIGHT_TEXT"]
        res["text_score"] = round(ts, 4)
        res["final_score"] = round((1 - wt) * res["final_score"] + wt * ts, 4)
        res["verdict"] = verdict_for(res["final_score"], cfg)
    else:
        res["text_score"] = None          # unreadable signature: text not used for ranking
    return res


def compare(prep_a: Prepared, prep_b: Prepared,
            emb_a: np.ndarray | None = None, emb_b: np.ndarray | None = None,
            cfg: dict | None = None) -> dict:
    cfg = cfg or DEFAULTS
    fa, fb = extract_features(prep_a.norm_mask), extract_features(prep_b.norm_mask)
    scores, details, reasons = compare_features(fa, fb, prep_a.norm_mask, prep_b.norm_mask)

    fw = cfg["FEATURE_WEIGHTS"]
    feature_score = sum(fw[k] * scores[k] for k in fw) / sum(fw.values())

    emb_score = None
    if emb_a is not None and emb_b is not None:
        emb_score = float(np.clip(np.dot(emb_a, emb_b), 0.0, 1.0))   # both are L2-normalised
        we, wf = cfg["WEIGHT_EMBEDDING"], cfg["WEIGHT_FEATURES"]
        final = (we * emb_score + wf * feature_score) / (we + wf)
    else:
        final = feature_score

    return {
        "final_score": round(final, 4),
        "embedding_score": None if emb_score is None else round(emb_score, 4),
        "feature_score": round(feature_score, 4),
        "verdict": verdict_for(final, cfg),
        "feature_scores": {k: round(v, 3) for k, v in scores.items()},
        "details": details,
        "reasons": reasons,
    }
