"""Search pipeline shared by the JSON API and the upload page.

query image -> embedding -> FAISS candidates -> forensic scores (slant, dots, underline,
loops, strokes ...) -> optional Qwen word/letter score -> ranked list.
"""
from __future__ import annotations

import numpy as np

from . import services
from .compare import add_text_score, compare
from .preprocess import decode_image, prepare, read_image
from .models import Signature


def prepare_upload(upload):
    img = decode_image(upload.read())
    crop = services.detector().crop(img)
    return crop, prepare(crop)


def get_transcript(row: Signature, prepared=None) -> str:
    """Qwen transcription of a stored signature, cached in the database."""
    if row.transcript is None:
        prepared = prepared or prepare(read_image(row.image.path))
        text = (services.vlm().transcribe(prepared.mask) or "").strip()
        row.transcript = text.replace("\n", " ")[:255]
        row.save(update_fields=["transcript"])
    return row.transcript


def _pct(v):
    return None if v is None else int(round(v * 100))


def breakdown(res: dict) -> list:
    """(label, percent) pairs shown next to each ranked result."""
    fs = res["feature_scores"]
    tc = res.get("text_comparison") if res.get("text_score") is not None else None
    return [
        ("Embedding", _pct(res["embedding_score"])),
        ("Words", _pct(tc["word_similarity"]) if tc else None),
        ("Letters", _pct(tc["letter_similarity"]) if tc else None),
        ("Slant", _pct(fs["slant"])),
        ("Dots", _pct(fs["dots"])),
        ("Underline", _pct(fs["underline"])),
        ("Loops", _pct(fs["loops"])),
        ("Strokes", _pct(fs["stroke_overlay"])),
    ]


def compare_with_row(prep, row: Signature, q_emb, q_text, cfg) -> dict:
    stored = prepare(read_image(row.image.path))
    s_emb = np.frombuffer(row.embedding, np.float32) if row.embedding else None
    res = compare(prep, stored, q_emb, s_emb, cfg)
    if q_text is not None:
        res = add_text_score(res, q_text, get_transcript(row, stored), cfg)
    res["row"] = row
    res["breakdown"] = breakdown(res)
    return res


def search(prep, kind: str, top_k: int, use_qwen: bool):
    """Returns (ranked results, query transcription or None)."""
    cfg = services.cfg()
    q_emb = services.embedder(kind).embed([prep.canvas])[0]
    hits = services.index(kind).search(q_emb, top_k)
    rows = Signature.objects.in_bulk([h[0] for h in hits])
    q_text = services.vlm().transcribe(prep.mask).strip() if use_qwen else None

    results = []
    for sid, cos in hits:
        row = rows.get(sid)
        if row is None:
            continue
        res = compare_with_row(prep, row, q_emb, q_text, cfg)
        res["faiss_cosine"] = round(cos, 4)
        results.append(res)
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results, q_text


def to_json(res: dict) -> dict:
    out = {k: v for k, v in res.items() if k not in ("row", "breakdown")}
    row = res["row"]
    out.update({"id": row.pk, "person_ref": row.person_ref, "person_name": row.person_name,
                "transcript": row.transcript})
    out["breakdown"] = {label: val for label, val in res["breakdown"]}
    return out
