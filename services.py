"""Lazy singletons so models load once per process."""
from __future__ import annotations

from pathlib import Path
from threading import Lock

from . import config as _config

_lock = Lock()
_state: dict = {"embedders": {}, "indexes": {}, "detector": None, "vlm": None}


def cfg() -> dict:
    return _config.load()


def detector():
    with _lock:
        if _state["detector"] is None:
            from .detector import SignatureDetector
            c = cfg()
            _state["detector"] = SignatureDetector(c["YOLO_WEIGHTS"], c["YOLO_CONF"], c["DEVICE"])
        return _state["detector"]


def embedder(kind: str):
    with _lock:
        if kind not in _state["embedders"]:
            from .embedder import Embedder
            c = cfg()
            _state["embedders"][kind] = Embedder(c["EMBEDDER_WEIGHTS"].get(kind), c["EMBED_DIM"], c["DEVICE"])
        return _state["embedders"][kind]


def index(kind: str):
    with _lock:
        c = cfg()
        if kind not in _state["indexes"]:
            from .index import VectorIndex
            _state["indexes"][kind] = VectorIndex(c["EMBED_DIM"], Path(c["INDEX_DIR"]) / f"{kind}.faiss")
        idx = _state["indexes"][kind]
        idx.reload_if_changed()
        return idx


def vlm():
    with _lock:
        if _state["vlm"] is None:
            from .vlm import QwenReader
            _state["vlm"] = QwenReader(cfg()["VLM_MODEL"])
        return _state["vlm"]
