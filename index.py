"""FAISS vector index; vector ID == Signature primary key in MySQL."""
from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int, path: str | Path):
        self.dim, self.path = dim, Path(path)
        self.mtime = 0.0
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))   # cosine on L2-normalised vectors
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self.index = faiss.read_index(str(self.path))
            self.mtime = self.path.stat().st_mtime

    def reload_if_changed(self) -> None:
        """Pick up writes made by another worker process."""
        if self.path.exists() and self.path.stat().st_mtime > self.mtime:
            self.load()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp))
        os.replace(tmp, self.path)                                # atomic swap
        self.mtime = self.path.stat().st_mtime

    def add(self, ids, vectors: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        self.index.remove_ids(ids)                                # makes add idempotent
        self.index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), ids)

    def remove(self, ids) -> None:
        self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def search(self, vec: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(np.ascontiguousarray(vec[None], dtype=np.float32),
                                        min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def clear(self) -> None:
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
