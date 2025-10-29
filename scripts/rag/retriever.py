import json
import os
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

ART_DIR = os.environ.get("ARTIFACTS", "artifacts")
INDEX_PATH = os.path.join(ART_DIR, "kb_faiss.index")
META_PATH = os.path.join(ART_DIR, "kb_meta.jsonl")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")


def _load_meta(path: str) -> List[Dict[str, Any]]:
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                meta.append(json.loads(line))
            except Exception:
                pass
    return meta


class Retriever:
    def __init__(self, k: int = 4):
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Meta file not found: {META_PATH}")

        self.index = faiss.read_index(INDEX_PATH)
        self.meta = _load_meta(META_PATH)
        self.k = k

        # E5: важно указывать префиксы query:/passage: + L2-нормировать
        self.model = SentenceTransformer(EMBED_MODEL)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(f"query: {text}", normalize_embeddings=True)  # 768-d
        return np.asarray(vec, dtype="float32")

    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = k or self.k
        qv = self.embed_query(query)
        D, I = self.index.search(qv[None, :], k)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx]
            text = m.get("chunk") or m.get("text") or m.get("excerpt") or ""
            path = m.get("path") or m.get("source_path") or m.get("source") or m.get("file") or ""
            base = os.path.splitext(os.path.basename(path))[0] if path else None
            title = m.get("title") or base or f"chunk-{idx}"

            results.append(
                {
                    "id": idx,
                    "score": float(score),
                    "title": title,
                    "path": path or "",
                    "text": text.strip(),
                    "chunk_index": m.get("chunk_index"),
                    "word_start": m.get("word_start"),
                    "word_end": m.get("word_end"),
                }
            )
        return results
