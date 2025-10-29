import argparse
import json
import os
import pathlib
import re
import time
from typing import Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- CPU-friendly defaults for 8 vCPU / 16 GB RAM ---
DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
DEFAULT_BATCH = int(os.environ.get("EMBED_BATCH", "96"))
DEFAULT_MAX_WORDS = int(os.environ.get("EMBED_MAX_WORDS", "200"))
DEFAULT_OVERLAP_WORDS = int(os.environ.get("EMBED_OVERLAP_WORDS", "40"))


def set_threads():
    try:
        import torch  # type: ignore

        torch.set_num_threads(min(os.cpu_count() or 1, 8))
    except Exception:
        pass
    try:
        faiss.omp_set_num_threads(min(os.cpu_count() or 1, 8))
    except Exception:
        pass


def read_text(path: pathlib.Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_by_words(text: str, max_words: int, overlap_words: int):
    tokens = re.split(r"(\s+)", text)
    word_positions = [i for i, tok in enumerate(tokens) if not tok.isspace() and tok != ""]
    if not word_positions:
        c = text.strip()
        if c:
            yield (c, 0, 0)
        return
    step_words = max(1, max_words - overlap_words)
    for start_w in range(0, len(word_positions), step_words):
        end_w = min(start_w + max_words, len(word_positions))
        start_tok = word_positions[start_w]
        end_tok = word_positions[end_w - 1] if end_w > start_w else word_positions[start_w]
        end_tok_inclusive = min(end_tok + 1, len(tokens))
        chunk = "".join(tokens[start_tok:end_tok_inclusive]).strip()
        if chunk:
            yield (chunk, start_w, end_w)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def main():
    ap = argparse.ArgumentParser(description="Build FAISS index (FlatIP cosine) from text KB (CPU-friendly, E5-ready).")
    ap.add_argument("--kb", default="knowledge_base", help="Folder with .txt/.md files")
    ap.add_argument("--out", default="artifacts", help="Output dir for index and metadata")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model (E5 recommended)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Encode batch size")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS, help="Max words per chunk")
    ap.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS, help="Words overlap between chunks")
    args = ap.parse_args()

    kb_dir = pathlib.Path(args.kb)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_threads()
    model = SentenceTransformer(args.model, device="cpu")
    dim = model.get_sentence_embedding_dimension()

    # cosine via L2-normalized vectors + inner product
    index = faiss.IndexFlatIP(dim)
    meta_path = out_dir / "kb_meta.jsonl"
    fw_meta = open(meta_path, "w", encoding="utf-8")

    files: List[pathlib.Path] = [p for p in kb_dir.rglob("*") if p.suffix.lower() in {".txt", ".md"}]
    files.sort()

    t0 = time.time()
    total_chunks = 0
    total_files = 0

    for fp in files:
        text = read_text(fp)
        if not text.strip():
            continue
        title = fp.stem
        total_files += 1

        local_chunks: List[str] = []
        local_metas: List[Dict] = []
        chunk_idx = 0

        for chunk, start_w, end_w in chunk_by_words(text, args.max_words, args.overlap_words):
            # E5 best practice: prepend "passage: "
            e5_doc = f"passage: {title}\n\n{chunk}"
            local_chunks.append(e5_doc)
            local_metas.append(
                {
                    "id": f"{fp.name}::chunk-{chunk_idx}",
                    "source_path": str(fp),
                    "title": title,
                    "chunk_index": chunk_idx,
                    "word_start": int(start_w),
                    "word_end": int(end_w),
                    "text": chunk,  # полный чанк (для гибридного поиска/цитат)
                    "excerpt": chunk[:400],
                }
            )
            chunk_idx += 1

            if len(local_chunks) >= args.batch:
                vecs = model.encode(
                    local_chunks, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False
                ).astype("float32")
                vecs = l2_normalize(vecs)
                index.add(vecs)
                for m in local_metas:
                    fw_meta.write(json.dumps(m, ensure_ascii=False) + "\n")
                total_chunks += len(local_chunks)
                local_chunks.clear()
                local_metas.clear()

        if local_chunks:
            vecs = model.encode(
                local_chunks, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False
            ).astype("float32")
            vecs = l2_normalize(vecs)
            index.add(vecs)
            for m in local_metas:
                fw_meta.write(json.dumps(m, ensure_ascii=False) + "\n")
            total_chunks += len(local_chunks)
            local_chunks.clear()
            local_metas.clear()

    fw_meta.close()
    index_path = out_dir / "kb_faiss.index"
    faiss.write_index(index, str(index_path))

    elapsed = time.time() - t0
    summary = {
        "model": args.model,
        "dimension": dim,
        "files": total_files,
        "chunks": total_chunks,
        "index_path": str(index_path),
        "meta_path": str(meta_path),
        "elapsed_sec": round(elapsed, 2),
        "chunking": {"max_words": args.max_words, "overlap_words": args.overlap_words},
        "cpu": {"vcpus": os.cpu_count(), "batch": args.batch},
        "vectorization": "E5 format: 'passage: {title}\\n\\n{chunk}'",
    }
    with open(out_dir / "kb_summary.json", "w", encoding="utf-8") as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
