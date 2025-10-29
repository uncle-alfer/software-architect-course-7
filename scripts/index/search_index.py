import argparse
import json
import pathlib
import re
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def load_meta(meta_path: pathlib.Path) -> List[Dict]:
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def norm(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"[^a-z0-9а-я_\\-\\s]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def toks(s: str) -> List[str]:
    return re.findall(r"[\\w\\-]+", norm(s), flags=re.UNICODE)


def ngrams3(s: str) -> set:
    s = norm(s).replace(" ", "_")
    return set([s[i : i + 3] for i in range(0, max(0, len(s) - 3 + 1))]) if len(s) >= 3 else set()


def lexical_score(query: str, text: str, title: str) -> float:
    q_tokens = set(toks(query))
    t_tokens = set(toks(text)) | set(toks(title))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens) / len(q_tokens)
    tf = sum((text.lower().count(tok) + title.lower().count(tok)) for tok in q_tokens)
    tf = min(tf, 10) / 10.0
    q3 = ngrams3(query)
    x3 = ngrams3(title + "\\n" + text)
    jacc = (len(q3 & x3) / len(q3 | x3)) if q3 and x3 else 0.0
    return 0.5 * overlap + 0.3 * tf + 0.2 * jacc


def title_boost(query: str, title: str, text: str) -> float:
    qn, tn = norm(query), norm(title)
    if not qn or not tn:
        return 0.0
    boost = 0.0
    if tn in qn or qn in tn:
        boost += 1.0
    if set(toks(qn)) & set(toks(tn)):
        boost += 0.5
    if tn and tn in norm(text):
        boost += 0.2
    return min(boost, 1.4)


def main():
    ap = argparse.ArgumentParser(description="Hybrid search (E5): vector (FAISS) + lexical + title-boost, CPU-only.")
    ap.add_argument("--out", default="artifacts", help="Dir with kb_faiss.index and kb_meta.jsonl")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cand-mult", type=int, default=8)
    args = ap.parse_args()

    index = faiss.read_index(str(pathlib.Path(args.out) / "kb_faiss.index"))
    metas = load_meta(pathlib.Path(args.out) / "kb_meta.jsonl")
    model = SentenceTransformer(args.model, device="cpu")

    # E5 best practice: prepend "query: "
    qvec = model.encode([f"query: {args.query}"], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    qvec = l2_normalize(qvec)

    top_cand = max(args.top_k * args.cand_mult, args.top_k)
    vscores, vids = index.search(qvec, top_cand)
    vscores, vids = vscores[0].tolist(), vids[0].tolist()

    # normalize vector scores to [0,1]
    if vscores:
        vmin, vmax = min(vscores), max(vscores)
        rng = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
        vnorm = [(s - vmin) / rng for s in vscores]
    else:
        vnorm = []

    cands: List[Tuple[float, int, float, float, float]] = []
    for sid, v in zip(vids, vnorm):
        if sid < 0 or sid >= len(metas):
            continue
        m = metas[int(sid)]
        text = m.get("text") or m.get("excerpt") or ""
        title = m.get("title", "")
        lex = lexical_score(args.query, text, title)
        tbst = title_boost(args.query, title, text)
        # E5 уже хорош по вектору: немного снижаем лексическую долю
        final = 0.70 * v + 0.15 * lex + 0.15 * tbst
        cands.append((final, int(sid), v, lex, tbst))

    cands.sort(key=lambda x: x[0], reverse=True)
    top = cands[: args.top_k]

    print(f"\nQuery: {args.query}\n")
    for i, (final, sid, v, lex, tbst) in enumerate(top, 1):
        m = metas[sid]
        print(
            f"[{i}] score={final:.4f} (vec={v:.3f}, lex={lex:.3f}, title={tbst:.2f}) | {m.get('title','?')}  ← {m.get('source_path','?')}"
        )
        ex = str(m.get("text") or m.get("excerpt") or "").replace("\n", " ")
        print(f"    {ex[:240]}...")
    print()


if __name__ == "__main__":
    main()
