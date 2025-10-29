import argparse
import json
import os
import textwrap
from typing import List, Dict, Any
from scripts.rag.retriever import Retriever
from scripts.rag.llm_backends import generate

FEW_SHOTS_PATH = os.getenv("FEW_SHOTS_PATH", "prompts/few_shots.jsonl")

SYSTEM_PROMPT = (
    "Ты - ассистент по внутренней базе знаний. Отвечай СТРОГО на русском языке, не смешивай другие языки. "
    "Сначала подумай скрыто, затем напечатай КРАТКИЕ шаги (до 3 пунктов) и итоговый ответ. "
    "Отвечай только на основании приведённого контекста. "
    "Если вопрос относится к твоей личности/модели, к общим фактам вне базы (дата, курс валют, погода и т.п.), "
    "или в контексте нет ответа - напиши: «Я не знаю». "
    "В конце укажи источники в виде номеров фрагментов."
)


def load_few_shots(path: str) -> List[Dict[str, str]]:
    shots = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "q" in obj and "a" in obj:
                        shots.append({"q": obj["q"], "a": obj["a"]})
                except Exception:
                    pass
    return shots[:2]


def build_prompt(user_query: str, hits: List[Dict[str, Any]], few_shots: List[Dict[str, str]]) -> str:
    ctx_blocks = []
    for i, h in enumerate(hits, 1):
        snippet = h["text"].strip().replace("\n", " ").strip()
        ctx_blocks.append(f"[{i}] {h['title']} - {snippet}")
    ctx = "\n".join(ctx_blocks) if ctx_blocks else "(контекст пуст)"

    fs = ""
    if few_shots:
        fs_lines = []
        for s in few_shots:
            fs_lines.append(f"Q: {s['q']}\nA: {s['a']}\n")
        fs = "\n".join(fs_lines)

    prompt = f"""\
Это контекст из базы знаний:
{ctx}

Если контекст недостаточен - ответь «Я не знаю».

Few-shot примеры:
{fs if fs else "(нет)"}

Теперь ответь на вопрос пользователя. Формат ответа:
1) Краткие шаги (до 3 пунктов), ссылаясь на фрагменты [номер].
2) Итоговый ответ.
3) Источники: перечисли номера фрагментов в квадратных скобках.

Вопрос: {user_query}
"""
    return textwrap.dedent(prompt)


def list_sources(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        p = h.get("path") or ""
        span = ""
        if "chunk_index" in h and h["chunk_index"] is not None:
            span = f" [chunk {h['chunk_index']}]"
        if p:
            lines.append(f"[{i}] {h['title']}{span} ({p}) score={h['score']:.3f}")
        else:
            lines.append(f"[{i}] {h['title']}{span} score={h['score']:.3f}")
    return "\n".join(lines)


def token_overlap(query: str, text: str) -> int:
    import re

    q = set(re.findall(r"\w{4,}", query.lower()))
    if not q:
        return 0
    t = " " + (text or "").lower() + " "
    return sum(1 for tok in q if tok in t)


def should_say_idk(
    query: str, hits: list, score_thresh: float, min_chars: int, min_overlap_hits: int = 1, consider_topn: int = 2
) -> bool:
    """Возвращаем True -> сказать 'Я не знаю'."""
    if not hits:
        return True
    if hits[0]["score"] < score_thresh:
        return True
    total_chars = sum(len(h.get("text", "")) for h in hits)
    if total_chars < min_chars:
        return True
    # хотя бы в одном из top-N фрагментов должны встречаться слова из вопроса
    overlap_ok = 0
    for h in hits[:consider_topn]:
        if token_overlap(query, h.get("text", "")) > 0:
            overlap_ok += 1
    return overlap_ok < min_overlap_hits


def ask_once(query: str, k: int, score_thresh: float, min_chars: int) -> None:
    retr = Retriever(k=k)
    shots = load_few_shots(FEW_SHOTS_PATH)
    hits = retr.search(query, k=k)

    if should_say_idk(query, hits, score_thresh, min_chars):
        print(
            "Краткие шаги:\n- Проверил контекст - релевантных фактов недостаточно.\nОтвет:\nЯ не знаю.\nИсточники:\n-"
        )
        return

    prompt = build_prompt(query, hits, shots)
    answer = generate(SYSTEM_PROMPT, prompt)
    print(answer.strip())
    print("\n--- Источники ---")
    print(list_sources(hits))


def repl(k: int, score_thresh: float, min_chars: int) -> None:
    print("RAG-бот запущен. Введите вопрос, 'exit' для выхода.")
    retr = Retriever(k=k)
    shots = load_few_shots(FEW_SHOTS_PATH)
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        hits = retr.search(q, k=k)
        if should_say_idk(q, hits, score_thresh, min_chars):
            print(
                "Краткие шаги:\n- Проверил контекст - релевантных фактов недостаточно.\nОтвет:\nЯ не знаю.\nИсточники:\n-"
            )
            continue
        prompt = build_prompt(q, hits, shots)
        answer = generate(SYSTEM_PROMPT, prompt)
        print(answer.strip())
        print("\n--- Источники ---")
        print(list_sources(hits))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ask", "repl"], default="repl")
    ap.add_argument("--q", help="вопрос (для mode=ask)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.28, help="минимальная похожесть top-1")
    ap.add_argument("--min-chars", type=int, default=400, help="минимум суммарных символов в контексте")
    args = ap.parse_args()

    if args.mode == "ask":
        if not args.q:
            raise SystemExit("Нужно задать --q 'вопрос'")
        ask_once(args.q, args.k, args.score_thresh, args.min_chars)
    else:
        repl(args.k, args.score_thresh, args.min_chars)
