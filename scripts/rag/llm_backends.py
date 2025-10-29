import os
import requests
from typing import List, Dict


def call_openai(messages: List[Dict[str, str]]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "600")),
    )
    return resp.choices[0].message.content.strip()


def call_ollama(messages: List[Dict[str, str]]) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    model = os.getenv("OLLAMA_MODEL", "llama3:instruct")
    options = {
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
        # Ограничим длину ответа, чтобы ускорить CPU-инференс
        "num_predict": int(os.getenv("LLM_MAX_TOKENS", "400")),
        # Умеренный контекст для экономии памяти/времени
        "num_ctx": int(os.getenv("LLM_NUM_CTX", "2048")),
    }
    payload = {"model": model, "messages": messages, "stream": False, "options": options}
    # Раздельный connect/read таймаут: 10s на соединение, до 600s на ответ (первый прогрев может быть долгим)
    read_timeout = float(os.getenv("LLM_HTTP_TIMEOUT", "600"))
    r = requests.post(url, json=payload, timeout=(10, read_timeout))
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "").strip()


def generate(system_prompt: str, user_prompt: str) -> str:
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    if backend == "ollama":
        return call_ollama(messages)
    return call_openai(messages)
