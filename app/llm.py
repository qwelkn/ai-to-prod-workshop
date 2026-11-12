"""Тонка обгортка над openai SDK (OpenAI-сумісний Gemini). Без streaming."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # max_retries=0: ретраями керуємо самі, щоб тримати бюджет часу
        _client = OpenAI(api_key=os.getenv("GEMINI_API_KEY", ""), base_url=BASE_URL,
                         timeout=8.0, max_retries=0)
    return _client


def chat(messages, timeout: float = 8.0, json_mode: bool = False, max_tokens: int = 400) -> str:
    kwargs = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().with_options(timeout=timeout).chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""