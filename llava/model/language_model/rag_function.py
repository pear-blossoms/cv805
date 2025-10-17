"""multimodal_wiki_extractor.py

A small reusable module that exposes a single importable function

    extract_entities_and_fetch_wiki(query: str, image_path: str, *, top_k_per_entity: int = 2, lang: str = 'en', model: str = 'gpt-4o-mini') -> List[str]

It will:
  - call an OpenAI chat model (multimodal) to extract named entities from the text+image
  - query wikipedia for each entity and return a list of page contents

Usage notes:
  - Configure OpenAI credentials via environment variables: OPENAI_API_KEY and optionally OPENAI_API_BASE
  - Install dependencies: pip install openai wikipedia

The module contains a CLI example under `if __name__ == '__main__'` for quick local testing.
"""

from typing import List, Optional
import os
import json
import base64
import re
import random
import time
import traceback
from mimetypes import guess_type

from openai import OpenAI
import wikipedia

# --- simple module-level client cache ---
_client: Optional[OpenAI] = None


def init_client(api_key: Optional[str] = None, api_base: Optional[str] = None) -> OpenAI:
    """Initialize and return a cached OpenAI client.

    The function reads from environment variables if explicit args are not provided:
      - OPENAI_API_KEY
      - OPENAI_API_BASE (optional)
    """
    global _client
    if _client is not None:
        return _client

    api_key = ""
    api_base = ""

    if not api_key:
        raise RuntimeError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key.")

    # If the user set an alternate base URL, pass it; otherwise default library behavior will be used.
    if api_base:
        _client = OpenAI(api_key=api_key, base_url=api_base)
    else:
        _client = OpenAI(api_key=api_key)
    return _client


def image_to_data_uri(image_path: str) -> str:
    """Convert an image file to a data URI (base64).

    Raises exceptions on I/O errors so callers can decide how to handle them.
    """
    mime_type, _ = guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _parse_retry_after_ms_from_error(err: Exception) -> Optional[int]:
    try:
        msg = str(err)
        m = re.search(r"(\d+)\s*ms", msg)
        if m:
            return int(m.group(1))
        m2 = re.search(r"(\d+)\s*seconds?", msg)
        if m2:
            return int(m2.group(1)) * 1000
    except Exception:
        return None
    return None


def call_chat_with_retries(client: OpenAI, **create_kwargs) -> object:
    """Call client.chat.completions.create with a retry loop for rate limits and server errors."""
    max_retries = 10
    initial_delay = 0.5
    backoff = 2.0
    max_delay = 60.0

    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(**create_kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "rate" in err_str and ("limit" in err_str or "429" in err_str)
            is_server_error = "502" in err_str or "503" in err_str or "504" in err_str
            is_timeout = "timeout" in err_str
            if not (is_rate_limit or is_server_error or is_timeout):
                raise

            retry_after_ms = _parse_retry_after_ms_from_error(e)
            if retry_after_ms:
                sleep_for = min(max_delay, retry_after_ms / 1000.0 + random.random() * 0.1)
            else:
                sleep_for = min(max_delay, initial_delay * (backoff ** (attempt - 1)) + random.random() * 0.5)
            # best-effort backoff
            time.sleep(sleep_for)
    raise RuntimeError("Max retries exceeded")


def extract_entities_multimodal(query: str, image_path: str, *, model: str = "gpt-4o-mini", temperature: float = 0.0) -> List[str]:
    """Given user text + image path, call the multimodal chat model and return a list of extracted entities.

    The model is expected to return a JSON array like ["Entity1", "Entity2"]. If the model responds with plain text,
    we try to parse JSON first and fall back to line-splitting.
    """
    client = init_client()
    try:
        data_uri = image_to_data_uri(image_path)
    except Exception:
        traceback.print_exc()
        raise

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert entity extraction assistant with multimodal capabilities.\n"
                "Given both the user's query text and an associated image, extract only named entities "
                "(people, locations, organizations, dates, objects, concepts) that are directly relevant.\n"
                "Return the extracted entities as a JSON array, e.g., [\"Entity1\", \"Entity2\"].\n"
                "DO NOT answer the question, only extract entities."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]

    resp = call_chat_with_retries(
        client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
    )

    reply = resp.choices[0].message.content.strip()
    try:
        ents = json.loads(reply)
        return [e for e in ents if isinstance(e, str)]
    except Exception:
        # fallback: split by lines and strip
        return [line.strip().strip('"') for line in reply.splitlines() if line.strip()]


def extract_entities_and_fetch_wiki(query: str, image_path: str, *, top_k_per_entity: int = 1, lang: str = "en", model: str = "gpt-4o-mini") -> List[str]:
    """Public function: extract entities from text+image and fetch corresponding Wikipedia page contents.

    Returns a list of page.content strings (may be empty if nothing found).
    """
    # return 'this is a string'#for debug
    entities = extract_entities_multimodal(query, image_path, model=model)
    if not entities:
        return []

    wikipedia.set_lang(lang)
    all_contents: List[str] = []

    for ent in entities:
        try:
            titles = wikipedia.search(ent)
            for title in titles[:top_k_per_entity]:
                try:
                    page = wikipedia.page(title)
                    all_contents.append(page.content)
                except wikipedia.DisambiguationError:
                    # skip ambiguous titles in this simple helper
                    continue
                except wikipedia.PageError:
                    continue
        except Exception:
            traceback.print_exc()
            continue
    _max_chars_joined = 20000  # 可调整：最终字符串最多保留多少字符

    # 过滤空内容并用两个换行符拼接
    joined = "\n\n".join([c for c in all_contents if c])

    if not joined:
        return []  # 没有内容时保持返回空列表

    # 直接按字符截断（Python 字符串按 Unicode code points 切片，安全）
    truncated = joined[:_max_chars_joined]

    # 返回单个字符串的列表，便于直接传给 embeddings API 的 input 参数
    return truncated


# --- CLI example for quick local testing ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract entities from (text + image) and fetch Wikipedia pages.")
    parser.add_argument("query", help="User query text")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--top_k", type=int, default=2, help="Top K wiki pages per entity")
    parser.add_argument("--lang", default="en", help="Wikipedia language code (default: en)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to call")
    args = parser.parse_args()

    try:
        contents = extract_entities_and_fetch_wiki(args.query, args.image, top_k_per_entity=args.top_k, lang=args.lang, model=args.model)
        print(json.dumps({"num_pages": len(contents)}))
    except Exception as e:
        print("Error:", e)
        raise
