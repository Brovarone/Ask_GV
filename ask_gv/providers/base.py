from __future__ import annotations
from typing import Any, Dict
import json
import requests
from ..defaults import DEFAULT_TIMEOUT
from ..utils import retry_request

def post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text}
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {json.dumps(data, ensure_ascii=False)[:2500]}")
    return data

def get_json(url: str, headers: Dict[str, str], timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    resp = requests.get(url, headers=headers, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text}
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {json.dumps(data, ensure_ascii=False)[:2500]}")
    return data

def extract_openai_like_text(data: Dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    try:
        return data["output_text"].strip()
    except Exception:
        pass
    return json.dumps(data, ensure_ascii=False)

def extract_gemini_text(data: Dict[str, Any]) -> str:
    try:
        return "\n".join(part.get("text", "") for part in data["candidates"][0]["content"]["parts"]).strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False)

def retry_post(url, headers, payload):
    return retry_request(lambda: post_json(url, headers, payload))
