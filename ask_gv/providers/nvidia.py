from __future__ import annotations
from typing import Any, Dict, List, Optional
from .base import retry_post


def call_nvidia(
    model: str,
    api_key: str,
    messages: List[Dict[str, str]],
    base_url: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if response_format:
        payload["response_format"] = response_format
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return retry_post(base_url.rstrip("/") + "/chat/completions", headers, payload)