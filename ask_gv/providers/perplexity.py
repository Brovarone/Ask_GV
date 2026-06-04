from __future__ import annotations
from typing import Any, Dict, List, Optional
from .base import retry_post

def call_perplexity(model: str, api_key: str, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return retry_post("https://api.perplexity.ai/chat/completions", headers, payload)
