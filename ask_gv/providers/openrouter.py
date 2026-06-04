from __future__ import annotations
from typing import Any, Dict, List, Optional
from .base import get_json, retry_post

def call_openrouter(model: str, api_key: str, messages: List[Dict[str, str]], temperature: float = 0.4, max_tokens: Optional[int] = None, response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if response_format:
        payload["response_format"] = response_format
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "https://localhost", "X-Title": "rpg-llm-modular"}
    return retry_post("https://openrouter.ai/api/v1/chat/completions", headers, payload)

def list_models(api_key: str) -> Dict[str, Any]:
    return get_json("https://openrouter.ai/api/v1/models", {"Authorization": f"Bearer {api_key}"})
