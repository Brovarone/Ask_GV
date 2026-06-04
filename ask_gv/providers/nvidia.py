from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple
from .base import retry_post


def call_nvidia(
    model: str,
    api_key: str,
    messages: List[Dict[str, str]],
    base_url: str,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[int, float, Dict[str, Any], Tuple[float, float]]] = None,
    retries: int = 2,
    backoff: float = 1.5,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
    }
    if max_tokens:
        payload['max_tokens'] = max_tokens
    if response_format:
        payload['response_format'] = response_format

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    return retry_post(
        base_url.rstrip('/') + '/chat/completions',
        headers,
        payload,
        timeout=timeout,
        retries=retries,
        base_sleep=backoff,
    )