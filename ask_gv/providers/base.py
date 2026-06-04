from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union
import json
import time
import requests
from ..defaults import DEFAULT_TIMEOUT

TimeoutType = Union[int, float, Tuple[float, float]]


class ProviderError(RuntimeError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderHTTPError(ProviderError):
    def __init__(self, status_code: int, message: str, data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.data = data or {}


class ProviderRateLimitError(ProviderHTTPError):
    pass


def _normalize_timeout(
    timeout: Optional[Union[int, float, Dict[str, Any], Tuple[float, float]]]
) -> TimeoutType:
    if timeout is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout, (int, float)):
        return timeout
    if isinstance(timeout, tuple):
        return timeout
    if isinstance(timeout, dict):
        connect = float(timeout.get('connect', 10))
        read = float(timeout.get('read', DEFAULT_TIMEOUT))
        return (connect, read)
    raise ValueError(f'Timeout non valido: {timeout}')


def _parse_response(resp: requests.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {'data': data}
    except Exception:
        return {'text': resp.text}


def _raise_for_status(resp: requests.Response, data: Dict[str, Any]) -> None:
    if resp.status_code < 400:
        return
    message = f"HTTP {resp.status_code}: {json.dumps(data, ensure_ascii=False)[:2500]}"
    if resp.status_code == 429:
        raise ProviderRateLimitError(resp.status_code, message, data)
    raise ProviderHTTPError(resp.status_code, message, data)


def request_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[int, float, Dict[str, Any], Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    normalized_timeout = _normalize_timeout(timeout)
    try:
        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=payload,
            timeout=normalized_timeout,
        )
    except requests.exceptions.ReadTimeout as e:
        raise ProviderTimeoutError(f'Read timeout su {url}: {e}') from e
    except requests.exceptions.ConnectTimeout as e:
        raise ProviderTimeoutError(f'Connect timeout su {url}: {e}') from e
    except requests.exceptions.Timeout as e:
        raise ProviderTimeoutError(f'Timeout su {url}: {e}') from e
    except requests.exceptions.RequestException as e:
        raise ProviderError(f'Errore di rete su {url}: {e}') from e

    data = _parse_response(resp)
    _raise_for_status(resp, data)
    return data


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, ProviderRateLimitError):
        return True
    if isinstance(exc, ProviderTimeoutError):
        return True
    if isinstance(exc, ProviderHTTPError) and exc.status_code in {502, 503, 504}:
        return True
    return False


def retry_request(fn, retries: int = 2, base_sleep: float = 1.5):
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt >= retries or not _should_retry(e):
                break
            time.sleep(base_sleep * (2 ** (attempt - 1)))
    assert last is not None
    raise last

def post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: Union[int, float, Tuple[float, float]] = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text}
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {json.dumps(data, ensure_ascii=False)[:2500]}")
    return data

def get_json(
    url: str,
    headers: Dict[str, str],
    timeout: Union[int, float, Tuple[float, float]] = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    resp = requests.get(url, headers=headers, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text}
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {json.dumps(data, ensure_ascii=False)[:2500]}")
    return data

def retry_post(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: Optional[Union[int, float, Dict[str, Any], Tuple[float, float]]] = None,
    retries: int = 2,
    base_sleep: float = 1.5,
) -> Dict[str, Any]:
    return retry_request(
        lambda: post_json(url, headers, payload, timeout=timeout),
        retries=retries,
        base_sleep=base_sleep,
    )

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

