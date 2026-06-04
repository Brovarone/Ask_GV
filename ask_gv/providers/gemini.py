from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import requests
from ..defaults import DEFAULT_TIMEOUT
from ..utils import retry_request
from .base import retry_post

def call_gemini_generate_content(model: str, api_key: str, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> Dict[str, object]:
    payload = {"systemInstruction": {"parts": [{"text": system_prompt}]}, "contents": [{"role": "user", "parts": [{"text": user_prompt}]}], "generationConfig": {"temperature": temperature}}
    return retry_post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}", {"Content-Type": "application/json"}, payload)

def call_gemini_generate_with_file_uri(model: str, api_key: str, system_prompt: str, file_uri: str, mime_type: str, question_prompt: str, temperature: float = 0.4) -> Dict[str, object]:
    payload = {"systemInstruction": {"parts": [{"text": system_prompt}]}, "contents": [{"role": "user", "parts": [{"fileData": {"mimeType": mime_type, "fileUri": file_uri}}, {"text": question_prompt}]}], "generationConfig": {"temperature": temperature}}
    return retry_post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}", {"Content-Type": "application/json"}, payload)

def call_gemini_upload_file(api_key: str, file_path: Path, mime_type: str = "text/markdown") -> Dict[str, object]:
    url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
    headers = {"X-Goog-Upload-Protocol": "multipart"}
    files = {"metadata": (None, json.dumps({"display_name": file_path.name}), "application/json"), "file": (file_path.name, file_path.read_bytes(), mime_type)}
    def _do():
        r = requests.post(url, headers=headers, files=files, timeout=DEFAULT_TIMEOUT)
        try:
            data = r.json()
        except Exception:
            data = {"text": r.text}
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data, ensure_ascii=False)[:2500]}")
        return data
    return retry_request(_do)
