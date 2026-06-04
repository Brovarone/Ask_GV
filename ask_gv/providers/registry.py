from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .base import extract_gemini_text, extract_openai_like_text
from .gemini import call_gemini_generate_content, call_gemini_generate_with_file_uri, call_gemini_upload_file
from .nvidia import call_nvidia
from .openrouter import call_openrouter
from .perplexity import call_perplexity

OPENAI_LIKE_PROVIDERS = {"openrouter", "perplexity", "nvidia"}


def provider_settings(settings: Dict[str, Any], provider: str) -> Dict[str, Any]:
    return settings.get("providers", {}).get(provider, {})


def provider_api_key(provider: str) -> str:
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY non impostata")
        return key
    if provider == "perplexity":
        key = os.getenv("PERPLEXITY_API_KEY")
        if not key:
            raise RuntimeError("PERPLEXITY_API_KEY non impostata")
        return key
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY non impostata")
        return key
    if provider == "nvidia":
        key = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY / NIM_API_KEY non impostata")
        return key
    raise RuntimeError(f"Provider non supportato: {provider}")


def extract_provider_text(provider: str, raw: Dict[str, Any]) -> str:
    if provider == "gemini":
        return extract_gemini_text(raw)
    if provider in OPENAI_LIKE_PROVIDERS:
        return extract_openai_like_text(raw)
    raise RuntimeError(f"Provider non supportato per extract text: {provider}")


def call_provider(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    settings: Dict[str, Any],
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    api_key = provider_api_key(provider)
    pcfg = provider_settings(settings, provider)

    if provider == "openrouter":
        raw = call_openrouter(model, api_key, messages, temperature, max_tokens, response_format=response_format)
    elif provider == "perplexity":
        raw = call_perplexity(model, api_key, messages, temperature, max_tokens)
    elif provider == "nvidia":
        base_url = pcfg.get("base_url", "https://integrate.api.nvidia.com/v1")
        raw = call_nvidia(model, api_key, messages, base_url, temperature, max_tokens, response_format=response_format)
    elif provider == "gemini":
        if len(messages) < 2:
            raise RuntimeError("Messaggi Gemini insufficienti")
        system_prompt = messages[0].get("content", "")
        user_prompt = messages[-1].get("content", "")
        raw = call_gemini_generate_content(model, api_key, system_prompt, user_prompt, temperature)
    else:
        raise RuntimeError(f"Provider non supportato: {provider}")

    return raw, extract_provider_text(provider, raw)


def call_gemini_summary_with_optional_file(
    model: str,
    summary_prompt: str,
    user_prompt: str,
    gemini_use_files: bool = False,
    consolidated_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], str]:
    api_key = provider_api_key("gemini")
    if gemini_use_files and consolidated_path is not None:
        up = call_gemini_upload_file(api_key, consolidated_path, "text/markdown")
        file_uri = up.get("file", {}).get("uri") or up.get("uri")
        mime = up.get("file", {}).get("mimeType", "text/markdown")
        if file_uri:
            raw = call_gemini_generate_with_file_uri(
                model,
                api_key,
                summary_prompt,
                file_uri,
                mime,
                "Analizza il file allegato e restituisci JSON valido secondo lo schema richiesto.",
            )
            return raw, extract_gemini_text(raw)
    raw = call_gemini_generate_content(model, api_key, summary_prompt, user_prompt)
    return raw, extract_gemini_text(raw)