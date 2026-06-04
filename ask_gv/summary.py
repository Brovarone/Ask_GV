from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from .defaults import DEFAULT_SUMMARY_MAX_CHARS, SUMMARY_PROMPT, SYSTEM_PROMPT
from .models import SourceDocument
from .retrieval import build_keyword_counter
from .utils import safe_json_loads, trim_text, write_text
from .providers.registry import call_gemini_summary_with_optional_file, call_provider


def build_summary_corpus(documents: List[SourceDocument], max_chars: int = DEFAULT_SUMMARY_MAX_CHARS) -> str:
    blocks: List[str] = []
    current = 0
    for d in documents:
        block = f"# FILE: {d.path} TITLE: {d.title} {trim_text(d.content, 16000)}"
        if current + len(block) > max_chars:
            break
        blocks.append(block)
        current += len(block)
    return "\n".join(blocks).strip()


def generate_summary(
    provider: str,
    model: str,
    documents: List[SourceDocument],
    settings: Dict[str, Any],
    gemini_use_files: bool = False,
    work_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    corpus = build_summary_corpus(documents)
    user_prompt = "Analizza il seguente corpus Markdown relativo a un gioco di ruolo tabletop e costruisci un summary tecnico." + corpus

    if provider == "gemini":
        consolidated = None
        if gemini_use_files and work_dir is not None:
            consolidated = work_dir / "gemini_summary_corpus.md"
            write_text(consolidated, corpus)
        raw, text = call_gemini_summary_with_optional_file(model, SUMMARY_PROMPT, user_prompt, gemini_use_files, consolidated)
        return safe_json_loads(text) or {"summary_text": text, "raw": raw}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw, text = call_provider(provider, model, messages, settings, temperature=0.2, max_tokens=1024, response_format={"type": "json_object"})
    return safe_json_loads(text) or {"summary_text": text, "raw": raw}


def fallback_summary(documents: List[SourceDocument], error: Exception) -> Dict[str, Any]:
    return {
        "summary_text": f"Summary automatica non disponibile: {error}",
        "keywords": list(build_keyword_counter(documents).keys())[:60],
        "ambiguities": [],
    }
