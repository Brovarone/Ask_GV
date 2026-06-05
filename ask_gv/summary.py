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

def build_summary_messages(provider: str, user_prompt: str, settings: Dict[str, Any]) -> List[dict]:
    """
    Costruisce la lista di messaggi da inviare a qualunque provider.
    Il formato è lo stesso di `call_provider` (system + user).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

def generate_summary(
    provider: str,
    model: str,
    documents: List[SourceDocument],
    settings: Dict[str, Any],
    work_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Genera un riassunto tecnico a partire da una lista di documenti Markdown.
    Il flusso è:
        1. Costruisci il prompt generico.
        2. Crea la lista di messaggi con ``build_summary_messages``.
        3. Chiama ``call_provider`` (già gestisce Gemini, OpenAI, NVIDIA, ecc.).
        4. Restituisci lo stesso schema di risposta usato da `fallback_summary`.
    """
    # 1️⃣  Costruisci il prompt generico (non più concatenato al corpo del documento)
    corpus = build_summary_corpus(documents)
    user_prompt = "Analizza il seguente corpus Markdown relativo a un gioco di ruolo tabletop e costruisci un summary tecnico." + corpus

    # 2️⃣  Prepara i messaggi da inviare al modello
    messages = build_summary_messages(provider, user_prompt, settings)

    # 3️⃣  Chiama il provider con i parametri presenti nella configurazione
    raw, text = call_provider(
        provider,
        model,
        messages,
        settings,
        temperature=settings.get("temperature", 0.2),
        max_tokens=settings.get("max_tokens", 1024),
        response_format={"type": "json_object"},
    )

    # 4️⃣  Normalizza la risposta (stessa logica di prima)
    return safe_json_loads(text) or {"summary_text": text, "raw": raw}

def fallback_summary(documents: List[SourceDocument], error: Exception) -> Dict[str, Any]:
    return {
        "summary_text": f"Summary automatica non disponibile: {error}",
        "keywords": list(build_keyword_counter(documents).keys())[:60],
        "ambiguities": [],
    }
