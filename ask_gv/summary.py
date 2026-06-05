from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from .defaults import DEFAULT_SUMMARY_MAX_CHARS, SUMMARY_PROMPT, SYSTEM_PROMPT
from .models import SourceDocument
from .retrieval import build_keyword_counter
from .utils import safe_json_loads, trim_text, write_text
from .providers.registry import call_gemini_summary_with_optional_file, call_provider
from .observability import get_logger

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
        4. Se la chiamata fallisce, usa ``fallback_summary``.
        5. Restituisci lo stesso schema di risposta usato da `fallback_summary`.
    """
    # 1️⃣  Costruisci il prompt generico (non più concatenato al corpo del documento)
    corpus = build_summary_corpus(documents)
    user_prompt = "Analizza il seguente corpus Markdown relativo a un gioco di ruolo tabletop e costruisci un summary tecnico."
    user_prompt += corpus

    # 2️⃣  Prepara i messaggi da inviare al modello
    messages = build_summary_messages(provider, user_prompt, settings)
    
    log = get_logger(__name__, "summary")
    log.info("=== INIZIO generate_summary ===")
    log.debug("provider = %s, model = %s", provider, model)
    log.debug("documents count: %d", len(documents))
    log.debug("settings: %s", settings)

    # 3️⃣  Chiama il provider con i parametri presenti nella configurazione
    try:
        raw, text = call_provider(
            provider,
            model,
            messages,
            settings,
            temperature=settings.get("temperature", 0.2),
            max_tokens=settings.get("max_tokens", 1024),
            response_format={"type": "json_object"},
        )
        
    except Exception as exc:
        # ---------------------------------------------------------
        # 4️⃣  Gestione dell’errore: fallback
        # ---------------------------------------------------------
        log.exception("generate_summary fallito – uso fallback")
        # il fallback ha bisogno dei documenti e dell’eccezione
        fallback = fallback_summary(documents, exc)
        # aggiungiamo eventuali info di logging per il fallback
        log.info("fallback_summary attivato – %s", exc)
        return fallback
    

    result = safe_json_loads(text) or {"summary_text": text, "raw": raw}
    log.debug("raw (raw) = %s", raw)          # valore restituito da call_provider
    log.debug("text (raw) = %s", text)        # testo grezzo
    log.debug("safe_json_loads result: %s", result)

    # 4️⃣  Normalizza la risposta (stessa logica di prima)
    return result

def fallback_summary(documents: List[SourceDocument], error: Exception) -> Dict[str, Any]:
    """
    Restituisce un dizionario di fallback quando la generazione del riassunto
    non è possibile (es. provider non disponibile, modello non supportato, ecc.).

    Il risultato è sempre un dict con le chiavi:
        - "summary_text": messaggio esplicativo
        - "keywords": le prime 60 keyword estratte dal corpus
        - "error": messaggio dell'eccezione (opzionale, per debugging)
    """
    # Estrai le keyword dal corpus (già presente in utils)
    keywords = list(build_keyword_counter(documents).keys())[:60]

    # Messaggio leggibile per l’utente
    summary_text = (
        "Summary automatica non disponibile. "
        "Controlla la configurazione dei provider e dei modelli."
    )

    # Opzionale: includi il messaggio di errore per il debug (puoi rimuoverlo in produzione)
    error_msg = f"Errore: {str(error)}"

    return {
        "summary_text": summary_text,
        "keywords": keywords,
        "error": error_msg,          # campo opzionale, utile per debug
    }
