#!/usr/bin/env python3
"""
multi_llm_rpg_rules.py

Pipeline completa per:
- leggere un repository GitHub oppure una lista di file Markdown locali
- estrarre, normalizzare e segmentare le regole di un GdR
- costruire un knowledge pack riusabile
- fare la stessa domanda a piu' provider/modelli
- ottenere risposte diverse ma confrontabili
- salvare tutto in JSON/Markdown/CSV

Supporto provider:
- OpenRouter
- Google Gemini
- Perplexity
- NVIDIA NIM (endpoint OpenAI-compatible, se disponibile per il modello scelto)

Obiettivi principali:
- ingestione robusta
- prompt standardizzati
- diversificazione controllata delle risposte
- caching dei chunk e dei summary
- output riproducibile e facile da estendere

Requisiti consigliati:
- Python 3.10+
- requests

Esempi:
1) Repo GitHub
   python multi_llm_rpg_rules.py \
     --repo https://github.com/utente/repo-regolamento \
     --question "Proponi una progressione magie piu' bilanciata dal livello 1 al 5" \
     --profiles rules_lawyer systems_designer gm_experience \
     --openrouter-model openai/gpt-4.1-mini \
     --openrouter-model anthropic/claude-3.7-sonnet \
     --gemini-model gemini-2.5-pro \
     --perplexity-model sonar

2) File locali
   python multi_llm_rpg_rules.py \
     --files ./rules/*.md ./docs/*.md \
     --question "Come rifaresti il sistema di vantaggio/svantaggio?"

3) Solo costruzione knowledge pack
   python multi_llm_rpg_rules.py --repo <url> --build-only

4) Riutilizzo di cache esistente
   python multi_llm_rpg_rules.py --cache-dir output/cache/<id> --question "..."

Variabili ambiente supportate:
- OPENROUTER_API_KEY
- GEMINI_API_KEY oppure GOOGLE_API_KEY
- PERPLEXITY_API_KEY
- NVIDIA_API_KEY oppure NIM_API_KEY

Note:
- Per ottenere risposte davvero diverse in modo utile, si usano profili di analisi diversi.
- Il sistema non dipende da embedding o database vettoriali: usa ranking lessicale locale.
- Pensato per essere leggibile, estendibile e facile da self-hostare.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


# =========================
# Configurazione generale
# =========================

APP_VERSION = "1.0.0"
DEFAULT_OUTPUT_ROOT = Path("output")
DEFAULT_TIMEOUT = 180
MAX_CONTEXT_CHARS = 32000
DEFAULT_TOP_CHUNKS = 8
DEFAULT_CHUNK_SIZE = 2200
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_SUMMARY_CHARS = 10000
USER_AGENT = f"multi-llm-rpg-rules/{APP_VERSION}"

PROFILE_PRESETS: Dict[str, str] = {
    "rules_lawyer": (
        "Analizza come un rules lawyer rigoroso. Dai priorita' a coerenza interna, eccezioni, conflitti tra regole, definizioni implicite,"
        " edge case e formulazioni ambigue. Quando proponi cambiamenti, minimizza le rotture del sistema esistente."
    ),
    "systems_designer": (
        "Analizza come un systems designer. Dai priorita' a bilanciamento, progressione, economia delle risorse, scalabilita', exploit possibili"
        " e impatto sistemico delle modifiche sul resto del regolamento."
    ),
    "gm_experience": (
        "Analizza come un game master esperto. Dai priorita' a facilita' d'uso al tavolo, velocita' di risoluzione, chiarezza per i giocatori,"
        " gestione del pacing e carico cognitivo del master."
    ),
    "narrative_designer": (
        "Analizza come un narrative designer. Dai priorita' a fantasy del personaggio, fiction first, tono, identita' delle classi/abilità"
        " e coerenza tra meccaniche e immaginario del gioco."
    ),
    "new_player": (
        "Analizza come un playtester nuovo al gioco. Dai priorita' a onboarding, leggibilita', intuibilita', punti di attrito e punti dove"
        " un giocatore inesperto potrebbe sbagliare interpretazione."
    ),
}

DEFAULT_IGNORE_PATTERNS = [
    ".git/*",
    ".github/*",
    "node_modules/*",
    "venv/*",
    ".venv/*",
    "dist/*",
    "build/*",
    "site/*",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.svg",
    "*.pdf",
    "*.zip",
    "*.7z",
    "*.mp3",
    "*.mp4",
]

SYSTEM_PROMPT = """Sei un assistente esperto di game design e analisi di regolamenti per giochi di ruolo tabletop.
Leggi il materiale fornito come fonte primaria. Non inventare regole assenti.
Se una conclusione e' incerta, dichiaralo esplicitamente.
Rispondi in italiano, in modo strutturato, concreto e utile allo sviluppo.
Usa sempre questa struttura:
1. Lettura del problema
2. Cosa emerge dalle regole fornite
3. Criticita' o opportunita'
4. Proposta operativa
5. Impatti collaterali / trade-off
6. Test consigliati al tavolo
Quando possibile cita i file o le sezioni da cui inferisci le conclusioni.
"""

SUMMARY_PROMPT = """Sei un analista di regolamenti GdR. Costruisci un summary tecnico ad alta densita' informativa.
Obiettivi:
- identificare tema, loop di gioco, risoluzione azioni, combattimento, progressione, risorse, condizioni, eccezioni
- elencare termini ricorrenti e concetti chiave
- segnalare ambiguita', incoerenze o aree poco definite
- produrre un riassunto utile per rispondere a domande di sviluppo future
Rispondi in JSON valido con questa forma:
{
  "game_identity": "...",
  "core_loops": ["..."],
  "resolution_rules": ["..."],
  "combat_rules": ["..."],
  "progression_rules": ["..."],
  "resource_economy": ["..."],
  "classes_or_roles": ["..."],
  "magic_or_powers": ["..."],
  "conditions_and_status": ["..."],
  "ambiguities": ["..."],
  "keywords": ["..."],
  "summary_text": "..."
}
"""


# =========================
# Data model
# =========================

@dataclass
class SourceDocument:
    path: str
    title: str
    content: str
    sha256: str
    chars: int
    headings: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    title: str
    text: str
    start_char: int
    end_char: int
    token_estimate: int
    headings: List[str] = field(default_factory=list)


@dataclass
class RankedChunk:
    chunk: Chunk
    score: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class ModelTarget:
    provider: str
    model: str
    profile: str


@dataclass
class ModelAnswer:
    provider: str
    model: str
    profile: str
    success: bool
    latency_s: float
    response_text: str
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


# =========================
# Utility
# =========================

def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "run"


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None

    match = re.search(r"(\{.*\})", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", "    ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_headings(md: str) -> List[str]:
    headings = []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            headings.append(m.group(2).strip())
    return headings


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def first_heading_or_filename(path: str, content: str) -> str:
    hs = extract_headings(content)
    if hs:
        return hs[0]
    return Path(path).stem


def split_markdown_into_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_heading = "intro"
    current_lines: List[str] = []

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = m.group(2).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return [(h, s) for h, s in sections if s.strip()]


def iter_files_recursive(base: Path, patterns: List[str]) -> Iterable[Path]:
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()
        if any(fnmatch.fnmatch(rel, patt) for patt in patterns):
            continue
        if p.suffix.lower() == ".md":
            yield p


# =========================
# Git / file ingest
# =========================

def clone_repo(repo_url: str, workdir: Path) -> Path:
    repo_dir = workdir / "repo"
    cmd = ["git", "clone", "--depth", "1", repo_url, str(repo_dir)]
    log(f"[clone] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return repo_dir


def expand_local_file_patterns(patterns: List[str]) -> List[Path]:
    resolved: List[Path] = []
    for patt in patterns:
        matched = list(Path(".").glob(patt)) if any(ch in patt for ch in "*?[]") else [Path(patt)]
        for p in matched:
            if p.is_file() and p.suffix.lower() == ".md":
                resolved.append(p.resolve())
    uniq = []
    seen = set()
    for p in resolved:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def load_documents_from_repo(repo_url: str, cache_root: Path) -> List[SourceDocument]:
    with tempfile.TemporaryDirectory(prefix="rpg_repo_") as tmp:
        repo_dir = clone_repo(repo_url, Path(tmp))
        docs: List[SourceDocument] = []
        for file_path in iter_files_recursive(repo_dir, DEFAULT_IGNORE_PATTERNS):
            rel = file_path.relative_to(repo_dir).as_posix()
            content = normalize_markdown(read_text_file(file_path))
            if not content.strip():
                continue
            docs.append(SourceDocument(
                path=rel,
                title=first_heading_or_filename(rel, content),
                content=content,
                sha256=sha256_text(content),
                chars=len(content),
                headings=extract_headings(content),
            ))
        return docs


def load_documents_from_files(files: List[str]) -> List[SourceDocument]:
    docs: List[SourceDocument] = []
    for p in expand_local_file_patterns(files):
        content = normalize_markdown(read_text_file(p))
        if not content.strip():
            continue
        docs.append(SourceDocument(
            path=str(p),
            title=first_heading_or_filename(str(p), content),
            content=content,
            sha256=sha256_text(content),
            chars=len(content),
            headings=extract_headings(content),
        ))
    return docs


# =========================
# Chunking e ranking
# =========================

def chunk_document(doc: SourceDocument, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Chunk]:
    sections = split_markdown_into_sections(doc.content)
    chunks: List[Chunk] = []
    chunk_index = 0

    for heading, section_text in sections:
        if len(section_text) <= chunk_size:
            cid = f"{slugify(doc.path)}-{chunk_index:04d}"
            chunks.append(Chunk(
                chunk_id=cid,
                source_path=doc.path,
                title=doc.title,
                text=section_text,
                start_char=0,
                end_char=len(section_text),
                token_estimate=estimate_tokens(section_text),
                headings=[heading],
            ))
            chunk_index += 1
            continue

        start = 0
        while start < len(section_text):
            end = min(len(section_text), start + chunk_size)
            window = section_text[start:end]
            if end < len(section_text):
                last_break = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
                if last_break > chunk_size // 2:
                    end = start + last_break + 1
                    window = section_text[start:end]
            cid = f"{slugify(doc.path)}-{chunk_index:04d}"
            chunks.append(Chunk(
                chunk_id=cid,
                source_path=doc.path,
                title=doc.title,
                text=window.strip(),
                start_char=start,
                end_char=end,
                token_estimate=estimate_tokens(window),
                headings=[heading],
            ))
            chunk_index += 1
            if end >= len(section_text):
                break
            start = max(0, end - overlap)

    return chunks


def build_keyword_counter(documents: List[SourceDocument]) -> Counter:
    cnt: Counter = Counter()
    for doc in documents:
        words = re.findall(r"[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_-]{2,}", doc.content.lower())
        cnt.update(words)
    return cnt


def query_terms(question: str) -> List[str]:
    terms = re.findall(r"[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_-]{2,}", question.lower())
    stop = {
        "che", "come", "quale", "quali", "delle", "della", "dello", "degli", "della", "nelle", "nello", "sugli",
        "per", "con", "una", "uno", "del", "dei", "dai", "alla", "allo", "dalla", "dalle", "sulle", "sulla",
        "sistema", "regole", "gioco", "gdr", "rpg"
    }
    return [t for t in terms if t not in stop]


def rank_chunks(question: str, chunks: List[Chunk], summary: Optional[Dict[str, Any]] = None, top_k: int = DEFAULT_TOP_CHUNKS) -> List[RankedChunk]:
    qterms = query_terms(question)
    qset = set(qterms)
    summary_keywords = set((summary or {}).get("keywords", [])[:40]) if isinstance(summary, dict) else set()
    ranked: List[RankedChunk] = []

    for chunk in chunks:
        text_low = chunk.text.lower()
        score = 0.0
        reasons: List[str] = []

        hits = sum(text_low.count(term) for term in qset)
        if hits:
            score += hits * 3.0
            reasons.append(f"match termini domanda={hits}")

        heading_hits = sum(1 for h in chunk.headings if any(t in h.lower() for t in qset))
        if heading_hits:
            score += heading_hits * 4.0
            reasons.append(f"match heading={heading_hits}")

        kw_hits = sum(1 for kw in summary_keywords if kw and kw.lower() in text_low)
        if kw_hits:
            score += min(kw_hits, 10) * 0.5
            reasons.append(f"match keywords summary={kw_hits}")

        if any(word in text_low for word in ["bilanci", "balance", "progress", "danno", "abilità", "skill", "spell", "magic", "incantes"]):
            score += 1.2

        score += min(chunk.token_estimate / 400, 2.0)
        ranked.append(RankedChunk(chunk=chunk, score=score, reasons=reasons))

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]


def compose_context(summary: Optional[Dict[str, Any]], ranked_chunks: List[RankedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts: List[str] = []

    if summary:
        parts.append("## SUMMARY STRUTTURATO")
        parts.append(json.dumps(summary, ensure_ascii=False, indent=2))

    parts.append("## ESTRATTI RILEVANTI")
    current = "\n\n".join(parts)

    for rc in ranked_chunks:
        block = (
            f"\n\n### CHUNK {rc.chunk.chunk_id}\n"
            f"source_path: {rc.chunk.source_path}\n"
            f"title: {rc.chunk.title}\n"
            f"headings: {', '.join(rc.chunk.headings)}\n"
            f"score: {rc.score:.2f}\n"
            f"text:\n{rc.chunk.text}\n"
        )
        if len(current) + len(block) > max_chars:
            break
        current += block

    return current


# =========================
# Provider clients
# =========================

def post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Tuple[Dict[str, Any], int]:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    status = resp.status_code
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text}
    if status >= 400:
        raise RuntimeError(f"HTTP {status}: {json.dumps(data, ensure_ascii=False)[:1200]}")
    return data, status


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


def call_openrouter(model: str, messages: List[Dict[str, str]], api_key: str, response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "multi-llm-rpg-rules",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
    }
    if response_format:
        payload["response_format"] = response_format
    data, _ = post_json(url, headers, payload)
    return data


def call_perplexity(model: str, messages: List[Dict[str, str]], api_key: str) -> Dict[str, Any]:
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
    }
    data, _ = post_json(url, headers, payload)
    return data


def call_nvidia_nim(model: str, messages: List[Dict[str, str]], api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    # Molti endpoint NIM espongono un'interfaccia OpenAI-compatible.
    # Il base_url puo' essere passato da CLI per adattarsi al modello/tenant specifico.
    url = (base_url or "https://integrate.api.nvidia.com/v1") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
    }
    data, _ = post_json(url, headers, payload)
    return data


def call_gemini_generate_content(model: str, api_key: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.5,
        }
    }
    data, _ = post_json(url, headers, payload)
    return data


def extract_gemini_text(data: Dict[str, Any]) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "\n".join(part.get("text", "") for part in parts).strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False)


# =========================
# Prompting
# =========================

def make_summary_messages(context: str) -> Tuple[str, str]:
    user = (
        "Analizza il seguente corpus Markdown relativo a un gioco di ruolo tabletop. "
        "Costruisci un summary tecnico adatto a domande future di design e sviluppo.\n\n"
        f"{context}"
    )
    return SUMMARY_PROMPT, user


def build_question_prompt(question: str, summary: Optional[Dict[str, Any]], ranked_chunks: List[RankedChunk], profile_instruction: str) -> str:
    context = compose_context(summary, ranked_chunks)
    return textwrap.dedent(f"""
    Contesto del progetto:
    Stai analizzando un corpus di regole GdR in Markdown gia' preprocessato.

    Profilo di risposta:
    {profile_instruction}

    Domanda di sviluppo:
    {question}

    Materiale di riferimento:
    {context}

    Vincoli:
    - usa solo il materiale fornito come base
    - se manca informazione, dichiaralo
    - distingui osservazioni, inferenze e proposte
    - quando possibile cita source_path o headings
    - non limitarti a teoria astratta: proponi modifiche operative
    """).strip()


def messages_for_openai_style(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# =========================
# Summary generation
# =========================

def build_corpus_for_summary(documents: List[SourceDocument], max_chars: int = 60000) -> str:
    blocks: List[str] = []
    current_len = 0
    for doc in documents:
        block = f"\n\n# FILE: {doc.path}\nTITLE: {doc.title}\n\n{trim_text(doc.content, 12000)}"
        if current_len + len(block) > max_chars:
            break
        blocks.append(block)
        current_len += len(block)
    return "".join(blocks).strip()


def generate_summary_with_provider(provider: str, model: str, documents: List[SourceDocument]) -> Dict[str, Any]:
    corpus = build_corpus_for_summary(documents)
    system_prompt, user_prompt = make_summary_messages(corpus)

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY non impostata")
        data = call_openrouter(
            model=model,
            messages=messages_for_openai_style(user_prompt),
            api_key=api_key,
            response_format={"type": "json_object"},
        )
        parsed = safe_json_loads(extract_openai_like_text(data))
        return parsed or {"summary_text": extract_openai_like_text(data)}

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY non impostata")
        data = call_gemini_generate_content(model=model, api_key=api_key, system_prompt=system_prompt, user_prompt=user_prompt)
        parsed = safe_json_loads(extract_gemini_text(data))
        return parsed or {"summary_text": extract_gemini_text(data)}

    raise RuntimeError(f"Provider summary non supportato: {provider}")


# =========================
# Knowledge pack
# =========================

def build_knowledge_pack(documents: List[SourceDocument], cache_dir: Path, summary_provider: str = "gemini", summary_model: str = "gemini-2.5-pro") -> Dict[str, Any]:
    ensure_dir(cache_dir)

    docs_json = [asdict(d) for d in documents]
    write_json(cache_dir / "documents.json", docs_json)

    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    write_json(cache_dir / "chunks.json", [asdict(c) for c in all_chunks])

    stats = {
        "documents": len(documents),
        "chunks": len(all_chunks),
        "total_chars": sum(d.chars for d in documents),
        "keywords_top": build_keyword_counter(documents).most_common(120),
    }
    write_json(cache_dir / "stats.json", stats)

    summary: Dict[str, Any] = {}
    try:
        summary = generate_summary_with_provider(summary_provider, summary_model, documents)
    except Exception as e:
        summary = {
            "summary_text": f"Summary automatica non disponibile: {e}",
            "keywords": [k for k, _ in build_keyword_counter(documents).most_common(50)],
            "ambiguities": [],
        }
    write_json(cache_dir / "summary.json", summary)

    manifest = {
        "version": APP_VERSION,
        "created_at": now_ts(),
        "summary_provider": summary_provider,
        "summary_model": summary_model,
        "documents": len(documents),
        "chunks": len(all_chunks),
    }
    write_json(cache_dir / "manifest.json", manifest)

    return {
        "documents": documents,
        "chunks": all_chunks,
        "summary": summary,
        "stats": stats,
        "manifest": manifest,
    }


def load_knowledge_pack(cache_dir: Path) -> Dict[str, Any]:
    docs_data = json.loads((cache_dir / "documents.json").read_text(encoding="utf-8"))
    chunks_data = json.loads((cache_dir / "chunks.json").read_text(encoding="utf-8"))
    summary = json.loads((cache_dir / "summary.json").read_text(encoding="utf-8"))
    stats = json.loads((cache_dir / "stats.json").read_text(encoding="utf-8")) if (cache_dir / "stats.json").exists() else {}
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8")) if (cache_dir / "manifest.json").exists() else {}

    documents = [SourceDocument(**d) for d in docs_data]
    chunks = [Chunk(**c) for c in chunks_data]
    return {
        "documents": documents,
        "chunks": chunks,
        "summary": summary,
        "stats": stats,
        "manifest": manifest,
    }


# =========================
# Multi-model querying
# =========================

def answer_with_target(target: ModelTarget, question: str, summary: Dict[str, Any], chunks: List[Chunk], nvidia_base_url: Optional[str] = None) -> ModelAnswer:
    ranked = rank_chunks(question, chunks, summary=summary, top_k=DEFAULT_TOP_CHUNKS)
    profile_instruction = PROFILE_PRESETS.get(target.profile, PROFILE_PRESETS["systems_designer"])
    prompt = build_question_prompt(question, summary, ranked, profile_instruction)

    t0 = time.time()
    try:
        if target.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY non impostata")
            raw = call_openrouter(target.model, messages_for_openai_style(prompt), api_key)
            text = extract_openai_like_text(raw)

        elif target.provider == "perplexity":
            api_key = os.getenv("PERPLEXITY_API_KEY")
            if not api_key:
                raise RuntimeError("PERPLEXITY_API_KEY non impostata")
            raw = call_perplexity(target.model, messages_for_openai_style(prompt), api_key)
            text = extract_openai_like_text(raw)

        elif target.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY non impostata")
            raw = call_gemini_generate_content(target.model, api_key, SYSTEM_PROMPT, prompt)
            text = extract_gemini_text(raw)

        elif target.provider == "nvidia":
            api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
            if not api_key:
                raise RuntimeError("NVIDIA_API_KEY / NIM_API_KEY non impostata")
            raw = call_nvidia_nim(target.model, messages_for_openai_style(prompt), api_key, base_url=nvidia_base_url)
            text = extract_openai_like_text(raw)

        else:
            raise RuntimeError(f"Provider non supportato: {target.provider}")

        latency = time.time() - t0
        return ModelAnswer(
            provider=target.provider,
            model=target.model,
            profile=target.profile,
            success=True,
            latency_s=latency,
            response_text=text,
            raw=raw,
        )
    except Exception as e:
        latency = time.time() - t0
        return ModelAnswer(
            provider=target.provider,
            model=target.model,
            profile=target.profile,
            success=False,
            latency_s=latency,
            response_text="",
            error=str(e),
            raw=None,
        )


def build_targets(args: argparse.Namespace) -> List[ModelTarget]:
    targets: List[ModelTarget] = []
    profiles = args.profiles or ["systems_designer", "rules_lawyer", "gm_experience"]

    def add(provider: str, models: List[str]) -> None:
        for i, model in enumerate(models):
            profile = profiles[i % len(profiles)]
            targets.append(ModelTarget(provider=provider, model=model, profile=profile))

    add("openrouter", args.openrouter_model or [])
    add("gemini", args.gemini_model or [])
    add("perplexity", args.perplexity_model or [])
    add("nvidia", args.nvidia_model or [])
    return targets


# =========================
# Reporting
# =========================

def render_markdown_report(question: str, answers: List[ModelAnswer], summary: Dict[str, Any], ranked_chunks: List[RankedChunk]) -> str:
    lines: List[str] = []
    lines.append(f"# Report multi-LLM\n")
    lines.append(f"**Domanda:** {question}\n")

    if summary:
        lines.append("## Summary tecnico")
        if isinstance(summary, dict):
            st = summary.get("summary_text") or json.dumps(summary, ensure_ascii=False, indent=2)
            lines.append(trim_text(st, DEFAULT_SUMMARY_CHARS))
            lines.append("")

    lines.append("## Chunk usati")
    for rc in ranked_chunks:
        lines.append(f"- `{rc.chunk.chunk_id}` | `{rc.chunk.source_path}` | score={rc.score:.2f} | headings={', '.join(rc.chunk.headings)}")
    lines.append("")

    lines.append("## Risposte")
    for ans in answers:
        lines.append(f"### {ans.provider} :: {ans.model} :: {ans.profile}")
        if ans.success:
            lines.append(f"- success: true")
            lines.append(f"- latency_s: {ans.latency_s:.2f}")
            lines.append("")
            lines.append(ans.response_text.strip())
            lines.append("")
        else:
            lines.append(f"- success: false")
            lines.append(f"- latency_s: {ans.latency_s:.2f}")
            lines.append(f"- error: {ans.error}")
            lines.append("")
    return "\n".join(lines)


def save_answers_bundle(run_dir: Path, question: str, answers: List[ModelAnswer], summary: Dict[str, Any], chunks: List[Chunk]) -> None:
    ensure_dir(run_dir)
    write_json(run_dir / "answers.json", [asdict(a) for a in answers])

    ranked = rank_chunks(question, chunks, summary=summary, top_k=DEFAULT_TOP_CHUNKS)
    write_json(run_dir / "ranked_chunks.json", [
        {
            "score": rc.score,
            "reasons": rc.reasons,
            **asdict(rc.chunk)
        }
        for rc in ranked
    ])

    md = render_markdown_report(question, answers, summary, ranked)
    write_text(run_dir / "report.md", md)

    with (run_dir / "answers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["provider", "model", "profile", "success", "latency_s", "error", "response_text"])
        writer.writeheader()
        for a in answers:
            writer.writerow({
                "provider": a.provider,
                "model": a.model,
                "profile": a.profile,
                "success": a.success,
                "latency_s": f"{a.latency_s:.2f}",
                "error": a.error or "",
                "response_text": a.response_text,
            })


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analizza regole GdR in Markdown e interroga piu' LLM.")

    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--repo", help="URL del repository GitHub da clonare")
    src.add_argument("--files", nargs="+", help="Lista di file .md o glob locali")

    p.add_argument("--cache-dir", help="Directory cache esistente da riutilizzare")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Cartella root output")
    p.add_argument("--question", help="Domanda di sviluppo da porre ai modelli")
    p.add_argument("--build-only", action="store_true", help="Costruisce solo il knowledge pack")

    p.add_argument("--profiles", nargs="+", choices=sorted(PROFILE_PRESETS.keys()), help="Profili di analisi da ciclare sui modelli")

    p.add_argument("--summary-provider", default="gemini", choices=["gemini", "openrouter"], help="Provider per costruire il summary iniziale")
    p.add_argument("--summary-model", default="gemini-2.5-pro", help="Modello usato per il summary iniziale")

    p.add_argument("--openrouter-model", action="append", help="Modello OpenRouter; ripeti il flag per piu' modelli")
    p.add_argument("--gemini-model", action="append", help="Modello Gemini; ripeti il flag per piu' modelli")
    p.add_argument("--perplexity-model", action="append", help="Modello Perplexity; ripeti il flag per piu' modelli")
    p.add_argument("--nvidia-model", action="append", help="Modello NVIDIA NIM; ripeti il flag per piu' modelli")
    p.add_argument("--nvidia-base-url", help="Base URL OpenAI-compatible per endpoint NVIDIA NIM custom")

    p.add_argument("--force-rebuild", action="store_true", help="Ricostruisce la cache anche se esiste")
    return p.parse_args()


def create_run_paths(output_root: Path, source_label: str) -> Tuple[Path, Path]:
    run_id = f"{now_ts()}-{slugify(source_label)[:40]}"
    run_dir = ensure_dir(output_root / "runs" / run_id)
    cache_dir = ensure_dir(output_root / "cache" / run_id)
    return run_dir, cache_dir


def detect_source_label(args: argparse.Namespace) -> str:
    if args.repo:
        return Path(args.repo.rstrip("/")).name.replace(".git", "") or "repo"
    if args.files:
        return "files-batch"
    if args.cache_dir:
        return Path(args.cache_dir).name
    return "run"


def main() -> int:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    source_label = detect_source_label(args)
    run_dir, default_cache_dir = create_run_paths(output_root, source_label)

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        kp = load_knowledge_pack(cache_dir)
    else:
        cache_dir = default_cache_dir
        if not args.repo and not args.files:
            print("Errore: serve --repo oppure --files, oppure --cache-dir.", file=sys.stderr)
            return 2

        if args.repo:
            documents = load_documents_from_repo(args.repo, cache_dir)
        else:
            documents = load_documents_from_files(args.files or [])

        if not documents:
            print("Nessun documento Markdown trovato.", file=sys.stderr)
            return 2

        kp = build_knowledge_pack(
            documents=documents,
            cache_dir=cache_dir,
            summary_provider=args.summary_provider,
            summary_model=args.summary_model,
        )

    write_json(run_dir / "knowledge_manifest.json", kp.get("manifest", {}))
    write_json(run_dir / "knowledge_summary.json", kp.get("summary", {}))

    if args.build_only:
        print(str(cache_dir))
        return 0

    if not args.question:
        print("Errore: serve --question se non usi --build-only.", file=sys.stderr)
        return 2

    targets = build_targets(args)
    if not targets:
        print("Errore: specifica almeno un modello target (--openrouter-model / --gemini-model / --perplexity-model / --nvidia-model).", file=sys.stderr)
        return 2

    answers: List[ModelAnswer] = []
    for target in targets:
        log(f"[ask] provider={target.provider} model={target.model} profile={target.profile}")
        ans = answer_with_target(
            target=target,
            question=args.question,
            summary=kp["summary"],
            chunks=kp["chunks"],
            nvidia_base_url=args.nvidia_base_url,
        )
        answers.append(ans)

    save_answers_bundle(run_dir, args.question, answers, kp["summary"], kp["chunks"])

    final_index = {
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "question": args.question,
        "targets": [asdict(t) for t in targets],
        "answers_ok": sum(1 for a in answers if a.success),
        "answers_fail": sum(1 for a in answers if not a.success),
    }
    write_json(run_dir / "run_index.json", final_index)

    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
