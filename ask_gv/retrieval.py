from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from .models import Chunk, RankedChunk, SourceDocument
from .utils import estimate_tokens, slugify

def split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    heading = "intro"
    lines: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            if lines:
                sections.append((heading, "\n".join(lines).strip()))
                lines = []
            heading = m.group(2).strip()
            lines.append(line)
        else:
            lines.append(line)
    if lines:
        sections.append((heading, "\n".join(lines).strip()))
    return [(h, s) for h, s in sections if s.strip()]

def chunk_document(doc: SourceDocument, chunk_size: int, overlap: int) -> List[Chunk]:
    out: List[Chunk] = []
    idx = 0
    for heading, sec in split_markdown_sections(doc.content):
        if len(sec) <= chunk_size:
            out.append(Chunk(f"{slugify(doc.path)}-{idx:04d}", doc.path, doc.title, sec, 0, len(sec), estimate_tokens(sec), [heading]))
            idx += 1
            continue
        start = 0
        while start < len(sec):
            end = min(len(sec), start + chunk_size)
            window = sec[start:end]
            if end < len(sec):
                lb = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
                if lb > chunk_size // 2:
                    end = start + lb + 1
                    window = sec[start:end]
            out.append(Chunk(f"{slugify(doc.path)}-{idx:04d}", doc.path, doc.title, window.strip(), start, end, estimate_tokens(window), [heading]))
            idx += 1
            if end >= len(sec):
                break
            start = max(0, end - overlap)
    return out

def build_keyword_counter(documents: List[SourceDocument]) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for d in documents:
        for w in re.findall(r"[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_-]{2,}", d.content.lower()):
            cnt[w] = cnt.get(w, 0) + 1
    return dict(sorted(cnt.items(), key=lambda kv: kv[1], reverse=True))

def query_terms(question: str) -> List[str]:
    terms = re.findall(r"[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_-]{2,}", question.lower())
    stop = {"che", "come", "quale", "quali", "delle", "della", "dello", "degli", "nelle", "nello", "per", "con", "una", "uno", "del", "dei", "dai", "alla", "allo", "dalla", "dalle", "sulle", "sulla", "sistema", "regole", "gioco", "gdr", "rpg"}
    return [t for t in terms if t not in stop]

def rank_chunks(question: str, chunks: List[Chunk], summary: Optional[Dict[str, Any]], top_k: int) -> List[RankedChunk]:
    qset = set(query_terms(question))
    sk = set((summary or {}).get("keywords", [])[:60]) if isinstance(summary, dict) else set()
    ranked: List[RankedChunk] = []
    for c in chunks:
        score = 0.0
        reasons: List[str] = []
        tl = c.text.lower()
        qhits = sum(tl.count(q) for q in qset)
        if qhits:
            score += qhits * 3.0
            reasons.append(f"qhits={qhits}")
        hhits = sum(1 for h in c.headings if any(q in h.lower() for q in qset))
        if hhits:
            score += hhits * 4.0
            reasons.append(f"hhits={hhits}")
        khits = sum(1 for kw in sk if kw and kw.lower() in tl)
        if khits:
            score += min(khits, 12) * 0.4
            reasons.append(f"khits={khits}")
        if any(w in tl for w in ["danno", "spell", "magic", "mana", "slot", "classe", "azione", "turno", "progress"]):
            score += 1.2
        score += min(c.token_estimate / 500, 2.0)
        ranked.append(RankedChunk(c, score, reasons))
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]

def compose_context(summary: Optional[Dict[str, Any]], ranked_chunks: List[RankedChunk], max_chars: int) -> str:
    parts: List[str] = []
    if summary:
        parts += ["## SUMMARY STRUTTURATO", json.dumps(summary, ensure_ascii=False, indent=2)]
    parts.append("## ESTRATTI RILEVANTI")
    current = "\n\n".join(parts)
    for rc in ranked_chunks:
        block = f"\n\n### CHUNK {rc.chunk.chunk_id}\nsource_path: {rc.chunk.source_path}\ntitle: {rc.chunk.title}\nheadings: {', '.join(rc.chunk.headings)}\nscore: {rc.score:.2f}\ntext:\n{rc.chunk.text}\n"
        if len(current) + len(block) > max_chars:
            break
        current += block
    return current
