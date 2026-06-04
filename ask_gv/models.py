from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

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
class Target:
    provider: str
    model: str
    profile: str
    temperature: float = 0.4
    max_tokens: Optional[int] = None
    enabled: bool = True
    label: Optional[str] = None

@dataclass
class Answer:
    id: str
    provider: str
    model: str
    profile: str
    success: bool
    latency_s: float
    response_text: str
    prompt_path: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
