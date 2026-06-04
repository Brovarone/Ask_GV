from __future__ import annotations
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower().strip())
    return re.sub(r"-+", "-", value).strip("-") or "run"

def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def trim_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def extract_headings(md: str) -> list[str]:
    out = []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            out.append(m.group(2).strip())
    return out

def first_heading_or_filename(path: str, content: str) -> str:
    hs = extract_headings(content)
    return hs[0] if hs else Path(path).stem

def safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    for pat in [r"```json\s*(\{.*?\}|\[.*?\])\s*```", r"(\{.*\})"]:
        m = re.search(pat, text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None

def retry_request(fn, retries: int = 3, base_sleep: float = 1.2):
    last = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt == retries:
                break
            time.sleep(base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.4))
    raise last
