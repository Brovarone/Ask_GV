from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from .models import Target
from .utils import read_text

def load_config(config_path: Path) -> Dict[str, Any]:
    return json.loads(read_text(config_path))

def targets_from_config(cfg: Dict[str, Any]) -> List[Target]:
    out: List[Target] = []
    for t in cfg.get("targets", []):
        if not t.get("enabled", True):
            continue
        out.append(Target(
            provider=t["provider"],
            model=t["model"],
            profile=t.get("profile", "systems_designer"),
            temperature=t.get("temperature", 0.4),
            max_tokens=t.get("max_tokens"),
            enabled=t.get("enabled", True),
            label=t.get("label"),
        ))
    return out
