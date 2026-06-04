from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from .models import Answer
from .retrieval import rank_chunks
from .utils import trim_text, write_json, write_text

def render_report(question: str, kp: Dict[str, Any], answers: List[Answer], judge: Optional[Dict[str, Any]], settings: Dict[str, Any]) -> str:
    ranked = rank_chunks(question, kp["chunks"], kp["summary"], settings.get("retrieval", {}).get("top_k", 8))
    lines = [
        "# Report multi-LLM modulare",
        "",
        f"**Domanda:** {question}",
        "",
        "## Knowledge pack",
        f"- documenti: {len(kp['documents'])}",
        f"- chunk: {len(kp['chunks'])}",
        "",
        "## Summary tecnico",
        trim_text(kp["summary"].get("summary_text", json.dumps(kp["summary"], ensure_ascii=False, indent=2)), 12000),
        "",
        "## Chunk principali",
    ]
    for rc in ranked:
        lines.append(f"- `{rc.chunk.chunk_id}` | `{rc.chunk.source_path}` | score={rc.score:.2f} | headings={', '.join(rc.chunk.headings)}")
    lines += ["", "## Risposte"]
    for a in answers:
        lines += [f"### {a.id}", f"- success: {str(a.success).lower()}", f"- latency_s: {a.latency_s:.2f}"]
        if a.error:
            lines.append(f"- error: {a.error}")
        lines.append("")
        if a.success:
            lines += [a.response_text, ""]
    if judge:
        lines += ["## Judge", json.dumps(judge, ensure_ascii=False, indent=2), ""]
    return "\n".join(lines)

def save_outputs(run_dir: Path, question: str, kp: Dict[str, Any], answers: List[Answer], judge: Optional[Dict[str, Any]], settings: Dict[str, Any]) -> None:
    write_json(run_dir / "answers.json", [a.to_dict() for a in answers])
    with (run_dir / "answers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "provider", "model", "profile", "success", "latency_s", "error", "response_text", "prompt_path"])
        w.writeheader()
        for a in answers:
            w.writerow({"id": a.id, "provider": a.provider, "model": a.model, "profile": a.profile, "success": a.success, "latency_s": f"{a.latency_s:.2f}", "error": a.error or "", "response_text": a.response_text, "prompt_path": a.prompt_path or ""})
    write_text(run_dir / "report.md", render_report(question, kp, answers, judge, settings))
