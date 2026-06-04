from __future__ import annotations
import argparse
from pathlib import Path
from .config import load_config, targets_from_config
from .defaults import DEFAULT_IGNORE_PATTERNS
from .ingest import load_documents_from_files, load_documents_from_repo
from .observability import get_logger, setup_logging, shutdown_logging
from .pipeline import build_knowledge_pack, create_run_dirs, load_knowledge_pack, run_judge, run_targets
from .reporting import save_outputs
from .utils import append_jsonl, ensure_dir, write_json
from dotenv import load_dotenv

def source_label(args) -> str:
    if args.repo:
        return Path(args.repo.rstrip("/")).name.replace(".git", "") or "repo"
    if args.files:
        return "files-batch"
    if args.cache_dir:
        return Path(args.cache_dir).name
    return "run"

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline modulare multi-LLM per regolamenti GdR.")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--repo")
    g.add_argument("--files", nargs="+")
    p.add_argument("--cache-dir")
    p.add_argument("--config", required=True)
    p.add_argument("--question")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--output-root", default="output")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    cfg = load_config(Path(args.config))
    output_root = ensure_dir(Path(args.output_root))
    run_dir, default_cache_dir = create_run_dirs(output_root, source_label(args))
    setup_logging(run_dir / "logs", run_dir.name)
    log = get_logger(__name__, "cli")
    log.info("cli started", extra={"event": "cli_started", "run_dir": str(run_dir)})
    log_path = run_dir / "run_log.jsonl"
    append_jsonl(log_path, {"event": "start"})
    try:
        if args.cache_dir:
            cache_dir = Path(args.cache_dir)
            log.info("loading cache", extra={"event": "cache_loading", "cache_dir": str(cache_dir)})
            kp = load_knowledge_pack(cache_dir)
            append_jsonl(log_path, {"event": "cache_loaded", "cache_dir": str(cache_dir)})
        else:
            if not args.repo and not args.files:
                raise SystemExit("Errore: specifica --repo oppure --files oppure --cache-dir")
            log.info("loading documents", extra={"event": "documents_loading", "path": args.repo or ",".join(args.files or [])})
            docs = load_documents_from_repo(args.repo, cfg.get("input", {}).get("ignore_patterns", DEFAULT_IGNORE_PATTERNS)) if args.repo else load_documents_from_files(args.files or [])
            if not docs:
                raise SystemExit("Nessun documento Markdown trovato.")
            cache_dir = default_cache_dir
            log.info("building knowledge pack", extra={"event": "knowledge_building", "count": len(docs), "cache_dir": str(cache_dir)})
            kp = build_knowledge_pack(docs, cache_dir, cfg)
            append_jsonl(log_path, {"event": "knowledge_built", "documents": len(docs), "cache_dir": str(cache_dir)})
        write_json(run_dir / "knowledge_summary.json", kp.get("summary", {}))
        write_json(run_dir / "knowledge_manifest.json", kp.get("manifest", {}))
        write_json(run_dir / "effective_config.json", cfg)
        if args.build_only:
            log.info("build only completed", extra={"event": "build_only_completed", "cache_dir": str(cache_dir)})
            print(str(cache_dir))
            return 0
        question = args.question or cfg.get("run", {}).get("question")
        if not question:
            raise SystemExit("Errore: manca la question, passala con --question o nel config JSON.")
        targets = targets_from_config(cfg)
        if not targets:
            raise SystemExit("Errore: nessun target abilitato nel config.")
        append_jsonl(log_path, {"event": "targets_ready", "count": len(targets)})
        log.info("running targets", extra={"event": "targets_running", "count": len(targets)})
        answers = run_targets(question, targets, kp, run_dir, cfg, log_path)
        judge = run_judge(question, answers, kp, cfg, run_dir)
        save_outputs(run_dir, question, kp, answers, judge, cfg)
        write_json(run_dir / "run_index.json", {"run_dir": str(run_dir), "cache_dir": str(cache_dir), "answers_ok": sum(1 for a in answers if a.success), "answers_fail": sum(1 for a in answers if not a.success)})
        log.info("run completed", extra={"event": "run_completed", "run_dir": str(run_dir), "cache_dir": str(cache_dir), "count": len(answers)})
        print(str(run_dir))
        return 0
    finally:
        shutdown_logging()

if __name__ == "__main__":
    raise SystemExit(main())
