from __future__ import annotations
import concurrent.futures as cf
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .defaults import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_CONTEXT_CHARS, DEFAULT_MAX_WORKERS, DEFAULT_TOP_K, PROFILE_PRESETS, SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT
from .models import Answer, Chunk, SourceDocument, Target
from .observability import get_logger
from .providers.base import ProviderTimeoutError, ProviderRateLimitError, ProviderHTTPError
from .providers.registry import call_provider
from .retrieval import build_keyword_counter, chunk_document, compose_context, rank_chunks
from .summary import fallback_summary, generate_summary
from .utils import append_jsonl, ensure_dir, now_ts, safe_json_loads, slugify, write_json, write_text

def build_knowledge_pack(documents: List[SourceDocument], cache_dir: Path, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Build the knowledge pack from documents, create chunks, generate summary, and write intermediate files.

    Args:
        documents: List of SourceDocument objects.
        cache_dir: Path where intermediate files (documents.json, chunks.json, etc.) will be stored.
        settings: Configuration dictionary containing retrieval and summary options.

    Returns:
        A dictionary with keys 'documents', 'chunks', 'summary', 'stats', and 'manifest'.
    """
    log = get_logger(__name__, "pipeline")
    log.info("knowledge pack start", extra={"event": "knowledge_pack_start", "count": len(documents), "cache_dir": str(cache_dir)})
    ensure_dir(cache_dir)

    # Chunking
    retrieval_cfg = settings.get("retrieval", {})
    chunk_size = retrieval_cfg.get('chunk_size', DEFAULT_CHUNK_SIZE)
    overlap = retrieval_cfg.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)
    chunks: List[Chunk] = []
    for d in documents:
        chunks.extend(chunk_document(d, chunk_size, overlap))

    summary_cfg  = settings.get('summary', {})
    
    summary = generate_summary(
        provider=summary_cfg.get('provider', 'gemini'),
        model=summary_cfg.get('model', 'gemini-2.5-pro'),
        documents=documents,
        settings=settings,
        work_dir=cache_dir,
    )
    log.info(
        'summary generated',
        extra={'event': 'summary_generated', 'provider': summary_cfg .get('provider', 'gemini'), 'model': summary_cfg .get('model', 'gemini-2.5-pro')},
    )           

    # Statistics
    stats = {
        'documents': len(documents),
        'chunks': len(chunks),
        'total_chars': sum(d.chars for d in documents),
        'keywords_top': list(build_keyword_counter(documents).items())[:150],
    }

    # Manifest
    manifest = {'created_at': now_ts(), 
                'documents': len(documents), 
                'chunks': len(chunks)}

    # Write intermediate files
    write_json(cache_dir / 'documents.json', [d.__dict__ for d in documents])
    write_json(cache_dir / 'chunks.json', [c.__dict__ for c in chunks])
    write_json(cache_dir / 'summary.json', summary)
    write_json(cache_dir / 'stats.json', stats)
    write_json(cache_dir / 'manifest.json', manifest)

    write_text(
        cache_dir / "corpus_full.md",
        "\n\n".join([f"# FILE: {d.path}\n\n{d.content}" for d in documents]),
    )

    log.info("knowledge pack completed", extra={"event": "knowledge_pack_completed", "count": len(chunks), "cache_dir": str(cache_dir)})
    return {"documents": documents, "chunks": chunks, "summary": summary, "stats": stats, "manifest": manifest}

def load_knowledge_pack(cache_dir: Path) -> Dict[str, Any]:
    documents = [SourceDocument(**x) for x in json.loads((cache_dir / "documents.json").read_text(encoding="utf-8"))]
    chunks = [Chunk(**x) for x in json.loads((cache_dir / "chunks.json").read_text(encoding="utf-8"))]
    summary = json.loads((cache_dir / "summary.json").read_text(encoding="utf-8"))
    stats = json.loads((cache_dir / "stats.json").read_text(encoding="utf-8")) if (cache_dir / "stats.json").exists() else {}
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8")) if (cache_dir / "manifest.json").exists() else {}
    return {"documents": documents, "chunks": chunks, "summary": summary, "stats": stats, "manifest": manifest}


def build_user_prompt(question: str, summary: Dict[str, Any], chunks: List[Chunk], profile: str, top_k: int, max_chars: int) -> str:
    ranked = rank_chunks(question, chunks, summary, top_k)
    context = compose_context(summary, ranked, max_chars)
    profile_instruction = PROFILE_PRESETS.get(profile, PROFILE_PRESETS["systems_designer"])
    return (
        "Contesto del progetto:"
        "Stai analizzando un corpus di regole GdR in Markdown gia' preprocessato."
        f"Profilo di risposta:{profile_instruction}"
        f"Domanda di sviluppo:{question}"
        f"Materiale di riferimento:{context}"
        "Vincoli:"
        "- usa il corpus fornito come base primaria"
        "- se manca qualche informazione, dichiaralo"
        "- distingui osservazioni, inferenze e proposte"
        "- cita file, heading o chunk quando utile"
        "- proponi modifiche operative, non solo teoria astratta"
    )


def openai_style_messages(user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def classify_provider_error(exc: Exception) -> str:
    if isinstance(exc, ProviderTimeoutError):
        return 'timeout'
    if isinstance(exc, ProviderRateLimitError):
        return 'rate_limit'
    if isinstance(exc, ProviderHTTPError):
        return f'http_{exc.status_code}'
    return 'generic'

def ask_target(target: Target, question: str, kp: Dict[str, Any], run_dir: Path, settings: Dict[str, Any]) -> Answer:
    log = get_logger(__name__, 'providers')
    r = settings.get('retrieval', {})
    user_prompt = build_user_prompt(
        question,
        kp['summary'],
        kp['chunks'],
        target.profile,
        r.get('top_k', DEFAULT_TOP_K),
        r.get('max_context_chars', DEFAULT_CONTEXT_CHARS),
    )
    prompt_path = ensure_dir(run_dir / 'prompts') / f"{slugify(target.provider)}-{slugify(target.model)}-{slugify(target.profile)}.txt"
    write_text(prompt_path, user_prompt)
    t0 = time.time()
    raw = None
    log.info(
        'target request start',
        extra={'event': 'target_request_start', 'provider': target.provider, 'model': target.model, 'profile': target.profile, 'target_id': f'{target.provider}:{target.model}:{target.profile}'},
    )
    try:
        raw, text = call_provider(
            provider=target.provider,
            model=target.model,
            messages=openai_style_messages(user_prompt),
            settings=settings,
            temperature=target.temperature,
            max_tokens=target.max_tokens,
        )
        latency = time.time() - t0
        log.info(
            'target request success',
            extra={'event': 'target_request_success', 'provider': target.provider, 'model': target.model, 'profile': target.profile, 'latency_s': round(latency, 3), 'status': 'ok'},
        )
        return Answer(f'{target.provider}:{target.model}:{target.profile}', target.provider, target.model, target.profile, True, latency, text, str(prompt_path), None, raw)
    except Exception as e:
        latency = time.time() - t0
        error_type = classify_provider_error(e)
        log.exception(
            'target request failed',
            extra={'event': 'target_request_failed', 'provider': target.provider, 'model': target.model, 'profile': target.profile, 'latency_s': round(latency, 3), 'status': 'error', 'error_type': error_type},
        )
        return Answer(f'{target.provider}:{target.model}:{target.profile}', target.provider, target.model, target.profile, False, latency, '', str(prompt_path), f'[{error_type}] {e}', raw)

def run_judge(question: str, answers: List[Answer], kp: Dict[str, Any], settings: Dict[str, Any], run_dir: Path) -> Optional[Dict[str, Any]]:
    log = get_logger(__name__, 'judge')
    j = settings.get('judge', {})
    provider = j.get('provider', 'openrouter')
    model = j.get('model')
    candidates = [a for a in answers if a.success]
    if not j.get('enabled', False) or not model or len(candidates) < 2:
        log.info('judge skipped', extra={'event': 'judge_skipped', 'count': len(candidates)})
        return None

    ranked = rank_chunks(question, kp['chunks'], kp['summary'], min(6, settings.get('retrieval', {}).get('top_k', DEFAULT_TOP_K)))
    context = compose_context(kp['summary'], ranked, 22000)
    candidate_blocks = "\n".join([f"## CANDIDATE {a.id} {a.response_text}" for a in candidates])
    judge_user = f"Domanda originale:{question} Corpus rilevante:{context} Risposte candidate:{candidate_blocks}"
    write_text(run_dir / "prompts" / "judge_prompt.txt", judge_user)
    try:
        raw, text = call_provider(
            provider,
            model,
            openai_style_messages(judge_user, JUDGE_SYSTEM_PROMPT),
            settings,
            temperature=0.2,
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
        parsed = safe_json_loads(text) or {"raw_text": text, "raw": raw}
        write_json(run_dir / "judge_result.json", parsed)
        log.info("judge completed", extra={"event": "judge_completed", "provider": provider, "model": model})
        return parsed
    except Exception as e:
        err = {"error": str(e)}
        write_json(run_dir / "judge_result.json", err)
        log.exception("judge failed", extra={"event": "judge_failed", "provider": provider, "model": model})
        return err

def create_run_dirs(output_root: Path, label: str) -> Tuple[Path, Path]:
    rid = f'{now_ts()}-{slugify(label)[:40]}'
    run_dir = ensure_dir(output_root / 'runs' / rid)
    cache_dir = ensure_dir(output_root / 'cache' / rid)
    ensure_dir(run_dir / 'prompts')
    return run_dir, cache_dir


def run_targets(question: str, targets: List[Target], kp: Dict[str, Any], run_dir: Path, cfg: Dict[str, Any], log_path: Path) -> List[Answer]:
    answers: List[Answer] = []
    max_workers = cfg.get('run', {}).get('max_workers', DEFAULT_MAX_WORKERS)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(ask_target, t, question, kp, run_dir, cfg) for t in targets]
        for fut in cf.as_completed(futs):
            ans = fut.result()
            answers.append(ans)
            append_jsonl(log_path, {'event': 'answer_done', 'id': ans.id, 'success': ans.success, 'latency_s': ans.latency_s, 'error': ans.error})
    answers.sort(key=lambda a: a.id)
    return answers
