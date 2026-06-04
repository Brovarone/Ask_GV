from __future__ import annotations
import atexit
import json
import logging
import logging.config
import os
import queue
import socket
import sys
import threading
from contextvars import ContextVar
from logging.handlers import QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional
RUN_ID: ContextVar[str] = ContextVar("run_id", default="-")
_listener: Optional[QueueListener] = None
_log_queue: Optional[queue.Queue] = None
class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = getattr(record, "run_id", None) or RUN_ID.get()
        record.component = getattr(record, "component", None) or "app"
        record.hostname = socket.gethostname()
        record.thread_name = threading.current_thread().name
        return True
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
            "component": getattr(record, "component", "app"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": getattr(record, "thread_name", record.threadName),
            "hostname": getattr(record, "hostname", None),
        }
        for key in ("event", "provider", "model", "profile", "latency_s", "status", "path", "count", "cache_dir", "run_dir", "target_id"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)
class PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} | {record.levelname:<8} | {getattr(record, 'run_id', '-')} | {getattr(record, 'component', 'app')} | {record.name} | {record.getMessage()}"
        extras = []
        for key in ("event", "provider", "model", "profile", "latency_s", "status", "path", "count"):
            if hasattr(record, key):
                extras.append(f"{key}={getattr(record, key)}")
        if extras:
            base += " | " + " ".join(extras)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base
class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.setdefault("component", self.extra.get("component", "app"))
        kwargs["extra"] = extra
        return msg, kwargs

def setup_logging(log_dir: Path, run_id: str, debug: bool = False, json_console: bool = False) -> None:
    global _listener, _log_queue
    log_dir.mkdir(parents=True, exist_ok=True)
    RUN_ID.set(run_id)
    level = "DEBUG" if debug else os.getenv("LOG_LEVEL", "INFO").upper()
    _log_queue = queue.Queue(-1)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(JsonFormatter() if json_console else PrettyFormatter())
    console_handler.addFilter(ContextFilter())
    file_handler = RotatingFileHandler(log_dir / "app.log.jsonl", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(JsonFormatter())
    file_handler.addFilter(ContextFilter())
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": False, "handlers": {"queue": {"class": "logging.handlers.QueueHandler", "queue": _log_queue}}, "root": {"level": level, "handlers": ["queue"]}})
    _listener = QueueListener(_log_queue, console_handler, file_handler, respect_handler_level=True)
    _listener.start()
    logging.getLogger(__name__).info("logging initialized", extra={"event": "logging_initialized", "path": str(log_dir)})
    atexit.register(shutdown_logging)

def shutdown_logging() -> None:
    global _listener
    if _listener is not None:
        try:
            logging.getLogger(__name__).info("logging shutdown", extra={"event": "logging_shutdown"})
        except Exception:
            pass
        _listener.stop()
        _listener = None

def get_logger(name: str, component: str = "app") -> ContextAdapter:
    return ContextAdapter(logging.getLogger(name), {"component": component})
