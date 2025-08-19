import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extra attributes (fields not in default LogRecord)
        for key, value in record.__dict__.items():
            if key in ("args", "msg", "levelno", "levelname", "name", "pathname", "filename",
                       "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                       "created", "msecs", "relativeCreated", "thread", "threadName",
                       "processName", "process"):
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with JSON output to both stdout and file."""
    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    
    # File handler
    file_handler = logging.FileHandler("logs/app_logging.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)


