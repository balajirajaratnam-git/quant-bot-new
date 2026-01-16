from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:default} placeholders."""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            var = m.group(1)
            default = m.group(2) if m.group(2) is not None else ""
            return os.environ.get(var, default)

        return _ENV_PATTERN.sub(repl, value)

    if isinstance(value, list):
        return [_expand_env(v) for v in value]

    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}

    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config and expand environment placeholders.

    If python-dotenv is installed and a .env exists in the current working directory,
    it is loaded once (non-destructively) to support local development.
    """
    # Optional .env support
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(override=False)
    except Exception:
        pass

    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must parse into a mapping/object")

    return _expand_env(raw)
