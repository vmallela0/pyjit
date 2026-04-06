"""Disk-based compilation cache for pyjit.

Caches the compile_loop_ir kwargs to ~/.cache/pyjit/ so that re-tracing
and re-analysis are skipped on subsequent process runs. The fast Cranelift
compilation step (~0.6 ms) still runs on every process start.

Cache key: sha256(co_code + co_consts_repr + arg_types + version)
Invalidation: automatic when bytecode, constants, or pyjit version changes.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

_CACHE_VERSION = "0.2.0"
_CACHE_DIR = Path.home() / ".cache" / "pyjit"


def _make_key(code: Any, arg_types: list[str]) -> str:
    """Stable 24-hex-char hash of (bytecode, consts, arg_types, version)."""
    h = hashlib.sha256()
    h.update(code.co_code)
    h.update(repr(list(code.co_consts)).encode())
    h.update(repr(list(code.co_names)).encode())
    h.update("|".join(arg_types).encode())
    h.update(_CACHE_VERSION.encode())
    return h.hexdigest()[:24]


def load_compile_args(code: Any, arg_types: list[str]) -> dict[str, Any] | None:
    """Return cached compile_loop_ir kwargs if present, else None."""
    path = _CACHE_DIR / f"{_make_key(code, arg_types)}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)  # type: ignore[no-any-return]
    except Exception:
        # Corrupted or incompatible entry — ignore silently.
        return None


def save_compile_args(code: Any, arg_types: list[str], compile_args: dict[str, Any]) -> None:
    """Persist compile_loop_ir kwargs to disk cache. Best-effort: never raises."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / f"{_make_key(code, arg_types)}.pkl"
        with path.open("wb") as f:
            pickle.dump(compile_args, f)
    except Exception:
        pass


def clear_cache() -> int:
    """Delete all cache entries. Returns the number of files removed."""
    if not _CACHE_DIR.exists():
        return 0
    count = 0
    for p in _CACHE_DIR.glob("*.pkl"):
        try:
            p.unlink()
            count += 1
        except Exception:
            pass
    return count


def cache_stats() -> dict[str, Any]:
    """Return cache statistics: entries, total size in bytes, and path."""
    if not _CACHE_DIR.exists():
        return {"entries": 0, "size_bytes": 0, "path": str(_CACHE_DIR)}
    files = list(_CACHE_DIR.glob("*.pkl"))
    total_bytes = sum(f.stat().st_size for f in files if f.exists())
    return {
        "entries": len(files),
        "size_bytes": total_bytes,
        "path": str(_CACHE_DIR),
    }
