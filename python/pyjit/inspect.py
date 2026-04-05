"""Inspection utilities for JIT-compiled functions."""

from __future__ import annotations

from typing import Any, Callable


def is_jit_compiled(fn: Callable[..., Any]) -> bool:
    """Check if a function has been JIT compiled."""
    return bool(getattr(fn, "_pyjit_compiled", False))


def get_warmup(fn: Callable[..., Any]) -> int | None:
    """Get the warmup threshold for a JIT-decorated function."""
    result: int | None = getattr(fn, "_pyjit_warmup", None)
    return result
