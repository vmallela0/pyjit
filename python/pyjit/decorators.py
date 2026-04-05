"""The @jit decorator - the primary user-facing API."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


@overload
def jit(fn: F) -> F: ...


@overload
def jit(*, warmup: int = 10) -> Callable[[F], F]: ...


def jit(
    fn: F | None = None,
    *,
    warmup: int = 10,
) -> F | Callable[[F], F]:
    """Mark a function for JIT compilation.

    Can be used as ``@jit`` or ``@jit(warmup=10)``.

    After ``warmup`` interpreted calls, the function is traced, compiled
    to native code via Cranelift, and subsequent calls execute natively.

    Args:
        fn: The function to decorate (when used as bare ``@jit``).
        warmup: Number of interpreted calls before compilation triggers.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        call_count = 0
        compiled_fn: Any = None

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count, compiled_fn

            # If we have compiled code and no kwargs, call native
            if compiled_fn is not None and not kwargs:
                try:
                    return compiled_fn(*args)
                except (TypeError, OverflowError):
                    # Type mismatch or overflow — fall back to CPython
                    return func(*args, **kwargs)

            call_count += 1

            # After warmup calls, try to compile
            if call_count == warmup and not kwargs:
                compiled_fn = _try_compile(func, args)

            return func(*args, **kwargs)

        wrapper._pyjit_warmup = warmup  # type: ignore[attr-defined]
        wrapper._pyjit_compiled = False  # type: ignore[attr-defined]
        wrapper._pyjit_get_compiled = lambda: compiled_fn  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator


def _try_compile(func: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    """Attempt to compile a function to native code."""
    try:
        from pyjit._compiler import compile_function

        result = compile_function(func, args)
        if result is not None:
            return result
    except Exception:
        pass
    return None
