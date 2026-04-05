"""The @jit decorator - the primary user-facing API."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

# i64 range for overflow detection
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


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
    If the compiled code's type assumptions are violated (different types,
    None values, bigint overflow), execution gracefully falls back to CPython.

    Args:
        fn: The function to decorate (when used as bare ``@jit``).
        warmup: Number of interpreted calls before compilation triggers.

    Returns:
        The decorated function.
    """

    def decorator(func: F) -> F:
        call_count = 0
        compiled_fn: Any = None
        compiled_types: tuple[type, ...] = ()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count, compiled_fn, compiled_types

            # If we have compiled code, try to use it
            if compiled_fn is not None and not kwargs:
                # Guard: check argument types match what we compiled for
                if _guard_types(args, compiled_types):
                    try:
                        return compiled_fn(*args)
                    except (TypeError, OverflowError, SystemError):
                        # Deopt: fall back to CPython on any native error
                        pass

                # Deopt path: CPython fallback
                return func(*args, **kwargs)

            call_count += 1

            # After warmup calls, try to compile
            if call_count == warmup and not kwargs:
                compiled_fn = _try_compile(func, args)
                if compiled_fn is not None:
                    compiled_types = tuple(type(a) for a in args)

            return func(*args, **kwargs)

        wrapper._pyjit_warmup = warmup  # type: ignore[attr-defined]
        wrapper._pyjit_compiled = False  # type: ignore[attr-defined]
        wrapper._pyjit_get_compiled = lambda: compiled_fn  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator


def _guard_types(args: tuple[Any, ...], expected: tuple[type, ...]) -> bool:
    """Type guard: check all arguments match the expected types.

    Also checks that int arguments fit in i64 range to prevent overflow.
    Returns True if it's safe to call the compiled code.
    """
    if len(args) != len(expected):
        return False

    for arg, exp_type in zip(args, expected):
        # Type must match exactly (no subclass polymorphism in native code)
        if type(arg) is not exp_type:
            return False

        # Overflow guard: Python ints can be arbitrarily large,
        # but native code uses i64. Check bounds.
        if exp_type is int and isinstance(arg, int):
            if not (_I64_MIN <= arg <= _I64_MAX):
                return False

    return True


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
