"""The @jit decorator - the primary user-facing API."""

from __future__ import annotations

import ctypes
import functools
from typing import Any, Callable, Optional, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])

# i64 range for overflow detection
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


@overload
def jit(fn: F) -> F:
    ...


@overload
def jit(*, warmup: int = 10, eager: bool = False) -> Callable[[F], F]:
    ...


def jit(
    fn: F | None = None,
    *,
    warmup: int = 10,
    eager: bool = False,
) -> F | Callable[[F], F]:
    """Mark a function for JIT compilation.

    Can be used as ``@jit``, ``@jit(warmup=10)``, or ``@jit(eager=True)``.

    After ``warmup`` interpreted calls, the function is traced, compiled
    to native code via Cranelift, and subsequent calls execute natively.
    With ``eager=True``, compilation is attempted on the very first call.
    If the compiled code's type assumptions are violated (different types,
    None values, bigint overflow), execution gracefully falls back to CPython.

    Args:
        fn: The function to decorate (when used as bare ``@jit``).
        warmup: Number of interpreted calls before compilation triggers.
        eager: If True, compile on the first call instead of after warmup.

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
                        native_args = _prepare_native_args(args, compiled_types)
                        return compiled_fn(*native_args)
                    except (TypeError, OverflowError, SystemError):
                        # Deopt: fall back to CPython on any native error
                        pass

                # Deopt path: CPython fallback
                return func(*args, **kwargs)

            call_count += 1

            # Eager mode: compile on first call; warmup mode: compile after N calls.
            should_compile = not kwargs and (
                (eager and call_count == 1) or (not eager and call_count == warmup)
            )
            if should_compile:
                compiled_fn = _try_compile(func, args)
                if compiled_fn is not None:
                    compiled_types = tuple(type(a) for a in args)
                    # Eager: immediately use the compiled version for this call too
                    if eager and _guard_types(args, compiled_types):
                        try:
                            native_args = _prepare_native_args(args, compiled_types)
                            return compiled_fn(*native_args)
                        except (TypeError, OverflowError, SystemError):
                            pass

            return func(*args, **kwargs)

        wrapper._pyjit_warmup = warmup  # type: ignore[attr-defined]
        wrapper._pyjit_eager = eager  # type: ignore[attr-defined]
        wrapper._pyjit_compiled = False  # type: ignore[attr-defined]
        wrapper._pyjit_get_compiled = lambda: compiled_fn  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)
    return decorator


def _get_np_ndarray() -> Optional[type]:
    """Return numpy.ndarray type if numpy is available, else None."""
    try:
        import numpy as np

        return np.ndarray
    except ImportError:
        return None


def _guard_types(args: tuple[Any, ...], expected: tuple[type, ...]) -> bool:
    """Type guard: check all arguments match the expected types.

    Also checks that int arguments fit in i64 range to prevent overflow.
    Returns True if it's safe to call the compiled code.
    """
    if len(args) != len(expected):
        return False

    np_ndarray: Optional[type] = _get_np_ndarray()

    for arg, exp_type in zip(args, expected):
        # numpy arrays: guard that the type and dtype still match
        if np_ndarray is not None and exp_type is np_ndarray:
            if not isinstance(arg, exp_type):
                return False
            continue

        # Type must match exactly (no subclass polymorphism in native code)
        if type(arg) is not exp_type:
            return False

        # Overflow guard: Python ints can be arbitrarily large,
        # but native code uses i64. Check bounds.
        if exp_type is int and isinstance(arg, int):
            if not (_I64_MIN <= arg <= _I64_MAX):
                return False

    return True


def _prepare_native_args(args: tuple[Any, ...], compiled_types: tuple[type, ...]) -> list[Any]:
    """Convert Python args to native-ABI values for the compiled function.

    - Lists: replaced with their ob_item pointer (PyObject** array) as i64
    - NumPy arrays: replaced with their raw data buffer pointer as i64
    - All other args pass through unchanged.
    """
    np_ndarray: Optional[type] = _get_np_ndarray()

    result: list[Any] = []
    for arg, t in zip(args, compiled_types):
        if t is list:
            # PyListObject layout: refcnt(8) + type*(8) + ob_size(8) + ob_item*(8)
            ob_item_ptr = ctypes.c_ssize_t.from_address(id(arg) + 24).value
            result.append(ob_item_ptr)
        elif np_ndarray is not None and t is np_ndarray:
            # Extract the raw typed data buffer pointer via __array_interface__
            data_ptr = arg.__array_interface__["data"][0]
            result.append(data_ptr)
        else:
            result.append(arg)
    return result


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
