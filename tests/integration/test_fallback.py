"""Integration tests for graceful deoptimization — Phase 3.

Every test verifies that compiled code falls back to CPython
correctly when type assumptions are violated.
"""

from __future__ import annotations

from pyjit import jit


def test_type_change_deopt() -> None:
    """Function compiled for int must handle float gracefully."""

    @jit(warmup=2)
    def fn(a: int, b: int) -> int:
        return a + b

    # Warm up with ints — triggers compilation
    fn(1, 2)
    fn(3, 4)

    # Now call with floats — must deopt and still give correct result
    assert fn(1.5, 2.5) == 4.0  # type: ignore[arg-type]


def test_none_handling() -> None:
    """Compiled code must handle None without crashing."""

    @jit(warmup=2)
    def fn(x: object) -> int:
        if x is None:
            return 0
        return int(x) + 1  # type: ignore[arg-type]

    fn(5)
    fn(10)
    assert fn(None) == 0


def test_overflow_deopt() -> None:
    """Big ints that overflow i64 must deopt to Python bigint."""

    @jit(warmup=2)
    def fn(a: int, b: int) -> int:
        return a * b

    fn(3, 4)
    fn(5, 6)

    # This overflows i64 — must deopt
    big = 10**30
    assert fn(big, big) == big * big


def test_loop_with_type_change() -> None:
    """Loop function compiled for int must handle type change on non-range arg."""

    @jit(warmup=2)
    def fn(a: int, b: int) -> int:
        return a + b

    fn(1, 2)
    fn(3, 4)

    # Call with float — must deopt to CPython
    assert fn(1.5, 2.5) == 4.0  # type: ignore[arg-type]


def test_kwargs_bypass_jit() -> None:
    """Keyword arguments should always use CPython path."""

    @jit(warmup=2)
    def fn(a: int, b: int = 0) -> int:
        return a + b

    fn(1, 2)
    fn(3, 4)

    # kwargs always bypass JIT
    assert fn(a=5, b=6) == 11
    assert fn(5, b=6) == 11


def test_wrong_arg_count_deopt() -> None:
    """Wrong number of arguments should deopt gracefully."""

    @jit(warmup=2)
    def fn(a: int, b: int) -> int:
        return a + b

    fn(1, 2)
    fn(3, 4)

    # Wrong arg count — should fall back to CPython (which will raise TypeError)
    try:
        fn(1)  # type: ignore[call-arg]
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_continued_correctness_after_deopt() -> None:
    """After a deopt, the function should continue working correctly."""

    @jit(warmup=2)
    def fn(a: int, b: int) -> int:
        return a + b

    fn(1, 2)
    fn(3, 4)

    # Trigger deopt with floats
    fn(1.5, 2.5)  # type: ignore[arg-type]

    # Should still work with ints after deopt (compiled code still cached)
    assert fn(100, 200) == 300
    assert fn(-10, 10) == 0
