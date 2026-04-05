"""Correctness oracle: JIT output must match CPython output.

Every @jit function must produce IDENTICAL output to CPython.
This is the oracle that validates everything.
"""

from __future__ import annotations

from pyjit import jit


def _assert_jit_matches_cpython(
    fn: object,
    args_list: list[tuple[int, ...]],
) -> None:
    """Run function with and without JIT, assert identical results."""
    jitted = jit(warmup=1)(fn)  # type: ignore[arg-type]
    # Warmup call to trigger compilation
    jitted(*args_list[0])
    jitted(*args_list[0])

    for args in args_list:
        cpython_result = fn(*args)  # type: ignore[operator]
        jit_result = jitted(*args)
        assert jit_result == cpython_result, (
            f"{fn.__name__}{args}: CPython={cpython_result}, JIT={jit_result}"  # type: ignore[union-attr]
        )


class TestArithmeticCorrectness:
    def test_add(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        _assert_jit_matches_cpython(fn, [(1, 2), (0, 0), (-1, 1)])

    def test_multiply(self) -> None:
        def fn(a: int, b: int) -> int:
            return a * b

        _assert_jit_matches_cpython(fn, [(3, 4), (0, 100), (-3, 7)])

    def test_mixed_arithmetic(self) -> None:
        def fn(a: int, b: int, c: int) -> int:
            return (a + b) * c - a

        _assert_jit_matches_cpython(fn, [(1, 2, 3), (0, 0, 0), (-5, 10, 3)])


class TestLoopCorrectness:
    def test_sum_range(self) -> None:
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        _assert_jit_matches_cpython(fn, [(0,), (1,), (10,), (1000,)])

    def test_sum_squares(self) -> None:
        def fn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i * i
            return total

        _assert_jit_matches_cpython(fn, [(0,), (1,), (100,), (10000,)])
