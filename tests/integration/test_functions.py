"""Integration tests for function inlining and call support."""
from __future__ import annotations

import math
from pyjit import jit


# --- helpers that should be inlinable ---


def square(x: int) -> int:
    return x * x


def add_one(x: int) -> int:
    return x + 1


def dot2(a: int, b: int) -> int:
    return a * a + b * b


class TestFunctionInlining:
    def test_inline_square(self) -> None:
        """Inline a single-arg pure function."""

        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += square(i)
            return s

        fn(0)
        assert fn(10) == sum(i * i for i in range(10))

    def test_inline_add_one(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += add_one(i)
            return s

        fn(0)
        assert fn(10) == sum(i + 1 for i in range(10))

    def test_inline_two_arg(self) -> None:
        """Inline a two-arg pure function."""

        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += dot2(i, i + 1)
            return s

        fn(0)
        assert fn(5) == sum(i * i + (i + 1) * (i + 1) for i in range(5))


class TestMathBuiltins:
    def test_math_sqrt_loop(self) -> None:
        """math.sqrt should be recognized and compiled to native sqrt."""

        @jit(warmup=1)
        def fn(n: int) -> float:
            s = 0.0
            for i in range(1, n + 1):
                s += math.sqrt(float(i))
            return s

        fn(1)
        expected = sum(math.sqrt(float(i)) for i in range(1, 11))
        assert abs(fn(10) - expected) < 1e-9

    def test_abs_builtin(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(-n, n):
                s += abs(i)
            return s

        fn(0)
        assert fn(5) == sum(abs(i) for i in range(-5, 5))

    def test_min_max_builtins(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += min(i, 5) + max(i, 3)
            return s

        fn(0)
        assert fn(10) == sum(min(i, 5) + max(i, 3) for i in range(10))


class TestFallbackBehavior:
    def test_compile_failure_falls_back(self) -> None:
        """A function that can't compile should silently fall back to CPython."""

        @jit(warmup=1)
        def fn(x: int) -> str:
            return str(x)  # can't compile — string ops not supported

        assert fn(42) == "42"
        assert fn(0) == "0"

    def test_warmup_then_jit(self) -> None:
        """After compiling for int, subsequent calls should use native code."""

        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        fn(5)  # warmup triggers compile
        assert fn(10) == 45
        assert fn(100) == sum(range(100))
