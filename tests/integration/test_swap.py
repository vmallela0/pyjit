"""Integration tests for STORE_FAST_STORE_FAST swap idiom (Fibonacci and variants)."""

from __future__ import annotations

from pyjit import jit


class TestSwap:
    def test_fib_basic(self) -> None:
        @jit(warmup=1)
        def fib(n: int) -> int:
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a

        fib(0)
        assert fib(0) == 0
        assert fib(1) == 1
        assert fib(10) == 55
        assert fib(20) == 6765

    def test_fib_returns_b(self) -> None:
        """Same swap but returns b instead of a."""

        @jit(warmup=1)
        def fib_b(n: int) -> int:
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return b

        fib_b(0)
        assert fib_b(1) == 1
        assert fib_b(9) == 55  # fib(10) is at b after 9 iterations

    def test_swap_accumulator(self) -> None:
        """Two vars where one accumulates and the other tracks previous."""

        @jit(warmup=1)
        def running_sum(n: int) -> int:
            prev, cur = 0, 0
            for i in range(n):
                prev, cur = cur, cur + i
            return cur

        running_sum(0)
        assert running_sum(5) == sum(range(5))

    def test_fib_large(self) -> None:
        @jit(warmup=1)
        def fib(n: int) -> int:
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a

        fib(0)
        assert fib(30) == 832040
