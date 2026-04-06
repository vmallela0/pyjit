"""Integration tests for arithmetic operations inside JIT-compiled loops."""

from __future__ import annotations

from pyjit import jit


class TestIntegerArithmetic:
    def test_add_in_loop(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s = s + i
            return s

        fn(0)
        assert fn(1000) == sum(range(1000))

    def test_subtract_in_loop(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s = s - i
            return s

        fn(0)
        assert fn(100) == -sum(range(100))

    def test_multiply_in_loop(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i * i
            return s

        fn(0)
        assert fn(10) == sum(i * i for i in range(10))

    def test_floor_divide(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(1, n + 1):
                s += n // i
            return s

        fn(1)
        assert fn(10) == sum(10 // i for i in range(1, 11))

    def test_modulo(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i % 3
            return s

        fn(0)
        assert fn(30) == sum(i % 3 for i in range(30))

    def test_mixed_ops(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += (i * i + i) // 2
            return s

        fn(0)
        assert fn(20) == sum((i * i + i) // 2 for i in range(20))

    def test_negative_accumulator(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += -i
            return s

        fn(0)
        assert fn(50) == -sum(range(50))

    def test_bitwise_and(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i & 3
            return s

        fn(0)
        assert fn(20) == sum(i & 3 for i in range(20))

    def test_bitwise_or(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i | 1
            return s

        fn(0)
        assert fn(10) == sum(i | 1 for i in range(10))

    def test_shift_ops(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += (i << 1) + (i >> 1)
            return s

        fn(0)
        assert fn(16) == sum((i << 1) + (i >> 1) for i in range(16))


class TestFloatArithmetic:
    def test_float_sum(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> float:
            s = 0.0
            for i in range(n):
                s += float(i)
            return s

        fn(0)
        result = fn(100)
        assert abs(result - sum(float(i) for i in range(100))) < 1e-9

    def test_float_multiply(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> float:
            s = 0.0
            for i in range(1, n + 1):
                s += float(i) * 0.5
            return s

        fn(1)
        assert abs(fn(10) - sum(i * 0.5 for i in range(1, 11))) < 1e-9
