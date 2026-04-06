"""Integration tests for bitwise operations and continue statement."""

from __future__ import annotations

from pyjit import jit


class TestBitwiseOps:
    def test_bitwise_and(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i & 3
            return s

        fn(0)
        assert fn(8) == sum(i & 3 for i in range(8))

    def test_bitwise_or(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i | 4
            return s

        fn(0)
        assert fn(8) == sum(i | 4 for i in range(8))

    def test_bitwise_xor(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i ^ 7
            return s

        fn(0)
        assert fn(8) == sum(i ^ 7 for i in range(8))

    def test_left_shift(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i << 2
            return s

        fn(0)
        assert fn(8) == sum(i << 2 for i in range(8))

    def test_right_shift(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i >> 1
            return s

        fn(0)
        assert fn(16) == sum(i >> 1 for i in range(16))

    def test_inplace_bitwise_and(self) -> None:
        @jit(warmup=1)
        def fn(n: int, mask: int) -> int:
            s = 0
            for i in range(n):
                v = i
                v &= mask
                s += v
            return s

        fn(0, 3)
        assert fn(8, 3) == sum(i & 3 for i in range(8))

    def test_chained_bitwise(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += (i & 0xFF) | (i << 1)
            return s

        fn(0)
        assert fn(16) == sum((i & 0xFF) | (i << 1) for i in range(16))

    def test_bitwise_with_param(self) -> None:
        @jit(warmup=1)
        def fn(n: int, mask: int) -> int:
            s = 0
            for i in range(n):
                s += i & mask
            return s

        fn(0, 0)
        assert fn(16, 7) == sum(i & 7 for i in range(16))

    def test_popcount_style(self) -> None:
        """Typical bit manipulation: count set bits via shift-and-mask."""

        @jit(warmup=1)
        def fn(x: int, n: int) -> int:
            count = 0
            for _ in range(n):
                count += x & 1
                x = x >> 1
            return count

        fn(0, 0)
        assert fn(0b10110101, 8) == 5


class TestContinue:
    def test_continue_basic(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i % 2 == 0:
                    continue
                s += i
            return s

        fn(0)
        assert fn(10) == sum(i for i in range(10) if i % 2 != 0)

    def test_continue_skip_odds(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i & 1 == 1:
                    continue
                s += i
            return s

        fn(0)
        assert fn(10) == sum(i for i in range(10) if i % 2 == 0)
