"""Integration tests for boolean and/or conditions inside JIT loops."""

from __future__ import annotations

from pyjit import jit


class TestAndCondition:
    def test_and_two_comparisons(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i > 2 and i < 8:
                    s += i
            return s

        fn(0)
        assert fn(10) == sum(i for i in range(10) if i > 2 and i < 8)

    def test_and_with_accumulate(self) -> None:
        @jit(warmup=1)
        def fn(n: int, lo: int, hi: int) -> int:
            s = 0
            for i in range(n):
                if i >= lo and i <= hi:
                    s += i
            return s

        fn(0, 0, 0)
        assert fn(20, 5, 15) == sum(i for i in range(20) if 5 <= i <= 15)

    def test_and_count(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            count = 0
            for i in range(n):
                if i % 3 == 0 and i > 0:
                    count += 1
            return count

        fn(0)
        assert fn(30) == sum(1 for i in range(30) if i % 3 == 0 and i > 0)


class TestOrCondition:
    def test_or_two_comparisons(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i < 3 or i > 7:
                    s += i
            return s

        fn(0)
        assert fn(10) == sum(i for i in range(10) if i < 3 or i > 7)

    def test_or_with_param(self) -> None:
        @jit(warmup=1)
        def fn(n: int, lo: int, hi: int) -> int:
            s = 0
            for i in range(n):
                if i < lo or i > hi:
                    s += i
            return s

        fn(0, 0, 0)
        assert fn(20, 5, 15) == sum(i for i in range(20) if i < 5 or i > 15)
