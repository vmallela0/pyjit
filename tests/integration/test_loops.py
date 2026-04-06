"""Integration tests for loop compilation patterns."""

from __future__ import annotations

from pyjit import jit


def _jit1(warmup: int = 1) -> object:
    return jit(warmup=warmup)


class TestForLoops:
    def test_sum_range(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        fn(0)
        assert fn(100) == sum(range(100))

    def test_range_start_stop(self) -> None:
        @jit(warmup=1)
        def fn(start: int, stop: int) -> int:
            s = 0
            for i in range(start, stop):
                s += i
            return s

        fn(0, 1)
        assert fn(3, 10) == sum(range(3, 10))

    def test_range_step(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(0, n, 2):
                s += i
            return s

        fn(0)
        assert fn(20) == sum(range(0, 20, 2))

    def test_range_const_limit(self) -> None:
        @jit(warmup=1)
        def fn() -> int:
            s = 0
            for i in range(100):
                s += i
            return s

        fn()
        assert fn() == sum(range(100))

    def test_product_accumulator(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            p = 1
            for i in range(1, n + 1):
                p = p * i
            return p

        fn(1)
        assert fn(5) == 120
        assert fn(1) == 1

    def test_nested_loops(self) -> None:
        @jit(warmup=1)
        def fn(n: int, m: int) -> int:
            s = 0
            for i in range(n):
                for j in range(m):
                    s += i + j
            return s

        fn(1, 1)
        expected = sum(i + j for i in range(4) for j in range(5))
        assert fn(4, 5) == expected

    def test_multiple_accumulators(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            p = 1
            for i in range(1, n + 1):
                s += i
                p = p * i
            return s + p

        fn(1)
        n = 6
        assert fn(n) == sum(range(1, n + 1)) + 720  # 6! = 720


class TestWhileLoops:
    def test_while_lt(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            i = 0
            s = 0
            while i < n:
                s += i
                i += 1
            return s

        fn(0)
        assert fn(100) == sum(range(100))

    def test_while_le(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            i = 1
            p = 1
            while i <= n:
                p = p * i
                i += 1
            return p

        fn(1)
        assert fn(5) == 120


class TestBreakInLoops:
    def test_break_on_condition(self) -> None:
        """break when loop counter hits a threshold."""

        @jit(warmup=1)
        def fn(n: int, limit: int) -> int:
            s = 0
            for i in range(n):
                if i >= limit:
                    break
                s += i
            return s

        fn(10, 5)
        assert fn(100, 10) == sum(range(10))
        assert fn(100, 0) == 0
        assert fn(5, 100) == sum(range(5))

    def test_early_exit_sum(self) -> None:
        """Verify sum stops when guard fires."""

        def reference(n: int, cap: int) -> int:
            s = 0
            for i in range(n):
                if i >= cap:
                    break
                s += i
            return s

        @jit(warmup=1)
        def fn(n: int, cap: int) -> int:
            s = 0
            for i in range(n):
                if i >= cap:
                    break
                s += i
            return s

        fn(10, 5)
        for n, cap in [(50, 20), (100, 100), (0, 5), (10, 3)]:
            assert fn(n, cap) == reference(n, cap), f"n={n} cap={cap}"
