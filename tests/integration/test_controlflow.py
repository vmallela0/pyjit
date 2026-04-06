"""Integration tests for control flow: conditionals, break, and select ops."""
from __future__ import annotations

from pyjit import jit


class TestConditionals:
    def test_conditional_accumulate(self) -> None:
        """if inside loop body — only accumulate even numbers."""

        def reference(n: int) -> int:
            s = 0
            for i in range(n):
                if i % 2 == 0:
                    s += i
            return s

        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i % 2 == 0:
                    s += i
            return s

        fn(0)
        for n in [0, 1, 10, 100]:
            assert fn(n) == reference(n), f"n={n}"

    def test_conditional_abs_value(self) -> None:
        """Compute sum of absolute values via if/else."""

        def reference(n: int) -> int:
            s = 0
            for i in range(-n, n):
                if i >= 0:
                    s += i
                else:
                    s += -i
            return s

        @jit(warmup=1)
        def fn(n: int) -> int:
            s = 0
            for i in range(-n, n):
                if i >= 0:
                    s += i
                else:
                    s += -i
            return s

        fn(0)
        for n in [0, 5, 20]:
            assert fn(n) == reference(n), f"n={n}"

    def test_max_so_far(self) -> None:
        """Running maximum via conditional update."""

        @jit(warmup=1)
        def fn(n: int) -> int:
            mx = 0
            for i in range(n):
                v = i * 3 - i * i + 10
                if v > mx:
                    mx = v
            return mx

        fn(0)
        expected = max(
            (i * 3 - i * i + 10 for i in range(20)), default=0
        )
        assert fn(20) == expected


class TestBreakPatterns:
    def test_find_first(self) -> None:
        """Break as early return — find first i where i*i > threshold."""

        def reference(n: int, thresh: int) -> int:
            for i in range(n):
                if i * i > thresh:
                    break
            return i  # type: ignore[possibly-undefined]

        @jit(warmup=1)
        def fn(n: int, thresh: int) -> int:
            result = 0
            for i in range(n):
                result = i
                if i * i > thresh:
                    break
            return result

        fn(100, 50)
        for n, thresh in [(100, 50), (100, 0), (5, 1000)]:
            assert fn(n, thresh) == reference(n, thresh), f"n={n} thresh={thresh}"

    def test_sum_until_negative(self) -> None:
        """Break stops accumulation when condition fires."""

        @jit(warmup=1)
        def fn(n: int, limit: int) -> int:
            s = 0
            for i in range(n):
                if i >= limit:
                    break
                s += i
            return s

        fn(100, 50)
        for n, limit in [(100, 50), (10, 100), (0, 5)]:
            expected = sum(range(min(n, limit)))
            assert fn(n, limit) == expected, f"n={n} limit={limit}"


class TestWhileControlFlow:
    def test_while_with_conditional(self) -> None:
        @jit(warmup=1)
        def fn(n: int) -> int:
            i = 0
            s = 0
            while i < n:
                if i % 2 == 0:
                    s += i
                i += 1
            return s

        fn(0)
        assert fn(20) == sum(i for i in range(20) if i % 2 == 0)
