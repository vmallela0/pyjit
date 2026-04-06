"""Integration tests for list and NumPy array operations."""

from __future__ import annotations

import pytest
from pyjit import jit


class TestListRead:
    def test_sum_list(self) -> None:
        @jit(warmup=1)
        def fn(data: list[int], n: int) -> int:
            s = 0
            for i in range(n):
                s += data[i]
            return s

        data = list(range(100))
        fn(data, 0)
        assert fn(data, 100) == sum(data)

    def test_dot_product(self) -> None:
        @jit(warmup=1)
        def fn(a: list[int], b: list[int], n: int) -> int:
            s = 0
            for i in range(n):
                s += a[i] * b[i]
            return s

        a = list(range(10))
        b = list(range(10, 20))
        fn(a, b, 0)
        assert fn(a, b, 10) == sum(x * y for x, y in zip(a, b))

    def test_list_with_counter(self) -> None:
        @jit(warmup=1)
        def fn(data: list[int], n: int) -> int:
            s = 0
            for i in range(n):
                s += data[i] + i
            return s

        data = list(range(20))
        fn(data, 0)
        assert fn(data, 20) == sum(data[i] + i for i in range(20))


class TestListWrite:
    def test_fill_list(self) -> None:
        @jit(warmup=1)
        def fn(data: list[int], n: int) -> int:
            for i in range(n):
                data[i] = i * 2
            return data[n - 1]

        data = [0] * 50
        fn(data, 1)
        result = fn(data, 50)
        assert result == 98
        assert data[10] == 20

    def test_accumulate_into_list(self) -> None:
        @jit(warmup=1)
        def fn(src: list[int], dst: list[int], n: int) -> int:
            for i in range(n):
                dst[i] = src[i] * src[i]
            return dst[n - 1]

        src = list(range(10))
        dst = [0] * 10
        fn(src, dst, 1)
        fn(src, dst, 10)
        assert dst == [i * i for i in range(10)]


try:
    import numpy as np  # type: ignore[import-untyped]  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestNumpyArrays:
    def test_sum_float_array(self) -> None:
        import numpy as np

        @jit(warmup=1)
        def fn(data: object, n: int) -> float:
            s = 0.0
            for i in range(n):
                s += data[i]  # type: ignore[index]
            return s

        arr = np.arange(100, dtype=np.float64)
        fn(arr, 0)
        result = fn(arr, 100)
        assert abs(result - float(np.sum(arr))) < 1e-9

    def test_sum_int_array(self) -> None:
        import numpy as np

        @jit(warmup=1)
        def fn(data: object, n: int) -> int:
            s = 0
            for i in range(n):
                s += data[i]  # type: ignore[index]
            return s

        arr = np.arange(100, dtype=np.int64)
        fn(arr, 0)
        assert fn(arr, 100) == int(np.sum(arr))
