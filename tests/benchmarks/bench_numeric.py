"""Benchmarks for numeric computation patterns.

Measures JIT vs CPython speedup on typical numeric kernels.
Run with: pytest tests/benchmarks/bench_numeric.py --benchmark-only -v
"""

from __future__ import annotations

from pyjit import jit


# ---- benchmark functions ----


def _poly_cpython(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i * i - 3 * i * i + 2 * i - 1
    return s


@jit(warmup=1)
def _poly_jit(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i * i - 3 * i * i + 2 * i - 1
    return s


def _dot_product_cpython(a: list[int], b: list[int], n: int) -> int:
    s = 0
    for i in range(n):
        s += a[i] * b[i]
    return s


@jit(warmup=1)
def _dot_product_jit(a: list[int], b: list[int], n: int) -> int:
    s = 0
    for i in range(n):
        s += a[i] * b[i]
    return s


def _floor_sum_cpython(n: int) -> int:
    s = 0
    for i in range(1, n + 1):
        s += n // i
    return s


@jit(warmup=1)
def _floor_sum_jit(n: int) -> int:
    s = 0
    for i in range(1, n + 1):
        s += n // i
    return s


# ---- warmup ----

N = 10_000
_list_a = list(range(N))
_list_b = list(range(N, 2 * N))

_poly_jit(10)
_dot_product_jit(_list_a, _list_b, 10)
_floor_sum_jit(10)


# ---- benchmarks ----


def test_bench_poly_cpython(benchmark: object) -> None:
    benchmark.pedantic(_poly_cpython, args=(N,), rounds=5)  # type: ignore[attr-defined]


def test_bench_poly_jit(benchmark: object) -> None:
    benchmark.pedantic(_poly_jit, args=(N,), rounds=5)  # type: ignore[attr-defined]


def test_bench_dot_product_cpython(benchmark: object) -> None:
    benchmark.pedantic(_dot_product_cpython, args=(_list_a, _list_b, N), rounds=5)  # type: ignore[attr-defined]


def test_bench_dot_product_jit(benchmark: object) -> None:
    benchmark.pedantic(_dot_product_jit, args=(_list_a, _list_b, N), rounds=5)  # type: ignore[attr-defined]


def test_bench_floor_sum_cpython(benchmark: object) -> None:
    benchmark.pedantic(_floor_sum_cpython, args=(N,), rounds=5)  # type: ignore[attr-defined]


def test_bench_floor_sum_jit(benchmark: object) -> None:
    benchmark.pedantic(_floor_sum_jit, args=(N,), rounds=5)  # type: ignore[attr-defined]
