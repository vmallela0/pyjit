"""Benchmarks for loop compilation patterns.

Run with: pytest tests/benchmarks/ --benchmark-only -v
"""

from __future__ import annotations

import pytest
from pyjit import jit


def _sum_range_cpython(n: int) -> int:
    s = 0
    for i in range(n):
        s += i
    return s


@jit(warmup=1)
def _sum_range_jit(n: int) -> int:
    s = 0
    for i in range(n):
        s += i
    return s


def _sum_squares_cpython(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i
    return s


@jit(warmup=1)
def _sum_squares_jit(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i
    return s


def _nested_loop_cpython(n: int, m: int) -> int:
    s = 0
    for i in range(n):
        for j in range(m):
            s += i * j
    return s


@jit(warmup=1)
def _nested_loop_jit(n: int, m: int) -> int:
    s = 0
    for i in range(n):
        for j in range(m):
            s += i * j
    return s


def _while_loop_cpython(n: int) -> int:
    i = 0
    s = 0
    while i < n:
        s += i
        i += 1
    return s


@jit(warmup=1)
def _while_loop_jit(n: int) -> int:
    i = 0
    s = 0
    while i < n:
        s += i
        i += 1
    return s


# Warmup all JIT functions
_sum_range_jit(10)
_sum_squares_jit(10)
_nested_loop_jit(5, 5)
_while_loop_jit(10)


N = 100_000
NM = 1_000


@pytest.fixture(params=["cpython", "jit"])
def backend(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def test_bench_sum_range_cpython(benchmark: object) -> None:
    benchmark.pedantic(_sum_range_cpython, args=(N,), rounds=10)  # type: ignore[attr-defined]


def test_bench_sum_range_jit(benchmark: object) -> None:
    benchmark.pedantic(_sum_range_jit, args=(N,), rounds=10)  # type: ignore[attr-defined]


def test_bench_sum_squares_cpython(benchmark: object) -> None:
    benchmark.pedantic(_sum_squares_cpython, args=(N,), rounds=10)  # type: ignore[attr-defined]


def test_bench_sum_squares_jit(benchmark: object) -> None:
    benchmark.pedantic(_sum_squares_jit, args=(N,), rounds=10)  # type: ignore[attr-defined]


def test_bench_nested_loop_cpython(benchmark: object) -> None:
    benchmark.pedantic(_nested_loop_cpython, args=(NM, NM), rounds=10)  # type: ignore[attr-defined]


def test_bench_nested_loop_jit(benchmark: object) -> None:
    benchmark.pedantic(_nested_loop_jit, args=(NM, NM), rounds=10)  # type: ignore[attr-defined]


def test_bench_while_loop_cpython(benchmark: object) -> None:
    benchmark.pedantic(_while_loop_cpython, args=(N,), rounds=10)  # type: ignore[attr-defined]


def test_bench_while_loop_jit(benchmark: object) -> None:
    benchmark.pedantic(_while_loop_jit, args=(N,), rounds=10)  # type: ignore[attr-defined]
