"""Speedup regression benchmarks: assert JIT is at least 5x faster than CPython.

These are the "headline" benchmarks. If JIT speedup drops below the threshold,
the test fails — this prevents silent performance regressions.

Run with: pytest tests/benchmarks/bench_vs_cpython.py -v
(Does NOT require --benchmark-only; uses timeit directly so it works in CI.)
"""
from __future__ import annotations

import timeit

import pytest
from pyjit import jit


def _timeit_ms(fn: object, *args: object, number: int = 5) -> float:
    """Run fn(*args) `number` times and return the best time in ms."""
    timer = timeit.Timer(lambda: fn(*args))  # type: ignore[operator]
    times = timer.repeat(repeat=3, number=number)
    return min(times) / number * 1000


# ---- functions under test ----


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


# Warmup
_sum_range_jit(10)
_sum_squares_jit(10)

N = 500_000
MIN_SPEEDUP = 5.0  # JIT should be at least 5x faster than CPython


def test_sum_range_speedup() -> None:
    """sum_range JIT should be at least 5x faster than CPython."""
    cpython_ms = _timeit_ms(_sum_range_cpython, N)
    jit_ms = _timeit_ms(_sum_range_jit, N)
    speedup = cpython_ms / jit_ms
    print(f"\nsum_range: CPython={cpython_ms:.2f}ms JIT={jit_ms:.2f}ms speedup={speedup:.1f}x")
    assert speedup >= MIN_SPEEDUP, (
        f"sum_range speedup {speedup:.1f}x is below threshold {MIN_SPEEDUP}x"
    )


def test_sum_squares_speedup() -> None:
    """sum_squares JIT should be at least 5x faster than CPython."""
    cpython_ms = _timeit_ms(_sum_squares_cpython, N)
    jit_ms = _timeit_ms(_sum_squares_jit, N)
    speedup = cpython_ms / jit_ms
    print(f"\nsum_squares: CPython={cpython_ms:.2f}ms JIT={jit_ms:.2f}ms speedup={speedup:.1f}x")
    assert speedup >= MIN_SPEEDUP, (
        f"sum_squares speedup {speedup:.1f}x is below threshold {MIN_SPEEDUP}x"
    )
