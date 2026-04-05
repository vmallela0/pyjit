"""Tests for Cranelift codegen — Phase 2."""

from __future__ import annotations

from pyjit._pyjit import build_ir, compile_ir, compile_loop_ir, trace_function

COUNTER = 2**64 - 1  # usize::MAX sentinel for loop counter


class TestSimpleCompilation:
    """Compile simple arithmetic functions and verify results."""

    def test_add(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        compiled = compile_ir(ir, "add")
        assert compiled(3, 4) == 7

    def test_sub(self) -> None:
        def fn(a: int, b: int) -> int:
            return a - b

        trace = trace_function(fn, (10, 3))
        ir = build_ir(trace)
        compiled = compile_ir(ir, "sub")
        assert compiled(10, 3) == 7

    def test_mul(self) -> None:
        def fn(a: int, b: int) -> int:
            return a * b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        compiled = compile_ir(ir, "mul")
        assert compiled(3, 4) == 12

    def test_mixed_arithmetic(self) -> None:
        def fn(a: int, b: int) -> int:
            return a * b + 1

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        compiled = compile_ir(ir, "mixed")
        assert compiled(3, 4) == 13
        assert compiled(10, 20) == 201

    def test_three_args(self) -> None:
        def fn(a: int, b: int, c: int) -> int:
            return (a + b) * c

        trace = trace_function(fn, (2, 3, 4))
        ir = build_ir(trace)
        compiled = compile_ir(ir, "three")
        assert compiled(2, 3, 4) == 20

    def test_negative_numbers(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (-5, 3))
        ir = build_ir(trace)
        compiled = compile_ir(ir, "neg")
        assert compiled(-5, 3) == -2
        assert compiled(-10, -20) == -30


class TestLoopCompilation:
    """Test native loop compilation."""

    def test_sum_range(self) -> None:
        jit_fn = compile_loop_ir(
            num_params=1,
            limit_param=0,
            num_locals=3,
            return_local=1,
            init_locals=[(1, 0)],
            body_ops=[("Add", 1, 1, COUNTER, False, 0)],
            func_name="sum_range",
        )
        assert jit_fn(0) == 0
        assert jit_fn(1) == 0
        assert jit_fn(5) == 10
        assert jit_fn(100) == 4950

    def test_sum_squares(self) -> None:
        jit_fn = compile_loop_ir(
            num_params=1,
            limit_param=0,
            num_locals=4,
            return_local=1,
            init_locals=[(1, 0), (2, 0), (3, 0)],
            body_ops=[
                ("Mul", 3, COUNTER, COUNTER, False, 0),
                ("Add", 1, 1, 3, False, 0),
            ],
            func_name="sum_squares",
        )
        assert jit_fn(0) == 0
        assert jit_fn(1) == 0
        assert jit_fn(5) == 30
        assert jit_fn(100) == 328350

    def test_large_loop(self) -> None:
        jit_fn = compile_loop_ir(
            num_params=1,
            limit_param=0,
            num_locals=3,
            return_local=1,
            init_locals=[(1, 0)],
            body_ops=[("Add", 1, 1, COUNTER, False, 0)],
        )
        assert jit_fn(1_000_000) == 499999500000


class TestEndToEndJit:
    """Test the full @jit decorator pipeline."""

    def test_jit_sum_squares(self) -> None:
        from pyjit import jit

        @jit(warmup=3)
        def sum_squares(n: int) -> int:
            total = 0
            for i in range(n):
                total += i * i
            return total

        # Warm up
        for _ in range(5):
            sum_squares(10)

        # Verify compiled version
        assert sum_squares(0) == 0
        assert sum_squares(5) == 30
        assert sum_squares(100) == 328350

    def test_jit_simple_add(self) -> None:
        from pyjit import jit

        @jit(warmup=3)
        def add(a: int, b: int) -> int:
            return a + b

        for _ in range(5):
            add(1, 2)

        assert add(3, 4) == 7
        assert add(0, 0) == 0
        assert add(-1, 1) == 0

    def test_jit_sum_range(self) -> None:
        from pyjit import jit

        @jit(warmup=3)
        def sum_range(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        for _ in range(5):
            sum_range(10)

        assert sum_range(0) == 0
        assert sum_range(10) == 45
        assert sum_range(1000) == 499500
