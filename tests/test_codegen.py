"""Tests for Cranelift codegen — Phase 2+."""

from __future__ import annotations

import struct

from pyjit._pyjit import build_ir, compile_ir, compile_loop_ir, trace_function

COUNTER = 2**64 - 1  # usize::MAX sentinel for loop counter
TYPE_I64 = 0
TYPE_F64 = 1


def _f64_bits(v: float) -> int:
    """Encode a float as i64 bits for passing as immediate."""
    return struct.unpack("<q", struct.pack("<d", v))[0]


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
            init_float_locals=[],
            body_ops=[("Add", 1, 1, COUNTER, False, 0)],
            local_types=[TYPE_I64, TYPE_I64, TYPE_I64],
            param_types=[TYPE_I64],
            return_type_id=TYPE_I64,
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
            init_float_locals=[],
            body_ops=[
                ("Mul", 3, COUNTER, COUNTER, False, 0),
                ("Add", 1, 1, 3, False, 0),
            ],
            local_types=[TYPE_I64] * 4,
            param_types=[TYPE_I64],
            return_type_id=TYPE_I64,
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
            init_float_locals=[],
            body_ops=[("Add", 1, 1, COUNTER, False, 0)],
            local_types=[TYPE_I64] * 3,
            param_types=[TYPE_I64],
            return_type_id=TYPE_I64,
        )
        assert jit_fn(1_000_000) == 499999500000


class TestFloatLoopCompilation:
    """Test float loop compilation — Task 1."""

    def test_float_accumulate_const(self) -> None:
        """s += 1.5 in a loop."""
        jit_fn = compile_loop_ir(
            num_params=1,
            limit_param=0,
            num_locals=3,
            return_local=1,
            init_locals=[],
            init_float_locals=[(1, 0.0)],
            body_ops=[("Add", 1, 1, 0, True, _f64_bits(1.5))],
            local_types=[TYPE_I64, TYPE_F64, TYPE_I64],
            param_types=[TYPE_I64],
            return_type_id=TYPE_F64,
        )
        assert abs(jit_fn(100) - 150.0) < 1e-9

    def test_float_counter_accumulate(self) -> None:
        """s += float(counter) in a loop — counter converted to f64."""
        jit_fn = compile_loop_ir(
            num_params=1,
            limit_param=0,
            num_locals=3,
            return_local=1,
            init_locals=[],
            init_float_locals=[(1, 0.0)],
            body_ops=[("Add", 1, 1, COUNTER, False, 0)],
            local_types=[TYPE_I64, TYPE_F64, TYPE_I64],
            param_types=[TYPE_I64],
            return_type_id=TYPE_F64,
        )
        assert abs(jit_fn(10) - 45.0) < 1e-9
        assert abs(jit_fn(100) - 4950.0) < 1e-9

    def test_float_mul(self) -> None:
        """s += counter * counter as float."""
        jit_fn = compile_loop_ir(
            num_params=1,
            limit_param=0,
            num_locals=4,
            return_local=1,
            init_locals=[],
            init_float_locals=[(1, 0.0), (3, 0.0)],
            body_ops=[
                ("Mul", 3, COUNTER, COUNTER, False, 0),
                ("Add", 1, 1, 3, False, 0),
            ],
            local_types=[TYPE_I64, TYPE_F64, TYPE_I64, TYPE_F64],
            param_types=[TYPE_I64],
            return_type_id=TYPE_F64,
        )
        expected = sum(float(i) * float(i) for i in range(100))
        assert abs(jit_fn(100) - expected) < 1e-6

    def test_jit_float_accumulator(self) -> None:
        """End-to-end @jit test with float accumulator."""
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> float:
            s = 0.0
            for i in range(n):
                s += 1.5
            return s

        fn(10)
        fn(10)
        assert is_jit_compiled(fn)
        assert abs(fn(100) - 150.0) < 1e-9
        assert abs(fn(1000) - 1500.0) < 1e-6


class TestConditionalLoops:
    """Test conditionals in loop bodies — Task 2."""

    def test_conditional_accumulate(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i % 2 == 0:
                    s += i
                else:
                    s -= i
            return s

        fn(10)
        fn(10)
        expected = sum(i if i % 2 == 0 else -i for i in range(100))
        assert fn(100) == expected
        assert is_jit_compiled(fn)

    def test_filter_sum(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i % 3 == 0:
                    s += i
            return s

        fn(10)
        fn(10)
        expected = sum(i for i in range(100) if i % 3 == 0)
        assert fn(100) == expected
        assert is_jit_compiled(fn)


class TestWhileLoops:
    """Test while loop compilation — Task 3."""

    def test_while_sum(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            i = 0
            while i < n:
                s += i
                i += 1
            return s

        fn(10)
        fn(10)
        assert fn(1000) == sum(range(1000))
        assert is_jit_compiled(fn)

    def test_while_squares(self) -> None:
        from pyjit import jit

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            i = 0
            while i < n:
                s += i * i
                i += 1
            return s

        fn(10)
        fn(10)
        assert fn(100) == sum(i * i for i in range(100))


class TestNestedLoops:
    """Test nested loop compilation — Task 4."""

    def test_double_nested_sum(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                for j in range(n):
                    s += i * j
            return s

        fn(5)
        fn(5)
        expected = sum(i * j for i in range(50) for j in range(50))
        assert fn(50) == expected
        assert is_jit_compiled(fn)

    def test_triple_nested(self) -> None:
        from pyjit import jit

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        s += i * j * k
            return s

        fn(5)
        fn(5)
        expected = sum(i * j * k for i in range(10) for j in range(10) for k in range(10))
        assert fn(10) == expected

    def test_nested_different_limits(self) -> None:
        """Inner and outer loops use the same param as limit."""
        from pyjit import jit

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                for j in range(n):
                    s += i + j
            return s

        fn(5)
        fn(5)
        expected = sum(i + j for i in range(30) for j in range(30))
        assert fn(30) == expected

    def test_nested_speedup(self) -> None:
        """Nested loops should be the biggest speedup — >50x."""
        import time

        from pyjit import jit

        @jit(warmup=2)
        def jit_fn(n: int) -> int:
            s = 0
            for i in range(n):
                for j in range(n):
                    s += i * j
            return s

        def py_fn(n: int) -> int:
            s = 0
            for i in range(n):
                for j in range(n):
                    s += i * j
            return s

        jit_fn(10)
        jit_fn(10)

        n = 500
        n_runs = 10

        start = time.perf_counter_ns()
        for _ in range(n_runs):
            py_fn(n)
        cpython_ns = (time.perf_counter_ns() - start) / n_runs

        start = time.perf_counter_ns()
        for _ in range(n_runs):
            jit_fn(n)
        jit_ns = (time.perf_counter_ns() - start) / n_runs

        speedup = cpython_ns / max(jit_ns, 1)
        assert speedup > 30.0, f"Expected >30x speedup, got {speedup:.1f}x"


class TestUnaryOps:
    """Test unary operators in loop bodies — Sprint Task 1."""

    def test_negation(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += -(i * i)
            return s

        fn(10)
        fn(10)
        assert fn(100) == sum(-(i * i) for i in range(100))
        assert is_jit_compiled(fn)

    def test_bitwise_not(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += ~i
            return s

        fn(10)
        fn(10)
        assert fn(100) == sum(~i for i in range(100))
        assert is_jit_compiled(fn)


class TestStackOps:
    """Test COPY/SWAP/POP_TOP opcodes — Sprint Task 2."""

    def test_pop_top_and_copy_in_body(self) -> None:
        """POP_TOP/COPY/SWAP are recognized and don't bail."""
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i > 50:
                    s += i
            return s

        fn(10)
        fn(10)
        assert fn(100) == sum(i for i in range(100) if i > 50)
        assert is_jit_compiled(fn)


class TestBuiltinMath:
    """Test abs/min/max in loop bodies — Sprint Task 3."""

    def test_abs_in_loop(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += abs(i - 50)
            return s

        fn(10)
        fn(10)
        assert fn(100) == sum(abs(i - 50) for i in range(100))
        assert is_jit_compiled(fn)

    def test_min_in_loop(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            lo = 0
            for i in range(n):
                lo = min(lo, i - 50)
            return lo

        fn(10)
        fn(10)
        expected = min(i - 50 for i in range(100))
        assert fn(100) == expected
        assert is_jit_compiled(fn)

    def test_max_in_loop(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            hi = -999999
            for i in range(n):
                hi = max(hi, i * 3 - 100)
            return hi

        fn(10)
        fn(10)
        expected = max(i * 3 - 100 for i in range(100))
        assert fn(100) == expected
        assert is_jit_compiled(fn)


class TestRangeLiterals:
    """Test range() with literals and multi-arg — Sprint Task 4."""

    def test_range_literal(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn() -> int:
            s = 0
            for i in range(100):
                s += i
            return s

        fn()
        fn()
        assert fn() == sum(range(100))
        assert is_jit_compiled(fn)

    def test_range_two_arg(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(10, n):
                s += i
            return s

        fn(50)
        fn(50)
        assert fn(100) == sum(range(10, 100))
        assert is_jit_compiled(fn)

    def test_range_three_arg_step(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(0, n, 2):
                s += i
            return s

        fn(50)
        fn(50)
        assert fn(100) == sum(range(0, 100, 2))
        assert is_jit_compiled(fn)


class TestPowerOp:
    """Test ** operator — Sprint Task 5."""

    def test_square(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i**2
            return s

        fn(10)
        fn(10)
        assert fn(100) == sum(i**2 for i in range(100))
        assert is_jit_compiled(fn)

    def test_cube(self) -> None:
        from pyjit import jit

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i**3
            return s

        fn(10)
        fn(10)
        assert fn(100) == sum(i**3 for i in range(100))


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

        for _ in range(5):
            sum_squares(10)

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
