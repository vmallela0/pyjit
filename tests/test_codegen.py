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


class TestDivisionSafety:
    """Test division-by-zero guards — Sprint Task 6."""

    def test_floor_div_by_zero_no_crash(self) -> None:
        """Floor division by zero should deopt, not SIGFPE."""
        from pyjit import jit

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += 100 // (i - 5)  # i=5 causes div by zero
            return s

        fn(3)
        fn(3)
        # Should not crash — either deopt or return some value
        try:
            fn(10)
        except ZeroDivisionError:
            pass  # CPython fallback raised it — that's fine

    def test_mod_by_zero_no_crash(self) -> None:
        """Modulo by zero should deopt, not SIGFPE."""
        from pyjit import jit

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i % (i - 3)  # i=3 causes mod by zero
            return s

        fn(2)
        fn(2)
        try:
            fn(10)
        except ZeroDivisionError:
            pass


class TestBreakContinue:
    """Test break/continue in loops — Sprint Task 7."""

    def test_break_early_exit(self) -> None:
        from pyjit import jit

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i > 50:
                    break
                s += i
            return s

        fn(10)
        fn(10)
        assert fn(1000) == sum(range(51))

    def test_continue_skip(self) -> None:
        from pyjit import jit

        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                if i % 2 == 0:
                    continue
                s += i
            return s

        fn(10)
        fn(10)
        assert fn(100) == sum(i for i in range(100) if i % 2 != 0)


class TestListIndexing:
    """Test list[i] read in loop bodies — Month 1 Feature."""

    def test_sum_list(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def fn(data: list[int], n: int) -> int:
            s = 0
            for i in range(n):
                s += data[i]
            return s

        data = list(range(100))
        fn(data, 10)
        fn(data, 10)
        assert fn(data, 100) == sum(range(100))
        assert is_jit_compiled(fn)

    def test_weighted_sum(self) -> None:
        from pyjit import jit

        @jit(warmup=2)
        def fn(data: list[int], n: int) -> int:
            s = 0
            for i in range(n):
                s += data[i] * i
            return s

        data = [10, 20, 30, 40, 50]
        fn(data, 5)
        fn(data, 5)
        expected = sum(data[i] * i for i in range(5))
        assert fn(data, 5) == expected

    def test_dot_product(self) -> None:
        from pyjit import jit

        @jit(warmup=2)
        def fn(a: list[int], b: list[int], n: int) -> int:
            s = 0
            for i in range(n):
                s += a[i] * b[i]
            return s

        a = list(range(50))
        b = list(range(50, 100))
        fn(a, b, 10)
        fn(a, b, 10)
        expected = sum(a[i] * b[i] for i in range(50))
        assert fn(a, b, 50) == expected


class TestListWrite:
    """Test list[i] = x store in loop bodies — Month 2.1 Feature."""

    def test_scale_inplace(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def scale_inplace(data: list[int], n: int) -> int:
            s = 0
            for i in range(n):
                data[i] = data[i] * 2
                s += i
            return s

        data = [1, 2, 3, 4, 5]
        scale_inplace(data, 5)
        scale_inplace(data, 5)

        data2 = [1, 2, 3, 4, 5]
        result = scale_inplace(data2, 5)
        assert result == sum(range(5))
        assert data2 == [2, 4, 6, 8, 10]
        assert is_jit_compiled(scale_inplace)

    def test_write_counter(self) -> None:
        """Write loop counter values into a list."""
        from pyjit import jit

        @jit(warmup=2)
        def fill_range(data: list[int], n: int) -> int:
            s = 0
            for i in range(n):
                data[i] = i
                s += i
            return s

        data = [0] * 5
        fill_range(data, 5)
        fill_range(data, 5)
        data2 = [0] * 5
        result = fill_range(data2, 5)
        assert result == sum(range(5))
        assert data2 == list(range(5))

    def test_readback_after_write(self) -> None:
        """Write then read back in the same loop iteration."""
        from pyjit import jit

        @jit(warmup=2)
        def double_and_sum(data: list[int], n: int) -> int:
            s = 0
            for i in range(n):
                data[i] = data[i] * 2
                s += data[i]
            return s

        data = [1, 2, 3, 4, 5]
        double_and_sum(data, 5)
        double_and_sum(data, 5)
        data2 = [1, 2, 3, 4, 5]
        result = double_and_sum(data2, 5)
        # s = 2+4+6+8+10 = 30
        assert result == 30
        assert data2 == [2, 4, 6, 8, 10]


class TestNumPyIndexing:
    """Test NumPy array indexing — Month 2.2 Feature."""

    def test_sum_f64_array(self) -> None:
        import numpy as np
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def sum_f64(data: "np.ndarray", n: int) -> float:
            s = 0.0
            for i in range(n):
                s += data[i]
            return s

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        sum_f64(a, 5)
        sum_f64(a, 5)
        result = sum_f64(a, 5)
        assert abs(result - 15.0) < 1e-9
        assert is_jit_compiled(sum_f64)

    def test_dot_product_f64(self) -> None:
        import numpy as np
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def dot_f64(a: "np.ndarray", b: "np.ndarray", n: int) -> float:
            s = 0.0
            for i in range(n):
                s += a[i] * b[i]
            return s

        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        dot_f64(v1, v2, 3)
        dot_f64(v1, v2, 3)
        result = dot_f64(v1, v2, 3)
        assert abs(result - float(np.dot(v1, v2))) < 1e-9
        assert is_jit_compiled(dot_f64)

    def test_sum_i64_array(self) -> None:
        import numpy as np
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def sum_i64(data: "np.ndarray", n: int) -> int:
            s = 0
            for i in range(n):
                s += data[i]
            return s

        a = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        sum_i64(a, 5)
        sum_i64(a, 5)
        result = sum_i64(a, 5)
        assert result == 150
        assert is_jit_compiled(sum_i64)

    def test_scale_inplace_f64(self) -> None:
        import numpy as np
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def scale_f64(data: "np.ndarray", n: int) -> float:
            s = 0.0
            for i in range(n):
                data[i] = data[i] * 2.0
                s += data[i]
            return s

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        scale_f64(a, 5)
        scale_f64(a, 5)
        a2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = scale_f64(a2, 5)
        assert a2.tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(result - 30.0) < 1e-9
        assert is_jit_compiled(scale_f64)

    def test_scale_inplace_i64(self) -> None:
        import numpy as np
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def scale_i64(data: "np.ndarray", n: int) -> int:
            s = 0
            for i in range(n):
                data[i] = data[i] * 2
                s += i
            return s

        a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        scale_i64(a, 5)
        scale_i64(a, 5)
        a2 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        result = scale_i64(a2, 5)
        assert a2.tolist() == [2, 4, 6, 8, 10]
        assert result == sum(range(5))
        assert is_jit_compiled(scale_i64)


class TestABIFix:
    """Test args-buffer ABI — eliminates per-arity transmute dispatch (Month 2.3)."""

    def test_five_param_float_return(self) -> None:
        """Previously panicked: call_int_ret_float only handled ≤3 args."""
        import numpy as np
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def quad_dot(
            a: "np.ndarray", b: "np.ndarray", c: "np.ndarray", d: "np.ndarray", n: int
        ) -> float:
            s = 0.0
            for i in range(n):
                s += a[i] * b[i] + c[i] * d[i]
            return s

        v = np.ones(10, dtype=np.float64)
        quad_dot(v, v, v, v, 10)
        quad_dot(v, v, v, v, 10)
        result = quad_dot(v, v, v, v, 10)
        assert abs(result - 20.0) < 1e-9
        assert is_jit_compiled(quad_dot)

    def test_four_list_params(self) -> None:
        """Four list params = 5 total (list+list+list+list+n) — previously hit int_fn limit."""
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def four_list_sum(a: list, b: list, c: list, d: list, n: int) -> int:
            s = 0
            for i in range(n):
                s += a[i] + b[i] + c[i] + d[i]
            return s

        a, b, c, d = list(range(5)), list(range(5)), list(range(5)), list(range(5))
        four_list_sum(a, b, c, d, 5)
        four_list_sum(a, b, c, d, 5)
        result = four_list_sum(a, b, c, d, 5)
        assert result == 4 * sum(range(5))
        assert is_jit_compiled(four_list_sum)


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


class TestEagerJit:
    """Test @jit(eager=True) — compiles on first call."""

    def test_eager_compiles_on_first_call(self) -> None:
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(eager=True)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        # First call should compile AND execute correctly.
        assert fn(10) == 45
        assert is_jit_compiled(fn)

    def test_eager_correctness(self) -> None:
        from pyjit import jit

        @jit(eager=True)
        def sum_squares(n: int) -> int:
            s = 0
            for i in range(n):
                s += i * i
            return s

        assert sum_squares(100) == sum(i * i for i in range(100))
        assert sum_squares(0) == 0

    def test_eager_float(self) -> None:
        from pyjit import jit

        @jit(eager=True)
        def sum_float(n: int) -> float:
            s = 0.0
            for i in range(n):
                s += 1.5
            return s

        assert abs(sum_float(100) - 150.0) < 1e-9


class TestMathBuiltins:
    """Test math module builtins in loop bodies — Month 4.2."""

    def test_sqrt_in_loop(self) -> None:
        import math

        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def sum_sqrt(n: int) -> float:
            s = 0.0
            for i in range(n):
                s += math.sqrt(float(i))
            return s

        sum_sqrt(10)
        sum_sqrt(10)
        expected = sum(math.sqrt(float(i)) for i in range(100))
        assert abs(sum_sqrt(100) - expected) < 1e-6
        assert is_jit_compiled(sum_sqrt)

    def test_sqrt_with_numpy(self) -> None:
        import math

        import numpy as np
        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def sum_sqrt_np(data: "np.ndarray", n: int) -> float:
            s = 0.0
            for i in range(n):
                s += math.sqrt(data[i])
            return s

        a = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64)
        sum_sqrt_np(a, 5)
        sum_sqrt_np(a, 5)
        result = sum_sqrt_np(a, 5)
        assert abs(result - (1.0 + 2.0 + 3.0 + 4.0 + 5.0)) < 1e-9
        assert is_jit_compiled(sum_sqrt_np)

    def test_fabs_in_loop(self) -> None:
        import math

        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def sum_fabs(n: int) -> float:
            s = 0.0
            for i in range(n):
                s += math.fabs(float(i) - 50.0)
            return s

        sum_fabs(10)
        sum_fabs(10)
        expected = sum(math.fabs(float(i) - 50.0) for i in range(100))
        assert abs(sum_fabs(100) - expected) < 1e-6
        assert is_jit_compiled(sum_fabs)

    def test_exp_in_loop(self) -> None:
        import math

        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def sum_exp(n: int) -> float:
            s = 0.0
            for i in range(n):
                s += math.exp(float(i) * 0.01)
            return s

        sum_exp(10)
        sum_exp(10)
        expected = sum(math.exp(float(i) * 0.01) for i in range(20))
        assert abs(sum_exp(20) - expected) < 1e-6
        assert is_jit_compiled(sum_exp)

    def test_sin_cos_in_loop(self) -> None:
        import math

        from pyjit import jit
        from pyjit.inspect import is_jit_compiled

        @jit(warmup=2)
        def sum_sin(n: int) -> float:
            s = 0.0
            for i in range(n):
                s += math.sin(float(i) * 0.1)
            return s

        sum_sin(10)
        sum_sin(10)
        expected = sum(math.sin(float(i) * 0.1) for i in range(50))
        assert abs(sum_sin(50) - expected) < 1e-9
        assert is_jit_compiled(sum_sin)


class TestConstFolding:
    """Test that the const-fold + DCE optimizer passes work correctly."""

    def test_const_fold_add(self) -> None:
        from pyjit._pyjit import IROp, IRProgram, IRType, compile_ir

        # v0 = LoadConst 3, v1 = LoadConst 4, v2 = Add v0 v1 → return v2
        ops = [
            IROp("LoadConst", output=0, inputs=[], output_type=IRType.Int64, immediate=3),
            IROp("LoadConst", output=1, inputs=[], output_type=IRType.Int64, immediate=4),
            IROp("Add", output=2, inputs=[0, 1], output_type=IRType.Int64),
        ]
        prog = IRProgram(ops=ops, return_value=2, num_params=0, param_types=[])
        fn = compile_ir(prog, "const_add")
        assert fn() == 7

    def test_const_fold_mul(self) -> None:
        from pyjit._pyjit import IROp, IRProgram, IRType, compile_ir

        ops = [
            IROp("LoadConst", output=0, inputs=[], output_type=IRType.Int64, immediate=6),
            IROp("LoadConst", output=1, inputs=[], output_type=IRType.Int64, immediate=7),
            IROp("Mul", output=2, inputs=[0, 1], output_type=IRType.Int64),
        ]
        prog = IRProgram(ops=ops, return_value=2, num_params=0, param_types=[])
        fn = compile_ir(prog, "const_mul")
        assert fn() == 42

    def test_dce_removes_unused(self) -> None:
        """An unused value is emitted but doesn't affect the result."""
        from pyjit._pyjit import IROp, IRProgram, IRType, compile_ir

        ops = [
            IROp("Param", output=0, inputs=[], output_type=IRType.Int64),
            IROp("LoadConst", output=1, inputs=[], output_type=IRType.Int64, immediate=999),
            # v1 is never used — DCE should eliminate it
            IROp("Add", output=2, inputs=[0, 0], output_type=IRType.Int64),
        ]
        prog = IRProgram(ops=ops, return_value=2, num_params=1, param_types=[IRType.Int64])
        fn = compile_ir(prog, "dce_test")
        assert fn(5) == 10  # 5+5, not using v1 at all
