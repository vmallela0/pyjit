"""Tests for the bytecode tracer — Phase 1."""

from __future__ import annotations

from pyjit._pyjit import trace_function


class TestTraceArithmetic:
    """Trace simple arithmetic functions and verify ops are recorded."""

    def test_trace_add_result(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        assert trace.result == 7

    def test_trace_add_has_binary_op(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        assert any(op.kind == "BINARY_OP" for op in trace.ops)

    def test_trace_add_arg_types(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        binary_ops = [op for op in trace.ops if op.kind == "BINARY_OP"]
        assert len(binary_ops) >= 1
        assert binary_ops[0].arg_types == ["int", "int"]

    def test_trace_multiply(self) -> None:
        def fn(a: int, b: int) -> int:
            return a * b

        trace = trace_function(fn, (5, 6))
        assert trace.result == 30
        assert any(op.kind == "BINARY_OP" for op in trace.ops)

    def test_trace_mixed_arithmetic(self) -> None:
        def fn(a: int, b: int, c: int) -> int:
            return (a + b) * c

        trace = trace_function(fn, (2, 3, 4))
        assert trace.result == 20
        binary_ops = [op for op in trace.ops if op.kind == "BINARY_OP"]
        assert len(binary_ops) == 2

    def test_trace_float_arithmetic(self) -> None:
        def fn(a: float, b: float) -> float:
            return a + b

        trace = trace_function(fn, (1.5, 2.5))
        assert trace.result == 4.0
        assert trace.input_types == ["float", "float"]
        binary_ops = [op for op in trace.ops if op.kind == "BINARY_OP"]
        assert len(binary_ops) >= 1
        assert binary_ops[0].arg_types == ["float", "float"]


class TestTraceLoop:
    """Trace loops and verify iteration recording."""

    def test_loop_result(self) -> None:
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        trace = trace_function(fn, (10,))
        assert trace.result == 45

    def test_loop_has_for_iter(self) -> None:
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        trace = trace_function(fn, (5,))
        assert any(op.kind == "FOR_ITER" for op in trace.ops)

    def test_loop_body_marked(self) -> None:
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        trace = trace_function(fn, (5,))
        loop_ops = [op for op in trace.ops if op.is_loop_body]
        assert len(loop_ops) > 0

    def test_loop_iterations_recorded(self) -> None:
        """Each iteration should produce separate trace ops."""

        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        trace = trace_function(fn, (3,))
        # 3 iterations: each has BINARY_OP (+=)
        binary_ops = [op for op in trace.ops if op.kind == "BINARY_OP" and op.is_loop_body]
        assert len(binary_ops) == 3

    def test_loop_jump_backward(self) -> None:
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        trace = trace_function(fn, (3,))
        assert any(op.kind == "JUMP_BACKWARD" for op in trace.ops)

    def test_nested_loop(self) -> None:
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                for j in range(n):
                    s += i * j
            return s

        trace = trace_function(fn, (3,))
        assert trace.result == 9  # sum of i*j for i,j in [0,1,2]
        # Should have multiple FOR_ITER ops
        for_iters = [op for op in trace.ops if op.kind == "FOR_ITER"]
        assert len(for_iters) > 3  # outer + inner iterations


class TestTraceBranching:
    """Trace branching and verify branch direction recording."""

    def test_branch_taken(self) -> None:
        def fn(x: int) -> int:
            if x > 0:
                return 1
            else:
                return -1

        trace = trace_function(fn, (5,))
        assert trace.result == 1
        assert any(op.kind == "COMPARE_OP" for op in trace.ops)

    def test_branch_not_taken(self) -> None:
        def fn(x: int) -> int:
            if x > 0:
                return 1
            else:
                return -1

        trace = trace_function(fn, (-3,))
        assert trace.result == -1

    def test_different_paths_different_traces(self) -> None:
        """Different inputs produce different trace paths."""

        def fn(x: int) -> str:
            if x > 10:
                return "big"
            elif x > 0:
                return "small"
            else:
                return "negative"

        t1 = trace_function(fn, (20,))
        t2 = trace_function(fn, (5,))
        t3 = trace_function(fn, (-1,))

        assert t1.result == "big"
        assert t2.result == "small"
        assert t3.result == "negative"

        # Different branch paths should have different ops
        assert len(t1.ops) != len(t3.ops) or [o.offset for o in t1.ops] != [
            o.offset for o in t3.ops
        ]


class TestTraceTypeRecording:
    """Verify type observation for various Python types."""

    def test_int_types(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (1, 2))
        assert trace.input_types == ["int", "int"]

    def test_float_types(self) -> None:
        def fn(a: float, b: float) -> float:
            return a + b

        trace = trace_function(fn, (1.0, 2.0))
        assert trace.input_types == ["float", "float"]

    def test_str_types(self) -> None:
        def fn(a: str, b: str) -> str:
            return a + b

        trace = trace_function(fn, ("hello", " world"))
        assert trace.result == "hello world"
        assert trace.input_types == ["str", "str"]

    def test_none_type(self) -> None:
        def fn(x: None) -> int:
            return 0

        trace = trace_function(fn, (None,))
        assert trace.input_types == ["NoneType"]

    def test_list_type(self) -> None:
        def fn(items: list[int]) -> int:
            return len(items)

        trace = trace_function(fn, ([1, 2, 3],))
        assert trace.input_types == ["list"]


class TestTraceMetadata:
    """Verify trace metadata is correctly recorded."""

    def test_func_name(self) -> None:
        def my_fancy_function(x: int) -> int:
            return x + 1

        trace = trace_function(my_fancy_function, (5,))
        assert trace.func_name == "my_fancy_function"

    def test_trace_len(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (1, 2))
        assert len(trace) == len(trace.ops)
        assert len(trace) > 0

    def test_trace_repr(self) -> None:
        def fn(a: int) -> int:
            return a

        trace = trace_function(fn, (42,))
        r = repr(trace)
        assert "fn" in r
        assert "ops=" in r

    def test_traceop_repr(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (1, 2))
        op = trace.ops[0]
        r = repr(op)
        assert "TraceOp" in r
        assert op.kind in r


class TestTraceEdgeCases:
    """Edge cases and robustness checks."""

    def test_empty_function(self) -> None:
        def fn() -> None:
            pass

        trace = trace_function(fn)
        assert trace.result is None
        assert trace.input_types == []

    def test_single_constant(self) -> None:
        def fn() -> int:
            return 42

        trace = trace_function(fn)
        assert trace.result == 42

    def test_many_args(self) -> None:
        def fn(a: int, b: int, c: int, d: int) -> int:
            return a + b + c + d

        trace = trace_function(fn, (1, 2, 3, 4))
        assert trace.result == 10
        assert trace.input_types == ["int", "int", "int", "int"]

    def test_while_loop(self) -> None:
        def fn(n: int) -> int:
            s = 0
            i = 0
            while i < n:
                s += i
                i += 1
            return s

        trace = trace_function(fn, (5,))
        assert trace.result == 10
        # While loops use COMPARE_OP + POP_JUMP_IF_FALSE + JUMP_BACKWARD
        assert any(op.kind == "COMPARE_OP" for op in trace.ops)
