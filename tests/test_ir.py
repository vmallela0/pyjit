"""Tests for IR generation — Phase 1.5."""

from __future__ import annotations

from pyjit._pyjit import IRType, build_ir, trace_function


class TestIRArithmetic:
    """Arithmetic traces produce correct IR ops."""

    def test_add_produces_ir(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        assert len(ir.ops) > 0

    def test_add_has_add_op(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        assert any(op.kind == "Add" for op in ir.ops)

    def test_multiply_has_mul_op(self) -> None:
        def fn(a: int, b: int) -> int:
            return a * b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        assert any(op.kind == "Mul" for op in ir.ops)

    def test_mixed_arithmetic(self) -> None:
        def fn(a: int, b: int, c: int) -> int:
            return (a + b) * c

        trace = trace_function(fn, (2, 3, 4))
        ir = build_ir(trace)
        kinds = [op.kind for op in ir.ops]
        assert "Add" in kinds
        assert "Mul" in kinds

    def test_subtraction(self) -> None:
        def fn(a: int, b: int) -> int:
            return a - b

        trace = trace_function(fn, (10, 3))
        ir = build_ir(trace)
        assert any(op.kind == "Sub" for op in ir.ops)


class TestIRSSAForm:
    """Verify correct SSA form: every value defined before use."""

    def test_ssa_arithmetic(self) -> None:
        def fn(a: int, b: int) -> int:
            return a * b + 1

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        _assert_valid_ssa(ir)

    def test_ssa_multi_op(self) -> None:
        def fn(a: int, b: int, c: int) -> int:
            x = a + b
            y = x * c
            return y - a

        trace = trace_function(fn, (1, 2, 3))
        ir = build_ir(trace)
        _assert_valid_ssa(ir)

    def test_ssa_loop(self) -> None:
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        trace = trace_function(fn, (5,))
        ir = build_ir(trace)
        _assert_valid_ssa(ir)

    def test_ssa_branching(self) -> None:
        def fn(x: int) -> int:
            if x > 0:
                return 1
            else:
                return -1

        trace = trace_function(fn, (5,))
        ir = build_ir(trace)
        _assert_valid_ssa(ir)


class TestIRGuards:
    """Verify guard insertion for type assumptions."""

    def test_guards_for_int_params(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        guards = [op for op in ir.ops if op.kind == "Guard"]
        assert len(guards) >= 2  # one guard per int param

    def test_guard_has_expected_type(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        guards = [op for op in ir.ops if op.kind == "Guard"]
        for guard in guards:
            assert guard.guard_type is not None
            assert guard.guard_type == IRType.Int64

    def test_float_guards(self) -> None:
        def fn(a: float, b: float) -> float:
            return a + b

        trace = trace_function(fn, (1.0, 2.0))
        ir = build_ir(trace)
        guards = [op for op in ir.ops if op.kind == "Guard"]
        assert len(guards) >= 2
        for guard in guards:
            assert guard.guard_type == IRType.Float64


class TestIRUnboxing:
    """Verify unbox/box operations for type specialization."""

    def test_int_unbox(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        unbox_ops = [op for op in ir.ops if op.kind == "UnboxInt"]
        assert len(unbox_ops) >= 2  # unbox a and b

    def test_float_unbox(self) -> None:
        def fn(a: float, b: float) -> float:
            return a + b

        trace = trace_function(fn, (1.0, 2.0))
        ir = build_ir(trace)
        unbox_ops = [op for op in ir.ops if op.kind == "UnboxFloat"]
        assert len(unbox_ops) >= 2

    def test_box_on_return(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        box_ops = [op for op in ir.ops if op.kind == "BoxInt"]
        assert len(box_ops) >= 1  # box the return value


class TestIRTypes:
    """Verify IR type propagation."""

    def test_int_arithmetic_stays_int(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        add_ops = [op for op in ir.ops if op.kind == "Add"]
        assert len(add_ops) == 1
        assert add_ops[0].output_type == IRType.Int64

    def test_param_types(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        assert ir.param_types == [IRType.Int64, IRType.Int64]
        assert ir.num_params == 2

    def test_float_param_types(self) -> None:
        def fn(a: float, b: float) -> float:
            return a + b

        trace = trace_function(fn, (1.0, 2.0))
        ir = build_ir(trace)
        assert ir.param_types == [IRType.Float64, IRType.Float64]


class TestIRProgram:
    """Test IRProgram metadata and utilities."""

    def test_return_value_set(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        assert ir.return_value is not None

    def test_len(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        assert len(ir) == len(ir.ops)

    def test_dump_contains_ops(self) -> None:
        def fn(a: int, b: int) -> int:
            return a + b

        trace = trace_function(fn, (3, 4))
        ir = build_ir(trace)
        dump = ir.dump()
        assert "Add" in dump
        assert "Param" in dump
        assert "return" in dump

    def test_repr(self) -> None:
        def fn(a: int) -> int:
            return a + 1

        trace = trace_function(fn, (5,))
        ir = build_ir(trace)
        r = repr(ir)
        assert "IRProgram" in r


class TestIRLoop:
    """Verify loop trace produces correct unrolled IR."""

    def test_loop_unrolled(self) -> None:
        """A loop with 3 iterations should produce 3 Add ops."""

        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        trace = trace_function(fn, (3,))
        ir = build_ir(trace)
        add_ops = [op for op in ir.ops if op.kind == "Add"]
        assert len(add_ops) == 3

    def test_loop_has_iter_ops(self) -> None:
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        trace = trace_function(fn, (3,))
        ir = build_ir(trace)
        iter_ops = [op for op in ir.ops if op.kind == "IterNext"]
        assert len(iter_ops) > 0


def _assert_valid_ssa(ir: object) -> None:
    """Assert that an IR program is in valid SSA form."""
    defined: set[int] = set()
    for op in ir.ops:  # type: ignore[attr-defined]
        for inp in op.inputs:
            assert inp in defined, (
                f"v{inp} used before definition in op '{op.kind}' "
                f"(defined so far: {sorted(defined)})"
            )
        if op.output is not None:
            defined.add(op.output)
