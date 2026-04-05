"""Bytecode-level tracer using sys.monitoring (PEP 669).

Records every bytecode instruction executed during a function call,
along with observed operand types. This is the foundation for the
tracing JIT — the recorded trace becomes input to IR generation.
"""

from __future__ import annotations

import opcode
import sys
from typing import Any, Callable

from pyjit._pyjit import Trace, TraceOp

# BINARY_OP sub-opcode to human-readable name
_BINARY_OP_NAMES: dict[int, str] = {}
if hasattr(opcode, "_nb_ops"):
    for i, (name, _symbol) in enumerate(opcode._nb_ops):
        _BINARY_OP_NAMES[i] = name

# COMPARE_OP sub-opcode names
_COMPARE_OP_NAMES: dict[int, str] = {
    0: "LT",
    1: "LE",
    2: "EQ",
    3: "NE",
    4: "GT",
    5: "GE",
}

# Opcodes that indicate the start/end of a loop body
_LOOP_START_OPS = {"FOR_ITER", "GET_ITER"}
_LOOP_END_OPS = {"END_FOR", "POP_ITER"}
_LOOP_BACK_OPS = {"JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"}

_TOOL_ID = sys.monitoring.PROFILER_ID


def _get_type_name(obj: object) -> str:
    """Get the type name of a Python object."""
    return type(obj).__name__


def record_trace(
    func: Callable[..., Any],
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> Trace:
    """Record a bytecode-level execution trace of a function call.

    Args:
        func: The Python function to trace.
        args: Positional arguments for the function call.
        kwargs: Keyword arguments for the function call.

    Returns:
        A Trace object containing the result and recorded operations.
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    target_code = func.__code__
    input_types = [_get_type_name(a) for a in args]

    # State for the tracer
    raw_ops: list[tuple[int, str, int]] = []  # (offset, opcode_name, arg)
    in_loop = False
    loop_depth = 0
    seen_offsets: set[int] = set()

    mon = sys.monitoring

    try:
        mon.use_tool_id(_TOOL_ID, "pyjit_tracer")
    except ValueError:
        # Tool ID already in use, free and reclaim
        mon.free_tool_id(_TOOL_ID)
        mon.use_tool_id(_TOOL_ID, "pyjit_tracer")

    def on_instruction(code: Any, offset: int) -> object:
        nonlocal in_loop, loop_depth

        if code is not target_code:
            return mon.DISABLE

        raw = code.co_code
        if offset >= len(raw):
            return None

        op = raw[offset]
        arg = raw[offset + 1] if offset + 1 < len(raw) else 0
        name = opcode.opname[op]

        # Track loop state
        if name in _LOOP_START_OPS:
            if name == "FOR_ITER":
                loop_depth += 1
                in_loop = True
        elif name in _LOOP_END_OPS:
            loop_depth = max(0, loop_depth - 1)
            if loop_depth == 0:
                in_loop = False

        # Detect loop via back-edge (revisiting an offset)
        if offset in seen_offsets and name in _LOOP_BACK_OPS:
            in_loop = True

        seen_offsets.add(offset)
        raw_ops.append((offset, name, arg))
        return None

    mon.register_callback(_TOOL_ID, mon.events.INSTRUCTION, on_instruction)
    mon.set_local_events(_TOOL_ID, target_code, mon.events.INSTRUCTION)

    try:
        result = func(*args, **kwargs)
    finally:
        mon.set_local_events(_TOOL_ID, target_code, 0)
        mon.register_callback(_TOOL_ID, mon.events.INSTRUCTION, None)
        mon.free_tool_id(_TOOL_ID)

    # Build TraceOp objects with type information
    trace_ops = _build_trace_ops(raw_ops, target_code, args, input_types, result)

    return Trace(
        result=result,
        ops=trace_ops,
        func_name=func.__name__,
        input_types=input_types,
    )


def _build_trace_ops(
    raw_ops: list[tuple[int, str, int]],
    code: Any,
    args: tuple[Any, ...],
    input_types: list[str],
    result: Any,
) -> list[TraceOp]:
    """Convert raw recorded ops into TraceOp objects with type annotations.

    Uses a shadow type stack to track operand types through the trace.
    """
    # Initialize local variable types from function arguments
    local_types: dict[int, str] = {}
    for i, t in enumerate(input_types):
        local_types[i] = t

    # Shadow type stack for tracking operand types
    type_stack: list[str] = []
    trace_ops: list[TraceOp] = []

    # Track which offsets are in loop bodies
    loop_offsets = _detect_loop_offsets(raw_ops)

    for offset, name, arg in raw_ops:
        arg_types: list[str] = []
        is_loop = offset in loop_offsets

        # Simulate type stack effects based on opcode
        if name == "RESUME":
            pass

        elif name in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"):
            var_idx = arg
            t = local_types.get(var_idx, "object")
            type_stack.append(t)
            arg_types = [t]

        elif name == "LOAD_FAST_BORROW_LOAD_FAST_BORROW":
            # Fused opcode: loads two locals onto the stack.
            # Encoding: first pushed = arg >> 4, second pushed = arg & 0xf
            idx_a = (arg >> 4) & 0xF
            idx_b = arg & 0xF
            t_a = local_types.get(idx_a, "object")
            t_b = local_types.get(idx_b, "object")
            type_stack.append(t_a)
            type_stack.append(t_b)
            arg_types = [t_a, t_b]

        elif name in ("STORE_FAST", "STORE_FAST_MAYBE_NULL"):
            if type_stack:
                t = type_stack.pop()
                local_types[arg] = t
                arg_types = [t]

        elif name == "BINARY_OP":
            if len(type_stack) >= 2:
                t_b = type_stack.pop()
                t_a = type_stack.pop()
                arg_types = [t_a, t_b]
                # Result type: int op int -> int, float involved -> float
                if t_a == "float" or t_b == "float":
                    type_stack.append("float")
                else:
                    type_stack.append(t_a)
            else:
                type_stack.append("object")

        elif name in ("COMPARE_OP", "CONTAINS_OP", "IS_OP"):
            if len(type_stack) >= 2:
                t_b = type_stack.pop()
                t_a = type_stack.pop()
                arg_types = [t_a, t_b]
            type_stack.append("bool")

        elif name in ("LOAD_CONST", "LOAD_SMALL_INT"):
            if name == "LOAD_SMALL_INT":
                type_stack.append("int")
            else:
                # Try to determine type from co_consts
                consts = code.co_consts
                if arg < len(consts):
                    type_stack.append(_get_type_name(consts[arg]))
                else:
                    type_stack.append("object")

        elif name in ("LOAD_GLOBAL",):
            # LOAD_GLOBAL may push NULL + value or just value
            type_stack.append("object")  # the global value

        elif name in ("CALL",):
            # CALL pops callable + args; we can't easily determine the return type
            # Pop the arguments and callable from stack
            n_call_args = arg
            for _ in range(n_call_args + 1):  # +1 for callable (+ possible NULL)
                if type_stack:
                    type_stack.pop()
            type_stack.append("object")  # return value type unknown

        elif name == "RETURN_VALUE":
            if type_stack:
                t = type_stack.pop()
                arg_types = [t]

        elif name in (
            "POP_JUMP_IF_FALSE",
            "POP_JUMP_IF_TRUE",
            "POP_JUMP_IF_NONE",
            "POP_JUMP_IF_NOT_NONE",
        ):
            if type_stack:
                t = type_stack.pop()
                arg_types = [t]

        elif name == "GET_ITER":
            if type_stack:
                t = type_stack.pop()
                arg_types = [t]
            type_stack.append("iterator")

        elif name == "FOR_ITER":
            # Pushes the next value from the iterator
            type_stack.append("int")  # assume int for range() iterators

        elif name in ("END_FOR", "POP_ITER"):
            if type_stack:
                type_stack.pop()

        elif name in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "JUMP_FORWARD", "NOT_TAKEN"):
            pass  # no stack effect

        elif name == "POP_TOP":
            if type_stack:
                type_stack.pop()

        elif name == "COPY":
            if type_stack:
                type_stack.append(type_stack[-1])

        elif name == "SWAP":
            if len(type_stack) >= 2:
                type_stack[-1], type_stack[-2] = type_stack[-2], type_stack[-1]

        elif name in ("UNARY_NEGATIVE", "UNARY_NOT", "UNARY_INVERT"):
            if type_stack:
                arg_types = [type_stack[-1]]
                # Result type is same as input for neg/invert, bool for not
                if name == "UNARY_NOT":
                    type_stack[-1] = "bool"

        else:
            # Unknown opcode — don't crash, just mark types as unknown
            pass

        op = TraceOp(
            kind=name,
            arg=arg,
            offset=offset,
            arg_types=arg_types,
            is_loop_body=is_loop,
        )
        trace_ops.append(op)

    return trace_ops


def _detect_loop_offsets(raw_ops: list[tuple[int, str, int]]) -> set[int]:
    """Detect which offsets are inside loop bodies.

    A loop body starts after FOR_ITER and ends at END_FOR/POP_ITER.
    Also detected via back-edges (JUMP_BACKWARD to a previously seen offset).
    """
    loop_offsets: set[int] = set()
    in_loop = False
    loop_depth = 0
    back_edge_targets: set[int] = set()

    # First pass: find back-edge targets
    seen: set[int] = set()
    for offset, name, _arg in raw_ops:
        if offset in seen and name in _LOOP_BACK_OPS:
            back_edge_targets.add(offset)
        seen.add(offset)

    # Second pass: mark loop body offsets
    for offset, name, _arg in raw_ops:
        if name == "FOR_ITER":
            loop_depth += 1
            in_loop = True
        elif name in _LOOP_END_OPS:
            loop_depth = max(0, loop_depth - 1)
            if loop_depth == 0:
                in_loop = False

        if in_loop:
            loop_offsets.add(offset)

    return loop_offsets
