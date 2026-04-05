"""Loop analysis and compilation orchestration.

Analyzes traces to detect loop patterns, extracts loop structure,
and delegates to the Cranelift backend for native compilation.
"""

from __future__ import annotations

import dis
from typing import Any, Callable

from pyjit._pyjit import (
    CompiledFunction,
    compile_loop_ir,
)

# BINARY_OP sub-opcode to IR op name mapping
_BINOP_MAP: dict[int, str] = {
    0: "Add",  # NB_ADD
    5: "Mul",  # NB_MULTIPLY
    10: "Sub",  # NB_SUBTRACT
    11: "Div",  # NB_TRUE_DIVIDE
    2: "FloorDiv",  # NB_FLOOR_DIVIDE
    6: "Mod",  # NB_REMAINDER
    13: "Add",  # NB_INPLACE_ADD
    18: "Mul",  # NB_INPLACE_MULTIPLY
    23: "Sub",  # NB_INPLACE_SUBTRACT
    24: "Div",  # NB_INPLACE_TRUE_DIVIDE
    15: "FloorDiv",  # NB_INPLACE_FLOOR_DIVIDE
    19: "Mod",  # NB_INPLACE_REMAINDER
}

COUNTER_SENTINEL = 2**64 - 1  # usize::MAX — means "use loop counter"

# Type IDs matching Rust side: 0 = i64, 1 = f64
TYPE_I64 = 0
TYPE_F64 = 1


def compile_function(
    func: Callable[..., Any],
    args: tuple[Any, ...],
) -> CompiledFunction | None:
    """Compile a Python function to native code.

    Supports int and float arguments for loop functions.
    Returns None if the function can't be compiled.
    """
    # Check all args are numeric (int or float)
    if not all(isinstance(a, (int, float)) for a in args):
        return None

    loop_result = _try_compile_loop(func, args)
    if loop_result is not None:
        return loop_result

    return None


def _try_compile_loop(
    func: Callable[..., Any],
    args: tuple[Any, ...],
) -> CompiledFunction | None:
    """Try to detect and compile a range-for loop pattern."""
    code = func.__code__
    instructions = list(dis.get_instructions(code))

    for_iter_idx = None
    for i, instr in enumerate(instructions):
        if instr.opname == "FOR_ITER":
            for_iter_idx = i
            break

    if for_iter_idx is None:
        return None

    range_param = _detect_range_param(instructions, for_iter_idx, code)
    if range_param is None:
        return None

    body_start = for_iter_idx + 1
    body_end = None
    for i in range(body_start, len(instructions)):
        if instructions[i].opname in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
            body_end = i
            break

    if body_end is None:
        return None

    if instructions[body_start].opname != "STORE_FAST":
        return None
    iter_var_slot = instructions[body_start].arg
    if iter_var_slot is None:
        return None

    body_ops = _extract_body_ops(
        instructions[body_start + 1 : body_end],
        code,
        iter_var_slot,
    )
    if body_ops is None:
        return None

    return_local = _find_return_local(instructions)
    if return_local is None:
        return None

    init_locals_int, init_locals_float, float_slots = _find_init_locals(
        instructions, for_iter_idx, code
    )

    # Determine param types
    num_params = code.co_argcount
    param_types: list[int] = []
    for i in range(num_params):
        if i < len(args) and isinstance(args[i], float):
            param_types.append(TYPE_F64)
        else:
            param_types.append(TYPE_I64)

    # Calculate num_locals
    base_locals = code.co_nlocals
    max_slot = max((op[1] for op in body_ops), default=0)
    max_src = max(
        (op[2] for op in body_ops if op[2] != COUNTER_SENTINEL),
        default=0,
    )
    max_src_b = max(
        (op[3] for op in body_ops if not op[4] and op[3] != COUNTER_SENTINEL),
        default=0,
    )
    num_locals = max(base_locals, max_slot + 1, max_src + 1, max_src_b + 1)

    # Build local_types: float slots are f64, everything else is i64
    local_types: list[int] = [TYPE_I64] * num_locals
    for slot in float_slots:
        if slot < num_locals:
            local_types[slot] = TYPE_F64
    # Params that are float
    for i, pt in enumerate(param_types):
        if i < num_locals:
            local_types[i] = pt
    # Temp slots used by body ops that produce float results inherit from their dst
    # If a body op's dst is a float slot, the temp intermediates should also be float
    for op_kind, dst, _sa, _sb, _imm, _iv in body_ops:
        if dst < num_locals and local_types[dst] == TYPE_F64:
            pass  # already set
        # If the op is Div (true division), result is float
        if op_kind == "Div" and dst < num_locals:
            local_types[dst] = TYPE_F64

    return_type_id = local_types[return_local] if return_local < len(local_types) else TYPE_I64

    try:
        return compile_loop_ir(
            num_params=num_params,
            limit_param=range_param,
            num_locals=num_locals,
            return_local=return_local,
            init_locals=init_locals_int,
            init_float_locals=init_locals_float,
            body_ops=body_ops,
            local_types=local_types,
            param_types=param_types,
            return_type_id=return_type_id,
            func_name=func.__name__,
        )
    except Exception:
        return None


def _detect_range_param(
    instructions: list[dis.Instruction],
    for_iter_idx: int,
    code: Any,
) -> int | None:
    """Detect `for i in range(param)` and return the param index."""
    idx = for_iter_idx - 1
    if idx < 0 or instructions[idx].opname != "GET_ITER":
        return None
    idx -= 1
    if idx < 0 or instructions[idx].opname != "CALL":
        return None
    idx -= 1
    if idx < 0:
        return None

    load_instr = instructions[idx]
    if load_instr.opname in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"):
        return load_instr.arg

    return None


def _extract_body_ops(
    body_instrs: list[dis.Instruction],
    code: Any,
    iter_var_slot: int,
) -> list[tuple[str, int, int, int, bool, int]] | None:
    """Extract loop body operations as (kind, dst, src_a, src_b, is_b_imm, imm).

    Uses a stack simulation to track operand flow.
    """
    stack: list[int | str] = []  # local slot indices or 'counter' or ('imm', val)
    ops: list[tuple[str, int, int, int, bool, int]] = []

    for instr in body_instrs:
        name = instr.opname
        arg = instr.arg if instr.arg is not None else 0

        if name in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"):
            if arg == iter_var_slot:
                stack.append("counter")
            else:
                stack.append(arg)

        elif name == "LOAD_FAST_BORROW_LOAD_FAST_BORROW":
            idx_a = (arg >> 4) & 0xF
            idx_b = arg & 0xF
            stack.append("counter" if idx_a == iter_var_slot else idx_a)
            stack.append("counter" if idx_b == iter_var_slot else idx_b)

        elif name == "LOAD_SMALL_INT":
            stack.append(("imm", arg))  # type: ignore[arg-type]

        elif name == "LOAD_CONST":
            # Handle float constants from co_consts
            consts = code.co_consts
            if arg < len(consts) and isinstance(consts[arg], (int, float)):
                stack.append(("imm", consts[arg]))  # type: ignore[arg-type]
            else:
                return None  # unsupported constant type

        elif name == "BINARY_OP":
            if len(stack) < 2:
                return None
            b_val = stack.pop()
            a_val = stack.pop()
            op_name = _BINOP_MAP.get(arg)
            if op_name is None:
                return None

            temp_slot = 100 + len(ops)

            src_a = (
                COUNTER_SENTINEL if a_val == "counter" else (a_val if isinstance(a_val, int) else 0)
            )
            if isinstance(b_val, tuple) and b_val[0] == "imm":
                imm_val = b_val[1]
                # For float immediates, encode as int bits
                if isinstance(imm_val, float):
                    import struct

                    imm_bits = struct.unpack("<q", struct.pack("<d", imm_val))[0]
                    ops.append((op_name, temp_slot, src_a, 0, True, imm_bits))
                else:
                    ops.append((op_name, temp_slot, src_a, 0, True, int(imm_val)))
            elif b_val == "counter":
                ops.append((op_name, temp_slot, src_a, COUNTER_SENTINEL, False, 0))
            elif isinstance(b_val, int):
                ops.append((op_name, temp_slot, src_a, b_val, False, 0))
            else:
                return None

            stack.append(temp_slot)

        elif name in ("STORE_FAST", "STORE_FAST_MAYBE_NULL"):
            if not stack:
                return None
            val = stack.pop()
            if isinstance(val, int):
                if ops and ops[-1][1] >= 100:
                    last = ops[-1]
                    ops[-1] = (last[0], arg, last[2], last[3], last[4], last[5])
            elif val == "counter":
                pass

        elif name in ("RESUME", "NOT_TAKEN", "NOP"):
            pass

        elif name == "CALL":
            # Handle float() conversion: CALL on float builtin with 1 arg
            # The arg on the stack is what gets converted to float
            # For now, just treat it as a passthrough (the value is already numeric)
            n_call_args = arg
            if n_call_args == 1 and stack:
                # float(x) or int(x) — passthrough for numeric types
                val = stack[-1]  # peek at the arg
                # Pop args + callable (CALL pops n_args + 1 for the callable)
                call_arg = stack.pop()
                if stack:
                    stack.pop()  # pop the callable (LOAD_GLOBAL float)
                stack.append(call_arg)
            else:
                return None

        elif name == "LOAD_GLOBAL":
            # Push a marker for the global (might be float/int builtin)
            stack.append(("global", arg))  # type: ignore[arg-type]

        else:
            return None

    return ops if ops else None


def _find_return_local(instructions: list[dis.Instruction]) -> int | None:
    """Find which local variable is returned."""
    for i, instr in enumerate(instructions):
        if instr.opname == "RETURN_VALUE" and i > 0:
            prev = instructions[i - 1]
            if prev.opname in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"):
                return prev.arg
    return None


def _find_init_locals(
    instructions: list[dis.Instruction],
    for_iter_idx: int,
    code: Any,
) -> tuple[list[tuple[int, int]], list[tuple[int, float]], set[int]]:
    """Find local variable initializations before the loop.

    Returns (int_inits, float_inits, float_slots).
    """
    int_inits: list[tuple[int, int]] = []
    float_inits: list[tuple[int, float]] = []
    float_slots: set[int] = set()

    for i in range(for_iter_idx):
        instr = instructions[i]
        if instr.opname == "STORE_FAST" and i > 0 and instr.arg is not None:
            prev = instructions[i - 1]
            if prev.opname == "LOAD_SMALL_INT" and prev.arg is not None:
                int_inits.append((instr.arg, prev.arg))
            elif prev.opname == "LOAD_CONST" and prev.arg is not None:
                consts = code.co_consts
                if prev.arg < len(consts):
                    val = consts[prev.arg]
                    if isinstance(val, float):
                        float_inits.append((instr.arg, val))
                        float_slots.add(instr.arg)
                    elif isinstance(val, int):
                        int_inits.append((instr.arg, val))

    return int_inits, float_inits, float_slots
