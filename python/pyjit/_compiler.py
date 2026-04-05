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
    2: "FloorDiv",  # NB_FLOOR_DIVIDE
    6: "Mod",  # NB_REMAINDER
    13: "Add",  # NB_INPLACE_ADD
    18: "Mul",  # NB_INPLACE_MULTIPLY
    23: "Sub",  # NB_INPLACE_SUBTRACT
    15: "FloorDiv",  # NB_INPLACE_FLOOR_DIVIDE
    19: "Mod",  # NB_INPLACE_REMAINDER
}

COUNTER_SENTINEL = 2**64 - 1  # usize::MAX — means "use loop counter"


def compile_function(
    func: Callable[..., Any],
    args: tuple[Any, ...],
) -> CompiledFunction | None:
    """Compile a Python function to native code.

    Detects whether the function contains a loop pattern and chooses
    the appropriate compilation strategy:
    - Simple functions: trace → IR → Cranelift
    - Loop functions: analyze loop → compile with native loop blocks

    Returns None if the function can't be compiled.
    """
    # Check if all args are ints (only supported type for now)
    if not all(isinstance(a, int) for a in args):
        return None

    # Try loop compilation first (the common case for speedups)
    loop_result = _try_compile_loop(func, args)
    if loop_result is not None:
        return loop_result

    # Linear compilation is only safe for pure arithmetic (no branches, no calls).
    # For now, only loop compilation is used — it gives the real speedups.
    return None


def _try_compile_loop(
    func: Callable[..., Any],
    args: tuple[Any, ...],
) -> CompiledFunction | None:
    """Try to detect and compile a range-for loop pattern."""
    code = func.__code__
    instructions = list(dis.get_instructions(code))

    # Find FOR_ITER instruction (indicates a for loop)
    for_iter_idx = None
    for i, instr in enumerate(instructions):
        if instr.opname == "FOR_ITER":
            for_iter_idx = i
            break

    if for_iter_idx is None:
        return None  # No for loop found

    # Detect the range() call pattern:
    # LOAD_GLOBAL range, LOAD_FAST n, CALL 1, GET_ITER, FOR_ITER
    range_param = _detect_range_param(instructions, for_iter_idx, code)
    if range_param is None:
        return None

    # Find the loop body (between FOR_ITER and JUMP_BACKWARD)
    body_start = for_iter_idx + 1
    body_end = None
    for i in range(body_start, len(instructions)):
        if instructions[i].opname in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
            body_end = i
            break

    if body_end is None:
        return None

    # Extract the iterator variable slot (STORE_FAST after FOR_ITER)
    if instructions[body_start].opname != "STORE_FAST":
        return None
    iter_var_slot = instructions[body_start].arg
    if iter_var_slot is None:
        return None

    # Analyze the loop body operations
    body_ops = _extract_body_ops(
        instructions[body_start + 1 : body_end],
        code,
        iter_var_slot,
    )
    if body_ops is None:
        return None

    # Find which local is returned (look for LOAD_FAST before RETURN_VALUE)
    return_local = _find_return_local(instructions)
    if return_local is None:
        return None

    # Find initialized locals (LOAD_SMALL_INT + STORE_FAST before the loop)
    init_locals = _find_init_locals(instructions, for_iter_idx)

    # Calculate num_locals — must be large enough for all temp slots used by body_ops
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
    num_params = code.co_argcount

    try:
        return compile_loop_ir(
            num_params=num_params,
            limit_param=range_param,
            num_locals=num_locals,
            return_local=return_local,
            init_locals=init_locals,
            body_ops=body_ops,
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
    # Walk backwards from FOR_ITER to find: LOAD_GLOBAL range, LOAD_FAST x, CALL 1, GET_ITER
    idx = for_iter_idx - 1
    if idx < 0 or instructions[idx].opname != "GET_ITER":
        return None
    idx -= 1
    if idx < 0 or instructions[idx].opname != "CALL":
        return None
    idx -= 1
    if idx < 0:
        return None

    # The LOAD_FAST before CALL is the range argument
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

        elif name == "BINARY_OP":
            if len(stack) < 2:
                return None
            b_val = stack.pop()
            a_val = stack.pop()
            op_name = _BINOP_MAP.get(arg)
            if op_name is None:
                return None

            # We need a temp local to store the result.
            # Use a high slot number that won't collide.
            temp_slot = 100 + len(ops)

            src_a = (
                COUNTER_SENTINEL if a_val == "counter" else (a_val if isinstance(a_val, int) else 0)
            )
            if isinstance(b_val, tuple) and b_val[0] == "imm":
                ops.append((op_name, temp_slot, src_a, 0, True, b_val[1]))
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
                # Rewrite the last op's dst to this local slot
                if ops and ops[-1][1] >= 100:
                    last = ops[-1]
                    ops[-1] = (last[0], arg, last[2], last[3], last[4], last[5])
                # Also handle direct store (load + store without op)
            elif val == "counter":
                # Storing counter to a local — this is the `i` assignment, skip
                pass

        elif name in ("RESUME", "NOT_TAKEN", "NOP"):
            pass

        else:
            # Unsupported opcode in loop body
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
) -> list[tuple[int, int]]:
    """Find local variable initializations before the loop."""
    inits: list[tuple[int, int]] = []
    for i in range(for_iter_idx):
        instr = instructions[i]
        if instr.opname == "STORE_FAST" and i > 0:
            prev = instructions[i - 1]
            if prev.opname == "LOAD_SMALL_INT":
                inits.append((instr.arg, prev.arg))  # type: ignore[arg-type]
    return inits
