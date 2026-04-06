"""Loop analysis and compilation orchestration.

Analyzes traces to detect loop patterns, extracts loop structure,
and delegates to the Cranelift backend for native compilation.
"""

from __future__ import annotations

import dis
import sys
from typing import Any, Callable

from pyjit._pyjit import (
    CompiledFunction,
    compile_loop_ir,
)

# ---------------------------------------------------------------------------
# Explain mode — thread-local log for @jit(explain=True) diagnostics
# ---------------------------------------------------------------------------

_explain_log: list[str] | None = None


def _log(msg: str) -> None:
    """Append a message to the explain log if explain mode is active."""
    if _explain_log is not None:
        _explain_log.append(msg)


# ---------------------------------------------------------------------------
# Inline slot base: inner-function locals are remapped above this threshold.
# Must be above the 100+ temp-slot range used by _make_binop.
# ---------------------------------------------------------------------------
_INLINE_SLOT_BASE = 500

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
    8: "Pow",  # NB_POWER
    21: "Pow",  # NB_INPLACE_POWER
    1: "BitAnd",  # NB_AND
    7: "BitOr",  # NB_OR
    12: "BitXor",  # NB_XOR
    3: "LShift",  # NB_LSHIFT
    9: "RShift",  # NB_RSHIFT
    14: "BitAnd",  # NB_INPLACE_AND
    20: "BitOr",  # NB_INPLACE_OR
    25: "BitXor",  # NB_INPLACE_XOR
    16: "LShift",  # NB_INPLACE_LSHIFT
    22: "RShift",  # NB_INPLACE_RSHIFT
}

COUNTER_SENTINEL = 2**64 - 1  # usize::MAX — means "use loop counter"

# Type IDs matching Rust side: 0 = i64, 1 = f64
TYPE_I64 = 0
TYPE_F64 = 1


def compile_function(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    explain_log: list[str] | None = None,
) -> CompiledFunction | None:
    """Compile a Python function to native code.

    Supports int and float arguments for loop functions.
    Returns None if the function can't be compiled.
    """
    global _explain_log
    _explain_log = explain_log

    def _is_supported(a: Any) -> bool:
        if isinstance(a, (int, float, list)):
            return True
        try:
            import numpy as np

            return isinstance(a, np.ndarray)
        except ImportError:
            return False

    if not all(_is_supported(a) for a in args):
        _log("bail: unsupported argument type(s)")
        return None

    arg_types = [type(a).__name__ for a in args]

    # ---- Disk cache check ----
    from pyjit._cache import load_compile_args, save_compile_args

    cached = load_compile_args(func.__code__, arg_types)
    if cached is not None:
        _log(f"cache hit for {func.__name__}({', '.join(arg_types)})")
        try:
            return compile_loop_ir(**cached)
        except Exception as e:
            _log(f"cache hit but recompile failed: {e}")
            # Fall through to normal compilation

    _log(f"cache miss — analyzing {func.__name__}({', '.join(arg_types)})")

    loop_args = _build_loop_compile_args(func, args)
    if loop_args is not None:
        _log("detected for-loop pattern")
        save_compile_args(func.__code__, arg_types, loop_args)
        try:
            return compile_loop_ir(**loop_args)
        except Exception as e:
            _log(f"for-loop compile_loop_ir failed: {e}")

    while_args = _build_while_loop_compile_args(func, args)
    if while_args is not None:
        _log("detected while-loop pattern")
        save_compile_args(func.__code__, arg_types, while_args)
        try:
            return compile_loop_ir(**while_args)
        except Exception as e:
            _log(f"while-loop compile_loop_ir failed: {e}")

    _log("bail: no compilable loop pattern found")
    return None


def _build_loop_compile_args(
    func: Callable[..., Any],
    args: tuple[Any, ...],
) -> dict[str, Any] | None:
    """Try to detect and compile a range-for loop pattern."""
    code = func.__code__
    instructions = list(dis.get_instructions(code))

    for_iter_idx = None
    for i, instr in enumerate(instructions):
        if instr.opname == "FOR_ITER":
            for_iter_idx = i
            break

    if for_iter_idx is None:
        _log("bail: no FOR_ITER found")
        return None

    range_spec = _detect_range_spec(instructions, for_iter_idx, code)
    if range_spec is None:
        _log("bail: range() pattern not detected")
        return None

    body_start = for_iter_idx + 1
    body_end = None
    # Find the LAST JUMP_BACKWARD before the OUTER END_FOR.
    # Must track nesting depth: inner FOR_ITER/END_FOR pairs are skipped.
    depth = 0
    for i in range(body_start, len(instructions)):
        op = instructions[i].opname
        if op == "FOR_ITER":
            depth += 1
        elif op == "END_FOR":
            if depth > 0:
                depth -= 1
            else:
                break  # this is our outer END_FOR
        elif op in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT") and depth == 0:
            body_end = i

    if body_end is None:
        _log("bail: could not find loop body end")
        return None

    if instructions[body_start].opname != "STORE_FAST":
        _log("bail: loop body doesn't start with STORE_FAST")
        return None
    iter_var_slot = instructions[body_start].arg
    if iter_var_slot is None:
        return None

    # Build numpy dtype map: param slot → element type ('f64' or 'i64')
    # Used by _extract_body_ops to choose typed vs boxed element ops.
    numpy_dtypes: dict[int, str] = {}
    try:
        import numpy as np

        for _slot in range(code.co_argcount):
            if _slot < len(args) and isinstance(args[_slot], np.ndarray):
                dt = args[_slot].dtype
                if dt == np.float64:
                    numpy_dtypes[_slot] = "f64"
                elif dt in (np.int64, np.int32, np.int16, np.int8):
                    numpy_dtypes[_slot] = "i64"
    except ImportError:
        pass

    body_ops = _extract_body_ops(
        instructions[body_start + 1 : body_end],
        code,
        iter_var_slot,
        numpy_dtypes,
        func_globals=func.__globals__,
    )
    if body_ops is None:
        _log("bail: body op extraction failed")
        return None

    # Prepend: store the loop counter into the iter var's local slot.
    # This is critical for nested loops where inner bodies reference
    # the outer counter via its local slot (not COUNTER_SENTINEL).
    body_ops.insert(0, ("StoreCounter", iter_var_slot, 0, 0, False, 0))

    return_local = _find_return_local(instructions)
    if return_local is None:
        _log("bail: return local not found")
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
            param_types.append(TYPE_I64)  # int, list, and numpy arrays all pass as i64 pointer

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
    # Propagate float types through temp slots:
    # 1. LoadElementF64 produces f64; Div always produces f64.
    # 2. Any arithmetic op whose first input is f64 also produces f64 (iterative fixpoint).
    _arith_ops = {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "TrueDiv",
        "FloorDiv",
        "Mod",
        "Pow",
        "Abs",
        "Min",
        "Max",
        "Neg",
    }
    _always_f64_ops = {"LoadElementF64", "Div", "Sqrt", "Sin", "Cos", "Exp", "Log", "Fabs", "ToF64"}
    for op_kind, dst, _sa, _sb, _imm, _iv in body_ops:
        if op_kind in _always_f64_ops and dst < num_locals:
            local_types[dst] = TYPE_F64

    # Iterative fixpoint: propagate f64 through arithmetic chains
    changed = True
    while changed:
        changed = False
        for op_kind, dst, src_a, _sb, _imm, _iv in body_ops:
            if op_kind not in _arith_ops:
                continue
            if dst >= num_locals:
                continue
            src_is_float = (
                src_a != COUNTER_SENTINEL and src_a < num_locals and local_types[src_a] == TYPE_F64
            )
            if src_is_float and local_types[dst] != TYPE_F64:
                local_types[dst] = TYPE_F64
                changed = True

    return_type_id = local_types[return_local] if return_local < len(local_types) else TYPE_I64

    # Resolve range spec to limit_param + start/step values
    # For constant limits, allocate a synthetic local slot
    stop_spec = range_spec["stop"]
    start_spec = range_spec["start"]
    step_spec = range_spec["step"]

    if stop_spec[0] == "param":
        limit_param = stop_spec[1]
    elif stop_spec[0] == "const":
        # Allocate a synthetic local for the constant limit
        limit_slot = num_locals
        num_locals += 1
        local_types.append(TYPE_I64)
        init_locals_int.append((limit_slot, stop_spec[1]))
        limit_param = limit_slot
    else:
        _log(f"bail: unsupported range stop spec: {stop_spec}")
        return None

    start_value = start_spec[1] if start_spec[0] == "const" else 0
    start_param = start_spec[1] if start_spec[0] == "param" else 2**64 - 1  # usize::MAX = not set
    step_value = step_spec[1] if step_spec[0] == "const" else 1

    return {
        "num_params": num_params,
        "limit_param": limit_param,
        "num_locals": num_locals,
        "return_local": return_local,
        "init_locals": init_locals_int,
        "init_float_locals": init_locals_float,
        "body_ops": body_ops,
        "local_types": local_types,
        "param_types": param_types,
        "return_type_id": return_type_id,
        "func_name": func.__name__,
        "start_value": start_value,
        "start_param": start_param,
        "step_value": step_value,
    }


def _build_while_loop_compile_args(
    func: Callable[..., Any],
    args: tuple[Any, ...],
) -> dict[str, Any] | None:
    """Detect `while i < n: body; i += 1` and compile as a range loop.

    Handles three patterns:
    - Python 3.14: single check at top, JUMP_BACKWARD → condition head
    - Python 3.13: LOAD_FAST_LOAD_FAST fused, double-check, JUMP_BACKWARD → body
    - Python 3.12: two separate LOAD_FAST, double-check, JUMP_BACKWARD → body
    """
    code = func.__code__
    instructions = list(dis.get_instructions(code))

    for i, instr in enumerate(instructions):
        if instr.opname != "COMPARE_OP":
            continue
        if i + 1 >= len(instructions):
            continue
        if instructions[i + 1].opname != "POP_JUMP_IF_FALSE":
            continue

        # Detect the load pattern before COMPARE_OP
        counter_slot: int | None = None
        limit_slot: int | None = None
        load_start_idx = i  # index of the first load instruction

        if i >= 1:
            prev = instructions[i - 1]
            if prev.opname in ("LOAD_FAST_BORROW_LOAD_FAST_BORROW", "LOAD_FAST_LOAD_FAST"):
                # Python 3.13/3.14: fused two-register load
                idx_a = (prev.arg >> 4) & 0xF if prev.arg is not None else 0
                idx_b = prev.arg & 0xF if prev.arg is not None else 0
                counter_slot = idx_a
                limit_slot = idx_b
                load_start_idx = i - 1
            elif (
                i >= 2
                and prev.opname in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK")
                and instructions[i - 2].opname
                in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK")
            ):
                # Python 3.12: two separate loads
                counter_slot = instructions[i - 2].arg if instructions[i - 2].arg is not None else 0
                limit_slot = prev.arg if prev.arg is not None else 0
                load_start_idx = i - 2

        if counter_slot is None or limit_slot is None:
            continue
        if limit_slot >= code.co_argcount:
            continue

        # Only support strict-less-than comparisons (< or >).
        # <=, >=, ==, != require off-by-one adjustments not yet supported.
        raw_arg = instr.arg if instr.arg is not None else 0
        cmp_type = raw_arg >> 5 if sys.version_info >= (3, 13) else raw_arg >> 4
        if cmp_type not in (0, 4):  # 0 = <, 4 = >
            continue

        loop_head_offset = instructions[load_start_idx].offset
        body_start = i + 2  # after POP_JUMP_IF_FALSE

        # Find body_end by locating the (last) JUMP_BACKWARD
        body_end: int | None = None
        for j in range(body_start, len(instructions)):
            jop = instructions[j].opname
            if jop not in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
                continue
            target = instructions[j].argval
            if target is None:
                continue

            if loop_head_offset <= target <= instr.offset:
                # Python 3.14: single-check — JUMP_BACKWARD targets the condition head
                body_end = j
                break

            if target >= instructions[body_start].offset:
                # Python 3.12/3.13: double-check — JUMP_BACKWARD targets body start.
                # Strip the trailing recheck (LOAD*, COMPARE_OP, POP_JUMP_IF_FALSE).
                stripped = _strip_while_recheck(instructions, j, counter_slot, limit_slot)
                if stripped is not None:
                    body_end = stripped
                    break

        if body_end is None:
            continue

        # Skip NOT_TAKEN/NOP at body start
        body_actual_start = body_start
        while body_actual_start < body_end and instructions[body_actual_start].opname in (
            "NOT_TAKEN",
            "NOP",
        ):
            body_actual_start += 1

        body_ops = _extract_body_ops(
            instructions[body_actual_start:body_end],
            code,
            counter_slot,
            func_globals=func.__globals__,
        )
        if body_ops is None:
            _log("bail: body op extraction failed (while loop)")
            continue

        return_local = _find_return_local(instructions)
        if return_local is None:
            _log("bail: return local not found (while loop)")
            continue

        init_locals_int, init_locals_float, float_slots = _find_init_locals(instructions, i, code)

        num_params = code.co_argcount
        param_types = [TYPE_I64] * num_params
        base_locals = code.co_nlocals
        max_slot = max((op[1] for op in body_ops), default=0)
        num_locals = max(base_locals, max_slot + 1)

        local_types_list: list[int] = [TYPE_I64] * num_locals
        for slot in float_slots:
            if slot < num_locals:
                local_types_list[slot] = TYPE_F64

        return_type_id = (
            local_types_list[return_local] if return_local < len(local_types_list) else TYPE_I64
        )

        return {
            "num_params": num_params,
            "limit_param": limit_slot,
            "num_locals": num_locals,
            "return_local": return_local,
            "init_locals": init_locals_int,
            "init_float_locals": init_locals_float,
            "body_ops": body_ops,
            "local_types": local_types_list,
            "param_types": param_types,
            "return_type_id": return_type_id,
            "func_name": func.__name__,
        }

    return None


def _strip_while_recheck(
    instructions: list[dis.Instruction],
    jump_backward_idx: int,
    counter_slot: int,
    limit_slot: int,
) -> int | None:
    """Return the index where a while-loop recheck starts, or None.

    In Python 3.12/3.13, while loops end with a duplicate condition check:
      [LOAD*, COMPARE_OP, POP_JUMP_IF_FALSE, JUMP_BACKWARD]
    We detect this pattern and return the index of LOAD* so the caller
    can exclude the recheck from body_ops.
    """
    j = jump_backward_idx
    if j < 2:
        return None
    if instructions[j - 1].opname != "POP_JUMP_IF_FALSE":
        return None
    if instructions[j - 2].opname != "COMPARE_OP":
        return None

    # Check for fused load at j-3
    if j >= 3 and instructions[j - 3].opname in (
        "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
        "LOAD_FAST_LOAD_FAST",
    ):
        rc = instructions[j - 3]
        a = (rc.arg >> 4) & 0xF if rc.arg is not None else 0
        b = rc.arg & 0xF if rc.arg is not None else 0
        if a == counter_slot and b == limit_slot:
            return j - 3

    # Check for two separate loads at j-4, j-3
    if (
        j >= 4
        and instructions[j - 3].opname
        in (
            "LOAD_FAST",
            "LOAD_FAST_BORROW",
            "LOAD_FAST_CHECK",
        )
        and instructions[j - 4].opname
        in (
            "LOAD_FAST",
            "LOAD_FAST_BORROW",
            "LOAD_FAST_CHECK",
        )
    ):
        arg_a = instructions[j - 4].arg
        arg_b = instructions[j - 3].arg
        a = arg_a if arg_a is not None else 0
        b = arg_b if arg_b is not None else 0
        if a == counter_slot and b == limit_slot:
            return j - 4

    return None


def _detect_range_spec(
    instructions: list[dis.Instruction],
    for_iter_idx: int,
    code: Any,
) -> dict[str, Any] | None:
    """Detect range() call pattern and return spec: {stop, start, step}.

    Each value is either ('param', idx) or ('const', value).
    Returns None if the pattern isn't recognized.
    """
    idx = for_iter_idx - 1
    if idx < 0 or instructions[idx].opname != "GET_ITER":
        return None
    idx -= 1
    if idx < 0 or instructions[idx].opname != "CALL":
        return None

    n_args = instructions[idx].arg
    if n_args is None or n_args < 1 or n_args > 3:
        return None

    def _resolve_load(instr: dis.Instruction) -> tuple[str, int] | None:
        if instr.opname in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"):
            return ("param", instr.arg if instr.arg is not None else 0)
        if instr.opname == "LOAD_SMALL_INT":
            return ("const", instr.arg if instr.arg is not None else 0)
        if instr.opname == "LOAD_CONST" and instr.arg is not None:
            consts = code.co_consts
            if instr.arg < len(consts) and isinstance(consts[instr.arg], int):
                return ("const", consts[instr.arg])
        return None

    # Collect the arguments (they are the N instructions before CALL, after LOAD_GLOBAL)
    arg_instrs: list[dis.Instruction] = []
    scan = idx - 1
    for _ in range(n_args):
        if scan < 0:
            return None
        arg_instrs.insert(0, instructions[scan])
        scan -= 1

    if n_args == 1:
        # range(stop)
        stop = _resolve_load(arg_instrs[0])
        if stop is None:
            return None
        return {"stop": stop, "start": ("const", 0), "step": ("const", 1)}

    if n_args == 2:
        # range(start, stop)
        start = _resolve_load(arg_instrs[0])
        stop = _resolve_load(arg_instrs[1])
        if start is None or stop is None:
            return None
        return {"stop": stop, "start": start, "step": ("const", 1)}

    if n_args == 3:
        # range(start, stop, step)
        start = _resolve_load(arg_instrs[0])
        stop = _resolve_load(arg_instrs[1])
        step = _resolve_load(arg_instrs[2])
        if start is None or stop is None or step is None:
            return None
        return {"stop": stop, "start": start, "step": step}

    return None


_CMP_MAP: dict[int, str] = {
    # (arg >> 4) & 0xf gives the comparison type
    # but the full arg also has flags. Common patterns:
    # 40 = Lt, 88 = Eq, 148 = Gt, etc. We match on (arg >> 4) & 0xf
}


def _resolve_val(val: Any, ops: list[tuple[str, int, int, int, bool, int]]) -> int:
    """Resolve a stack value to a local slot or COUNTER_SENTINEL."""
    if val == "counter":
        return COUNTER_SENTINEL
    if isinstance(val, int):
        return val
    return 0


def _make_binop(
    op_name: str,
    a_val: Any,
    b_val: Any,
    ops: list[tuple[str, int, int, int, bool, int]],
) -> int | None:
    """Emit a binary op and return the temp slot, or None on failure."""
    temp_slot = 100 + len(ops)
    src_a = _resolve_val(a_val, ops)

    if isinstance(b_val, tuple) and len(b_val) == 2 and b_val[0] == "imm":
        imm_val = b_val[1]
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

    return temp_slot


def _extract_body_ops(
    body_instrs: list[dis.Instruction],
    code: Any,
    iter_var_slot: int,
    numpy_dtypes: dict[int, str] | None = None,
    func_globals: dict[str, Any] | None = None,
) -> list[tuple[str, int, int, int, bool, int]] | None:
    """Extract loop body operations as (kind, dst, src_a, src_b, is_b_imm, imm).

    Uses a stack simulation. Handles conditionals by detecting the if/else
    pattern and generating Select ops.
    """
    stack: list[Any] = []
    ops: list[tuple[str, int, int, int, bool, int]] = []
    i = 0
    instrs = body_instrs

    while i < len(instrs):
        instr = instrs[i]
        name = instr.opname
        arg = instr.arg if instr.arg is not None else 0

        if name in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"):
            stack.append("counter" if arg == iter_var_slot else arg)

        elif name in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"):
            # Python 3.13 uses LOAD_FAST_LOAD_FAST; Python 3.14 uses LOAD_FAST_BORROW_LOAD_FAST_BORROW
            idx_a = (arg >> 4) & 0xF
            idx_b = arg & 0xF
            stack.append("counter" if idx_a == iter_var_slot else idx_a)
            stack.append("counter" if idx_b == iter_var_slot else idx_b)

        elif name == "LOAD_SMALL_INT":
            stack.append(("imm", arg))

        elif name == "LOAD_CONST":
            consts = code.co_consts
            if arg < len(consts) and isinstance(consts[arg], (int, float)):
                stack.append(("imm", consts[arg]))
            else:
                return None

        elif name == "BINARY_SUBSCR":
            # Python 3.12/3.13: separate opcode for subscript access (3.14 merged into BINARY_OP arg=26)
            if len(stack) < 2:
                return None
            b_val = stack.pop()
            a_val = stack.pop()
            container = _resolve_val(a_val, ops)
            index = _resolve_val(b_val, ops)
            temp_slot = 100 + len(ops)
            ndtype = (numpy_dtypes or {}).get(container)
            if ndtype == "f64":
                ops.append(("LoadElementF64", temp_slot, container, index, False, 0))
            elif ndtype == "i64":
                ops.append(("LoadElementI64", temp_slot, container, index, False, 0))
            else:
                ops.append(("LoadElement", temp_slot, container, index, False, 0))
            stack.append(temp_slot)

        elif name == "BINARY_OP":
            if len(stack) < 2:
                return None
            b_val = stack.pop()
            a_val = stack.pop()

            if arg == 26:  # NB_SUBSCR: container[index]
                # a_val = container (should be a param slot), b_val = index
                container = _resolve_val(a_val, ops)
                index = _resolve_val(b_val, ops)
                temp_slot = 100 + len(ops)
                # Use typed load for NumPy arrays, boxed load for Python lists
                ndtype = (numpy_dtypes or {}).get(container)
                if ndtype == "f64":
                    ops.append(("LoadElementF64", temp_slot, container, index, False, 0))
                elif ndtype == "i64":
                    ops.append(("LoadElementI64", temp_slot, container, index, False, 0))
                else:
                    ops.append(("LoadElement", temp_slot, container, index, False, 0))
                stack.append(temp_slot)
            else:
                op_name = _BINOP_MAP.get(arg)
                if op_name is None:
                    return None
                result = _make_binop(op_name, a_val, b_val, ops)
                if result is None:
                    return None
                stack.append(result)

        elif name == "COMPARE_OP":
            if len(stack) < 2:
                return None
            b_val = stack.pop()
            a_val = stack.pop()
            # Comparison type encoding changed in Python 3.13+ (arg >> 5) vs 3.12 (arg >> 4)
            cmp_op = arg >> 5 if sys.version_info >= (3, 13) else arg >> 4
            cmp_names = {0: "CmpLt", 1: "CmpLe", 2: "CmpEq", 3: "CmpNe", 4: "CmpGt", 5: "CmpGe"}
            cmp_name = cmp_names.get(cmp_op)
            if cmp_name is None:
                return None
            result = _make_binop(cmp_name, a_val, b_val, ops)
            if result is None:
                return None
            stack.append(("cmp", result))

        elif name in ("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"):
            if not stack or not (isinstance(stack[-1], tuple) and stack[-1][0] == "cmp"):
                return None
            cmp_slot = stack.pop()[1]

            # Find the true/false branch ranges
            jump_target = instrs[i].argval
            if name == "POP_JUMP_IF_FALSE":
                # Fall-through = true branch, jump target = false/merge
                true_start_idx = i + 1
            else:
                # POP_JUMP_IF_TRUE: fall-through = skip, jump target = true body
                true_start_idx = i + 1

            # Find the true branch end (JUMP_BACKWARD or JUMP_FORWARD from true branch)
            true_end_idx = None
            false_start_idx = None
            for j in range(true_start_idx, len(instrs)):
                if instrs[j].offset == jump_target:
                    # This is where the jump target points
                    true_end_idx = j
                    break
                if instrs[j].opname in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
                    true_end_idx = j
                    # Check if there's a false branch at the jump target
                    for k in range(j + 1, len(instrs)):
                        if instrs[k].offset == jump_target:
                            false_start_idx = k
                            break
                    break
                if instrs[j].opname == "JUMP_FORWARD":
                    true_end_idx = j
                    # False branch starts at the POP_JUMP target
                    for k in range(j + 1, len(instrs)):
                        if instrs[k].offset == jump_target:
                            false_start_idx = k
                            break
                    break

            if true_end_idx is None:
                return None

            # ---- break detection ----
            # If the true branch is just a JUMP_FORWARD to outside the body,
            # this is `if cond: break`. Emit BreakIf and skip to false branch.
            body_offsets = {x.offset for x in instrs}
            if (
                true_end_idx is not None
                and instrs[true_end_idx].opname == "JUMP_FORWARD"
                and true_start_idx == true_end_idx  # nothing between POP_JUMP and the JUMP_FORWARD
                and instrs[true_end_idx].argval is not None
                and instrs[true_end_idx].argval not in body_offsets
            ):
                # Simple `if cond: break` — emit BreakIf op
                # POP_JUMP_IF_FALSE: break when cond is TRUE  → invert=0
                # POP_JUMP_IF_TRUE:  break when cond is FALSE → invert=1
                invert = 1 if name == "POP_JUMP_IF_TRUE" else 0
                ops.append(("BreakIf", cmp_slot, invert, 0, False, 0))
                _log(f"  detected break: BreakIf(cmp={cmp_slot}, invert={invert})")
                # Continue from where the false branch starts (jump_target in instrs)
                new_i = len(instrs)
                for k, x in enumerate(instrs):
                    if x.offset == jump_target:
                        new_i = k
                        break
                i = new_i
                continue

            # Extract true branch instructions
            true_instrs = instrs[true_start_idx:true_end_idx]
            true_instrs = [x for x in true_instrs if x.opname not in ("NOT_TAKEN", "NOP")]

            # Extract true branch body ops recursively
            true_ops = _extract_body_ops(
                true_instrs, code, iter_var_slot, func_globals=func_globals
            )

            # If POP_JUMP_IF_TRUE, the "true" branch is actually the skip (false path)
            # and the jump target is the real body. Swap them.
            if name == "POP_JUMP_IF_TRUE":
                # true_instrs is the "skip" path (between fall-through and jump target)
                # The real body is at jump_target onwards until JUMP_BACKWARD
                real_body_start = None
                real_body_end = None
                for j in range(i + 1, len(instrs)):
                    if instrs[j].offset == jump_target:
                        real_body_start = j
                        break
                if real_body_start is not None:
                    for j in range(real_body_start, len(instrs)):
                        if instrs[j].opname in (
                            "JUMP_BACKWARD",
                            "JUMP_BACKWARD_NO_INTERRUPT",
                        ):
                            real_body_end = j
                            break

                if real_body_start is not None:
                    if real_body_end is None:
                        real_body_end = len(instrs)
                    real_instrs = instrs[real_body_start:real_body_end]
                    true_ops = _extract_body_ops(
                        real_instrs, code, iter_var_slot, func_globals=func_globals
                    )
                    i_skip = max(real_body_end, len(instrs))
                    false_start_idx = None  # no else for POP_JUMP_IF_TRUE
                else:
                    true_ops = None
                    i_skip = true_end_idx + 1
                    false_start_idx = None
            else:
                i_skip = true_end_idx + 1

            # Determine false branch
            false_ops: list[tuple[str, int, int, int, bool, int]] | None = None
            if false_start_idx is not None:
                # Find false branch end
                false_end_idx = None
                for j in range(false_start_idx, len(instrs)):
                    if instrs[j].opname in (
                        "JUMP_BACKWARD",
                        "JUMP_BACKWARD_NO_INTERRUPT",
                    ):
                        false_end_idx = j
                        break
                if false_end_idx is None:
                    false_end_idx = len(instrs)  # extends to end of body
                false_instrs = instrs[false_start_idx:false_end_idx]
                false_ops = _extract_body_ops(
                    false_instrs, code, iter_var_slot, func_globals=func_globals
                )
                i_skip = max(false_end_idx, len(instrs))

            # Emit: CondStart + true_ops + [CondElse + false_ops] + CondEnd
            invert = 0  # we've already handled inversion above
            ops.append(("CondStart", cmp_slot, invert, 0, False, 0))
            if true_ops:
                ops.extend(true_ops)
            if false_ops:
                ops.append(("CondElse", 0, 0, 0, False, 0))
                ops.extend(false_ops)
            ops.append(("CondEnd", 0, 0, 0, False, 0))

            i = i_skip
            continue

        elif name in (
            "JUMP_BACKWARD",
            "JUMP_BACKWARD_NO_INTERRUPT",
            "JUMP_FORWARD",
        ):
            pass  # skip jumps handled by conditional extraction above

        elif name in ("STORE_FAST", "STORE_FAST_MAYBE_NULL"):
            if not stack:
                return None
            val = stack.pop()
            if isinstance(val, int):
                if ops and ops[-1][1] >= 100:
                    # Rename the last temp-slot op to write directly to arg
                    last = ops[-1]
                    ops[-1] = (last[0], arg, last[2], last[3], last[4], last[5])
                elif val != arg:
                    # Local-to-local copy (e.g. `mx = v`): emit "Add src, 0"
                    ops.append(("Add", arg, val, 0, True, 0))
            elif isinstance(val, tuple) and len(val) == 2 and val[0] == "imm":
                # Store a constant into a local: emit a LoadConst op
                imm_val = val[1]
                if isinstance(imm_val, float):
                    import struct

                    imm_bits = struct.unpack("<q", struct.pack("<d", imm_val))[0]
                    ops.append(("LoadConst", arg, 0, 0, True, imm_bits))
                else:
                    ops.append(("LoadConst", arg, 0, 0, True, int(imm_val)))
            elif val == "counter":
                # Copy loop counter into a named local: v = i → v = counter + 0
                ops.append(("Add", arg, COUNTER_SENTINEL, 0, True, 0))

        elif name == "UNARY_NEGATIVE":
            if not stack:
                return None
            val = stack.pop()
            temp_slot = 100 + len(ops)
            src = _resolve_val(val, ops)
            ops.append(("Neg", temp_slot, src, 0, False, 0))
            stack.append(temp_slot)

        elif name == "UNARY_INVERT":
            if not stack:
                return None
            val = stack.pop()
            temp_slot = 100 + len(ops)
            src = _resolve_val(val, ops)
            ops.append(("BitNot", temp_slot, src, 0, False, 0))
            stack.append(temp_slot)

        elif name == "UNARY_NOT":
            if not stack:
                return None
            val = stack.pop()
            temp_slot = 100 + len(ops)
            src = _resolve_val(val, ops)
            ops.append(("Not", temp_slot, src, 0, False, 0))
            stack.append(temp_slot)

        elif name == "COPY":
            # COPY arg=N: copy the Nth item from TOS (1-indexed)
            depth = arg if arg else 1
            if len(stack) >= depth:
                stack.append(stack[-depth])

        elif name == "SWAP":
            # SWAP arg=N: swap TOS with the Nth item from TOS (1-indexed)
            depth = arg if arg else 2
            if len(stack) >= depth:
                stack[-1], stack[-depth] = stack[-depth], stack[-1]

        elif name == "POP_TOP":
            if stack:
                stack.pop()

        elif name in ("RESUME", "NOT_TAKEN", "NOP"):
            pass

        elif name == "CALL":
            n_call_args = arg
            # Check if this is a call to a known builtin
            builtin_handled = False
            _math_unary = {
                "math.sqrt": "Sqrt",
                "math.sin": "Sin",
                "math.cos": "Cos",
                "math.exp": "Exp",
                "math.log": "Log",
                "math.fabs": "Fabs",
            }
            if n_call_args in (1, 2) and len(stack) >= n_call_args + 1:
                # Stack: [..., NULL?, callable, arg1, ...argN]
                # The callable is at -(n_call_args + 1), but NULL placeholders may shift it.
                callable_pos = -(n_call_args + 1)
                callable_val = stack[callable_pos] if abs(callable_pos) <= len(stack) else None
                # If callable slot is a None (NULL), look one deeper for the real callable.
                if callable_val is None:
                    deeper = callable_pos - 1
                    if abs(deeper) <= len(stack):
                        callable_val = stack[deeper]
                        callable_pos = deeper

                if isinstance(callable_val, tuple) and callable_val[0] == "builtin":
                    builtin_name = callable_val[1]
                    if n_call_args == 1 and builtin_name == "abs":
                        arg1 = stack.pop()  # the argument
                        # Pop everything from TOS down to and including the callable.
                        while stack and stack[-1] != ("builtin", "abs"):
                            stack.pop()
                        if stack:
                            stack.pop()  # pop callable
                        if stack and stack[-1] is None:
                            stack.pop()  # pop NULL below callable
                        temp_slot = 100 + len(ops)
                        src = _resolve_val(arg1, ops)
                        ops.append(("Abs", temp_slot, src, 0, False, 0))
                        stack.append(temp_slot)
                        builtin_handled = True
                    elif n_call_args == 1 and builtin_name in _math_unary:
                        arg1 = stack.pop()
                        while stack and stack[-1] != ("builtin", builtin_name):
                            stack.pop()
                        if stack:
                            stack.pop()  # pop callable
                        if stack and stack[-1] is None:
                            stack.pop()  # pop NULL below callable
                        temp_slot = 100 + len(ops)
                        src = _resolve_val(arg1, ops)
                        ops.append((_math_unary[builtin_name], temp_slot, src, 0, False, 0))
                        stack.append(temp_slot)
                        builtin_handled = True
                    elif n_call_args == 2 and builtin_name in ("min", "max"):
                        arg2 = stack.pop()
                        arg1 = stack.pop()
                        while stack and stack[-1] != ("builtin", builtin_name):
                            stack.pop()
                        if stack:
                            stack.pop()  # pop callable
                        if stack and stack[-1] is None:
                            stack.pop()  # pop NULL below callable
                        op_name = "Min" if builtin_name == "min" else "Max"
                        result = _make_binop(op_name, arg1, arg2, ops)
                        if result is None:
                            return None
                        stack.append(result)
                        builtin_handled = True

            if not builtin_handled:
                # Try user-function inlining (for global functions with __code__)
                inlined = False
                if n_call_args >= 1 and func_globals is not None:
                    inline_callable_pos = -(n_call_args + 1)
                    inline_callable = (
                        stack[inline_callable_pos]
                        if abs(inline_callable_pos) <= len(stack)
                        else None
                    )
                    if inline_callable is None and abs(inline_callable_pos) < len(stack):
                        inline_callable = stack[inline_callable_pos - 1]

                    if isinstance(inline_callable, tuple) and inline_callable[0] == "global":
                        gname_idx = inline_callable[1] >> 1
                        gnames = code.co_names
                        if gname_idx < len(gnames):
                            fn_obj = func_globals.get(gnames[gname_idx])
                            if (
                                fn_obj is not None
                                and callable(fn_obj)
                                and hasattr(fn_obj, "__code__")
                                and not isinstance(fn_obj, type)
                            ):
                                # Pop args, then callable and any NULL placeholder
                                inline_args: list[Any] = []
                                for _ in range(n_call_args):
                                    if stack:
                                        inline_args.insert(0, stack.pop())
                                while stack and stack[-1] in (None, inline_callable):
                                    stack.pop()
                                inline_base = _INLINE_SLOT_BASE + (len(ops) // 10 + 1) * 10
                                inline_result = _try_inline_call(
                                    fn_obj.__code__, inline_args, inline_base, ops, code
                                )
                                if inline_result is not None:
                                    extra_ops, ret_val = inline_result
                                    ops.extend(extra_ops)
                                    stack.append(ret_val)
                                    inlined = True
                                    _log(f"  inlined {gnames[gname_idx]}")

                if not inlined:
                    # Fall back: handle float()/int() type conversions.
                    if n_call_args == 1 and stack:
                        call_arg = stack.pop()  # pop arg
                        popped_callable = stack.pop() if stack else None  # pop callable
                        if stack and stack[-1] is None:
                            stack.pop()  # pop NULL placeholder if present
                        callable_name = (
                            popped_callable[1]
                            if isinstance(popped_callable, tuple) and len(popped_callable) == 2
                            else None
                        )
                        if callable_name == "float":
                            temp_slot = 100 + len(ops)
                            src = _resolve_val(call_arg, ops)
                            ops.append(("ToF64", temp_slot, src, 0, False, 0))
                            stack.append(temp_slot)
                        elif callable_name in ("int", "range"):
                            stack.append(call_arg)
                        else:
                            return None  # unknown callable — bail out
                    else:
                        return None

        elif name == "LOAD_GLOBAL":
            # Resolve the global name to check for known builtins
            name_idx = arg >> 1  # CPython 3.12+: lower bit is NULL flag
            names = code.co_names
            if name_idx < len(names):
                gname = names[name_idx]
                if gname in ("abs", "min", "max", "int", "float", "range"):
                    # Push NULL (for CALL protocol) + builtin marker
                    if arg & 1:  # NULL flag set
                        stack.append(None)  # NULL placeholder
                    stack.append(("builtin", gname))
                elif gname == "math":
                    # Push module marker — LOAD_ATTR will replace it with the specific func
                    stack.append(("module", "math"))
                else:
                    stack.append(("global", arg))
            else:
                stack.append(("global", arg))

        elif name == "LOAD_DEREF":
            # Cell or free variable access (e.g. math imported in an outer scope).
            # Use the argval (resolved name) rather than the raw index.
            var_name = instr.argval if instr.argval is not None else ""
            if var_name == "math":
                stack.append(("module", "math"))
            else:
                return None  # can't handle arbitrary free-variable access

        elif name == "LOAD_ATTR":
            # CPython 3.12+: arg encodes (namei << 1 | method_flag)
            attr_idx = arg >> 1
            names = code.co_names
            if not stack:
                return None
            obj = stack[-1]
            if (
                isinstance(obj, tuple)
                and obj[0] == "module"
                and obj[1] == "math"
                and attr_idx < len(names)
            ):
                attr_name = names[attr_idx]
                if attr_name in ("sqrt", "sin", "cos", "exp", "log", "fabs"):
                    stack.pop()  # pop the module
                    if arg & 1:  # method flag: push NULL then callable
                        stack.append(None)
                    stack.append(("builtin", f"math.{attr_name}"))
                else:
                    return None  # unsupported math attribute
            else:
                return None  # can't handle arbitrary attribute access

        elif name == "PUSH_NULL":
            stack.append(None)  # NULL for CALL protocol

        elif name == "FOR_ITER":
            # Nested loop! Extract inner loop structure.
            # The stack should have: [("global", range), limit_slot, call_result, iterator]
            # from preceding: LOAD_GLOBAL range, LOAD_FAST x, CALL 1, GET_ITER
            # Pop the iterator/call/global markers from the stack
            stack.clear()  # these are intermediate values, not needed

            # Next instruction should be STORE_FAST for inner iter variable
            if i + 1 >= len(instrs):
                return None
            inner_store = instrs[i + 1]
            if inner_store.opname != "STORE_FAST" or inner_store.arg is None:
                return None
            inner_iter_slot = inner_store.arg

            # Find the range param: look backwards for LOAD_FAST before CALL/GET_ITER
            inner_limit_slot = None
            for back in range(i - 1, -1, -1):
                bi = instrs[back]
                if bi.opname in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"):
                    inner_limit_slot = bi.arg
                    break
                if bi.opname in ("LOAD_GLOBAL", "CALL", "GET_ITER"):
                    continue
                break
            if inner_limit_slot is None:
                return None

            # Find inner body: from STORE_FAST+1 to the LAST JUMP_BACKWARD
            # before the matching END_FOR (depth-aware)
            inner_body_start = i + 2
            inner_body_end = None
            inner_depth = 0
            for j in range(inner_body_start, len(instrs)):
                jop = instrs[j].opname
                if jop == "FOR_ITER":
                    inner_depth += 1
                elif jop == "END_FOR":
                    if inner_depth > 0:
                        inner_depth -= 1
                    else:
                        break  # matching END_FOR for this FOR_ITER
                elif jop in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT") and inner_depth == 0:
                    inner_body_end = j
            if inner_body_end is None:
                return None

            # Find END_FOR + POP_ITER after inner JUMP_BACKWARD
            skip_to = inner_body_end + 1
            while skip_to < len(instrs) and instrs[skip_to].opname in (
                "END_FOR",
                "POP_ITER",
            ):
                skip_to += 1

            # Recursively extract inner body ops
            inner_body_instrs = instrs[inner_body_start:inner_body_end]
            inner_ops = _extract_body_ops(
                inner_body_instrs, code, inner_iter_slot, func_globals=func_globals
            )
            if inner_ops is None:
                return None

            # Emit: LoopStart + inner ops + LoopEnd
            ops.append(("LoopStart", inner_limit_slot, inner_iter_slot, 0, False, 0))
            ops.extend(inner_ops)
            ops.append(("LoopEnd", 0, 0, 0, False, 0))

            i = skip_to
            continue

        elif name == "STORE_SUBSCR":
            # data[i] = value  →  stack before: [value, container, index]
            # TOS=index, TOS1=container, TOS2=value
            if len(stack) < 3:
                return None
            key = stack.pop()  # index
            container = stack.pop()  # container
            val = stack.pop()  # value to store
            container_slot = _resolve_val(container, ops)
            index_ref = _resolve_val(key, ops)
            value_slot = _resolve_val(val, ops)
            # Use typed store for NumPy arrays, boxed store for Python lists
            ndtype = (numpy_dtypes or {}).get(container_slot)
            if ndtype == "f64":
                ops.append(("StoreElementF64", container_slot, index_ref, value_slot, False, 0))
            elif ndtype == "i64":
                ops.append(("StoreElementI64", container_slot, index_ref, value_slot, False, 0))
            else:
                ops.append(("StoreElement", container_slot, index_ref, value_slot, False, 0))
            # no result pushed to stack

        elif name in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "JUMP_FORWARD"):
            pass  # skip jumps that are part of control flow

        elif name in ("GET_ITER", "CALL", "END_FOR", "POP_ITER"):
            pass  # skip loop setup/teardown instructions

        else:
            return None

        i += 1

    return ops if ops else None


def _try_inline_call(
    inner_code: Any,
    arg_vals: list[Any],
    base_slot: int,
    outer_ops: list[tuple[str, int, int, int, bool, int]],
    outer_code: Any,
) -> tuple[list[tuple[str, int, int, int, bool, int]], Any] | None:
    """Try to inline a simple pure function into the loop body.

    Args:
        inner_code: __code__ of the function to inline.
        arg_vals:   actual argument values (slot ints / "counter" / ("imm", v)).
        base_slot:  base slot number for remapping inner locals (avoids collision).
        outer_ops:  accumulated outer ops (used for temp slot counting).
        outer_code: outer code object (unused, kept for future use).

    Returns (new_ops, return_val) where return_val is a slot int or tuple,
    or None if inlining is not safe/possible.
    """
    import struct as _struct

    num_params = inner_code.co_argcount
    if num_params != len(arg_vals):
        return None

    inner_instrs = list(dis.get_instructions(inner_code))

    # Reject functions with loops, calls, or other complex ops
    for instr in inner_instrs:
        if instr.opname in (
            "FOR_ITER",
            "GET_ITER",
            "CALL",
            "IMPORT_NAME",
            "IMPORT_FROM",
            "RAISE_VARARGS",
            "YIELD_VALUE",
            "LOAD_GLOBAL",
            "LOAD_ATTR",
        ):
            return None

    stack: list[Any] = []
    ops: list[tuple[str, int, int, int, bool, int]] = []
    # Temp slots start at base_slot + 100 so they never collide with param remaps
    temp_base = base_slot + 100

    for instr in inner_instrs:
        name = instr.opname
        arg = instr.arg if instr.arg is not None else 0

        if name in ("RESUME", "NOT_TAKEN", "NOP"):
            continue

        elif name in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"):
            if arg < num_params:
                stack.append(arg_vals[arg])
            else:
                stack.append(base_slot + arg)

        elif name in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"):
            idx_a = (arg >> 4) & 0xF
            idx_b = arg & 0xF
            stack.append(arg_vals[idx_a] if idx_a < num_params else base_slot + idx_a)
            stack.append(arg_vals[idx_b] if idx_b < num_params else base_slot + idx_b)

        elif name == "LOAD_SMALL_INT":
            stack.append(("imm", arg))

        elif name == "LOAD_CONST":
            consts = inner_code.co_consts
            if arg < len(consts) and isinstance(consts[arg], (int, float)):
                stack.append(("imm", consts[arg]))
            else:
                return None

        elif name == "BINARY_OP":
            if len(stack) < 2:
                return None
            b_val = stack.pop()
            a_val = stack.pop()
            op_name = _BINOP_MAP.get(arg)
            if op_name is None:
                return None
            temp_slot = temp_base + len(ops)
            src_a = _resolve_val(a_val, ops)
            if isinstance(b_val, tuple) and len(b_val) == 2 and b_val[0] == "imm":
                imm_val = b_val[1]
                if isinstance(imm_val, float):
                    imm_bits = _struct.unpack("<q", _struct.pack("<d", imm_val))[0]
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

        elif name == "UNARY_NEGATIVE":
            if not stack:
                return None
            val = stack.pop()
            temp_slot = temp_base + len(ops)
            src = _resolve_val(val, ops)
            ops.append(("Neg", temp_slot, src, 0, False, 0))
            stack.append(temp_slot)

        elif name in ("STORE_FAST", "STORE_FAST_MAYBE_NULL"):
            if not stack:
                return None
            val = stack.pop()
            rslot = base_slot + arg  # always remap inner locals
            if isinstance(val, int) and ops:
                last = ops[-1]
                ops[-1] = (last[0], rslot, last[2], last[3], last[4], last[5])
            elif isinstance(val, tuple) and len(val) == 2 and val[0] == "imm":
                imm_val = val[1]
                if isinstance(imm_val, float):
                    imm_bits = _struct.unpack("<q", _struct.pack("<d", imm_val))[0]
                    ops.append(("LoadConst", rslot, 0, 0, True, imm_bits))
                else:
                    ops.append(("LoadConst", rslot, 0, 0, True, int(imm_val)))

        elif name == "RETURN_VALUE":
            if not stack:
                return None
            return ops, stack[-1]

        else:
            return None  # unsupported — give up

    return None  # no RETURN_VALUE encountered


def _extract_conditional_branches(
    instrs: list[dis.Instruction],
    jump_idx: int,
    cmp_slot: int,
    code: Any,
    iter_var_slot: int,
    existing_ops: list[tuple[str, int, int, int, bool, int]],
    invert: bool = False,
) -> tuple[list[tuple[str, int, int, int, bool, int]], int, dict[int, int]] | None:
    """Extract if/else branches and generate Select ops.

    Returns (new_ops, next_instruction_index, modified_locals) or None.
    """
    jump_instr = instrs[jump_idx]
    jump_target = jump_instr.argval  # byte offset of else branch

    # Find where the true branch ends (JUMP_BACKWARD or JUMP_FORWARD)
    true_start = jump_idx + 1
    true_end = None
    for j in range(true_start, len(instrs)):
        if instrs[j].opname in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "JUMP_FORWARD"):
            true_end = j
            break
        if instrs[j].offset == jump_target:
            # No true branch — this is an if-without-else
            true_end = j
            break

    if true_end is None:
        return None

    # Find the false branch
    false_start = None
    for j in range(true_end, len(instrs)):
        if instrs[j].offset == jump_target:
            false_start = j
            break

    # Extract true branch ops
    true_ops: list[tuple[str, int, int, int, bool, int]] = []
    true_stores: dict[int, int] = {}  # local_slot -> temp_slot_with_value

    true_branch = instrs[true_start:true_end]
    # Skip NOT_TAKEN at the start
    true_branch = [i for i in true_branch if i.opname not in ("NOT_TAKEN", "NOP")]

    true_result = _extract_branch_ops(true_branch, code, iter_var_slot, existing_ops, true_ops)
    if true_result is None:
        return None
    true_stores = true_result

    # Extract false branch ops (if exists)
    false_ops: list[tuple[str, int, int, int, bool, int]] = []
    false_stores: dict[int, int] = {}
    after_false = true_end + 1  # default: skip past jump

    if false_start is not None:
        # Find end of false branch
        false_end = None
        for j in range(false_start, len(instrs)):
            if instrs[j].opname in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
                false_end = j
                break
        if false_end is None:
            false_end = len(instrs)

        false_branch = instrs[false_start:false_end]
        # Pass existing_ops + true_ops so false branch gets unique temp slots
        combined_existing = list(existing_ops) + true_ops
        false_result = _extract_branch_ops(
            false_branch, code, iter_var_slot, combined_existing, false_ops
        )
        if false_result is None:
            return None
        false_stores = false_result
        after_false = false_end + 1
    else:
        after_false = true_end + 1

    # Generate Select ops for each modified local
    all_ops: list[tuple[str, int, int, int, bool, int]] = []
    all_ops.extend(true_ops)
    all_ops.extend(false_ops)

    all_modified = set(true_stores.keys()) | set(false_stores.keys())
    for local_slot in sorted(all_modified):
        true_val = true_stores.get(local_slot, local_slot)  # unchanged = original value
        false_val = false_stores.get(local_slot, local_slot)  # unchanged = original value
        # Select: dst = cmp_slot ? true_val : false_val
        # Encode as: ("Select", dst, cmp_slot, true_val, false=false_val_in_imm, 0)
        # We need to encode the false_val. Use a different encoding:
        # ("Select", dst, true_val, false_val, False, cmp_slot)
        if invert:
            all_ops.append(("Select", local_slot, false_val, true_val, False, cmp_slot))
        else:
            all_ops.append(("Select", local_slot, true_val, false_val, False, cmp_slot))

    return all_ops, after_false, {}


def _extract_branch_ops(
    branch_instrs: list[dis.Instruction],
    code: Any,
    iter_var_slot: int,
    existing_ops: list[tuple[str, int, int, int, bool, int]],
    branch_ops: list[tuple[str, int, int, int, bool, int]],
) -> dict[int, int] | None:
    """Extract ops from a single branch. Returns {local_slot: temp_slot} for STORE_FASTs."""
    stack: list[Any] = []
    stores: dict[int, int] = {}
    # Start temp slots after existing + branch ops
    base_temp = 100 + len(existing_ops) + len(branch_ops)

    for instr in branch_instrs:
        name = instr.opname
        arg = instr.arg if instr.arg is not None else 0

        if name in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"):
            stack.append("counter" if arg == iter_var_slot else arg)

        elif name in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"):
            idx_a = (arg >> 4) & 0xF
            idx_b = arg & 0xF
            stack.append("counter" if idx_a == iter_var_slot else idx_a)
            stack.append("counter" if idx_b == iter_var_slot else idx_b)

        elif name == "LOAD_SMALL_INT":
            stack.append(("imm", arg))

        elif name == "BINARY_OP":
            if len(stack) < 2:
                return None
            b_val = stack.pop()
            a_val = stack.pop()
            op_name = _BINOP_MAP.get(arg)
            if op_name is None:
                return None
            temp_slot = base_temp + len(branch_ops)
            src_a = _resolve_val(a_val, branch_ops)
            if isinstance(b_val, tuple) and len(b_val) == 2 and b_val[0] == "imm":
                imm_val = b_val[1]
                if isinstance(imm_val, float):
                    import struct

                    imm_bits = struct.unpack("<q", struct.pack("<d", imm_val))[0]
                    branch_ops.append((op_name, temp_slot, src_a, 0, True, imm_bits))
                else:
                    branch_ops.append((op_name, temp_slot, src_a, 0, True, int(imm_val)))
            elif b_val == "counter":
                branch_ops.append((op_name, temp_slot, src_a, COUNTER_SENTINEL, False, 0))
            elif isinstance(b_val, int):
                branch_ops.append((op_name, temp_slot, src_a, b_val, False, 0))
            else:
                return None
            stack.append(temp_slot)

        elif name in ("STORE_FAST", "STORE_FAST_MAYBE_NULL"):
            if not stack:
                return None
            val = stack.pop()
            if isinstance(val, int):
                # Rewrite last op dst if it was a temp
                if branch_ops and branch_ops[-1][1] >= 100:
                    last = branch_ops[-1]
                    temp = last[1]
                    branch_ops[-1] = (last[0], temp, last[2], last[3], last[4], last[5])
                stores[arg] = val if val < 100 else branch_ops[-1][1]
            elif val == "counter":
                pass

        elif name in ("NOT_TAKEN", "NOP", "RESUME"):
            pass

        else:
            return None

    return stores


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
