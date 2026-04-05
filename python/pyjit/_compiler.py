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

    while_result = _try_compile_while_loop(func, args)
    if while_result is not None:
        return while_result

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

    # Prepend: store the loop counter into the iter var's local slot.
    # This is critical for nested loops where inner bodies reference
    # the outer counter via its local slot (not COUNTER_SENTINEL).
    body_ops.insert(0, ("StoreCounter", iter_var_slot, 0, 0, False, 0))

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


def _try_compile_while_loop(
    func: Callable[..., Any],
    args: tuple[Any, ...],
) -> CompiledFunction | None:
    """Detect `while i < n: body; i += 1` and compile as a range loop."""
    code = func.__code__
    instructions = list(dis.get_instructions(code))

    # Look for pattern: COMPARE_OP + POP_JUMP_IF_FALSE + ... + JUMP_BACKWARD
    # where the JUMP_BACKWARD target is the COMPARE_OP
    for i, instr in enumerate(instructions):
        if instr.opname != "COMPARE_OP":
            continue
        if i + 1 >= len(instructions):
            continue
        next_instr = instructions[i + 1]
        if next_instr.opname != "POP_JUMP_IF_FALSE":
            continue

        # Found a while loop candidate. Check for JUMP_BACKWARD pointing back
        # to the compare or its preceding load instruction
        loop_head_offset = instructions[i - 1].offset if i > 0 else instr.offset
        body_start = i + 2  # after POP_JUMP_IF_FALSE
        body_end = None
        for j in range(body_start, len(instructions)):
            if instructions[j].opname in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
                target = instructions[j].argval
                if target is not None and loop_head_offset <= target <= instr.offset:
                    body_end = j
                    break

        if body_end is None:
            continue

        # Detect: the comparison is `local_i < local_n` (a param)
        # Look at what's loaded before COMPARE_OP
        if i < 1:
            continue
        prev = instructions[i - 1]
        if prev.opname == "LOAD_FAST_BORROW_LOAD_FAST_BORROW":
            idx_a = (prev.arg >> 4) & 0xF if prev.arg is not None else 0
            idx_b = prev.arg & 0xF if prev.arg is not None else 0
            counter_slot = idx_a
            limit_slot = idx_b
        else:
            continue

        # limit_slot should be a param
        if limit_slot >= code.co_argcount:
            continue

        # Skip NOT_TAKEN after POP_JUMP_IF_FALSE
        body_actual_start = body_start
        while body_actual_start < body_end and instructions[body_actual_start].opname in (
            "NOT_TAKEN",
            "NOP",
        ):
            body_actual_start += 1

        # Extract body ops (reuse for-loop body extractor)
        body_ops = _extract_body_ops(
            instructions[body_actual_start:body_end],
            code,
            counter_slot,  # treat the while counter as the "iter var"
        )
        if body_ops is None:
            continue

        # Find return local
        return_local = _find_return_local(instructions)
        if return_local is None:
            continue

        # Find init locals
        init_locals_int, init_locals_float, float_slots = _find_init_locals(instructions, i, code)

        # The while loop counter is managed by body ops (i += 1 is in the body).
        # We can use the same compile_loop infrastructure with the counter as a regular local.
        # BUT: we need to tell compile_loop that the "loop limit" param controls the loop.
        # The difference from for-range: the counter lives in a local, not the hardware counter.
        # We can reuse compile_loop by having the body ops include the i += 1, and
        # the counter_slot IS the loop variable tracked by compile_loop's counter.

        # Actually, the simplest: detect that body contains `i += 1` and strip it,
        # then compile as range(limit_slot) loop.
        # Check if body_ops has an Add to counter_slot with imm 1
        # (This would be generated as ("Add", counter_slot, counter_slot, 0, True, 1)
        # but since counter_slot maps to COUNTER_SENTINEL, it'd be different)

        # For now, use compile_loop directly — the while counter IS the range counter
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

        try:
            return compile_loop_ir(
                num_params=num_params,
                limit_param=limit_slot,
                num_locals=num_locals,
                return_local=return_local,
                init_locals=init_locals_int,
                init_float_locals=init_locals_float,
                body_ops=body_ops,
                local_types=local_types_list,
                param_types=param_types,
                return_type_id=return_type_id,
                func_name=func.__name__,
            )
        except Exception:
            continue

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

        elif name == "LOAD_FAST_BORROW_LOAD_FAST_BORROW":
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

        elif name == "BINARY_OP":
            if len(stack) < 2:
                return None
            b_val = stack.pop()
            a_val = stack.pop()
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
            # Comparison type is in upper bits
            cmp_op = arg >> 5
            cmp_names = {0: "CmpLt", 1: "CmpLe", 2: "CmpEq", 3: "CmpNe", 4: "CmpGt", 5: "CmpGe"}
            cmp_name = cmp_names.get(cmp_op)
            if cmp_name is None:
                return None
            result = _make_binop(cmp_name, a_val, b_val, ops)
            if result is None:
                return None
            stack.append(("cmp", result))

        elif name in ("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"):
            # Start of conditional — extract true and false branches
            if not stack or not (isinstance(stack[-1], tuple) and stack[-1][0] == "cmp"):
                return None
            cmp_slot = stack.pop()[1]

            # Find the branch targets by scanning ahead
            invert = name == "POP_JUMP_IF_TRUE"
            cond_result = _extract_conditional_branches(
                instrs, i, cmp_slot, code, iter_var_slot, ops, invert
            )
            if cond_result is None:
                return None

            new_ops, skip_to, modified_locals = cond_result
            ops.extend(new_ops)
            # Update the loop counter for the outer loop
            i = skip_to
            continue

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
            n_call_args = arg
            if n_call_args == 1 and stack:
                call_arg = stack.pop()
                if stack:
                    stack.pop()
                stack.append(call_arg)
            else:
                return None

        elif name == "LOAD_GLOBAL":
            stack.append(("global", arg))

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
            inner_ops = _extract_body_ops(inner_body_instrs, code, inner_iter_slot)
            if inner_ops is None:
                return None

            # Emit: LoopStart + inner ops + LoopEnd
            ops.append(("LoopStart", inner_limit_slot, inner_iter_slot, 0, False, 0))
            ops.extend(inner_ops)
            ops.append(("LoopEnd", 0, 0, 0, False, 0))

            i = skip_to
            continue

        elif name in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "JUMP_FORWARD"):
            pass  # skip jumps that are part of control flow

        elif name in ("GET_ITER", "CALL", "END_FOR", "POP_ITER"):
            pass  # skip loop setup/teardown instructions

        else:
            return None

        i += 1

    return ops if ops else None


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

        elif name == "LOAD_FAST_BORROW_LOAD_FAST_BORROW":
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
