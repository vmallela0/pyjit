//! Cranelift JIT compilation — translates IR to native machine code.

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::ir::ops::{IROp, IRProgram, ValueId};
use crate::ir::types::IRType;

/// Called from JIT-compiled code to box a native i64 into a Python int (PyObject*).
/// Returns the PyObject* as i64. Handles the full integer range.
unsafe extern "C" fn pyjit_box_int(val: i64) -> i64 {
    pyo3::ffi::PyLong_FromLong(val) as i64
}

// Math helpers — wrap libm/std via Rust's f64 methods.
extern "C" fn pyjit_sin(x: f64) -> f64 { x.sin() }
extern "C" fn pyjit_cos(x: f64) -> f64 { x.cos() }
extern "C" fn pyjit_exp(x: f64) -> f64 { x.exp() }
extern "C" fn pyjit_log(x: f64) -> f64 { x.ln() }

/// Refs to runtime helper functions, passed into emit_body_ops.
struct RuntimeRefs {
    box_int: cranelift_codegen::ir::FuncRef,
    sin: cranelift_codegen::ir::FuncRef,
    cos: cranelift_codegen::ir::FuncRef,
    exp: cranelift_codegen::ir::FuncRef,
    log: cranelift_codegen::ir::FuncRef,
}

/// A compiled native function ready for execution.
pub struct CompiledCode {
    /// The JIT module that owns the compiled code memory.
    /// Must be kept alive as long as `fn_ptr` is in use.
    pub _module: JITModule,
    /// Pointer to the compiled native code.
    pub fn_ptr: *const u8,
    /// Number of function parameters.
    pub num_params: usize,
    /// Types of the parameters.
    pub param_types: Vec<IRType>,
    /// Type of the return value.
    pub return_type: IRType,
}

// Safety: The JITModule and fn_ptr are safe to send between threads.
// The compiled code is immutable once finalized.
unsafe impl Send for CompiledCode {}

/// Map an IRType to a Cranelift type.
fn ir_type_to_cl(ty: &IRType) -> types::Type {
    match ty {
        IRType::Int64 => types::I64,
        IRType::Float64 => types::F64,
        IRType::Bool => types::I8,
        IRType::PyObject => types::I64,
        IRType::Void => types::I64,
    }
}

/// Compile an IRProgram to native code via Cranelift.
pub fn compile(program: &IRProgram) -> Result<CompiledCode, String> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| e.to_string())?;
    let isa_builder =
        cranelift_native::builder().map_err(|e| format!("host ISA not available: {e}"))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| e.to_string())?;

    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);

    let native_params: Vec<IRType> = program.param_types.clone();
    let native_return = determine_return_type(program);

    // ABI: fn(*const u64) -> u64 — args-buffer calling convention (same as compile_loop)
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64)); // args: *const u64
    sig.returns.push(AbiParam::new(types::I64)); // return: u64 bits

    let func_id = module
        .declare_function("jit_fn", Linkage::Local, &sig)
        .map_err(|e| e.to_string())?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let block0 = builder.create_block();
        builder.append_block_params_for_function_params(block0);
        builder.switch_to_block(block0);
        builder.seal_block(block0);

        let args_ptr = builder.block_params(block0)[0];

        // Unpack each param from the args buffer
        let block_params: Vec<cranelift_codegen::ir::Value> = native_params
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let raw = builder.ins().load(types::I64, MemFlags::trusted(), args_ptr, (i * 8) as i32);
                match ty {
                    IRType::Float64 => builder.ins().bitcast(types::F64, MemFlags::new(), raw),
                    _ => raw,
                }
            })
            .collect();

        let mut values: HashMap<ValueId, cranelift_codegen::ir::Value> = HashMap::new();
        let mut param_idx = 0;

        for op in &program.ops {
            translate_op(&mut builder, op, &mut values, &block_params, &mut param_idx);
        }

        let ret_bits = if let Some(ret_val) = program.return_value {
            if let Some(&cl_val) = values.get(&ret_val) {
                if native_return == IRType::Float64 {
                    builder.ins().bitcast(types::I64, MemFlags::new(), cl_val)
                } else {
                    cl_val
                }
            } else {
                builder.ins().iconst(types::I64, 0)
            }
        } else {
            builder.ins().iconst(types::I64, 0)
        };
        builder.ins().return_(&[ret_bits]);

        builder.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| e.to_string())?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| e.to_string())?;

    let fn_ptr = module.get_finalized_function(func_id);

    Ok(CompiledCode {
        _module: module,
        fn_ptr,
        num_params: native_params.len(),
        param_types: native_params,
        return_type: native_return,
    })
}

fn translate_op(
    builder: &mut FunctionBuilder,
    op: &IROp,
    values: &mut HashMap<ValueId, cranelift_codegen::ir::Value>,
    block_params: &[cranelift_codegen::ir::Value],
    param_idx: &mut usize,
) {
    match op.kind.as_str() {
        "Param" => {
            if let Some(output) = op.output {
                if *param_idx < block_params.len() {
                    values.insert(output, block_params[*param_idx]);
                }
            }
        }

        "UnboxInt" | "UnboxFloat" => {
            if let Some(output) = op.output {
                if *param_idx < block_params.len() {
                    values.insert(output, block_params[*param_idx]);
                    *param_idx += 1;
                }
            }
        }

        "BoxInt" | "BoxFloat" => {
            if let Some(output) = op.output {
                if let Some(&input_val) = op.inputs.first().and_then(|id| values.get(id)) {
                    values.insert(output, input_val);
                }
            }
        }

        "Guard" | "GuardTrue" | "GuardFalse" | "GuardNotNone" => {}

        "LoadConst" => {
            if let Some(output) = op.output {
                let imm = op.immediate.unwrap_or(0);
                let cl_type = ir_type_to_cl(&op.output_type);
                let val = if cl_type == types::F64 {
                    builder.ins().f64const(imm as f64)
                } else {
                    builder.ins().iconst(cl_type, imm)
                };
                values.insert(output, val);
            }
        }

        "Add" => emit_binary(builder, op, values, |b, l, r, ty| {
            if ty == types::F64 { b.ins().fadd(l, r) } else { b.ins().iadd(l, r) }
        }),
        "Sub" => emit_binary(builder, op, values, |b, l, r, ty| {
            if ty == types::F64 { b.ins().fsub(l, r) } else { b.ins().isub(l, r) }
        }),
        "Mul" => emit_binary(builder, op, values, |b, l, r, ty| {
            if ty == types::F64 { b.ins().fmul(l, r) } else { b.ins().imul(l, r) }
        }),
        "FloorDiv" => emit_binary(builder, op, values, |b, l, r, _| b.ins().sdiv(l, r)),
        "Mod" => emit_binary(builder, op, values, |b, l, r, _| b.ins().srem(l, r)),
        "TrueDiv" => emit_binary(builder, op, values, |b, l, r, _| b.ins().fdiv(l, r)),
        "BitAnd" => emit_binary(builder, op, values, |b, l, r, _| b.ins().band(l, r)),
        "BitOr" => emit_binary(builder, op, values, |b, l, r, _| b.ins().bor(l, r)),
        "BitXor" => emit_binary(builder, op, values, |b, l, r, _| b.ins().bxor(l, r)),
        "LShift" => emit_binary(builder, op, values, |b, l, r, _| b.ins().ishl(l, r)),
        "RShift" => emit_binary(builder, op, values, |b, l, r, _| b.ins().sshr(l, r)),

        "Neg" => {
            if let Some(output) = op.output {
                if let Some(&v) = op.inputs.first().and_then(|id| values.get(id)) {
                    let cl_type = ir_type_to_cl(&op.output_type);
                    let val = if cl_type == types::F64 {
                        builder.ins().fneg(v)
                    } else {
                        builder.ins().ineg(v)
                    };
                    values.insert(output, val);
                }
            }
        }

        "Lt" => emit_icmp(builder, op, values, IntCC::SignedLessThan),
        "Le" => emit_icmp(builder, op, values, IntCC::SignedLessThanOrEqual),
        "Eq" => emit_icmp(builder, op, values, IntCC::Equal),
        "Ne" => emit_icmp(builder, op, values, IntCC::NotEqual),
        "Gt" => emit_icmp(builder, op, values, IntCC::SignedGreaterThan),
        "Ge" => emit_icmp(builder, op, values, IntCC::SignedGreaterThanOrEqual),

        "Not" => {
            if let Some(output) = op.output {
                if let Some(&v) = op.inputs.first().and_then(|id| values.get(id)) {
                    let one = builder.ins().iconst(types::I8, 1);
                    let val = builder.ins().bxor(v, one);
                    values.insert(output, val);
                }
            }
        }

        _ => {
            if let Some(output) = op.output {
                let val = builder.ins().iconst(types::I64, 0);
                values.insert(output, val);
            }
        }
    }
}

fn emit_binary<F>(
    builder: &mut FunctionBuilder,
    op: &IROp,
    values: &mut HashMap<ValueId, cranelift_codegen::ir::Value>,
    emit_fn: F,
) where
    F: FnOnce(
        &mut FunctionBuilder,
        cranelift_codegen::ir::Value,
        cranelift_codegen::ir::Value,
        types::Type,
    ) -> cranelift_codegen::ir::Value,
{
    if let Some(output) = op.output {
        if op.inputs.len() >= 2 {
            if let (Some(&lhs), Some(&rhs)) = (values.get(&op.inputs[0]), values.get(&op.inputs[1]))
            {
                let cl_type = ir_type_to_cl(&op.output_type);
                let val = emit_fn(builder, lhs, rhs, cl_type);
                values.insert(output, val);
            }
        }
    }
}

fn emit_icmp(
    builder: &mut FunctionBuilder,
    op: &IROp,
    values: &mut HashMap<ValueId, cranelift_codegen::ir::Value>,
    cc: IntCC,
) {
    if let Some(output) = op.output {
        if op.inputs.len() >= 2 {
            if let (Some(&lhs), Some(&rhs)) = (values.get(&op.inputs[0]), values.get(&op.inputs[1]))
            {
                let val = builder.ins().icmp(cc, lhs, rhs);
                values.insert(output, val);
            }
        }
    }
}

/// Compile a numeric loop to native code with proper Cranelift loop blocks.
/// Supports arbitrarily nested loops via LoopStart/LoopEnd markers in body_ops.
#[allow(clippy::too_many_arguments)]
pub fn compile_loop(
    num_params: usize,
    limit_param: usize,
    num_locals: usize,
    return_local: usize,
    init_locals: &[(usize, i64)],
    init_float_locals: &[(usize, f64)],
    body_ops: &[(String, usize, usize, usize, bool, i64)],
    local_types: &[u8],
    param_types_vec: &[u8],
    return_type_id: u8,
    start_value: i64,
    step_value: i64,
    start_param: usize,
) -> Result<CompiledCode, String> {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").map_err(|e| e.to_string())?;
    let isa_builder = cranelift_native::builder().map_err(|e| format!("host ISA: {e}"))?;
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).map_err(|e| e.to_string())?;
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    // Register runtime helpers.
    jit_builder.symbol("pyjit_box_int", pyjit_box_int as *const u8);
    jit_builder.symbol("pyjit_sin", pyjit_sin as *const u8);
    jit_builder.symbol("pyjit_cos", pyjit_cos as *const u8);
    jit_builder.symbol("pyjit_exp", pyjit_exp as *const u8);
    jit_builder.symbol("pyjit_log", pyjit_log as *const u8);
    let mut module = JITModule::new(jit_builder);

    // Declare pyjit_box_int: (i64) -> i64
    let mut box_int_sig = module.make_signature();
    box_int_sig.params.push(AbiParam::new(types::I64));
    box_int_sig.returns.push(AbiParam::new(types::I64));
    let box_int_id = module
        .declare_function("pyjit_box_int", Linkage::Import, &box_int_sig)
        .map_err(|e| e.to_string())?;

    // Declare math helpers: (f64) -> f64
    let mut f64_f64_sig = module.make_signature();
    f64_f64_sig.params.push(AbiParam::new(types::F64));
    f64_f64_sig.returns.push(AbiParam::new(types::F64));
    let sin_id = module.declare_function("pyjit_sin", Linkage::Import, &f64_f64_sig).map_err(|e| e.to_string())?;
    let cos_id = module.declare_function("pyjit_cos", Linkage::Import, &f64_f64_sig).map_err(|e| e.to_string())?;
    let exp_id = module.declare_function("pyjit_exp", Linkage::Import, &f64_f64_sig).map_err(|e| e.to_string())?;
    let log_id = module.declare_function("pyjit_log", Linkage::Import, &f64_f64_sig).map_err(|e| e.to_string())?;

    let local_cl_type = |slot: usize| -> types::Type {
        if *local_types.get(slot).unwrap_or(&0) == 1 { types::F64 } else { types::I64 }
    };

    // ABI: always fn(*const u64) -> u64 — args packed as raw u64 bits, return is raw u64 bits.
    // This eliminates the per-arity transmute dispatch and removes the arg-count limit.
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64)); // args: *const u64
    sig.returns.push(AbiParam::new(types::I64)); // return: u64 bits

    let func_id = module.declare_function("jit_loop", Linkage::Local, &sig).map_err(|e| e.to_string())?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    let mut func_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

    let block_entry = b.create_block();
    let block_header = b.create_block();
    let block_body = b.create_block();
    let block_exit = b.create_block();

    // block_exit has params for all locals so BreakIf can pass current state
    for slot in 0..num_locals {
        b.append_block_param(block_exit, local_cl_type(slot));
    }

    // --- Entry: unpack args from the *const u64 buffer ---
    b.append_block_params_for_function_params(block_entry);
    b.switch_to_block(block_entry);
    b.seal_block(block_entry);

    let args_ptr = b.block_params(block_entry)[0];

    // Unpack each param from args_ptr[i * 8]
    let unpacked_params: Vec<cranelift_codegen::ir::Value> = (0..num_params).map(|i| {
        let raw = b.ins().load(types::I64, MemFlags::trusted(), args_ptr, (i * 8) as i32);
        if *param_types_vec.get(i).unwrap_or(&0) == 1 {
            // f64 param: reinterpret i64 bits as f64
            b.ins().bitcast(types::F64, MemFlags::new(), raw)
        } else {
            raw
        }
    }).collect();

    let mut init_vals: Vec<cranelift_codegen::ir::Value> = Vec::new();
    for slot in 0..num_locals {
        let lt = local_cl_type(slot);
        let val = if slot < unpacked_params.len() {
            unpacked_params[slot]
        } else if lt == types::F64 {
            let v = init_float_locals.iter().find(|&&(s, _)| s == slot).map(|&(_, v)| v).unwrap_or(0.0);
            b.ins().f64const(v)
        } else {
            let v = init_locals.iter().find(|&&(s, _)| s == slot).map(|&(_, v)| v).unwrap_or(0);
            b.ins().iconst(types::I64, v)
        };
        init_vals.push(val);
    }

    // Get the loop limit from the appropriate local (may be a param or a constant-init'd local)
    let limit = init_vals[limit_param];

    let counter_init = if start_param != usize::MAX && start_param < init_vals.len() {
        init_vals[start_param]
    } else {
        b.ins().iconst(types::I64, start_value)
    };
    let mut args: Vec<BlockArg> = vec![BlockArg::Value(counter_init)];
    args.extend(init_vals.iter().map(|&v| BlockArg::Value(v)));
    b.ins().jump(block_header, &args);

    // --- Header ---
    b.append_block_param(block_header, types::I64);
    for slot in 0..num_locals { b.append_block_param(block_header, local_cl_type(slot)); }
    b.switch_to_block(block_header);
    let hp: Vec<cranelift_codegen::ir::Value> = b.block_params(block_header).to_vec();
    let counter = hp[0];
    let header_locals: Vec<cranelift_codegen::ir::Value> = hp[1..].to_vec();

    let cond = b.ins().icmp(IntCC::SignedLessThan, counter, limit);
    let exit_args_hdr: Vec<BlockArg> = header_locals.iter().map(|&v| BlockArg::Value(v)).collect();
    b.ins().brif(cond, block_body, &[], block_exit, &exit_args_hdr);

    // --- Body (recursive, handles nested loops) ---
    b.switch_to_block(block_body);
    b.seal_block(block_body);

    let rt = RuntimeRefs {
        box_int: module.declare_func_in_func(box_int_id, b.func),
        sin: module.declare_func_in_func(sin_id, b.func),
        cos: module.declare_func_in_func(cos_id, b.func),
        exp: module.declare_func_in_func(exp_id, b.func),
        log: module.declare_func_in_func(log_id, b.func),
    };

    let mut body_locals = header_locals.clone();
    emit_body_ops(&mut b, body_ops, &mut body_locals, counter, limit, block_exit, &local_cl_type, num_locals, &rt);

    // Increment counter by step, jump back to header
    let step = b.ins().iconst(types::I64, step_value);
    let counter_next = b.ins().iadd(counter, step);
    let mut back: Vec<BlockArg> = vec![BlockArg::Value(counter_next)];
    back.extend(body_locals.iter().map(|&v| BlockArg::Value(v)));
    b.ins().jump(block_header, &back);

    // --- Exit: return as raw u64 bits ---
    b.switch_to_block(block_exit);
    b.seal_block(block_exit);
    b.seal_block(block_header);
    let exit_params: Vec<cranelift_codegen::ir::Value> = b.block_params(block_exit).to_vec();
    let ret_val = if return_local < exit_params.len() {
        exit_params[return_local]
    } else {
        b.ins().iconst(types::I64, 0)
    };
    let ret_bits = if return_type_id == 1 {
        // f64 return: reinterpret as i64 bits
        b.ins().bitcast(types::I64, MemFlags::new(), ret_val)
    } else {
        ret_val
    };
    b.ins().return_(&[ret_bits]);

    b.finalize();

    module.define_function(func_id, &mut ctx).map_err(|e| e.to_string())?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let fn_ptr = module.get_finalized_function(func_id);

    let param_ir_types: Vec<IRType> = param_types_vec.iter().map(|&t| if t == 1 { IRType::Float64 } else { IRType::Int64 }).collect();
    let ret_ir = if return_type_id == 1 { IRType::Float64 } else { IRType::Int64 };

    Ok(CompiledCode { _module: module, fn_ptr, num_params, param_types: param_ir_types, return_type: ret_ir })
}

/// Recursively emit body ops, handling LoopStart/LoopEnd for nested loops.
///
/// LoopStart: ("LoopStart", limit_slot, iter_var_slot, 0, False, 0)
///   — creates an inner loop header/body/exit, counter goes 0..locals[limit_slot]
/// LoopEnd: ("LoopEnd", 0, 0, 0, False, 0)
///   — closes the inner loop, resumes outer body
#[allow(clippy::too_many_arguments)]
fn emit_body_ops(
    b: &mut FunctionBuilder,
    ops: &[(String, usize, usize, usize, bool, i64)],
    locals: &mut [cranelift_codegen::ir::Value],
    counter: cranelift_codegen::ir::Value,
    limit: cranelift_codegen::ir::Value,
    block_exit: cranelift_codegen::ir::Block,
    local_cl_type: &dyn Fn(usize) -> types::Type,
    num_locals: usize,
    rt: &RuntimeRefs,
) {
    let counter_slot = usize::MAX;
    let mut cmp_results: HashMap<usize, cranelift_codegen::ir::Value> = HashMap::new();
    // Stack for nested conditionals: (merge_block, pre-condition locals)
    let mut cond_stack: Vec<(cranelift_codegen::ir::Block, Vec<cranelift_codegen::ir::Value>)> = Vec::new();
    let mut i = 0;

    while i < ops.len() {
        let (ref kind, dst, src_a, src_b, is_b_imm, imm) = ops[i];

        if kind == "LoopStart" {
            // dst = limit_slot (which local holds the inner loop's limit)
            // src_a = inner iter_var_slot (not used in compiled code — counter is separate)
            let inner_limit = if dst < locals.len() { locals[dst] } else { limit };

            // Find matching LoopEnd
            let inner_end = find_loop_end(ops, i);
            let inner_ops = &ops[i + 1..inner_end];

            // Create inner loop blocks
            let inner_header = b.create_block();
            let inner_body = b.create_block();
            let inner_exit = b.create_block();

            // Init inner counter, jump to inner header
            let inner_counter_init = b.ins().iconst(types::I64, 0);
            let mut jump_args: Vec<BlockArg> = vec![BlockArg::Value(inner_counter_init)];
            jump_args.extend(locals.iter().map(|&v| BlockArg::Value(v)));
            b.ins().jump(inner_header, &jump_args);

            // Inner header: [inner_counter, locals...]
            b.append_block_param(inner_header, types::I64);
            for slot in 0..num_locals { b.append_block_param(inner_header, local_cl_type(slot)); }
            b.switch_to_block(inner_header);
            let ihp: Vec<cranelift_codegen::ir::Value> = b.block_params(inner_header).to_vec();
            let inner_counter = ihp[0];
            let inner_locals: Vec<cranelift_codegen::ir::Value> = ihp[1..].to_vec();

            let inner_cond = b.ins().icmp(IntCC::SignedLessThan, inner_counter, inner_limit);
            // When inner loop exits, pass current locals to exit block
            let exit_args: Vec<BlockArg> = inner_locals.iter().map(|&v| BlockArg::Value(v)).collect();
            b.ins().brif(inner_cond, inner_body, &[], inner_exit, &exit_args);

            // Inner body
            b.switch_to_block(inner_body);
            b.seal_block(inner_body);

            let mut inner_body_locals = inner_locals.clone();
            // Store inner counter into its iter var slot so nested loops can reference it
            let inner_iter_slot = src_a; // LoopStart encodes iter_var_slot in src_a
            if inner_iter_slot < inner_body_locals.len() {
                inner_body_locals[inner_iter_slot] = inner_counter;
            }
            // RECURSE: process inner ops (may contain more LoopStart/LoopEnd)
            emit_body_ops(b, inner_ops, &mut inner_body_locals, inner_counter, inner_limit, inner_exit, local_cl_type, num_locals, rt);

            // Increment inner counter, jump back
            let one = b.ins().iconst(types::I64, 1);
            let inner_next = b.ins().iadd(inner_counter, one);
            let mut inner_back: Vec<BlockArg> = vec![BlockArg::Value(inner_next)];
            inner_back.extend(inner_body_locals.iter().map(|&v| BlockArg::Value(v)));
            b.ins().jump(inner_header, &inner_back);

            // Inner exit: locals flow out to continue the outer body
            // The inner_exit block receives locals from inner_header when loop is done
            for slot in 0..num_locals { b.append_block_param(inner_exit, local_cl_type(slot)); }
            b.switch_to_block(inner_exit);
            b.seal_block(inner_exit);
            b.seal_block(inner_header);

            // Update outer locals from inner exit block params
            let exit_params: Vec<cranelift_codegen::ir::Value> = b.block_params(inner_exit).to_vec();
            for (slot, &val) in exit_params.iter().enumerate() {
                if slot < locals.len() {
                    locals[slot] = val;
                }
            }

            i = inner_end + 1;
            continue;
        }

        if kind == "CondStart" {
            let cmp_val = cmp_results.get(&dst).copied()
                .unwrap_or_else(|| if dst < locals.len() { locals[dst] } else { counter });

            let true_block = b.create_block();
            let merge_block = b.create_block();

            // merge_block has params: one per local
            for slot in 0..num_locals { b.append_block_param(merge_block, local_cl_type(slot)); }

            // brif: if cond → true_block (no args), else → merge with CURRENT locals
            let merge_false_args: Vec<BlockArg> = locals.iter().map(|&v| BlockArg::Value(v)).collect();
            b.ins().brif(cmp_val, true_block, &[], merge_block, &merge_false_args);

            b.switch_to_block(true_block);
            b.seal_block(true_block);

            cond_stack.push((merge_block, locals.to_vec()));

            i += 1;
            continue;
        }

        if kind == "CondElse" {
            // End of true branch. Jump to a NEW merge, switch to false (using pre-cond locals).
            if let Some((old_merge, _pre_locals)) = cond_stack.pop() {
                // The old merge_block was the if-only merge. We no longer need it as merge
                // because now there's an else. Create a new merge.
                let new_merge = b.create_block();
                for slot in 0..num_locals { b.append_block_param(new_merge, local_cl_type(slot)); }

                // True branch → new_merge
                let true_args: Vec<BlockArg> = locals.iter().map(|&v| BlockArg::Value(v)).collect();
                b.ins().jump(new_merge, &true_args);

                // The old_merge is now the else block (it receives pre-cond locals from brif)
                b.switch_to_block(old_merge);
                b.seal_block(old_merge);

                // Restore locals from old_merge block params (the pre-condition values)
                let mp: Vec<cranelift_codegen::ir::Value> = b.block_params(old_merge).to_vec();
                for (slot, &val) in mp.iter().enumerate() {
                    if slot < locals.len() { locals[slot] = val; }
                }

                cond_stack.push((new_merge, Vec::new()));
            }
            i += 1;
            continue;
        }

        if kind == "CondEnd" {
            if let Some((merge_block, _)) = cond_stack.pop() {
                let args: Vec<BlockArg> = locals.iter().map(|&v| BlockArg::Value(v)).collect();
                b.ins().jump(merge_block, &args);

                b.switch_to_block(merge_block);
                b.seal_block(merge_block);

                let mp: Vec<cranelift_codegen::ir::Value> = b.block_params(merge_block).to_vec();
                for (slot, &val) in mp.iter().enumerate() {
                    if slot < locals.len() { locals[slot] = val; }
                }
            }
            i += 1;
            continue;
        }

        if kind == "LoadConst" {
            // Store a constant into a local slot
            if dst < locals.len() {
                let lt = local_cl_type(dst);
                let val = if lt == types::F64 {
                    b.ins().f64const(f64::from_bits(imm as u64))
                } else {
                    b.ins().iconst(types::I64, imm)
                };
                locals[dst] = val;
            }
            i += 1;
            continue;
        }

        if kind == "StoreCounter" {
            if dst < locals.len() {
                locals[dst] = counter;
            }
            i += 1;
            continue;
        }

        if kind == "LoopEnd" {
            // Should not reach here — LoopEnd is consumed by LoopStart processing
            i += 1;
            continue;
        }

        if kind == "BreakIf" {
            // dst = cmp_slot, src_a = invert (0 = exit if true, 1 = exit if false)
            let cond_val = cmp_results.get(&dst).copied()
                .unwrap_or_else(|| b.ins().iconst(types::I8, 0));
            let invert = src_a != 0;

            let continue_block = b.create_block();
            let exit_args: Vec<BlockArg> = locals.iter().map(|&v| BlockArg::Value(v)).collect();

            if invert {
                // Exit if cond is false (0)
                b.ins().brif(cond_val, continue_block, &[], block_exit, &exit_args);
            } else {
                // Exit if cond is true (1)
                b.ins().brif(cond_val, block_exit, &exit_args, continue_block, &[]);
            }

            b.switch_to_block(continue_block);
            b.seal_block(continue_block);
            i += 1;
            continue;
        }

        if kind == "LoadElement" {
            // dst = result slot, src_a = ob_item pointer slot, src_b = index slot (or counter)
            let base_ptr = if src_a < locals.len() { locals[src_a] } else { counter };
            let index = if src_b == counter_slot {
                counter
            } else if src_b < locals.len() {
                locals[src_b]
            } else {
                counter
            };

            // elem_addr = ob_item + index * 8  (each element is a pointer)
            let eight = b.ins().iconst(types::I64, 8);
            let byte_offset = b.ins().imul(index, eight);
            let elem_addr = b.ins().iadd(base_ptr, byte_offset);

            // Load the PyObject* at ob_item[index]
            let pyobj_ptr = b.ins().load(types::I64, MemFlags::trusted(), elem_addr, 0);

            // Unbox CPython integer: lv_tag at offset 16, ob_digit[0] at offset 24
            // lv_tag layout: bits[3+] = ndigits, bit[1] = sign (1=negative), bit[2] = immortal
            let lv_tag = b.ins().load(types::I64, MemFlags::trusted(), pyobj_ptr, 16);
            let three = b.ins().iconst(types::I64, 3);
            let ndigits = b.ins().ushr(lv_tag, three);

            let digit_i32 = b.ins().load(types::I32, MemFlags::trusted(), pyobj_ptr, 24);
            let digit = b.ins().sextend(types::I64, digit_i32);

            let one64 = b.ins().iconst(types::I64, 1);
            let zero64 = b.ins().iconst(types::I64, 0);
            let lv_shifted = b.ins().ushr(lv_tag, one64);
            let sign_bit = b.ins().band(lv_shifted, one64);
            let neg_digit = b.ins().ineg(digit);
            let is_neg = b.ins().icmp(IntCC::NotEqual, sign_bit, zero64);
            let value_if_nonzero = b.ins().select(is_neg, neg_digit, digit);

            let is_zero_int = b.ins().icmp(IntCC::Equal, ndigits, zero64);
            let value = b.ins().select(is_zero_int, zero64, value_if_nonzero);

            if dst < locals.len() {
                locals[dst] = value;
            }
            i += 1;
            continue;
        }

        if kind == "LoadElementF64" || kind == "LoadElementI64" {
            // NumPy typed load — no boxing, just a raw memory load
            let base_ptr = if src_a < locals.len() { locals[src_a] } else { counter };
            let index = if src_b == counter_slot {
                counter
            } else if src_b < locals.len() {
                locals[src_b]
            } else {
                counter
            };

            let eight = b.ins().iconst(types::I64, 8);
            let byte_offset = b.ins().imul(index, eight);
            let elem_addr = b.ins().iadd(base_ptr, byte_offset);

            let val = if kind == "LoadElementF64" {
                b.ins().load(types::F64, MemFlags::trusted(), elem_addr, 0)
            } else {
                b.ins().load(types::I64, MemFlags::trusted(), elem_addr, 0)
            };
            if dst < locals.len() {
                locals[dst] = val;
            }
            i += 1;
            continue;
        }

        if kind == "StoreElementF64" || kind == "StoreElementI64" {
            // NumPy typed store — no boxing, raw memory write
            let base_ptr = if dst < locals.len() { locals[dst] } else { counter };
            let index = if src_a == counter_slot {
                counter
            } else if src_a < locals.len() {
                locals[src_a]
            } else {
                counter
            };
            let value = if src_b == counter_slot {
                counter
            } else if src_b < locals.len() {
                locals[src_b]
            } else {
                counter
            };

            let eight = b.ins().iconst(types::I64, 8);
            let byte_offset = b.ins().imul(index, eight);
            let elem_addr = b.ins().iadd(base_ptr, byte_offset);

            b.ins().store(MemFlags::trusted(), value, elem_addr, 0);
            i += 1;
            continue;
        }

        if kind == "StoreElement" {
            // dst = container_slot (ob_item ptr), src_a = index, src_b = value to store
            let base_ptr = if dst < locals.len() { locals[dst] } else { counter };
            let index = if src_a == counter_slot {
                counter
            } else if src_a < locals.len() {
                locals[src_a]
            } else {
                counter
            };
            let value = if src_b == counter_slot {
                counter
            } else if src_b < locals.len() {
                locals[src_b]
            } else {
                counter
            };

            // elem_addr = ob_item + index * 8
            let eight = b.ins().iconst(types::I64, 8);
            let byte_offset = b.ins().imul(index, eight);
            let elem_addr = b.ins().iadd(base_ptr, byte_offset);

            // Box the i64 value into a Python int object via pyjit_box_int
            let call_inst = b.ins().call(rt.box_int, &[value]);
            let pyobj_ptr = b.inst_results(call_inst)[0];

            // Store the PyObject* into ob_item[index]
            b.ins().store(MemFlags::trusted(), pyobj_ptr, elem_addr, 0);
            i += 1;
            continue;
        }

        // Explicit float conversion: int → f64
        if kind == "ToF64" {
            let val = if src_a < locals.len() { locals[src_a] } else { counter };
            let result = if local_cl_type(src_a) == types::F64 {
                val  // already f64
            } else {
                b.ins().fcvt_from_sint(types::F64, val)  // i64 → f64
            };
            if dst < locals.len() { locals[dst] = result; }
            i += 1;
            continue;
        }

        // Math unary ops
        if kind == "Sqrt" || kind == "Sin" || kind == "Cos" || kind == "Exp" || kind == "Log" || kind == "Fabs" {
            let val = if src_a < locals.len() { locals[src_a] } else { counter };
            // Ensure the input is f64
            let fval = if local_cl_type(src_a) == types::F64 {
                val
            } else {
                b.ins().fcvt_from_sint(types::F64, val)
            };
            let result = match kind.as_str() {
                "Sqrt" => b.ins().sqrt(fval),
                "Fabs" => b.ins().fabs(fval),
                "Sin" => { let c = b.ins().call(rt.sin, &[fval]); b.inst_results(c)[0] }
                "Cos" => { let c = b.ins().call(rt.cos, &[fval]); b.inst_results(c)[0] }
                "Exp" => { let c = b.ins().call(rt.exp, &[fval]); b.inst_results(c)[0] }
                "Log" => { let c = b.ins().call(rt.log, &[fval]); b.inst_results(c)[0] }
                _ => fval,
            };
            if dst < locals.len() { locals[dst] = result; }
            i += 1;
            continue;
        }

        // Handle Select
        if kind == "Select" {
            let cmp_slot = imm as usize;
            if let Some(&cond_val) = cmp_results.get(&cmp_slot) {
                let tv = if src_a < locals.len() { locals[src_a] } else { counter };
                let fv = if src_b < locals.len() { locals[src_b] } else { counter };
                let sel = b.ins().select(cond_val, tv, fv);
                if dst < locals.len() { locals[dst] = sel; }
            }
            i += 1;
            continue;
        }

        // Regular arithmetic/comparison op
        let dst_is_float = local_cl_type(dst) == types::F64;

        let a = if src_a == counter_slot {
            if dst_is_float { b.ins().fcvt_from_sint(types::F64, counter) } else { counter }
        } else if src_a < locals.len() {
            locals[src_a]
        } else { counter };

        let bv = if is_b_imm {
            if dst_is_float { b.ins().f64const(f64::from_bits(imm as u64)) } else { b.ins().iconst(types::I64, imm) }
        } else if src_b == counter_slot {
            if dst_is_float { b.ins().fcvt_from_sint(types::F64, counter) } else { counter }
        } else if src_b < locals.len() {
            locals[src_b]
        } else { counter };

        if dst_is_float {
            let result = match kind.as_str() {
                "Add" => b.ins().fadd(a, bv),
                "Sub" => b.ins().fsub(a, bv),
                "Mul" => b.ins().fmul(a, bv),
                "Div" | "TrueDiv" => b.ins().fdiv(a, bv),
                _ => b.ins().fadd(a, bv),
            };
            if dst < locals.len() { locals[dst] = result; }
        } else {
            match kind.as_str() {
                "CmpLt" | "CmpLe" | "CmpEq" | "CmpNe" | "CmpGt" | "CmpGe" => {
                    let cc = match kind.as_str() {
                        "CmpLt" => IntCC::SignedLessThan,
                        "CmpLe" => IntCC::SignedLessThanOrEqual,
                        "CmpEq" => IntCC::Equal,
                        "CmpNe" => IntCC::NotEqual,
                        "CmpGt" => IntCC::SignedGreaterThan,
                        _ => IntCC::SignedGreaterThanOrEqual,
                    };
                    let v = b.ins().icmp(cc, a, bv);
                    cmp_results.insert(dst, v);
                }
                "Neg" => {
                    let result = if dst_is_float { b.ins().fneg(a) } else { b.ins().ineg(a) };
                    if dst < locals.len() { locals[dst] = result; }
                }
                "Abs" => {
                    let result = if dst_is_float {
                        b.ins().fabs(a)
                    } else {
                        let zero = b.ins().iconst(types::I64, 0);
                        let is_neg = b.ins().icmp(IntCC::SignedLessThan, a, zero);
                        let neg_a = b.ins().ineg(a);
                        b.ins().select(is_neg, neg_a, a)
                    };
                    if dst < locals.len() { locals[dst] = result; }
                }
                "Min" => {
                    let result = if dst_is_float {
                        b.ins().fmin(a, bv)
                    } else {
                        let cond = b.ins().icmp(IntCC::SignedLessThan, a, bv);
                        b.ins().select(cond, a, bv)
                    };
                    if dst < locals.len() { locals[dst] = result; }
                }
                "Max" => {
                    let result = if dst_is_float {
                        b.ins().fmax(a, bv)
                    } else {
                        let cond = b.ins().icmp(IntCC::SignedGreaterThan, a, bv);
                        b.ins().select(cond, a, bv)
                    };
                    if dst < locals.len() { locals[dst] = result; }
                }
                "Pow" => {
                    // For small constant exponents, emit repeated multiply
                    let result = if is_b_imm && (0..=8).contains(&imm) {
                        let exp = imm as usize;
                        match exp {
                            0 => if dst_is_float { b.ins().f64const(1.0) } else { b.ins().iconst(types::I64, 1) },
                            1 => a,
                            _ => {
                                let mut r = a;
                                for _ in 1..exp {
                                    r = if dst_is_float { b.ins().fmul(r, a) } else { b.ins().imul(r, a) };
                                }
                                r
                            }
                        }
                    } else {
                        // Non-constant or large exponent: return 0 (will be wrong, but guarded)
                        if dst_is_float { b.ins().f64const(0.0) } else { b.ins().iconst(types::I64, 0) }
                    };
                    if dst < locals.len() { locals[dst] = result; }
                }
                "BitNot" => {
                    let result = b.ins().bnot(a);
                    if dst < locals.len() { locals[dst] = result; }
                }
                "Not" => {
                    let one = b.ins().iconst(types::I8, 1);
                    let result = b.ins().bxor(a, one);
                    if dst < locals.len() { locals[dst] = result; }
                }
                _ => {
                    let result = match kind.as_str() {
                        "Add" => b.ins().iadd(a, bv),
                        "Sub" => b.ins().isub(a, bv),
                        "Mul" => b.ins().imul(a, bv),
                        "BitAnd" => b.ins().band(a, bv),
                        "BitOr"  => b.ins().bor(a, bv),
                        "BitXor" => b.ins().bxor(a, bv),
                        "LShift" => b.ins().ishl(a, bv),
                        "RShift" => b.ins().sshr(a, bv),
                        "FloorDiv" => {
                            // Guard: replace 0 divisor with 1 to prevent SIGFPE
                            let zero = b.ins().iconst(types::I64, 0);
                            let one = b.ins().iconst(types::I64, 1);
                            let is_zero = b.ins().icmp(IntCC::Equal, bv, zero);
                            let safe_bv = b.ins().select(is_zero, one, bv);
                            let div_result = b.ins().sdiv(a, safe_bv);
                            b.ins().select(is_zero, zero, div_result)
                        }
                        "Mod" => {
                            let zero = b.ins().iconst(types::I64, 0);
                            let one = b.ins().iconst(types::I64, 1);
                            let is_zero = b.ins().icmp(IntCC::Equal, bv, zero);
                            let safe_bv = b.ins().select(is_zero, one, bv);
                            let mod_result = b.ins().srem(a, safe_bv);
                            b.ins().select(is_zero, zero, mod_result)
                        }
                        _ => b.ins().iadd(a, bv),
                    };
                    if dst < locals.len() { locals[dst] = result; }
                }
            }
        }

        i += 1;
    }
}

/// Find the matching LoopEnd for a LoopStart at position `start`.
fn find_loop_end(ops: &[(String, usize, usize, usize, bool, i64)], start: usize) -> usize {
    let mut depth = 0;
    for (j, op) in ops.iter().enumerate().skip(start) {
        if op.0 == "LoopStart" { depth += 1; }
        if op.0 == "LoopEnd" {
            depth -= 1;
            if depth == 0 { return j; }
        }
    }
    ops.len() // fallback
}

fn determine_return_type(program: &IRProgram) -> IRType {
    if let Some(ret_val) = program.return_value {
        for op in program.ops.iter().rev() {
            if op.output == Some(ret_val) {
                return match op.kind.as_str() {
                    "BoxInt" => IRType::Int64,
                    "BoxFloat" => IRType::Float64,
                    _ => op.output_type.clone(),
                };
            }
        }
    }
    IRType::Int64
}

