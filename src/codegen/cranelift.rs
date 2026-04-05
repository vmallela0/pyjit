//! Cranelift JIT compilation — translates IR to native machine code.

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::{AbiParam, InstBuilder};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::ir::ops::{IROp, IRProgram, ValueId};
use crate::ir::types::IRType;

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

    let mut sig = module.make_signature();
    for ty in &native_params {
        sig.params.push(AbiParam::new(ir_type_to_cl(ty)));
    }
    sig.returns.push(AbiParam::new(ir_type_to_cl(&native_return)));

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

        let mut values: HashMap<ValueId, cranelift_codegen::ir::Value> = HashMap::new();
        let block_params: Vec<cranelift_codegen::ir::Value> =
            builder.block_params(block0).to_vec();
        let mut param_idx = 0;

        for op in &program.ops {
            translate_op(&mut builder, op, &mut values, &block_params, &mut param_idx);
        }

        if let Some(ret_val) = program.return_value {
            if let Some(&cl_val) = values.get(&ret_val) {
                builder.ins().return_(&[cl_val]);
            } else {
                let zero = builder.ins().iconst(types::I64, 0);
                builder.ins().return_(&[zero]);
            }
        } else {
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().return_(&[zero]);
        }

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
///
/// Produces: fn(param0, param1, ...) -> i64 that runs:
///   init locals from params and constants
///   for counter in 0..limit:
///       execute body_ops(counter, locals)
///   return locals[return_local]
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
) -> Result<CompiledCode, String> {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").map_err(|e| e.to_string())?;
    let isa_builder = cranelift_native::builder().map_err(|e| format!("host ISA: {e}"))?;
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).map_err(|e| e.to_string())?;
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);

    let cl_type_for = |type_id: u8| -> types::Type {
        if type_id == 1 { types::F64 } else { types::I64 }
    };
    let local_cl_type = |slot: usize| -> types::Type {
        cl_type_for(*local_types.get(slot).unwrap_or(&0))
    };

    let mut sig = module.make_signature();
    for i in 0..num_params {
        sig.params.push(AbiParam::new(cl_type_for(*param_types_vec.get(i).unwrap_or(&0))));
    }
    sig.returns.push(AbiParam::new(cl_type_for(return_type_id)));

    let func_id = module.declare_function("jit_loop", Linkage::Local, &sig).map_err(|e| e.to_string())?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    let mut func_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

    let block_entry = b.create_block();
    let block_header = b.create_block();
    let block_body = b.create_block();
    let block_exit = b.create_block();

    // --- Entry block ---
    b.append_block_params_for_function_params(block_entry);
    b.switch_to_block(block_entry);
    b.seal_block(block_entry);

    let params: Vec<cranelift_codegen::ir::Value> = b.block_params(block_entry).to_vec();
    let limit = params[limit_param];

    // Initialize locals with correct types
    let mut init_vals: Vec<cranelift_codegen::ir::Value> = Vec::new();
    for slot in 0..num_locals {
        let lt = local_cl_type(slot);
        let val = if slot < params.len() {
            params[slot]
        } else if lt == types::F64 {
            let v = init_float_locals.iter().find(|&&(s, _)| s == slot).map(|&(_, v)| v).unwrap_or(0.0);
            b.ins().f64const(v)
        } else {
            let v = init_locals.iter().find(|&&(s, _)| s == slot).map(|&(_, v)| v).unwrap_or(0);
            b.ins().iconst(types::I64, v)
        };
        init_vals.push(val);
    }

    let counter_init = b.ins().iconst(types::I64, 0);
    let mut header_args: Vec<BlockArg> = vec![BlockArg::Value(counter_init)];
    header_args.extend(init_vals.iter().map(|&v| BlockArg::Value(v)));
    b.ins().jump(block_header, &header_args);

    // Loop header: [counter(i64), local0, local1, ...]
    b.append_block_param(block_header, types::I64); // counter always i64
    for slot in 0..num_locals {
        b.append_block_param(block_header, local_cl_type(slot));
    }
    b.switch_to_block(block_header);
    let hp: Vec<cranelift_codegen::ir::Value> = b.block_params(block_header).to_vec();
    let counter = hp[0];
    let header_locals: Vec<cranelift_codegen::ir::Value> = hp[1..].to_vec();

    let cond = b.ins().icmp(IntCC::SignedLessThan, counter, limit);
    b.ins().brif(cond, block_body, &[], block_exit, &[]);

    // --- Loop body ---
    b.switch_to_block(block_body);
    b.seal_block(block_body);

    let mut body_locals = header_locals.clone();
    let counter_slot = usize::MAX;
    // Comparison results stored separately (i8 type)
    let mut cmp_results: HashMap<usize, cranelift_codegen::ir::Value> = HashMap::new();

    for (kind, dst, src_a, src_b, is_b_imm, imm) in body_ops {
        // Handle Select op specially: ("Select", dst, true_slot, false_slot, _, cmp_slot)
        if kind == "Select" {
            let cmp_slot = *imm as usize;
            if let Some(&cond) = cmp_results.get(&cmp_slot) {
                let true_val = if *src_a < body_locals.len() { body_locals[*src_a] }
                    else if *src_a == counter_slot { counter }
                    else { body_locals.get(*src_a).copied().unwrap_or(counter) };
                let false_val = if *src_b < body_locals.len() { body_locals[*src_b] }
                    else if *src_b == counter_slot { counter }
                    else { body_locals.get(*src_b).copied().unwrap_or(counter) };
                let selected = b.ins().select(cond, true_val, false_val);
                if *dst < body_locals.len() {
                    body_locals[*dst] = selected;
                }
            }
            continue;
        }

        let dst_is_float = local_cl_type(*dst) == types::F64;

        let a = if *src_a == counter_slot {
            if dst_is_float { b.ins().fcvt_from_sint(types::F64, counter) } else { counter }
        } else if *src_a < body_locals.len() {
            body_locals[*src_a]
        } else if dst_is_float {
            b.ins().fcvt_from_sint(types::F64, counter)
        } else {
            counter
        };

        let bv = if *is_b_imm {
            if dst_is_float { b.ins().f64const(f64::from_bits(*imm as u64)) } else { b.ins().iconst(types::I64, *imm) }
        } else if *src_b == counter_slot {
            if dst_is_float { b.ins().fcvt_from_sint(types::F64, counter) } else { counter }
        } else if *src_b < body_locals.len() {
            body_locals[*src_b]
        } else if dst_is_float {
            b.ins().fcvt_from_sint(types::F64, counter)
        } else {
            counter
        };

        let result = if dst_is_float {
            match kind.as_str() {
                "Add" => b.ins().fadd(a, bv),
                "Sub" => b.ins().fsub(a, bv),
                "Mul" => b.ins().fmul(a, bv),
                "Div" | "TrueDiv" => b.ins().fdiv(a, bv),
                _ => b.ins().fadd(a, bv),
            }
        } else {
            match kind.as_str() {
                "Add" => b.ins().iadd(a, bv),
                "Sub" => b.ins().isub(a, bv),
                "Mul" => b.ins().imul(a, bv),
                "FloorDiv" => b.ins().sdiv(a, bv),
                "Mod" => b.ins().srem(a, bv),
                "CmpLt" => { let v = b.ins().icmp(IntCC::SignedLessThan, a, bv); cmp_results.insert(*dst, v); continue; }
                "CmpLe" => { let v = b.ins().icmp(IntCC::SignedLessThanOrEqual, a, bv); cmp_results.insert(*dst, v); continue; }
                "CmpEq" => { let v = b.ins().icmp(IntCC::Equal, a, bv); cmp_results.insert(*dst, v); continue; }
                "CmpNe" => { let v = b.ins().icmp(IntCC::NotEqual, a, bv); cmp_results.insert(*dst, v); continue; }
                "CmpGt" => { let v = b.ins().icmp(IntCC::SignedGreaterThan, a, bv); cmp_results.insert(*dst, v); continue; }
                "CmpGe" => { let v = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, bv); cmp_results.insert(*dst, v); continue; }
                _ => b.ins().iadd(a, bv),
            }
        };

        if *dst < body_locals.len() {
            body_locals[*dst] = result;
        }
    }

    let one = b.ins().iconst(types::I64, 1);
    let counter_next = b.ins().iadd(counter, one);
    let mut back_args: Vec<BlockArg> = vec![BlockArg::Value(counter_next)];
    back_args.extend(body_locals.iter().map(|&v| BlockArg::Value(v)));
    b.ins().jump(block_header, &back_args);

    // --- Exit block ---
    b.switch_to_block(block_exit);
    b.seal_block(block_exit);
    b.seal_block(block_header);

    let ret_val = header_locals[return_local];
    b.ins().return_(&[ret_val]);

    b.finalize();

    module.define_function(func_id, &mut ctx).map_err(|e| e.to_string())?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| e.to_string())?;
    let fn_ptr = module.get_finalized_function(func_id);

    let param_ir_types: Vec<IRType> = param_types_vec.iter().map(|&t| {
        if t == 1 { IRType::Float64 } else { IRType::Int64 }
    }).collect();
    let ret_ir_type = if return_type_id == 1 { IRType::Float64 } else { IRType::Int64 };

    Ok(CompiledCode {
        _module: module,
        fn_ptr,
        num_params,
        param_types: param_ir_types,
        return_type: ret_ir_type,
    })
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

