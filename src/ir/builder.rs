//! Trace → IR builder. Converts a recorded bytecode trace into typed SSA IR.

use pyo3::prelude::*;

use super::ops::{IROp, IRProgram, ValueId};
use super::types::IRType;
use crate::tracer::trace::{Trace, TraceOp};

/// BINARY_OP argument values from CPython 3.12+
const NB_ADD: i32 = 0;
const NB_AND: i32 = 1;
const NB_FLOOR_DIVIDE: i32 = 2;
const NB_LSHIFT: i32 = 3;
const NB_MULTIPLY: i32 = 5;
const NB_REMAINDER: i32 = 6;
const NB_OR: i32 = 7;
const NB_RSHIFT: i32 = 9;
const NB_SUBTRACT: i32 = 10;
const NB_TRUE_DIVIDE: i32 = 11;
const NB_XOR: i32 = 12;
const NB_INPLACE_ADD: i32 = 13;
const NB_INPLACE_AND: i32 = 14;
const NB_INPLACE_FLOOR_DIVIDE: i32 = 15;
const NB_INPLACE_LSHIFT: i32 = 16;
const NB_INPLACE_MULTIPLY: i32 = 18;
const NB_INPLACE_REMAINDER: i32 = 19;
const NB_INPLACE_OR: i32 = 20;
const NB_INPLACE_RSHIFT: i32 = 22;
const NB_INPLACE_SUBTRACT: i32 = 23;
const NB_INPLACE_TRUE_DIVIDE: i32 = 24;
const NB_INPLACE_XOR: i32 = 25;

/// Map a BINARY_OP argument to an IR operation name.
fn binary_op_name(arg: i32) -> &'static str {
    match arg {
        NB_ADD | NB_INPLACE_ADD => "Add",
        NB_SUBTRACT | NB_INPLACE_SUBTRACT => "Sub",
        NB_MULTIPLY | NB_INPLACE_MULTIPLY => "Mul",
        NB_TRUE_DIVIDE | NB_INPLACE_TRUE_DIVIDE => "TrueDiv",
        NB_FLOOR_DIVIDE | NB_INPLACE_FLOOR_DIVIDE => "FloorDiv",
        NB_REMAINDER | NB_INPLACE_REMAINDER => "Mod",
        NB_AND | NB_INPLACE_AND => "BitAnd",
        NB_OR | NB_INPLACE_OR => "BitOr",
        NB_XOR | NB_INPLACE_XOR => "BitXor",
        NB_LSHIFT | NB_INPLACE_LSHIFT => "LShift",
        NB_RSHIFT | NB_INPLACE_RSHIFT => "RShift",
        _ => "BinaryOp",
    }
}

/// Determine the output type of a binary operation.
fn binary_result_type(op: &str, lhs: &IRType, rhs: &IRType) -> IRType {
    match op {
        "TrueDiv" => IRType::Float64,
        _ => {
            if *lhs == IRType::Float64 || *rhs == IRType::Float64 {
                IRType::Float64
            } else {
                lhs.clone()
            }
        }
    }
}

/// COMPARE_OP argument decoding.
/// Python 3.12 encodes the comparison in bits 4+ (arg >> 4).
/// Python 3.13+ encodes it in bits 5+ (arg >> 5) with bit 4 as a bool-conversion flag.
fn compare_op_name(arg: i32, py_minor: u8) -> &'static str {
    let idx = if py_minor >= 13 { (arg >> 5) & 0xf } else { (arg >> 4) & 0xf };
    match idx {
        0 => "Lt",
        1 => "Le",
        2 => "Eq",
        3 => "Ne",
        4 => "Gt",
        5 => "Ge",
        _ => "Compare",
    }
}

/// State for the IR builder.
struct Builder {
    ops: Vec<IROp>,
    next_value: ValueId,
    /// Maps local variable slot → current SSA value
    locals: Vec<Option<ValueId>>,
    /// Maps SSA value → its type
    value_types: Vec<IRType>,
    /// Shadow stack of SSA values (mirrors Python's value stack)
    stack: Vec<ValueId>,
    /// Number of function parameters
    _num_params: u32,
    /// Types of function parameters
    param_types: Vec<IRType>,
}

impl Builder {
    fn new(num_params: u32, param_types: Vec<IRType>) -> Self {
        Self {
            ops: Vec::new(),
            next_value: 0,
            locals: vec![None; 256], // generous default
            value_types: Vec::new(),
            stack: Vec::new(),
            _num_params: num_params,
            param_types,
        }
    }

    fn alloc_value(&mut self, ty: IRType) -> ValueId {
        let id = self.next_value;
        self.next_value += 1;
        // Grow value_types to accommodate
        while self.value_types.len() <= id as usize {
            self.value_types.push(IRType::Void);
        }
        self.value_types[id as usize] = ty;
        id
    }

    fn get_type(&self, v: ValueId) -> IRType {
        self.value_types
            .get(v as usize)
            .cloned()
            .unwrap_or(IRType::PyObject)
    }

    fn emit(&mut self, kind: &str, inputs: Vec<ValueId>, output_type: IRType, label: &str) -> ValueId {
        let output = self.alloc_value(output_type.clone());
        self.ops.push(IROp {
            kind: kind.to_string(),
            output: Some(output),
            inputs,
            output_type,
            immediate: None,
            guard_type: None,
            label: label.to_string(),
        });
        output
    }

    fn emit_with_immediate(
        &mut self,
        kind: &str,
        inputs: Vec<ValueId>,
        output_type: IRType,
        immediate: i64,
        label: &str,
    ) -> ValueId {
        let output = self.alloc_value(output_type.clone());
        self.ops.push(IROp {
            kind: kind.to_string(),
            output: Some(output),
            inputs,
            output_type,
            immediate: Some(immediate),
            guard_type: None,
            label: label.to_string(),
        });
        output
    }

    fn emit_guard(&mut self, input: ValueId, expected_type: IRType, label: &str) {
        self.ops.push(IROp {
            kind: "Guard".to_string(),
            output: None,
            inputs: vec![input],
            output_type: IRType::Void,
            immediate: None,
            guard_type: Some(expected_type),
            label: label.to_string(),
        });
    }

    fn emit_void(&mut self, kind: &str, inputs: Vec<ValueId>, label: &str) {
        self.ops.push(IROp {
            kind: kind.to_string(),
            output: None,
            inputs,
            output_type: IRType::Void,
            immediate: None,
            guard_type: None,
            label: label.to_string(),
        });
    }

    fn push(&mut self, v: ValueId) {
        self.stack.push(v);
    }

    fn pop(&mut self) -> Option<ValueId> {
        self.stack.pop()
    }

    fn store_local(&mut self, slot: usize, v: ValueId) {
        if slot >= self.locals.len() {
            self.locals.resize(slot + 1, None);
        }
        self.locals[slot] = Some(v);
    }

    fn load_local(&self, slot: usize) -> Option<ValueId> {
        self.locals.get(slot).copied().flatten()
    }
}

/// Build an IR program from a trace.
///
/// This is the main entry point called from Python via `build_ir(trace)`.
#[pyfunction]
pub fn build_ir(py: Python<'_>, trace: &Bound<'_, Trace>) -> PyResult<IRProgram> {
    let trace = trace.borrow();
    let ops: Vec<TraceOp> = trace.ops.clone();
    let input_types: Vec<IRType> = trace
        .input_types
        .iter()
        .map(|t| IRType::from_type_name(t))
        .collect();

    let py_minor = py.version_info().minor;
    let num_params = input_types.len() as u32;
    let mut builder = Builder::new(num_params, input_types.clone());

    // Create parameter values and emit guards for each
    for (i, ty) in input_types.iter().enumerate() {
        let param = builder.emit(
            "Param",
            vec![],
            IRType::PyObject,
            &format!("param_{}", i),
        );
        builder.emit_guard(param, ty.clone(), &format!("guard_param_{}", i));

        // Unbox the parameter to its native type
        let unboxed = match ty {
            IRType::Int64 => builder.emit("UnboxInt", vec![param], IRType::Int64, &format!("unbox_{}", i)),
            IRType::Float64 => builder.emit("UnboxFloat", vec![param], IRType::Float64, &format!("unbox_{}", i)),
            _ => param,
        };
        builder.store_local(i, unboxed);
    }

    let mut return_value: Option<ValueId> = None;

    for trace_op in &ops {
        match trace_op.kind.as_str() {
            "RESUME" | "NOT_TAKEN" => {}

            "LOAD_FAST" | "LOAD_FAST_BORROW" | "LOAD_FAST_CHECK" => {
                let slot = trace_op.arg as usize;
                if let Some(v) = builder.load_local(slot) {
                    builder.push(v);
                }
            }

            "LOAD_FAST_LOAD_FAST" | "LOAD_FAST_BORROW_LOAD_FAST_BORROW" => {
                // Python 3.13 uses LOAD_FAST_LOAD_FAST; Python 3.14 uses LOAD_FAST_BORROW_LOAD_FAST_BORROW
                let idx_a = ((trace_op.arg >> 4) & 0xf) as usize;
                let idx_b = (trace_op.arg & 0xf) as usize;
                if let Some(v) = builder.load_local(idx_a) {
                    builder.push(v);
                }
                if let Some(v) = builder.load_local(idx_b) {
                    builder.push(v);
                }
            }

            "STORE_FAST" | "STORE_FAST_MAYBE_NULL" => {
                if let Some(v) = builder.pop() {
                    builder.store_local(trace_op.arg as usize, v);
                }
            }

            "LOAD_SMALL_INT" => {
                let v = builder.emit_with_immediate(
                    "LoadConst",
                    vec![],
                    IRType::Int64,
                    trace_op.arg as i64,
                    &format!("const_{}", trace_op.arg),
                );
                builder.push(v);
            }

            "LOAD_CONST" => {
                // We don't have access to co_consts here, so emit a generic LoadConst
                let ty = if !trace_op.arg_types.is_empty() {
                    IRType::from_type_name(&trace_op.arg_types[0])
                } else {
                    IRType::PyObject
                };
                let v = builder.emit_with_immediate(
                    "LoadConst",
                    vec![],
                    ty,
                    trace_op.arg as i64,
                    "const",
                );
                builder.push(v);
            }

            "BINARY_OP" => {
                let rhs = builder.pop();
                let lhs = builder.pop();
                if let (Some(l), Some(r)) = (lhs, rhs) {
                    let op_name = binary_op_name(trace_op.arg);
                    let l_type = builder.get_type(l);
                    let r_type = builder.get_type(r);
                    let result_type = binary_result_type(op_name, &l_type, &r_type);
                    let v = builder.emit(op_name, vec![l, r], result_type, op_name);
                    builder.push(v);
                }
            }

            "COMPARE_OP" => {
                let rhs = builder.pop();
                let lhs = builder.pop();
                if let (Some(l), Some(r)) = (lhs, rhs) {
                    let op_name = compare_op_name(trace_op.arg, py_minor);
                    let v = builder.emit(op_name, vec![l, r], IRType::Bool, op_name);
                    builder.push(v);
                }
            }

            "RETURN_VALUE" => {
                if let Some(v) = builder.pop() {
                    // Box the return value back to PyObject if it's unboxed
                    let ty = builder.get_type(v);
                    let boxed = match ty {
                        IRType::Int64 => builder.emit("BoxInt", vec![v], IRType::PyObject, "box_return"),
                        IRType::Float64 => builder.emit("BoxFloat", vec![v], IRType::PyObject, "box_return"),
                        _ => v,
                    };
                    return_value = Some(boxed);
                }
            }

            "POP_JUMP_IF_FALSE" | "POP_JUMP_IF_TRUE" => {
                if let Some(v) = builder.pop() {
                    let guard_kind = if trace_op.kind == "POP_JUMP_IF_FALSE" {
                        "GuardTrue"
                    } else {
                        "GuardFalse"
                    };
                    builder.emit_void(guard_kind, vec![v], "branch_guard");
                }
            }

            "POP_JUMP_IF_NONE" | "POP_JUMP_IF_NOT_NONE" => {
                if let Some(v) = builder.pop() {
                    builder.emit_void("GuardNotNone", vec![v], "none_guard");
                }
            }

            // Loop ops — for now, we unroll loops in the trace so these are
            // just markers. The IR is a linear trace with guards.
            "FOR_ITER" => {
                // The iterator produces a value
                let v = builder.emit("IterNext", vec![], IRType::Int64, "iter_next");
                builder.push(v);
            }

            "GET_ITER" => {
                // Replace the iterable with an iterator
                if let Some(_iterable) = builder.pop() {
                    let v = builder.emit("GetIter", vec![], IRType::PyObject, "get_iter");
                    builder.push(v);
                }
            }

            "END_FOR" | "POP_ITER" => {
                builder.pop();
            }

            "JUMP_BACKWARD" | "JUMP_BACKWARD_NO_INTERRUPT" | "JUMP_FORWARD" => {
                // In a linear trace, jumps are no-ops — the trace is already linearized
            }

            "LOAD_GLOBAL" => {
                let v = builder.emit_with_immediate(
                    "LoadGlobal",
                    vec![],
                    IRType::PyObject,
                    trace_op.arg as i64,
                    "load_global",
                );
                builder.push(v);
            }

            "CALL" => {
                // Pop args + callable
                let n_args = trace_op.arg as usize;
                let mut args = Vec::new();
                for _ in 0..n_args {
                    if let Some(v) = builder.pop() {
                        args.push(v);
                    }
                }
                args.reverse();
                // Pop callable (and possible NULL)
                let _callable = builder.pop();
                let v = builder.emit("Call", args, IRType::PyObject, "call");
                builder.push(v);
            }

            "POP_TOP" => {
                builder.pop();
            }

            "COPY" => {
                if let Some(&top) = builder.stack.last() {
                    builder.push(top);
                }
            }

            "SWAP" => {
                let len = builder.stack.len();
                if len >= 2 {
                    builder.stack.swap(len - 1, len - 2);
                }
            }

            "UNARY_NEGATIVE" => {
                if let Some(v) = builder.pop() {
                    let ty = builder.get_type(v);
                    let result = builder.emit("Neg", vec![v], ty, "neg");
                    builder.push(result);
                }
            }

            "UNARY_NOT" => {
                if let Some(v) = builder.pop() {
                    let result = builder.emit("Not", vec![v], IRType::Bool, "not");
                    builder.push(result);
                }
            }

            _ => {
                // Unknown opcode — emit a generic fallback
                // This ensures the IR is always complete even for unhandled ops
            }
        }
    }

    // Drop the borrow before we use `py`
    drop(trace);

    Ok(IRProgram {
        ops: builder.ops,
        return_value,
        num_params,
        param_types: builder.param_types,
    })
}
