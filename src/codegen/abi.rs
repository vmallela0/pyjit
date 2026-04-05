//! ABI and calling conventions — bridge between Python objects and native code.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::ir::ops::IRProgram;
use crate::ir::types::IRType;

use super::cranelift::{compile, compile_loop, CompiledCode};

/// A compiled function that can be called from Python.
///
/// Wraps native code produced by Cranelift. Handles argument extraction
/// (PyObject → native) and result boxing (native → PyObject).
#[pyclass(unsendable)]
pub struct CompiledFunction {
    code: CompiledCode,
    func_name: String,
}

#[pymethods]
impl CompiledFunction {
    fn __repr__(&self) -> String {
        format!(
            "CompiledFunction({}, params={}, return={:?})",
            self.func_name, self.code.num_params, self.code.return_type
        )
    }

    /// Call the compiled native function with Python arguments.
    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        if args.len() != self.code.num_params {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{}() takes {} arguments, got {}",
                self.func_name,
                self.code.num_params,
                args.len()
            )));
        }

        // Extract arguments to native types
        let mut i64_args: Vec<i64> = Vec::new();
        let mut f64_args: Vec<f64> = Vec::new();
        let all_int = self.code.param_types.iter().all(|t| *t == IRType::Int64);
        let all_float = self.code.param_types.iter().all(|t| *t == IRType::Float64);

        if all_int {
            for (i, ty) in self.code.param_types.iter().enumerate() {
                match ty {
                    IRType::Int64 => {
                        let val: i64 = args.get_item(i)?.extract()?;
                        i64_args.push(val);
                    }
                    _ => unreachable!(),
                }
            }
            let result = self.call_int_fn(&i64_args);
            Ok(result.into_pyobject(py)?.into_any().unbind())
        } else if all_float {
            for (i, ty) in self.code.param_types.iter().enumerate() {
                match ty {
                    IRType::Float64 => {
                        let val: f64 = args.get_item(i)?.extract()?;
                        f64_args.push(val);
                    }
                    _ => unreachable!(),
                }
            }
            let result = self.call_float_fn(&f64_args);
            Ok(result.into_pyobject(py)?.into_any().unbind())
        } else {
            // Mixed types — fall back to interpreting
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Mixed parameter types not yet supported in JIT",
            ))
        }
    }
}

impl CompiledFunction {
    /// Call the native function with i64 arguments.
    fn call_int_fn(&self, args: &[i64]) -> i64 {
        let ptr = self.code.fn_ptr;
        unsafe {
            match args.len() {
                0 => {
                    let f: unsafe extern "C" fn() -> i64 = std::mem::transmute(ptr);
                    f()
                }
                1 => {
                    let f: unsafe extern "C" fn(i64) -> i64 = std::mem::transmute(ptr);
                    f(args[0])
                }
                2 => {
                    let f: unsafe extern "C" fn(i64, i64) -> i64 = std::mem::transmute(ptr);
                    f(args[0], args[1])
                }
                3 => {
                    let f: unsafe extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(ptr);
                    f(args[0], args[1], args[2])
                }
                4 => {
                    let f: unsafe extern "C" fn(i64, i64, i64, i64) -> i64 =
                        std::mem::transmute(ptr);
                    f(args[0], args[1], args[2], args[3])
                }
                5 => {
                    let f: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64 =
                        std::mem::transmute(ptr);
                    f(args[0], args[1], args[2], args[3], args[4])
                }
                6 => {
                    let f: unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
                        std::mem::transmute(ptr);
                    f(args[0], args[1], args[2], args[3], args[4], args[5])
                }
                _ => panic!("too many arguments (max 6)"),
            }
        }
    }

    /// Call the native function with f64 arguments.
    fn call_float_fn(&self, args: &[f64]) -> f64 {
        let ptr = self.code.fn_ptr;
        unsafe {
            match args.len() {
                0 => {
                    let f: unsafe extern "C" fn() -> f64 = std::mem::transmute(ptr);
                    f()
                }
                1 => {
                    let f: unsafe extern "C" fn(f64) -> f64 = std::mem::transmute(ptr);
                    f(args[0])
                }
                2 => {
                    let f: unsafe extern "C" fn(f64, f64) -> f64 = std::mem::transmute(ptr);
                    f(args[0], args[1])
                }
                3 => {
                    let f: unsafe extern "C" fn(f64, f64, f64) -> f64 = std::mem::transmute(ptr);
                    f(args[0], args[1], args[2])
                }
                4 => {
                    let f: unsafe extern "C" fn(f64, f64, f64, f64) -> f64 =
                        std::mem::transmute(ptr);
                    f(args[0], args[1], args[2], args[3])
                }
                _ => panic!("too many float arguments (max 4)"),
            }
        }
    }
}

/// Compile an IR program and return a callable CompiledFunction.
#[pyfunction]
pub fn compile_ir(program: &IRProgram, func_name: Option<String>) -> PyResult<CompiledFunction> {
    let code = compile(program).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(CompiledFunction {
        code,
        func_name: func_name.unwrap_or_else(|| "jit_fn".to_string()),
    })
}

/// Compile a loop function with native loop blocks for maximum performance.
///
/// # Arguments
/// * `num_params` — number of function parameters
/// * `limit_param` — index of the param that is the loop limit
/// * `num_locals` — total local variable slots
/// * `return_local` — which local to return
/// * `init_locals` — list of (slot, initial_value) for non-param locals
/// * `body_ops` — list of (kind, dst, src_a, src_b, is_b_imm, imm_val) tuples
/// * `func_name` — optional name for debugging
#[pyfunction]
#[pyo3(signature = (num_params, limit_param, num_locals, return_local, init_locals, body_ops, func_name=None))]
pub fn compile_loop_ir(
    num_params: usize,
    limit_param: usize,
    num_locals: usize,
    return_local: usize,
    init_locals: Vec<(usize, i64)>,
    body_ops: Vec<(String, usize, usize, usize, bool, i64)>,
    func_name: Option<String>,
) -> PyResult<CompiledFunction> {
    let code = compile_loop(
        num_params,
        limit_param,
        num_locals,
        return_local,
        &init_locals,
        &body_ops,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(CompiledFunction {
        code,
        func_name: func_name.unwrap_or_else(|| "jit_loop".to_string()),
    })
}
