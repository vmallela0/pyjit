//! ABI and calling conventions — bridge between Python objects and native code.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::ir::ops::IRProgram;
use crate::ir::types::IRType;

use super::cranelift::{compile, compile_loop, CompiledCode};

/// A compiled function that can be called from Python.
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

    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        if args.len() != self.code.num_params {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{}() takes {} arguments, got {}",
                self.func_name, self.code.num_params, args.len()
            )));
        }

        // Extract arguments to raw u64 buffer (both i64 and f64 fit in 8 bytes)
        let mut raw_args: Vec<u64> = Vec::with_capacity(self.code.num_params);
        for (i, ty) in self.code.param_types.iter().enumerate() {
            match ty {
                IRType::Int64 => {
                    let val: i64 = args.get_item(i)?.extract()?;
                    raw_args.push(val as u64);
                }
                IRType::Float64 => {
                    let val: f64 = args.get_item(i)?.extract()?;
                    raw_args.push(val.to_bits());
                }
                _ => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Unsupported parameter type",
                    ));
                }
            }
        }

        let result_raw = self.call_raw(&raw_args);

        match self.code.return_type {
            IRType::Float64 => {
                let val = f64::from_bits(result_raw);
                Ok(val.into_pyobject(py)?.into_any().unbind())
            }
            _ => {
                let val = result_raw as i64;
                Ok(val.into_pyobject(py)?.into_any().unbind())
            }
        }
    }
}

impl CompiledFunction {
    fn call_raw(&self, args: &[u64]) -> u64 {
        // ABI: fn(*const u64) -> u64 — args passed as a packed buffer, return is raw bits.
        // This works for any number of params and any mix of i64/f64 types.
        let ptr = self.code.fn_ptr;
        unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*const u64) -> u64>(ptr)(args.as_ptr())
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
#[pyfunction]
#[pyo3(signature = (num_params, limit_param, num_locals, return_local, init_locals, init_float_locals, body_ops, local_types, param_types, return_type_id, func_name=None, start_value=0, step_value=1))]
#[allow(clippy::too_many_arguments)]
pub fn compile_loop_ir(
    num_params: usize,
    limit_param: usize,
    num_locals: usize,
    return_local: usize,
    init_locals: Vec<(usize, i64)>,
    init_float_locals: Vec<(usize, f64)>,
    body_ops: Vec<(String, usize, usize, usize, bool, i64)>,
    local_types: Vec<u8>,
    param_types: Vec<u8>,
    return_type_id: u8,
    func_name: Option<String>,
    start_value: i64,
    step_value: i64,
) -> PyResult<CompiledFunction> {
    let code = compile_loop(
        num_params,
        limit_param,
        num_locals,
        return_local,
        &init_locals,
        &init_float_locals,
        &body_ops,
        &local_types,
        &param_types,
        return_type_id,
        start_value,
        step_value,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(CompiledFunction {
        code,
        func_name: func_name.unwrap_or_else(|| "jit_loop".to_string()),
    })
}
