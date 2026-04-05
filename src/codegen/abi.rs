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
        let all_int = self.code.param_types.iter().all(|t| *t == IRType::Int64);
        let ret_float = self.code.return_type == IRType::Float64;

        if all_int && !ret_float {
            self.call_int_fn(args) as u64
        } else if all_int && ret_float {
            self.call_int_ret_float(args).to_bits()
        } else {
            self.call_mixed(args)
        }
    }

    #[allow(clippy::missing_transmute_annotations)]
    fn call_int_fn(&self, args: &[u64]) -> i64 {
        let ptr = self.code.fn_ptr;
        let a = |i: usize| args[i] as i64;
        unsafe {
            match args.len() {
                0 => std::mem::transmute::<_, unsafe extern "C" fn() -> i64>(ptr)(),
                1 => std::mem::transmute::<_, unsafe extern "C" fn(i64) -> i64>(ptr)(a(0)),
                2 => std::mem::transmute::<_, unsafe extern "C" fn(i64, i64) -> i64>(ptr)(a(0), a(1)),
                3 => std::mem::transmute::<_, unsafe extern "C" fn(i64, i64, i64) -> i64>(ptr)(a(0), a(1), a(2)),
                4 => std::mem::transmute::<_, unsafe extern "C" fn(i64, i64, i64, i64) -> i64>(ptr)(a(0), a(1), a(2), a(3)),
                5 => std::mem::transmute::<_, unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64>(ptr)(a(0), a(1), a(2), a(3), a(4)),
                6 => std::mem::transmute::<_, unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64>(ptr)(a(0), a(1), a(2), a(3), a(4), a(5)),
                7 => std::mem::transmute::<_, unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64>(ptr)(a(0), a(1), a(2), a(3), a(4), a(5), a(6)),
                8 => std::mem::transmute::<_, unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64>(ptr)(a(0), a(1), a(2), a(3), a(4), a(5), a(6), a(7)),
                _ => panic!("too many int arguments (max 8)"),
            }
        }
    }

    #[allow(clippy::missing_transmute_annotations)]
    fn call_int_ret_float(&self, args: &[u64]) -> f64 {
        let ptr = self.code.fn_ptr;
        let a = |i: usize| args[i] as i64;
        unsafe {
            match args.len() {
                0 => std::mem::transmute::<_, unsafe extern "C" fn() -> f64>(ptr)(),
                1 => std::mem::transmute::<_, unsafe extern "C" fn(i64) -> f64>(ptr)(a(0)),
                2 => std::mem::transmute::<_, unsafe extern "C" fn(i64, i64) -> f64>(ptr)(a(0), a(1)),
                3 => std::mem::transmute::<_, unsafe extern "C" fn(i64, i64, i64) -> f64>(ptr)(a(0), a(1), a(2)),
                _ => panic!("too many args for int->float signature (max 3)"),
            }
        }
    }

    #[allow(clippy::missing_transmute_annotations)]
    fn call_mixed(&self, args: &[u64]) -> u64 {
        let ptr = self.code.fn_ptr;
        let types = &self.code.param_types;
        let ret_float = self.code.return_type == IRType::Float64;
        unsafe {
            match (args.len(), ret_float) {
                (1, true) if types[0] == IRType::Int64 => {
                    std::mem::transmute::<_, unsafe extern "C" fn(i64) -> f64>(ptr)(args[0] as i64).to_bits()
                }
                (1, false) if types[0] == IRType::Float64 => {
                    std::mem::transmute::<_, unsafe extern "C" fn(f64) -> i64>(ptr)(f64::from_bits(args[0])) as u64
                }
                (2, true) if types[0] == IRType::Int64 && types[1] == IRType::Int64 => {
                    std::mem::transmute::<_, unsafe extern "C" fn(i64, i64) -> f64>(ptr)(args[0] as i64, args[1] as i64).to_bits()
                }
                (2, false) if types[0] == IRType::Float64 && types[1] == IRType::Float64 => {
                    std::mem::transmute::<_, unsafe extern "C" fn(f64, f64) -> f64>(ptr)(f64::from_bits(args[0]), f64::from_bits(args[1])).to_bits()
                }
                _ => 0, // unsupported — Python guard should prevent reaching here
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
#[pyfunction]
#[pyo3(signature = (num_params, limit_param, num_locals, return_local, init_locals, init_float_locals, body_ops, local_types, param_types, return_type_id, func_name=None))]
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
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(CompiledFunction {
        code,
        func_name: func_name.unwrap_or_else(|| "jit_loop".to_string()),
    })
}
