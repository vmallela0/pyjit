//! Trace recording — hooks into Python execution to record bytecode traces.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::trace::Trace;

/// Trace a Python function's execution, recording every bytecode operation.
///
/// # Arguments
/// * `func` — The Python function to trace.
/// * `args` — Positional arguments to call the function with.
///
/// # Returns
/// A `Trace` containing the result and the sequence of recorded operations.
#[pyfunction]
#[pyo3(signature = (func, args=None, kwargs=None))]
pub fn trace_function(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    args: Option<&Bound<'_, PyTuple>>,
    kwargs: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<Trace>> {
    let tracer_mod = py.import("pyjit._tracer")?;
    let record_trace = tracer_mod.getattr("record_trace")?;

    let call_args = match (args, kwargs) {
        (Some(a), Some(k)) => PyTuple::new(py, [func.as_any(), a.as_any(), k])?,
        (Some(a), None) => PyTuple::new(py, [func.as_any(), a.as_any()])?,
        (None, Some(k)) => {
            let empty = PyTuple::empty(py);
            PyTuple::new(py, [func.as_any(), empty.as_any(), k])?
        }
        (None, None) => PyTuple::new(py, [func.as_any()])?,
    };

    let trace_obj = record_trace.call1(call_args)?;
    let trace: Bound<'_, Trace> = trace_obj.cast_into()?;
    Ok(trace.unbind())
}
