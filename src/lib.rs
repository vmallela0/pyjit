use pyo3::prelude::*;

mod tracer;
mod ir;
#[allow(unused)]
mod optimizer;
mod codegen;
#[allow(unused)]
mod guards;
#[allow(unused)]
mod runtime;

/// The native extension module for pyjit.
#[pymodule]
fn _pyjit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Tracer types
    m.add_class::<tracer::trace::TraceOp>()?;
    m.add_class::<tracer::trace::Trace>()?;
    m.add_function(wrap_pyfunction!(tracer::recorder::trace_function, m)?)?;

    // IR types
    m.add_class::<ir::types::IRType>()?;
    m.add_class::<ir::ops::IROp>()?;
    m.add_class::<ir::ops::IRProgram>()?;
    m.add_function(wrap_pyfunction!(ir::builder::build_ir, m)?)?;

    // Codegen
    m.add_class::<codegen::abi::CompiledFunction>()?;
    m.add_function(wrap_pyfunction!(codegen::abi::compile_ir, m)?)?;
    m.add_function(wrap_pyfunction!(codegen::abi::compile_loop_ir, m)?)?;

    Ok(())
}
