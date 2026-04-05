//! IR type system for the JIT compiler.

use pyo3::prelude::*;

/// The types that the IR can represent.
/// These map to machine-level representations for codegen.
#[pyclass(eq, from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IRType {
    /// 64-bit signed integer (unboxed from Python int).
    Int64,
    /// 64-bit floating point (unboxed from Python float).
    Float64,
    /// Boolean value.
    Bool,
    /// An opaque Python object pointer (boxed, GC-managed).
    PyObject,
    /// No value (used for void-returning operations like stores).
    Void,
}

impl IRType {
    /// Convert a Python type name string to an IRType.
    pub fn from_type_name(name: &str) -> Self {
        match name {
            "int" => IRType::Int64,
            "float" => IRType::Float64,
            "bool" => IRType::Bool,
            _ => IRType::PyObject,
        }
    }
}

#[pymethods]
impl IRType {
    fn __repr__(&self) -> &'static str {
        match self {
            IRType::Int64 => "IRType.Int64",
            IRType::Float64 => "IRType.Float64",
            IRType::Bool => "IRType.Bool",
            IRType::PyObject => "IRType.PyObject",
            IRType::Void => "IRType.Void",
        }
    }

    /// Python-visible version of from_type_name.
    #[staticmethod]
    #[pyo3(name = "from_type_name")]
    fn py_from_type_name(name: &str) -> Self {
        Self::from_type_name(name)
    }
}
