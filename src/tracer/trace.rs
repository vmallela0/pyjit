//! Trace IR data structures — the recorded execution trace of a Python function.

use pyo3::prelude::*;

/// A single recorded bytecode operation from a trace.
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct TraceOp {
    /// The bytecode opcode name (e.g., "BINARY_OP", "LOAD_FAST").
    #[pyo3(get)]
    pub kind: String,

    /// The bytecode argument value.
    #[pyo3(get)]
    pub arg: i32,

    /// The byte offset in co_code where this instruction lives.
    #[pyo3(get)]
    pub offset: i32,

    /// Observed types of operands (e.g., ["int", "int"] for a binary op).
    #[pyo3(get)]
    pub arg_types: Vec<String>,

    /// Whether this instruction is part of a loop body.
    #[pyo3(get)]
    pub is_loop_body: bool,
}

#[pymethods]
impl TraceOp {
    #[new]
    fn new(kind: String, arg: i32, offset: i32, arg_types: Vec<String>, is_loop_body: bool) -> Self {
        Self {
            kind,
            arg,
            offset,
            arg_types,
            is_loop_body,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TraceOp(kind={:?}, arg={}, offset={}, arg_types={:?}, is_loop_body={})",
            self.kind, self.arg, self.offset, self.arg_types, self.is_loop_body
        )
    }
}

/// A complete execution trace of a Python function.
#[pyclass]
pub struct Trace {
    /// The return value of the traced function call.
    #[pyo3(get)]
    pub result: Py<PyAny>,

    /// The recorded sequence of bytecode operations.
    #[pyo3(get)]
    pub ops: Vec<TraceOp>,

    /// The name of the traced function.
    #[pyo3(get)]
    pub func_name: String,

    /// Types of the function arguments as observed during this trace.
    #[pyo3(get)]
    pub input_types: Vec<String>,
}

#[pymethods]
impl Trace {
    #[new]
    fn new(
        result: Py<PyAny>,
        ops: Vec<TraceOp>,
        func_name: String,
        input_types: Vec<String>,
    ) -> Self {
        Self {
            result,
            ops,
            func_name,
            input_types,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Trace(func={:?}, ops={}, input_types={:?})",
            self.func_name,
            self.ops.len(),
            self.input_types,
        )
    }

    /// Number of recorded operations.
    fn __len__(&self) -> usize {
        self.ops.len()
    }
}
