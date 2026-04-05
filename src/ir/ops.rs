//! IR operation definitions for the JIT compiler.

use pyo3::prelude::*;

use super::types::IRType;

/// SSA value identifier. Each value is defined exactly once.
pub type ValueId = u32;

/// An operation in the IR. Each op produces at most one output value
/// and consumes zero or more input values.
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct IROp {
    /// The kind of operation.
    #[pyo3(get)]
    pub kind: String,

    /// The SSA value this operation defines (None for void ops).
    #[pyo3(get)]
    pub output: Option<u32>,

    /// SSA values this operation consumes.
    #[pyo3(get)]
    pub inputs: Vec<u32>,

    /// The type of the output value.
    #[pyo3(get)]
    pub output_type: IRType,

    /// Additional constant data (e.g., the immediate value for LoadConst).
    #[pyo3(get)]
    pub immediate: Option<i64>,

    /// For guard ops: which variable index this guards.
    #[pyo3(get)]
    pub guard_type: Option<IRType>,

    /// Human-readable label for debugging.
    #[pyo3(get)]
    pub label: String,
}

#[pymethods]
impl IROp {
    #[new]
    #[pyo3(signature = (kind, output=None, inputs=vec![], output_type=IRType::Void, immediate=None, guard_type=None, label=String::new()))]
    fn new(
        kind: String,
        output: Option<u32>,
        inputs: Vec<u32>,
        output_type: IRType,
        immediate: Option<i64>,
        guard_type: Option<IRType>,
        label: String,
    ) -> Self {
        Self {
            kind,
            output,
            inputs,
            output_type,
            immediate,
            guard_type,
            label,
        }
    }

    fn __repr__(&self) -> String {
        let out = match self.output {
            Some(v) => format!("v{}", v),
            None => "_".to_string(),
        };
        let ins: Vec<String> = self.inputs.iter().map(|v| format!("v{}", v)).collect();
        format!(
            "{} = {} {} : {:?}",
            out,
            self.kind,
            ins.join(", "),
            self.output_type
        )
    }
}

/// A complete IR program — a linear sequence of SSA operations.
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct IRProgram {
    /// The sequence of IR operations.
    #[pyo3(get)]
    pub ops: Vec<IROp>,

    /// The SSA value that holds the function's return value.
    #[pyo3(get)]
    pub return_value: Option<u32>,

    /// The number of function parameters.
    #[pyo3(get)]
    pub num_params: u32,

    /// Types of the function parameters.
    #[pyo3(get)]
    pub param_types: Vec<IRType>,
}

#[pymethods]
impl IRProgram {
    #[new]
    fn new(
        ops: Vec<IROp>,
        return_value: Option<u32>,
        num_params: u32,
        param_types: Vec<IRType>,
    ) -> Self {
        Self {
            ops,
            return_value,
            num_params,
            param_types,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IRProgram(ops={}, params={}, return=v{})",
            self.ops.len(),
            self.num_params,
            self.return_value.map_or("none".to_string(), |v| v.to_string())
        )
    }

    fn __len__(&self) -> usize {
        self.ops.len()
    }

    /// Pretty-print the IR for debugging.
    fn dump(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "IR: {} params {:?}, {} ops",
            self.num_params,
            self.param_types,
            self.ops.len()
        ));
        for op in &self.ops {
            lines.push(format!("  {}", op.__repr__()));
        }
        if let Some(rv) = self.return_value {
            lines.push(format!("  return v{}", rv));
        }
        lines.join("\n")
    }
}
