//! Dead code elimination: remove ops whose output is never consumed.

use std::collections::HashSet;

use crate::ir::ops::{IRProgram, ValueId};

/// Remove operations whose output value is never used as input anywhere,
/// and is not the return value. Void ops (no output) are always kept.
pub fn dead_code_elim(program: &mut IRProgram) {
    // Collect the set of all values that are actually consumed.
    let mut used: HashSet<ValueId> = HashSet::new();

    if let Some(rv) = program.return_value {
        used.insert(rv);
    }

    for op in &program.ops {
        for &input in &op.inputs {
            used.insert(input);
        }
    }

    // Retain an op if it has no output (side-effectful) or its output is used.
    program.ops.retain(|op| match op.output {
        None => true,
        Some(out) => used.contains(&out),
    });
}
