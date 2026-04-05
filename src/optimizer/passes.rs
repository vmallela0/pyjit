//! Optimizer pass pipeline: run all optimization passes over an IRProgram.

use crate::ir::ops::IRProgram;

use super::const_fold::const_fold;
use super::dce::dead_code_elim;

/// Run the full optimizer pipeline over the given program (in-place).
///
/// Passes:
///   1. Constant folding — evaluate pure ops on known constants.
///   2. Dead code elimination — remove ops whose output is never used.
pub fn optimize(program: &mut IRProgram) {
    const_fold(program);
    dead_code_elim(program);
}
