//! Constant folding pass: evaluate pure ops on known constants at compile time.

use std::collections::HashMap;

use crate::ir::ops::{IRProgram, ValueId};

/// Fold `LoadConst op LoadConst` → `LoadConst result` where both inputs are constants.
pub fn const_fold(program: &mut IRProgram) {
    // Map from ValueId → known constant i64 value.
    let mut known: HashMap<ValueId, i64> = HashMap::new();

    for op in program.ops.iter_mut() {
        match op.kind.as_str() {
            "LoadConst" => {
                if let (Some(out), Some(imm)) = (op.output, op.immediate) {
                    known.insert(out, imm);
                }
            }

            "Add" | "Sub" | "Mul" | "FloorDiv" | "Mod" | "BitAnd" | "BitOr" | "BitXor"
            | "LShift" | "RShift" => {
                if op.inputs.len() >= 2 {
                    let a = op.inputs[0];
                    let b = op.inputs[1];
                    if let (Some(&va), Some(&vb)) = (known.get(&a), known.get(&b)) {
                        let result = match op.kind.as_str() {
                            "Add" => va.wrapping_add(vb),
                            "Sub" => va.wrapping_sub(vb),
                            "Mul" => va.wrapping_mul(vb),
                            "FloorDiv" => {
                                if vb != 0 { va.wrapping_div(vb) } else { 0 }
                            }
                            "Mod" => {
                                if vb != 0 { va.wrapping_rem(vb) } else { 0 }
                            }
                            "BitAnd" => va & vb,
                            "BitOr" => va | vb,
                            "BitXor" => va ^ vb,
                            "LShift" => va.wrapping_shl(vb as u32),
                            "RShift" => va >> (vb as u32),
                            _ => return,
                        };
                        if let Some(out) = op.output {
                            known.insert(out, result);
                            op.kind = "LoadConst".to_string();
                            op.inputs.clear();
                            op.immediate = Some(result);
                        }
                    }
                }
            }

            "Neg" => {
                if let Some(&input_id) = op.inputs.first() {
                    if let Some(&v) = known.get(&input_id) {
                        let result = v.wrapping_neg();
                        if let Some(out) = op.output {
                            known.insert(out, result);
                            op.kind = "LoadConst".to_string();
                            op.inputs.clear();
                            op.immediate = Some(result);
                        }
                    }
                }
            }

            _ => {}
        }
    }
}
