//! CPython bytecode decoder — parses co_code into structured instructions.
//!
//! Phase 1: Bytecode decoding is handled on the Python side via the `dis` module
//! and `sys.monitoring`. This module will be expanded in later phases to support
//! Rust-native bytecode analysis for IR generation.
