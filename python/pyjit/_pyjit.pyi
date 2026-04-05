"""Type stubs for the native _pyjit extension module."""

from __future__ import annotations

from enum import Enum
from typing import Any

__version__: str

# --- Tracer types ---

class TraceOp:
    kind: str
    arg: int
    offset: int
    arg_types: list[str]
    is_loop_body: bool

    def __init__(
        self,
        kind: str,
        arg: int,
        offset: int,
        arg_types: list[str],
        is_loop_body: bool,
    ) -> None: ...
    def __repr__(self) -> str: ...

class Trace:
    result: Any
    ops: list[TraceOp]
    func_name: str
    input_types: list[str]

    def __init__(
        self,
        result: Any,
        ops: list[TraceOp],
        func_name: str,
        input_types: list[str],
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

def trace_function(
    func: Any,
    args: tuple[Any, ...] | None = None,
    kwargs: Any | None = None,
) -> Trace: ...

# --- IR types ---

class IRType(Enum):
    Int64 = ...
    Float64 = ...
    Bool = ...
    PyObject = ...
    Void = ...

    @staticmethod
    def from_type_name(name: str) -> IRType: ...

class IROp:
    kind: str
    output: int | None
    inputs: list[int]
    output_type: IRType
    immediate: int | None
    guard_type: IRType | None
    label: str

    def __init__(
        self,
        kind: str,
        output: int | None = None,
        inputs: list[int] | None = None,
        output_type: IRType = ...,
        immediate: int | None = None,
        guard_type: IRType | None = None,
        label: str = "",
    ) -> None: ...
    def __repr__(self) -> str: ...

class IRProgram:
    ops: list[IROp]
    return_value: int | None
    num_params: int
    param_types: list[IRType]

    def __init__(
        self,
        ops: list[IROp],
        return_value: int | None,
        num_params: int,
        param_types: list[IRType],
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def dump(self) -> str: ...

def build_ir(trace: Trace) -> IRProgram: ...

# --- Codegen types ---

class CompiledFunction:
    def __repr__(self) -> str: ...
    def __call__(self, *args: Any) -> Any: ...

def compile_ir(program: IRProgram, func_name: str | None = None) -> CompiledFunction: ...
def compile_loop_ir(
    num_params: int,
    limit_param: int,
    num_locals: int,
    return_local: int,
    init_locals: list[tuple[int, int]],
    init_float_locals: list[tuple[int, float]],
    body_ops: list[tuple[str, int, int, int, bool, int]],
    local_types: list[int],
    param_types: list[int],
    return_type_id: int,
    func_name: str | None = None,
) -> CompiledFunction: ...
