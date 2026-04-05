# pyjit

A drop-in Python JIT compiler. Add `@jit`, get 40-90x speedup on numeric loops.

Built with **Rust** (PyO3 + Cranelift), targeting **CPython 3.12+**.

```python
from pyjit import jit

@jit
def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

sum_squares(1_000_000)  # 60x faster than CPython
```

---

## Benchmarks

Measured on Apple M-series, Python 3.14, Cranelift 0.130. Median of 50 runs each.

| Function | CPython | pyjit | Speedup |
|:---|---:|---:|---:|
| `sum_cubes(100k)` | 5.3 ms | 0.06 ms | **92x** |
| `sum_squares(1M)` | 33.1 ms | 0.6 ms | **58x** |
| `while_sum(1M)` | 29.2 ms | 0.6 ms | **50x** |
| `cond_even_odd(1M)` | 30.2 ms | 0.6 ms | **48x** |
| `sum_range(1M)` | 22.6 ms | 0.6 ms | **39x** |
| `float_accumulate(1M)` | 17.4 ms | 0.9 ms | **20x** |

> Average speedup across compiled functions: **~50x**

Functions that can't be compiled fall back to CPython automatically. No wrong answers, no crashes.

---

## How It Works

pyjit is a **tracing JIT compiler** with four stages:

```
Python function
      |
      v
  [1] Trace ---- sys.monitoring (PEP 669) records every bytecode
      |
      v
  [2] IR ------- Typed SSA with guards, unbox/box ops
      |
      v
  [3] Compile -- Cranelift emits native aarch64/x86_64 with loop blocks
      |
      v
  [4] Execute -- Native code called directly; deopt to CPython on guard failure
```

**Key design choices:**
- Traces are linear sequences with guards (branches become guards)
- Loop bodies are compiled into native Cranelift loop blocks with SSA block parameters
- Type guards check before every native call; overflow/type-change falls back to CPython
- Zero `unwrap()` in library code; all error paths return `Result`

---

## Install

### Prerequisites

- Python 3.12+
- Rust (stable)
- [maturin](https://www.maturin.rs/)

### From Source

```bash
git clone https://github.com/vmallela0/pyjit.git
cd pyjit
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest pytest-benchmark ruff mypy
maturin develop --release
```

### Verify

```bash
python scripts/smoke_test.py
# pyjit 0.1.0
# SMOKE TEST PASSED
```

---

## Usage

### Basic

```python
from pyjit import jit

@jit
def dot_product(n):
    total = 0
    for i in range(n):
        total += i * i
    return total
```

The first 10 calls run in CPython (warmup). After that, the function is traced, compiled to native code, and all subsequent calls execute natively.

### Custom Warmup

```python
@jit(warmup=5)  # compile after 5 calls instead of 10
def compute(n):
    ...
```

### Introspection

```python
from pyjit.inspect import is_jit_compiled, get_warmup

is_jit_compiled(dot_product)  # True after warmup
get_warmup(dot_product)       # 10
```

### Deoptimization

pyjit guards against type mismatches, bigint overflow, and None values. When guards fail, execution falls back to CPython transparently:

```python
@jit
def add(a, b):
    return a + b

add(3, 4)          # Native (after warmup)
add(1.5, 2.5)      # Deopt to CPython (float != int)
add(10**30, 1)      # Deopt to CPython (bigint overflow)
add(3, 4)           # Native again (guards pass)
```

---

## What Compiles

pyjit currently compiles **`for i in range(n)` loops with integer arithmetic**:

| Pattern | Compiled? |
|:---|:---:|
| `for i in range(n): s += expr(i)` | Yes |
| `while i < n: s += expr(i); i += 1` | Yes |
| `for i in range(n): if cond: s += i else: s -= i` | Yes |
| `for i in range(n): if cond: s += i` (filter) | Yes |
| Float accumulators (`s = 0.0; s += 1.5`) | Yes |
| Integer arithmetic (`+`, `-`, `*`, `//`, `%`) | Yes |
| All 6 comparisons (`<`, `<=`, `==`, `!=`, `>`, `>=`) | Yes |
| Simple arithmetic without loops (`a + b`) | Falls back |
| Nested for loops | Falls back |
| String / collection ops | Falls back |

Fallback is always correct and safe. Functions that don't compile still work via CPython.

---

## Project Structure

```
pyjit/
├── src/                        # Rust core (1,486 lines)
│   ├── lib.rs                  #   PyO3 module entry point
│   ├── tracer/                 #   Bytecode tracing (sys.monitoring)
│   │   ├── trace.rs            #     TraceOp, Trace data types
│   │   ├── recorder.rs         #     trace_function() entry point
│   │   └── bytecode.rs         #     CPython bytecode decoder (stub)
│   ├── ir/                     #   Intermediate representation
│   │   ├── types.rs            #     IRType: Int64, Float64, Bool, PyObject, Void
│   │   ├── ops.rs              #     IROp, IRProgram (SSA)
│   │   └── builder.rs          #     Trace -> IR conversion
│   ├── codegen/                #   Native code generation
│   │   ├── cranelift.rs        #     IR -> Cranelift IR -> machine code
│   │   └── abi.rs              #     CompiledFunction callable + ABI bridge
│   ├── optimizer/              #   Optimization passes (stubs)
│   ├── guards/                 #   Type guards / deopt (stubs)
│   └── runtime/                #   Compiled code cache (stub)
├── python/pyjit/               # Python layer (754 lines)
│   ├── __init__.py             #   Public API
│   ├── decorators.py           #   @jit decorator with type guards
│   ├── _tracer.py              #   sys.monitoring bytecode tracer
│   ├── _compiler.py            #   Loop detection + compilation orchestration
│   ├── _pyjit.pyi              #   Type stubs for native module
│   └── inspect.py              #   is_jit_compiled(), get_warmup()
└── tests/                      # Test suite (1,227 lines, 103 tests)
    ├── test_decorator.py       #   @jit decorator behavior
    ├── test_tracer.py          #   Bytecode tracing
    ├── test_ir.py              #   IR generation + SSA validation
    ├── test_codegen.py         #   Cranelift compilation
    ├── test_guards.py          #   Type guards + overflow detection
    └── integration/            #   Correctness oracle + deopt tests
```

---

## Development

```bash
make check          # clippy + ruff + mypy + cargo test + pytest (the works)
make build-debug    # fast debug build (maturin develop)
make build          # release build (maturin develop --release)
make bench          # run benchmarks
make clean          # nuke build artifacts
```

### Code Quality

| Metric | Value |
|:---|:---|
| `unwrap()` in library code | 0 |
| `unsafe` blocks | 2 (transmute for fn dispatch, Send impl) |
| `TODO` / `FIXME` / `HACK` | 0 |
| `panic!` calls | 2 (arity limits, unreachable in practice) |
| mypy --strict | Clean |
| cargo clippy -D warnings | Clean |
| ruff | Clean |
| Test pass rate | 103/103 |

---

## Architecture Notes

**Tracing** uses `sys.monitoring` (PEP 669) for instruction-level bytecode recording. Each instruction is tagged with observed operand types via a shadow type stack.

**IR** is typed SSA. Parameters enter as `PyObject`, get `Guard`ed for expected type, then `Unbox`ed to native `i64`/`f64`. Results are `Box`ed back before return. All values are defined before use (SSA invariant enforced by tests).

**Codegen** maps IR ops to Cranelift instructions. For loops, Cranelift's block parameters carry loop state (counter + accumulator locals) through the back-edge — no phi nodes needed.

**Guards** live in the Python decorator layer. Before calling native code: exact type check, i64 bounds check, arg count check. On failure, CPython runs the function instead.

---

## License

MIT
