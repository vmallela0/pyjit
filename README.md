# pyjit

A drop-in Python JIT compiler. Add `@jit`, get 40-100x speedup on numeric loops.

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
| `ray_tracer(800x800)` | 132.8 ms | 2.3 ms | **57x** |
| `while_sum(1M)` | 29.2 ms | 0.6 ms | **50x** |
| `nested_ij(1000)` | 29.0 ms | 0.6 ms | **49x** |
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
- Nested loops use recursive block generation — arbitrary nesting depth
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

### Loop Types

| Pattern | Compiled? |
|:---|:---:|
| `for i in range(n)` | Yes |
| `for i in range(100)` (literal limit) | Yes |
| `for i in range(10, n)` (start + stop) | Yes |
| `for i in range(0, n, 2)` (start + stop + step) | Yes |
| `while i < n: ... i += 1` | Yes |
| Nested loops (arbitrary depth) | Yes |

### Operations in Loop Bodies

| Pattern | Compiled? |
|:---|:---:|
| Arithmetic: `+`, `-`, `*`, `//`, `%`, `**` | Yes |
| Comparisons: `<`, `<=`, `==`, `!=`, `>`, `>=` | Yes |
| Unary: `-x`, `~x`, `not x` | Yes |
| Builtins: `abs()`, `min()`, `max()` | Yes |
| Conditionals: `if/else`, `if`-only (filter) | Yes |
| Float accumulators (`s = 0.0; s += 1.5`) | Yes |
| Constants in body (`dz = -500`) | Yes |

### Not Yet Supported (falls back to CPython)

| Pattern | Status |
|:---|:---|
| `break` / `continue` | Planned |
| Chained comparisons (`10 <= x < 20`) | Planned |
| List/dict operations | Falls back |
| String operations | Falls back |
| Arbitrary function calls | Falls back |

Fallback is always correct and safe. Functions that don't compile still work via CPython.

---

## Real-World Example: Ray Tracer

A sphere intersection ray tracer with 7 parameters, 18 locals, complex arithmetic, and conditionals — compiled at **57x speedup**:

```python
@jit
def render_scanline(width, height, row, sx, sy, sz, sr_sq):
    total = 0
    for col in range(width):
        dx = col - width // 2
        dy = row - height // 2
        dz = -500
        len_sq = dx*dx + dy*dy + dz*dz
        b_val = -2 * (dx*sx + dy*sy + dz*sz)
        c_coeff = sx*sx + sy*sy + sz*sz - sr_sq
        disc = b_val*b_val - 4*len_sq*c_coeff
        if disc > 0:
            total += 128
    return total
```

| Resolution | CPython | pyjit | Speedup |
|:---|---:|---:|---:|
| 200x200 | 14.5 ms | 0.3 ms | **43x** |
| 400x400 | 34.1 ms | 0.7 ms | **46x** |
| 800x800 | 132.8 ms | 2.3 ms | **57x** |

---

## Project Structure

```
pyjit/
├── src/                        # Rust core
│   ├── lib.rs                  #   PyO3 module entry point
│   ├── tracer/                 #   Bytecode tracing (sys.monitoring)
│   ├── ir/                     #   Intermediate representation (typed SSA)
│   ├── codegen/                #   Native code generation (Cranelift)
│   ├── optimizer/              #   Optimization passes (stubs)
│   ├── guards/                 #   Type guards / deopt (stubs)
│   └── runtime/                #   Compiled code cache (stub)
├── python/pyjit/               # Python layer
│   ├── __init__.py             #   Public API
│   ├── decorators.py           #   @jit decorator with type guards
│   ├── _tracer.py              #   sys.monitoring bytecode tracer
│   ├── _compiler.py            #   Loop detection + compilation orchestration
│   ├── _pyjit.pyi              #   Type stubs for native module
│   └── inspect.py              #   is_jit_compiled(), get_warmup()
└── tests/                      # Test suite (126 tests)
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
| `panic!` calls | 2 (arity limits, guarded in practice) |
| mypy --strict | Clean |
| cargo clippy -D warnings | Clean |
| ruff | Clean |
| Test pass rate | 126/126 |

---

## Architecture Notes

**Tracing** uses `sys.monitoring` (PEP 669) for instruction-level bytecode recording. Each instruction is tagged with observed operand types via a shadow type stack.

**IR** is typed SSA. Parameters enter as `PyObject`, get `Guard`ed for expected type, then `Unbox`ed to native `i64`/`f64`. Results are `Box`ed back before return. All values are defined before use (SSA invariant enforced by tests).

**Codegen** maps IR ops to Cranelift instructions. Loops use Cranelift's block parameters to carry state (counter + locals) through the back-edge. Nested loops are handled recursively via `LoopStart`/`LoopEnd` markers — each level creates its own header/body/exit block triple. Conditionals use `CondStart`/`CondElse`/`CondEnd` with proper SSA merge blocks.

**Guards** live in the Python decorator layer. Before calling native code: exact type check, i64 bounds check, arg count check. On failure, CPython runs the function instead.

---

## License

MIT
