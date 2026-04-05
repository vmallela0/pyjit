# pyjit

A drop-in tracing JIT compiler for Python. Add `@jit`, get 20-113x speedup on numeric loops.

Built with **Rust** ([PyO3](https://pyo3.rs) + [Cranelift](https://cranelift.dev)), targeting **CPython 3.12+**.

```python
from pyjit import jit

@jit
def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

sum_squares(1_000_000)  # 59x faster than CPython
```

If a function can't be compiled, it falls back to CPython automatically. No wrong answers, no crashes.

---

## Benchmarks

Apple M-series, Python 3.14, Cranelift 0.130, release build. Median of 100 runs.

### Numeric Loops

| Benchmark | CPython | pyjit | Speedup |
|:---|---:|---:|---:|
| `i ** 3` sum (1M iters) | 64.6 ms | 0.57 ms | **113x** |
| `i * i * i` sum (1M iters) | 59.1 ms | 0.60 ms | **99x** |
| `abs(i - n//2)` sum (1M iters) | 48.1 ms | 0.57 ms | **84x** |
| `i * i` sum (1M iters) | 33.6 ms | 0.57 ms | **59x** |
| `while i < n: s += i` (1M iters) | 29.1 ms | 0.57 ms | **51x** |
| `s += i` sum (1M iters) | 23.1 ms | 0.57 ms | **40x** |

### Nested Loops

| Benchmark | CPython | pyjit | Speedup |
|:---|---:|---:|---:|
| `s += i * j`, 1000x1000 | 28.2 ms | 0.58 ms | **49x** |
| `s += i * j`, 500x500 | 6.6 ms | 0.15 ms | **45x** |

### Conditionals + Control Flow

| Benchmark | CPython | pyjit | Speedup |
|:---|---:|---:|---:|
| `if i % 2 == 0: s += i else: s -= i` (1M) | 30.7 ms | 1.1 ms | **27x** |
| `if i % 3 == 0: s += i` filter (1M) | 28.1 ms | 1.1 ms | **26x** |
| `if i % 2 == 0: continue` (1M) | 30.0 ms | 1.1 ms | **26x** |
| Float accumulator `s += 1.5` (1M) | 17.7 ms | 0.86 ms | **21x** |

### Ray Tracer (sphere intersection + shading, 7 params, 18 locals)

| Resolution | CPython | pyjit | Speedup |
|:---|---:|---:|---:|
| 200x200 (nested loop, full frame) | 8.5 ms | 0.11 ms | **78x** |
| 400x400 | 34.3 ms | 0.45 ms | **76x** |
| Single scanline, 800 pixels | 170 us | 3.0 us | **57x** |

### vs Numba

On a real workload (ray tracer, 800 pixels), pyjit is **3.4x slower** than Numba at execution but **146x faster** at compilation:

| | pyjit | Numba |
|:---|---:|---:|
| Execution (ray tracer) | 3.0 us | 0.9 us |
| First compile time | **0.6 ms** | 86 ms |

Numba's LLVM backend auto-vectorizes (SIMD), giving it a large edge on simple arithmetic loops. On complex loops with branches, the gap narrows to 3-6x. pyjit's Cranelift backend prioritizes compilation speed over peak throughput.

---

## How It Works

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
  [4] Execute -- Native code called directly; deopt on guard failure
```

- Loop bodies compile to Cranelift loop blocks with SSA block parameters
- Nested loops use recursive block generation (arbitrary depth)
- Conditionals use branch blocks with proper SSA merge
- Type guards (exact type match + i64 bounds) check before every native call
- Guard failure transparently falls back to CPython
- Zero `unwrap()` in library code

---

## Install

```bash
git clone https://github.com/vmallela0/pyjit.git
cd pyjit
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest pytest-benchmark ruff mypy
maturin develop --release
python scripts/smoke_test.py  # verify
```

Requires Python 3.12+, Rust stable, and [maturin](https://www.maturin.rs/).

---

## Usage

```python
from pyjit import jit

@jit
def compute(n):
    s = 0
    for i in range(n):
        s += i * i
    return s

# First 10 calls: CPython (warmup)
# After warmup: compiled to native, all subsequent calls run natively
```

### Options

```python
@jit(warmup=5)  # compile after 5 calls instead of 10
def compute(n): ...
```

### Introspection

```python
from pyjit.inspect import is_jit_compiled, get_warmup

is_jit_compiled(compute)  # True after warmup
get_warmup(compute)       # 10
```

### Safe Deoptimization

```python
@jit
def add(a, b):
    return a + b

add(3, 4)        # native (int + int)
add(1.5, 2.5)    # deopt to CPython (float args)
add(10**30, 1)    # deopt to CPython (bigint overflow)
add(3, 4)        # native again
```

---

## What Compiles

### Loop Types

| Pattern | Status |
|:---|:---:|
| `for i in range(n)` | Compiled |
| `for i in range(100)` (literal) | Compiled |
| `for i in range(10, n)` (start, stop) | Compiled |
| `for i in range(0, n, 2)` (start, stop, step) | Compiled |
| `while i < n: ... i += 1` | Compiled |
| Nested loops (arbitrary depth) | Compiled |
| `continue` in loop body | Compiled |

### Operations in Loop Bodies

| Pattern | Status |
|:---|:---:|
| `+`, `-`, `*`, `//`, `%`, `**` | Compiled |
| `<`, `<=`, `==`, `!=`, `>`, `>=` | Compiled |
| `-x`, `~x`, `not x` | Compiled |
| `abs()`, `min()`, `max()` | Compiled |
| `if`/`else`, `if`-only filters | Compiled |
| Float accumulators | Compiled |
| Constants in body (`dz = -500`) | Compiled |
| Division-by-zero (guarded, no crash) | Compiled |

### Falls Back to CPython

| Pattern | Reason |
|:---|:---|
| `break` with early return | Return inside loop not compiled |
| Chained comparisons (`10 <= x < 20`) | Complex bytecode pattern |
| List / dict / set operations | Requires boxing |
| String operations | Not numeric |
| Arbitrary function calls in body | Only `abs`/`min`/`max` inlined |
| Classes, closures, generators | Out of scope |

Fallback is always correct. Functions that can't compile still produce the right answer via CPython.

---

## Project Structure

```
pyjit/
├── src/                        # Rust (~1,700 lines)
│   ├── lib.rs                  #   PyO3 module entry
│   ├── tracer/                 #   Bytecode tracing (sys.monitoring)
│   ├── ir/                     #   Typed SSA intermediate representation
│   ├── codegen/                #   Cranelift native code generation
│   │   ├── cranelift.rs        #     Loop compiler with nested/conditional blocks
│   │   └── abi.rs              #     Native call dispatch (up to 8 params)
│   ├── optimizer/              #   (stubs)
│   ├── guards/                 #   (stubs)
│   └── runtime/                #   (stubs)
├── python/pyjit/               # Python (~900 lines)
│   ├── decorators.py           #   @jit with warmup, type guards, deopt
│   ├── _compiler.py            #   Bytecode analysis, loop/body extraction
│   ├── _tracer.py              #   sys.monitoring instruction recorder
│   └── inspect.py              #   is_jit_compiled(), get_warmup()
└── tests/                      # 130 tests
    ├── test_codegen.py         #   Compilation + correctness
    ├── test_guards.py          #   Type guards + overflow
    ├── test_tracer.py          #   Bytecode tracing
    ├── test_ir.py              #   SSA IR generation
    └── integration/            #   Correctness oracle + deopt
```

---

## Development

```bash
make check       # clippy + ruff + mypy --strict + cargo test + pytest
make build       # maturin develop --release
make build-debug # maturin develop (faster iteration)
make clean       # rm -rf target/ dist/
```

| Metric | Value |
|:---|:---|
| `unwrap()` in library code | 0 |
| `unsafe` blocks | 2 (justified) |
| `TODO` / `FIXME` / `HACK` | 0 |
| mypy --strict | Clean |
| clippy -D warnings | Clean |
| Tests | 130/130 |

---

## License

MIT
