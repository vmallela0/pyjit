# pyjit: Prototype to Product — 12-Month Roadmap

## Context

pyjit is a 3,400-line tracing JIT compiler (Rust/PyO3/Cranelift) that gets 20-113x speedup on numeric Python loops. It works today on one machine (macOS ARM64, Python 3.14) with no packaging. The goal: in 12 months, `pip install pyjit` works everywhere and `@jit` compiles 60-80% of numeric hot loops without modification.

The competitive edge over Numba: zero config (no type annotations, no nopython mode), instant compilation (0.6ms vs 86ms), no LLVM dependency (5MB wheel vs 100MB), and fast CPython version tracking.

---

## Q1 (Months 1-3): "It Works on Real Code"

The #1 bail-out in real code is `data[i]` — list/array indexing inside loops. Without it, pyjit only compiles functions that compute from counters. Fix that, ship an alpha.

### Month 1: CI + PyPI + List Read

**1.1 GitHub Actions CI** (3 days)
- Matrix: Python {3.12, 3.13, 3.14} x {macOS-arm64, ubuntu-x86_64}
- `maturin generate-ci github` as baseline, add `make check`
- Files: `.github/workflows/ci.yml`

**1.2 PyPI alpha** (2 days)
- Proper pyproject.toml metadata (author, license-file, readme, classifiers)
- `maturin publish` to PyPI as `pyjit==0.1.0a1`
- macOS arm64 + Linux x86_64 wheels initially

**1.3 `data[i]` — list read in loops** (2 weeks) **[HIGHEST IMPACT]**
- `_compiler.py`: handle `BINARY_SUBSCR` opcode → emit `LoadElement` body op
- `cranelift.rs`: `LoadElement` → Cranelift `load` from base pointer + index * 8
- `decorators.py`: for list params, extract internal `ob_item` pointer via ctypes, pass as i64
- Guard: bounds-check index against list length, check list identity hasn't changed
- Unlocks: `for i in range(n): total += data[i]` — the most common numeric pattern

### Month 2: List Write + NumPy + ABI Fix

**2.1 `data[i] = x` — list write in loops** (1 week)
- `STORE_SUBSCR` → `StoreElement` body op → Cranelift `store`

**2.2 NumPy array indexing** (2 weeks)
- Detect `numpy.ndarray` params, extract data pointer via `__array_interface__`
- Emit typed loads/stores based on dtype (float64, int64)
- Guard: contiguous C-order, dtype match

**2.3 Fix ABI — eliminate panic on >8 args** (1 week)
- Replace hardcoded transmute dispatch with pointer-to-args-buffer approach
- All args passed as `*const u64` + count, Cranelift function unpacks

### Month 3: Break + Optimizer + Alpha Polish

**3.1 `break` with early return** (1.5 weeks)
- Emit jump to loop exit block on break condition
- Propagate return value through exit path

**3.2 Constant folding + DCE** (1 week)
- Fill in `optimizer/const_fold.rs` and `optimizer/dce.rs`
- Standard SSA passes, wire into compilation pipeline

**3.3 Correctness fuzzer** (1 week)
- Hypothesis-based: generate random numeric functions, assert JIT == CPython
- Fill in empty test stubs

**Q1 exit:** `pip install pyjit==0.1.0a3`. List/numpy indexing. ~200 tests. CI on 2 platforms x 3 Python versions.

---

## Q2 (Months 4-6): "It's Useful"

### Month 4: Function Inlining

**4.1 Inline simple pure functions in loops** (2.5 weeks)
```python
def helper(x, y): return x*x + y*y

@jit
def fn(data, n):
    s = 0
    for i in range(n):
        s += helper(data[i], data[i+1])
    return s
```
- Resolve callable in `CALL`, get its `__code__`, verify it's simple (no loops/side effects)
- Inline by substituting callee bytecode into caller's body ops
- Guard: function identity hasn't changed

**4.2 `math` module builtins** (1 week)
- `math.sqrt`, `math.sin`, `math.cos`, `math.exp`, `math.log`
- Cranelift `sqrt` instruction + libm calls for trig/exp/log

**4.3 Python 3.12/3.13 compatibility** (1 week)
- Version-gate bytecode encodings that changed (COMPARE_OP, superinstructions)
- CI green on all three versions

### Month 5: Code Cache + Performance

**5.1 Compiled code cache to disk** (2 weeks)
- Hash: `co_code` + `co_consts` + arg types → cache key
- Store compiled machine code in `~/.cache/pyjit/<hash>.bin`
- mmap with exec permission on cache hit
- Invalidate on source change or pyjit version change
- Fill in `runtime/cache.rs`

**5.2 `@jit(eager=True)` — compile on first call** (2 days)
- Skip warmup for functions where compilation is cheap

**5.3 Benchmark regression tracking** (1 week)
- Fill in benchmark stubs, CI job for perf tracking

### Month 6: Docs + Windows + Beta

**6.1 Documentation site** (1.5 weeks)
- mkdocs: quickstart, API ref, supported patterns, Numba comparison, architecture

**6.2 Windows x86_64 support** (1 week)
- Add to CI, fix platform issues, publish wheel

**6.3 Beta release `0.2.0b1`** (0.5 weeks)

**Q2 exit:** 6 platform/Python combos. Function inlining. math builtins. Code cache. Docs. ~300 tests. 40-50% numeric coverage.

---

## Q3 (Months 7-9): "It's Reliable"

### Month 7: Guard Hardening

**7.1 Proper type guard framework in Rust** (2 weeks)
- `guards/types.rs`: structured guard representation
- `guards/deopt.rs`: deopt machinery (entry-guards first, mid-loop deopt in Q4)

**7.2 Integer overflow detection in compiled code** (1 week)
- Cranelift `iadd_cout` / manual overflow check after arithmetic
- Branch to deopt path on overflow

**7.3 Fuzz testing** (1 week)
- libFuzzer for Rust compiler core
- Nightly CI job

### Month 8: Diagnostics + Platform Expansion

**8.1 `@jit(explain=True)` — compilation diagnostics** (1.5 weeks)
- When compilation fails, tell the user WHY and WHERE
- Annotate all 46 bail-out points with structured error messages

**8.2 macOS x86_64 + Linux aarch64** (1 week)
- CI expansion, platform-specific fixes

**8.3 Chained comparisons** (1 week)
- Recognize `COMPARE_OP + COPY + POP_JUMP_IF_FALSE + COMPARE_OP` pattern

### Month 9: Performance + RC

**9.1 SIMD vectorization exploration** (2 weeks, OPTIONAL)
- Manual vectorization for simple accumulator patterns using Cranelift `i64x2`/`f64x2`
- Cut if ROI is low

**9.2 Release candidate `0.9.0rc1`** (1 week)

**Q3 exit:** Overflow detection. Diagnostic mode. All 6 platforms. Fuzz testing. >300 tests. RC published.

---

## Q4 (Months 10-12): "pip install and forget"

### Month 10: Advanced Features

**10.1 `@jit` on class methods** (1 week)
- Recognize `self.attr` reads, strip or pass `self`

**10.2 Mid-loop deoptimization** (2 weeks, CAN DEFER)
- Save SSA state at deopt points, reconstruct Python frame, resume interpreter
- The hardest single feature. Defer to post-1.0 if needed.

**10.3 Profiling hooks** (1 week)
- `@jit(profile=True)`: compilation time, deopt count, speedup estimate

### Month 11: Stability

**11.1 Thread safety audit** (1 week)
**11.2 Common patterns: enumerate, zip, sum, tuple unpacking** (2 weeks)
**11.3 `PYJIT_DISABLE=1` env var, error recovery** (1 week)

### Month 12: Ship 1.0

**12.1 1.0.0 release** — wheels for all platforms, semver promise
**12.2 CONTRIBUTING.md + architecture docs** — enable external contributors
**12.3 Coverage validation** — test against 100 real-world numeric functions, target 60-80%

---

## What Gets Cut If Behind Schedule

1. Mid-loop deoptimization (Month 10) — entry guards are sufficient for 1.0
2. SIMD vectorization (Month 9) — nice-to-have, not essential
3. enumerate/zip (Month 11) — workaround: use range(len())
4. Class method support (Month 10) — workaround: standalone function

## What Cannot Be Cut

1. List/array indexing (Month 1-2) — without this, pyjit is a toy
2. CI + PyPI (Month 1) — no distribution = no users
3. Python 3.12-3.14 compat (Month 4) — single-version = no adoption
4. Code cache (Month 5) — recompilation on every restart = unacceptable UX
5. Diagnostic mode (Month 8) — "it silently falls back and I don't know why" = users leave

## The One-Line Pitch at 1.0

"pip install pyjit, add @jit to your numeric functions, get 20-100x speedup with zero configuration and instant compilation."
