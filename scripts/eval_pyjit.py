#!/usr/bin/env python3
"""
pyjit evaluation script.

Tests 30+ functions for correctness (JIT vs CPython output match)
and benchmarks representative functions for speedup measurement.
"""

import random
import statistics
import time
import traceback
from pyjit import jit
from pyjit.inspect import is_jit_compiled

# ============================================================
# PART 1: CORRECTNESS ORACLE (30+ functions)
# ============================================================

PASS = 0
FAIL = 0
DEOPT = 0  # fell back to CPython (correct but not compiled)

def check(name, fn, args_list, warmup=3):
    """Test @jit(fn) produces same output as fn for all args."""
    global PASS, FAIL, DEOPT
    jitted = jit(warmup=warmup)(fn)
    # Warmup with first args
    for _ in range(warmup + 1):
        try:
            jitted(*args_list[0])
        except Exception:
            pass

    compiled = is_jit_compiled(jitted)
    all_ok = True
    for args in args_list:
        try:
            expected = fn(*args)
        except Exception as e:
            # If CPython raises, JIT should too or deopt
            try:
                jitted(*args)
                # JIT didn't raise — that's a bug
                print(f"  FAIL {name}{args}: CPython raised {e}, JIT didn't")
                FAIL += 1
                all_ok = False
            except type(e):
                pass  # both raised, good
            continue

        try:
            got = jitted(*args)
        except Exception as e:
            print(f"  FAIL {name}{args}: JIT raised {e}")
            FAIL += 1
            all_ok = False
            continue

        if got != expected:
            print(f"  FAIL {name}{args}: expected {expected}, got {got}")
            FAIL += 1
            all_ok = False

    if all_ok:
        tag = "JIT" if compiled else "deopt"
        if not compiled:
            DEOPT += 1
        PASS += 1
        print(f"  OK   {name:40s} [{tag}]")
    return compiled


print("=" * 60)
print("CORRECTNESS: JIT vs CPython output matching")
print("=" * 60)

# --- Arithmetic (simple) ---
check("add", lambda a, b: a + b, [(1,2),(0,0),(-1,1),(10**6,10**6)])
check("sub", lambda a, b: a - b, [(10,3),(0,0),(-5,3),(100,-100)])
check("mul", lambda a, b: a * b, [(3,4),(0,100),(-3,7),(10**9,10**9)])
check("floor_div", lambda a, b: a // b, [(7,2),(10,3),(-7,2),(100,7)])
check("mod", lambda a, b: a % b, [(7,3),(10,5),(-7,3),(100,7)])
check("mixed_arith", lambda a, b, c: (a + b) * c - a, [(1,2,3),(0,0,0),(-5,10,3)])
check("multi_op", lambda a, b: a * b + a - b, [(3,4),(0,0),(-2,5)])

# --- Loops ---
def sum_range(n):
    s = 0
    for i in range(n):
        s += i
    return s

def sum_squares(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

def sum_cubes(n):
    total = 0
    for i in range(n):
        total += i * i * i
    return total

def countdown(n):
    s = 0
    for i in range(n):
        s += n - i
    return s

check("sum_range", sum_range, [(0,),(1,),(10,),(100,),(1000,)])
check("sum_squares", sum_squares, [(0,),(1,),(5,),(100,),(10000,)])
check("sum_cubes", sum_cubes, [(0,),(1,),(5,),(100,)])
check("countdown", countdown, [(0,),(1,),(10,),(100,)])

# --- Control flow ---
def abs_val(x):
    if x >= 0:
        return x
    return -x

def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

check("abs_val", abs_val, [(5,),(0,),(-3,),(10**9,)])
check("clamp", clamp, [(5,0,10),(-5,0,10),(15,0,10),(0,0,10)])
check("sign", sign, [(5,),(0,),(-3,)])

# --- Recursive ---
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

check("fib", fib, [(0,),(1,),(5,),(10,)])
check("factorial", factorial, [(0,),(1,),(5,),(10,)])

# --- String ops ---
check("str_concat", lambda a, b: a + b, [("hello"," world"),("","")])
check("str_repeat", lambda s, n: s * n, [("ab",3),("x",0),("",5)])
check("str_len", lambda s: len(s), [("hello",),("",),("a"*100,)])

# --- Collections ---
check("list_sum", lambda items: sum(items), [([1,2,3],),([],),([10],)])
check("list_len", lambda items: len(items), [([1,2,3],),([],)])
check("tuple_sum", lambda t: sum(t), [((1,2,3),),((10,),)])

# --- Edge cases ---
check("zero_div_guard", lambda a, b: a // b if b != 0 else 0, [(10,3),(10,0),(0,0)])
check("negative_range", sum_range, [(0,),(-5,)])

# --- Bigints ---
check("bigint_add", lambda a, b: a + b, [(10**30, 10**30), (2**63, 1)])
check("bigint_mul", lambda a, b: a * b, [(10**18, 10**18), (2**62, 4)])

# --- Type polymorphism ---
def poly_add(a, b):
    return a + b

check("poly_int", poly_add, [(1,2),(100,200)])
check("poly_float", poly_add, [(1.5,2.5),(0.0,0.0)])
check("poly_str", poly_add, [("a","b"),("hello"," world")])

# --- Randomized ---
random.seed(42)
def rand_arith(a, b, c, d):
    return ((a + b) * (c - d)) // max(abs(a), 1) + (b % max(abs(c), 1))

rand_args = [tuple(random.randint(-1000,1000) for _ in range(4)) for _ in range(50)]
check("fuzz_arith", rand_arith, rand_args)

def rand_loop(n, scale):
    total = 0
    for i in range(abs(n) % 200):
        total += i * scale
    return total

rand_loop_args = [(random.randint(0,500), random.randint(-100,100)) for _ in range(30)]
check("fuzz_loop", rand_loop, rand_loop_args)

print()
print(f"Results: {PASS} passed, {FAIL} failed, {DEOPT} fell back to CPython (deopt)")
print()

# ============================================================
# PART 2: BENCHMARKS
# ============================================================

def bench(name, fn, args, n_runs=50):
    """Benchmark fn vs @jit(fn). Returns (cpython_median_ns, jit_median_ns, speedup)."""
    jitted = jit(warmup=5)(fn)
    # Warmup JIT
    for _ in range(10):
        jitted(*args)

    # CPython
    cpython_times = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        fn(*args)
        cpython_times.append(time.perf_counter_ns() - start)

    # JIT
    jit_times = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        jitted(*args)
        jit_times.append(time.perf_counter_ns() - start)

    cp_med = statistics.median(cpython_times)
    jit_med = statistics.median(jit_times)
    speedup = cp_med / max(jit_med, 1)
    compiled = is_jit_compiled(jitted)
    return cp_med, jit_med, speedup, compiled

print("=" * 60)
print("BENCHMARKS: CPython vs @jit (median of 50 runs)")
print("=" * 60)
print(f"{'Function':30s} {'CPython':>12s} {'JIT':>12s} {'Speedup':>10s} {'Compiled':>10s}")
print("-" * 75)

results = []
benchmarks = [
    ("sum_range(100k)", sum_range, (100_000,)),
    ("sum_range(1M)", sum_range, (1_000_000,)),
    ("sum_squares(100k)", sum_squares, (100_000,)),
    ("sum_squares(1M)", sum_squares, (1_000_000,)),
    ("sum_cubes(100k)", sum_cubes, (100_000,)),
    ("countdown(1M)", countdown, (1_000_000,)),
    ("add(3,4)", lambda a, b: a + b, (3, 4)),
    ("mixed(2,3,4)", lambda a, b, c: (a+b)*c - a, (2, 3, 4)),
    ("fib(20)", fib, (20,)),
    ("str_concat", lambda a, b: a + b, ("hello", " world")),
]

for name, fn, args in benchmarks:
    cp, jt, sp, compiled = bench(name, fn, args)
    tag = "YES" if compiled else "no"
    cp_str = f"{cp/1e3:.1f}us" if cp > 1000 else f"{cp:.0f}ns"
    jt_str = f"{jt/1e3:.1f}us" if jt > 1000 else f"{jt:.0f}ns"
    print(f"{name:30s} {cp_str:>12s} {jt_str:>12s} {sp:>9.1f}x {tag:>10s}")
    results.append((name, sp, compiled))

print()
compiled_speedups = [sp for _, sp, c in results if c]
all_speedups = [sp for _, sp, _ in results]
if compiled_speedups:
    print(f"Compiled functions: avg={statistics.mean(compiled_speedups):.1f}x, "
          f"min={min(compiled_speedups):.1f}x, max={max(compiled_speedups):.1f}x")
print(f"All functions:      avg={statistics.mean(all_speedups):.1f}x, "
      f"min={min(all_speedups):.1f}x, max={max(all_speedups):.1f}x")

# ============================================================
# PART 3: COMPILATION COVERAGE
# ============================================================
print()
print("=" * 60)
print("COMPILATION COVERAGE")
print("=" * 60)

test_fns = [
    ("simple add", lambda a,b: a+b, (1,2)),
    ("simple mul", lambda a,b: a*b, (1,2)),
    ("3-arg expr", lambda a,b,c: (a+b)*c, (1,2,3)),
    ("4-arg expr", lambda a,b,c,d: a+b+c+d, (1,2,3,4)),
    ("for-range sum", sum_range, (10,)),
    ("for-range squares", sum_squares, (10,)),
    ("for-range cubes", sum_cubes, (10,)),
    ("while loop", lambda n: (lambda: [s:=0, [None for i in range(n) if not (s:=s+i)], s][-1])(), (10,)),
    ("nested for", lambda n: sum(i*j for i in range(n) for j in range(n)), (5,)),
    ("if/else", lambda x: 1 if x > 0 else -1, (5,)),
    ("recursive fib", fib, (10,)),
    ("string op", lambda s: s.upper(), ("hello",)),
    ("list comp", lambda n: [i*i for i in range(n)], (10,)),
    ("dict create", lambda: {"a": 1, "b": 2}, ()),
    ("float arith", lambda a,b: a+b, (1.5, 2.5)),
    ("mixed int/float", lambda a,b: a+b, (1, 2.5)),
]

compiled_count = 0
for name, fn, args in test_fns:
    jitted = jit(warmup=2)(fn)
    for _ in range(5):
        try:
            jitted(*args)
        except Exception:
            pass
    c = is_jit_compiled(jitted)
    compiled_count += c
    print(f"  {'[JIT]' if c else '[CPy]':6s} {name}")

print(f"\nCompiled: {compiled_count}/{len(test_fns)} ({100*compiled_count/len(test_fns):.0f}%)")
