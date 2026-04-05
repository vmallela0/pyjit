#!/usr/bin/env python3
"""Smoke test: verify pyjit loads and @jit is a no-op passthrough."""

import pyjit
from pyjit import jit


@jit
def add(a: int, b: int) -> int:
    return a + b


assert add(2, 3) == 5
assert add.__name__ == "add"
print(f"pyjit {pyjit.__version__}")
print("SMOKE TEST PASSED")
