"""Tests for type guards — Phase 3."""

from __future__ import annotations

from pyjit import jit
from pyjit.inspect import is_jit_compiled


class TestTypeGuards:
    """Verify type guards check argument types before calling native code."""

    def test_int_guard_passes(self) -> None:
        @jit(warmup=2)
        def fn(a: int, b: int) -> int:
            return a + b

        fn(1, 2)
        fn(3, 4)  # triggers compilation
        assert fn(5, 6) == 11  # uses compiled code

    def test_float_args_deopt(self) -> None:
        """Function compiled for int must handle float gracefully."""

        @jit(warmup=2)
        def fn(a: int, b: int) -> int:
            return a + b

        # Warm up with ints
        fn(1, 2)
        fn(3, 4)

        # Now call with floats — must deopt and still give correct result
        assert fn(1.5, 2.5) == 4.0  # type: ignore[arg-type]

    def test_none_arg_deopt(self) -> None:
        """Compiled code must handle None without crashing."""

        @jit(warmup=2)
        def fn(x: object) -> int:
            if x is None:
                return 0
            return int(x) + 1  # type: ignore[arg-type]

        fn(5)
        fn(10)

        # Call with None — must deopt, not crash
        assert fn(None) == 0

    def test_string_arg_deopt(self) -> None:
        """Compiled code must handle string args gracefully."""

        @jit(warmup=2)
        def fn(a: int, b: int) -> int:
            return a + b

        fn(1, 2)
        fn(3, 4)

        # String args — must deopt to CPython
        assert fn("hello", " world") == "hello world"  # type: ignore[arg-type]

    def test_mixed_types_deopt(self) -> None:
        """Int function called with float must deopt."""

        @jit(warmup=2)
        def fn(a: int, b: int) -> int:
            return a + b

        fn(1, 2)
        fn(3, 4)

        # Call with floats — must deopt to CPython
        result = fn(1.5, 2.5)  # type: ignore[arg-type]
        assert result == 4.0


class TestOverflowGuards:
    """Verify overflow detection for bigint arguments."""

    def test_small_ints_pass(self) -> None:
        @jit(warmup=2)
        def fn(a: int, b: int) -> int:
            return a + b

        fn(1, 2)
        fn(3, 4)
        assert fn(100, 200) == 300

    def test_bigint_deopt(self) -> None:
        """Big ints that overflow i64 must deopt to Python bigint."""

        @jit(warmup=2)
        def fn(a: int, b: int) -> int:
            return a * b

        fn(3, 4)
        fn(5, 6)

        # This overflows i64 — must deopt
        big = 10**30
        assert fn(big, big) == big * big

    def test_negative_bigint_deopt(self) -> None:
        """Negative bigints must also deopt."""

        @jit(warmup=2)
        def fn(a: int, b: int) -> int:
            return a + b

        fn(1, 2)
        fn(3, 4)

        big = -(2**63) - 1  # just below i64 min
        assert fn(big, 0) == big

    def test_i64_max_boundary(self) -> None:
        """Values exactly at i64 boundary should work natively."""

        @jit(warmup=2)
        def fn(a: int, b: int) -> int:
            return a + b

        fn(1, 2)
        fn(3, 4)

        # i64 max = 2^63 - 1, should fit
        i64_max = 2**63 - 1
        assert fn(i64_max, 0) == i64_max


class TestCompilationStatus:
    """Verify JIT compilation status tracking."""

    def test_not_compiled_before_warmup(self) -> None:
        @jit(warmup=100)
        def fn(x: int) -> int:
            return x + 1

        fn(1)
        assert not is_jit_compiled(fn)

    def test_compiled_after_warmup_with_loop(self) -> None:
        @jit(warmup=2)
        def fn(n: int) -> int:
            s = 0
            for i in range(n):
                s += i
            return s

        fn(10)
        fn(10)
        assert is_jit_compiled(fn)

    def test_non_compilable_stays_uncompiled(self) -> None:
        """Functions that can't be compiled should still work via CPython."""

        @jit(warmup=2)
        def fn(x: str) -> str:
            return x.upper()

        fn("hello")
        fn("world")
        assert not is_jit_compiled(fn)
        assert fn("test") == "TEST"
