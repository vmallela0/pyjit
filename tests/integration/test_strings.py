"""Integration tests verifying string-producing functions gracefully fall back.

pyjit only compiles numeric (int/float) loops. Functions that work on strings,
lists of strings, or other non-numeric types must fall back to CPython silently.
"""

from __future__ import annotations

from pyjit import jit


def test_string_function_fallback() -> None:
    """String operations must fall back to CPython — no crash, correct result."""

    @jit(warmup=1)
    def fn(s: str, n: int) -> str:
        return s * n

    assert fn("ab", 3) == "ababab"
    assert fn("x", 0) == ""


def test_list_of_strings_fallback() -> None:
    @jit(warmup=1)
    def fn(words: list[str]) -> str:
        return " ".join(words)

    result = fn(["hello", "world"])
    assert result == "hello world"


def test_mixed_types_fallback() -> None:
    """A function mixing numeric and string ops falls back correctly."""

    @jit(warmup=1)
    def fn(n: int) -> str:
        return str(n * 2)

    assert fn(21) == "42"
    assert fn(0) == "0"
