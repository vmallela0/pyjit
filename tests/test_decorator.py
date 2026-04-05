"""Tests for the @jit decorator — Phase 0."""

from __future__ import annotations

import pyjit
from pyjit import jit
from pyjit.inspect import get_warmup, is_jit_compiled


class TestBareDecorator:
    """Test @jit used without parentheses."""

    def test_returns_correct_result(self) -> None:
        @jit
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_preserves_name(self) -> None:
        @jit
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    def test_preserves_docstring(self) -> None:
        @jit
        def my_func() -> None:
            """My docstring."""

        assert my_func.__doc__ == "My docstring."

    def test_not_yet_compiled(self) -> None:
        @jit
        def my_func() -> int:
            return 42

        assert not is_jit_compiled(my_func)

    def test_default_warmup(self) -> None:
        @jit
        def my_func() -> int:
            return 42

        assert get_warmup(my_func) == 10


class TestDecoratorWithArgs:
    """Test @jit(warmup=N) used with parentheses."""

    def test_custom_warmup(self) -> None:
        @jit(warmup=100)
        def my_func() -> int:
            return 42

        assert get_warmup(my_func) == 100

    def test_function_still_works(self) -> None:
        @jit(warmup=5)
        def multiply(a: int, b: int) -> int:
            return a * b

        assert multiply(3, 4) == 12

    def test_preserves_name_with_args(self) -> None:
        @jit(warmup=5)
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"


class TestNativeModule:
    """Test that the native _pyjit module loads correctly."""

    def test_version_exists(self) -> None:
        assert hasattr(pyjit, "__version__")

    def test_version_is_string(self) -> None:
        assert isinstance(pyjit.__version__, str)

    def test_version_value(self) -> None:
        assert pyjit.__version__ == "0.1.0"


class TestEdgeCases:
    """Edge cases for the decorator."""

    def test_kwargs(self) -> None:
        @jit
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        assert greet("World") == "Hello, World!"
        assert greet("World", greeting="Hi") == "Hi, World!"

    def test_returns_none(self) -> None:
        @jit
        def noop() -> None:
            pass

        assert noop() is None

    def test_recursive(self) -> None:
        @jit
        def fib(n: int) -> int:
            if n <= 1:
                return n
            return fib(n - 1) + fib(n - 2)

        assert fib(10) == 55

    def test_positional_and_keyword(self) -> None:
        @jit
        def fn(a: int, b: int, c: int = 0) -> int:
            return a + b + c

        assert fn(1, 2) == 3
        assert fn(1, 2, c=10) == 13
