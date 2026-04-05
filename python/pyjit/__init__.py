"""pyjit - A tracing JIT compiler for Python."""

from pyjit._pyjit import __version__
from pyjit.decorators import jit

__all__ = ["jit", "__version__"]
