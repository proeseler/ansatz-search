"""Batched NumPy statevector backend: fast for small circuits, exact adjoint gradients. The default backend."""

from ansatz_search.backends.base import register_providers

from .compiler import NumpyCompiler, NumpyProgram
from .gradients import NumpyGradientProvider
from .states import NumpyStateProvider

register_providers(NumpyProgram, gradient=NumpyGradientProvider, state=NumpyStateProvider)

__all__ = ["NumpyCompiler", "NumpyProgram", "NumpyGradientProvider", "NumpyStateProvider"]
