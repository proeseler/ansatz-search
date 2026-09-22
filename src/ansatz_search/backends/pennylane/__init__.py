"""PennyLane backend (optional dependency: pip install ansatz-search[pennylane])."""

from ansatz_search.backends.base import register_providers

from .compiler import PennyLaneCompiler, PennyLaneProgram
from .gradients import PennyLaneGradientProvider
from .states import PennyLaneStateProvider

register_providers(PennyLaneProgram, gradient=PennyLaneGradientProvider, state=PennyLaneStateProvider)

__all__ = ["PennyLaneCompiler", "PennyLaneProgram", "PennyLaneGradientProvider", "PennyLaneStateProvider"]
