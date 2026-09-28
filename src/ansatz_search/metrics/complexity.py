"""Complexity: how large an ansatz is, from its parameter count, gate count and depth."""

from __future__ import annotations

from .base import Metric


class Complexity(Metric):
    """Size of an ansatz: distinct parameters + gates + depth.

    Computed from the backend-neutral AnsatzSpec, so it needs no simulation and
    works with every backend. `value` is the raw count; `cost` divides it by
    ``max_param + max_gates + max_depth``, clipped to [0, 1], so smaller
    circuits cost less. Set the maxima to the search limits (SearchProblem's
    max_param, max_gates, max_depth) to use the full [0, 1] range.
    """

    label = "Complexity (params + gates + depth)"

    def __init__(self, max_param: int = 10, max_gates: int = 20, max_depth: int = 10):
        if min(max_param, max_gates, max_depth) <= 0:
            raise ValueError("max_param, max_gates and max_depth must be positive.")
        super().__init__()
        self.max_param = max_param
        self.max_gates = max_gates
        self.max_depth = max_depth

    @property
    def name(self) -> str:
        return "complexity"

    def value(self, qc, spec=None) -> float:
        if spec is None:
            raise ValueError("Complexity is computed from the AnsatzSpec; pass `spec`.")
        num_params = len({ref.index for block in spec for ref in block.param_refs})
        return float(num_params + len(spec) + spec.depth)

    def cost(self, value: float) -> float:
        return float(min(max(value / (self.max_param + self.max_gates + self.max_depth), 0.0), 1.0))
