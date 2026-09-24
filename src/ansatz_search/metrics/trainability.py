"""Trainability: the gradient-to-noise ratio of an ansatz (arXiv:2603.14451, Sec. II B)."""

from __future__ import annotations

from typing import Any, Sequence, Union

from ansatz_search.backends.base import GradientProvider

from .base import Metric
from .gate_error import GateError
from .gradient_variance import GradientVariance
from ..utils.parameter_sampler import ParameterSampler
from ..utils.statevector_sampler import StateSampler


class Trainability(Metric):
    """Gradient variance per unit of gate error: ``ratio = Var[dC] / Pr(err)``.

    Var[dC] is GradientVariance's value (clipped gradients, unbiased sample
    variance, averaged over parameters, observables and initial states) and
    Pr(err) is GateError's value. A ratio above a threshold tau_BP means that
    gradient fluctuations exceed the expected noise level, i.e. the circuit
    avoids a (noise-induced) barren plateau.

    `value` is the raw ratio (higher is better). `cost` is
    ``max(1 - ratio / max_ratio, 0)``. For the paper's trainability loss
    (Eq. 20), ``max(1 - ratio / tau_BP, 0)``, set the threshold on the cost
    function, e.g. ``thresholds={"trainability": 2.5}``; `max_ratio` must be at
    least that threshold. The paper uses tau_BP between 0.083 and 8.

    The defaults follow the paper: parameters uniform in [0, 2pi), 11,806
    samples, p1_err=0.001, p2_err=0.01, gates counted as written. The paper
    estimated gradients by finite differences (step 1e-7); the backends here
    compute them exactly.
    """

    label = "Trainability (gradient-to-noise ratio)"
    higher_is_better = True

    def __init__(
        self,
        # In the format the gradient provider expects, e.g. "ZIII" (Z on qubit 0).
        observables: Any,
        num_of_obs: int = 1,
        parameter_sampler: Union[str, ParameterSampler] = "uniform",
        num_of_param_samples: int = 11806,
        state_sampler: Union[str, StateSampler] = None,
        num_of_state_samples: int = 1,
        # None: the provider of the backend that compiled the circuit (NumPy by default).
        gradient_provider: GradientProvider | None = None,
        p1_err: float = 0.001,
        p2_err: float = 0.01,
        basis_gates: Sequence[str] | None = None,
        optimization_level: int = 1,
        max_ratio: float = 10.0,
        seed: int = None,
    ):
        if not max_ratio > 0:
            raise ValueError("max_ratio must be positive.")
        self._gradient_variance = GradientVariance(
            observables=observables,
            num_of_obs=num_of_obs,
            parameter_sampler=parameter_sampler,
            num_of_param_samples=num_of_param_samples,
            state_sampler=state_sampler,
            num_of_state_samples=num_of_state_samples,
            gradient_provider=gradient_provider,
            seed=seed,
        )
        self._gate_error = GateError(p1_err=p1_err, p2_err=p2_err, basis_gates=basis_gates,
                                     optimization_level=optimization_level)
        self.max_ratio = max_ratio

    @property
    def name(self) -> str:
        return "trainability"

    def value(self, qc, spec=None) -> float:
        error = self._gate_error.value(qc, spec)  # cheap: fail before computing gradients
        if error <= 0:
            raise ValueError("The gate error is 0 (no gates, or p1_err = p2_err = 0), "
                             "so the gradient-to-noise ratio is undefined.")
        return self._gradient_variance.value(qc, spec) / error

    def cost(self, value: float) -> float:
        return max(1.0 - value / self.max_ratio, 0.0)
