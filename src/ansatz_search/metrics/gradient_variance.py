from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ansatz_search.backends.base import GradientProvider, default_provider

from .base import SampledMetric
from ..utils.parameter_sampler import ParameterSampler
from ..utils.statevector_sampler import StateSampler


class GradientVariance(SampledMetric):
    """Expected gradient variance of a parameterized circuit.

    `value` is the raw expected variance (higher is better: further from a
    barren plateau). `cost` normalizes it to ``max(1 - variance / max_variance, 0)``,
    which rises towards 1 as the variance vanishes. Gradients are clipped to
    [-1, 1], which bounds the variance by 1, hence the default `max_variance`.
    A "trainable enough" threshold belongs to the cost function, not here.
    """

    label = "Gradient variance"
    higher_is_better = True

    def __init__(
        self,
        # In the format the gradient provider expects: a Pauli sum such as
        # {"ZZZZ": 1.0} for the NumPy/PennyLane backends.
        observables: Any,
        num_of_obs: int = 1,
        parameter_sampler: Union[str, ParameterSampler] = "gaussian",
        num_of_param_samples: int = 11806,
        state_sampler: Union[str, StateSampler] = None,
        num_of_state_samples: int = 1,
        # None: the provider of the backend that compiled the circuit (NumPy by default).
        gradient_provider: GradientProvider | None = None,
        max_workers: int = 32,
        seed: int = None,
        max_variance: float = 1.0,
    ):
        if not max_variance > 0:
            raise ValueError("max_variance must be positive.")
        super().__init__(
            parameter_sampler=parameter_sampler,
            num_of_param_samples=num_of_param_samples,
            state_sampler=state_sampler,
            num_of_state_samples=num_of_state_samples,
            max_workers=max_workers,
            seed=seed,
        )
        self._obs_arg = observables
        self._num_obs = num_of_obs
        self._gradient_provider = gradient_provider
        self.max_variance = max_variance

    @property
    def name(self) -> str:
        return "gradient_variance"

    def cost(self, value: float) -> float:
        # ddof=1 can push the variance slightly above max_variance; never go negative.
        return max(1.0 - value / self.max_variance, 0.0)

    def value_from_samples(self, ansatz: QuantumCircuit, param_samples: np.ndarray, stv_vecs: Sequence[Statevector]) -> float:
        """Expected gradient variance over parameter samples, averaged over everything else."""
        provider = default_provider(ansatz, "gradient") if self._gradient_provider is None else self._gradient_provider
        grads = provider.gradients(
            ansatz,
            observables=self._obs_arg,
            param_samples=param_samples,
            states=stv_vecs,
        )
        np.clip(grads, -1, 1, out=grads)

        grad_var = np.var(grads, ddof=1, axis=2)
        exp_grad = np.mean(grad_var, axis=2)
        exp_grad = np.mean(exp_grad, axis=1)
        exp_grad = np.mean(exp_grad, axis=0)
        return float(exp_grad)
