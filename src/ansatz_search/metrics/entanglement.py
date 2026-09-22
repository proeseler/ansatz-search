"""Entanglement capability: average Meyer-Wallach entanglement of an ansatz's output states."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from ansatz_search.backends.base import StateProvider, default_provider

from .base import SampledMetric
from ..utils.parameter_sampler import ParameterSampler
from ..utils.statevector_sampler import StateSampler


def meyer_wallach(states: np.ndarray, num_qubits: int) -> np.ndarray:
    """Meyer-Wallach Q for a batch of pure states, shape (B, 2**n) -> (B,).

    Q = 2 * (1 - mean_k Tr(rho_k^2)), where rho_k is the reduced state of
    qubit k. Vectorized: rho_k comes from splitting each amplitude index at
    bit k, without any partial trace. Q is 0 for product states and 1 for
    e.g. GHZ states.
    """
    psi = np.asarray(states, dtype=complex)
    B = psi.shape[0]
    purity = np.zeros(B)
    for k in range(num_qubits):
        # (bits above k, bit k, bits below k); qubit k is bit 1 << k
        v = psi.reshape(B, 2 ** (num_qubits - 1 - k), 2, 2 ** k)
        psi0, psi1 = v[:, :, 0, :], v[:, :, 1, :]
        rho00 = np.sum(np.abs(psi0) ** 2, axis=(1, 2))
        rho11 = np.sum(np.abs(psi1) ** 2, axis=(1, 2))
        rho01 = np.sum(psi0 * np.conj(psi1), axis=(1, 2))
        purity += rho00 ** 2 + rho11 ** 2 + 2 * np.abs(rho01) ** 2  # Tr(rho_k^2)
    # Clip rounding noise (e.g. -4e-17 for product states) back into [0, 1].
    return np.clip(2 * (1 - purity / num_qubits), 0.0, 1.0)


class Entanglement(SampledMetric):
    """Average Meyer-Wallach entanglement Q over parameter samples (and initial states).

    `value` is the raw average Q in [0, 1] (higher is more entangling). `cost`
    is ``max(1 - Q / reference, 0)``: 0 once the ansatz reaches `reference`,
    1 for an ansatz that creates no entanglement. By default `reference` is
    the Haar-random average (D - 2) / (D + 1), D = 2**num_qubits;
    ``reference=1.0`` gives ``1 - Q``.
    """

    label = "Entanglement (Meyer–Wallach Q)"
    higher_is_better = True

    def __init__(
        self,
        num_qubits: int,
        # None: the provider of the backend that compiled the circuit (NumPy by default).
        state_provider: StateProvider | None = None,
        parameter_sampler: Union[str, ParameterSampler] = "uniform",
        num_of_param_samples: int = 1000,
        state_sampler: Union[str, StateSampler] = None,
        num_of_state_samples: int = 1,
        reference: float | None = None,
        seed: int = None,
    ):
        D = 2 ** num_qubits
        self.reference = (D - 2) / (D + 1) if reference is None else reference
        if not 0 < self.reference <= 1:
            raise ValueError(
                f"reference must be in (0, 1], got {self.reference} "
                "(the Haar default is 0 for a single qubit, which cannot be entangled)."
            )
        super().__init__(
            parameter_sampler=parameter_sampler,
            num_of_param_samples=num_of_param_samples,
            state_sampler=state_sampler,
            num_of_state_samples=num_of_state_samples,
            seed=seed,
        )
        self.num_qubits = num_qubits
        self._state_provider = state_provider

    @property
    def name(self) -> str:
        return "entanglement"

    def value(self, qc, spec=None) -> float:
        num_qubits = self._circuit_shape(qc, spec)[0]
        if num_qubits != self.num_qubits:
            raise ValueError(f"Entanglement was set up for {self.num_qubits} qubits, got a {num_qubits}-qubit circuit.")
        return super().value(qc, spec)

    def value_from_samples(self, ansatz, param_samples: np.ndarray, stv_vecs: Sequence) -> float:
        """Average Q over all parameter samples, then over the initial states."""
        provider = default_provider(ansatz, "state") if self._state_provider is None else self._state_provider
        per_state = [
            np.mean(meyer_wallach(provider.states(ansatz, param_samples, state), self.num_qubits))
            for state in stv_vecs
        ]
        return float(np.mean(per_state))

    def cost(self, value: float) -> float:
        return float(max(1.0 - value / self.reference, 0.0))
