"""Expressibility: how close an ansatz's output-state fidelities are to Haar-random ones."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from scipy.stats import entropy

from ansatz_search.backends.base import FidelityProvider, StateProvider, default_provider

from .base import SampledMetric
from ..utils.parameter_sampler import ParameterSampler
from ..utils.statevector_sampler import StateSampler


class Expressibility(SampledMetric):
    """KL divergence between the ansatz's fidelity distribution and the Haar one (Sim et al., 2019).

    The parameter samples are split into consecutive pairs (theta_a, theta_b);
    each pair gives one fidelity F = |<psi(theta_a)|psi(theta_b)>|^2, computed
    from output states of the backend's `state_provider`. The fidelities are
    histogrammed into `num_bins` bins and compared with the exact Haar fidelity
    distribution, Beta(1, D - 1) for D = 2**num_qubits.

    `value` is the raw KL divergence (lower means more expressive). `cost`
    normalizes it to [0, 1]: 0 at the noise floor (the KL divergence that even
    exactly Haar-random states reach with this many samples) and 1 at the KL
    divergence of a circuit whose fidelities are all 1. With `log_scale`, the
    same range is mapped on a log10 scale, which spreads out small divergences.

    `epsilon` > 0 histograms only [0, t_max] with t_max chosen so that a Haar
    fidelity exceeds it with probability `epsilon`, and collects the rest in
    one overflow bin.

    `fidelity_provider` measures the fidelities directly instead of computing
    them from statevectors, which is how expressibility runs on quantum
    hardware (e.g. QiskitFidelityProvider, by compute–uncompute).
    """

    label = "Expressibility (KL divergence)"
    log_axis = True  # plot axis only; the `log_scale` argument sets the cost mapping

    def __init__(
        self,
        num_qubits: int,
        # None: the provider of the backend that compiled the circuit (NumPy by default).
        state_provider: StateProvider | None = None,
        parameter_sampler: Union[str, ParameterSampler] = "uniform",
        num_of_param_samples: int = 10000,
        state_sampler: Union[str, StateSampler] = None,
        num_of_state_samples: int = 1,
        num_bins: int = 75,
        epsilon: float = 0.0,
        log_scale: bool = False,
        floor_repetitions: int = 200,
        seed: int = None,
        # Set to measure fidelities (e.g. on hardware) instead of computing them from statevectors.
        fidelity_provider: FidelityProvider | None = None,
    ):
        if num_of_param_samples < 4 or num_of_param_samples % 2:
            raise ValueError("num_of_param_samples must be even (samples form pairs) and at least 4.")
        if not 0 <= epsilon < 1:
            raise ValueError("epsilon must be in [0, 1).")
        super().__init__(
            parameter_sampler=parameter_sampler,
            num_of_param_samples=num_of_param_samples,
            state_sampler=state_sampler,
            num_of_state_samples=num_of_state_samples,
            seed=seed,
        )
        self.num_qubits = num_qubits
        self._state_provider = state_provider
        self._fidelity_provider = fidelity_provider
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.log_scale = log_scale
        self.floor_repetitions = floor_repetitions

        D = 2 ** num_qubits
        # A Haar fidelity exceeds t_max with probability epsilon (t_max = 1 for epsilon = 0).
        self.t_max = 1 - epsilon ** (1 / (D - 1))
        self.haar_distr = self._haar_distribution()
        # KL divergence when every fidelity lands in the last bin (e.g. F = 1 always).
        self.max_value = float(np.log(1 / self.haar_distr[-1]))
        self._noise_floor = None

    @property
    def name(self) -> str:
        return "expressibility"

    def _haar_distribution(self) -> np.ndarray:
        """Bin probabilities of Haar fidelities: CDF(F) = 1 - (1 - F)**(D - 1)."""
        D = 2 ** self.num_qubits
        regular_bins = self.num_bins - 1 if self.epsilon else self.num_bins
        edges = np.linspace(0, self.t_max, regular_bins + 1)
        probs = (1 - edges[:-1]) ** (D - 1) - (1 - edges[1:]) ** (D - 1)
        if self.epsilon:
            probs = np.append(probs, self.epsilon)  # overflow bin: P(F > t_max)
        return probs / probs.sum()

    def _histogram(self, fidelities: np.ndarray) -> np.ndarray:
        # Rounding can push a fidelity of 1 slightly above 1, out of the histogram range.
        F = np.clip(fidelities, 0.0, 1.0)
        if self.epsilon:
            hist = np.histogram(F, bins=self.num_bins - 1, range=(0, self.t_max))[0]
            hist = np.append(hist, np.sum(F > self.t_max))
        else:
            hist = np.histogram(F, bins=self.num_bins, range=(0, 1))[0]
        return hist / hist.sum()

    def _kl(self, fidelities: np.ndarray) -> float:
        return float(entropy(pk=self._histogram(fidelities), qk=self.haar_distr))

    @property
    def noise_floor(self) -> float:
        """Mean KL divergence of N/2 exactly Haar-distributed fidelities.

        This is the best value reachable with this many samples and bins, so
        `cost` subtracts it. Computed once, from the metric's seed.
        """
        if self._noise_floor is None:
            D = 2 ** self.num_qubits
            rng = np.random.default_rng(self.seed)
            u = rng.random((self.floor_repetitions, self.num_of_param_samples // 2))
            haar_fidelities = 1 - u ** (1 / (D - 1))  # inverse CDF of Beta(1, D - 1)
            self._noise_floor = float(np.mean([self._kl(F) for F in haar_fidelities]))
        return self._noise_floor

    def value(self, qc, spec=None) -> float:
        num_qubits = self._circuit_shape(qc, spec)[0]
        if num_qubits != self.num_qubits:
            raise ValueError(f"Expressibility was set up for {self.num_qubits} qubits, got a {num_qubits}-qubit circuit.")
        return super().value(qc, spec)

    def value_from_samples(self, ansatz, param_samples: np.ndarray, stv_vecs: Sequence) -> float:
        """Raw KL divergence, averaged over the initial states."""
        return float(np.mean([self._kl(self._fidelities(ansatz, param_samples, state)) for state in stv_vecs]))

    def _fidelities(self, ansatz, param_samples: np.ndarray, state) -> np.ndarray:
        """Fidelities of consecutive sample pairs: measured by the fidelity provider, else from statevectors."""
        if self._fidelity_provider is not None:
            measured = self._fidelity_provider.fidelities(ansatz, param_samples[0::2], param_samples[1::2], state)
            return np.asarray(measured, dtype=float)
        provider = default_provider(ansatz, "state") if self._state_provider is None else self._state_provider
        psi = provider.states(ansatz, param_samples, state)
        a, b = psi[0::2], psi[1::2]  # consecutive samples form a pair
        return np.abs(np.sum(np.conj(a) * b, axis=1)) ** 2

    def cost(self, value: float) -> float:
        floor, top = self.noise_floor, self.max_value
        if self.log_scale:
            ratio = np.log10(max(value, floor) / floor) / np.log10(top / floor)
        else:
            ratio = (value - floor) / (top - floor)
        return float(np.clip(ratio, 0.0, 1.0))
