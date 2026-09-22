"""Fidelities between circuit outputs by compute–uncompute, runnable on quantum hardware."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ansatz_search.backends.base import FidelityProvider

from ._common import (checked_state, for_device, parameter_columns, portable, preparation, require_circuit,
                      sample_matrix)


class QiskitFidelityProvider(FidelityProvider):
    """Measure F = |<psi(b)|psi(a)>|^2 by compute–uncompute.

    The circuit U(a) is followed by U(b)†, all qubits are measured, and F is the
    fraction of shots that return 0...0. The circuit is transpiled once, and all
    pairs of a call (up to `max_bindings` per job) go to the sampler together.

    - No arguments: Qiskit's StatevectorSampler (exact simulation plus shot noise).
    - `backend=`: transpiles for that device and, without an explicit `sampler`,
      runs there with qiskit-ibm-runtime's SamplerV2: a fake IBM device (local
      simulation with its calibrated noise) or real hardware. Needs the `ibm`
      extra. `seed` fixes the transpiled qubit layout and makes simulated shots
      reproducible; shot noise on real hardware stays random.
    """

    def __init__(
        self,
        sampler: Any = None,
        backend: Any = None,
        shots: int = 4000,
        optimization_level: int = 1,
        seed: int | None = None,
        max_bindings: int = 10_000,
    ):
        self.sampler = sampler
        self.backend = backend
        self.shots = shots
        self.optimization_level = optimization_level
        self.seed = seed
        self.max_bindings = max_bindings

    def _sampler(self):
        if self.sampler is not None:
            return self.sampler
        if self.backend is not None:
            from qiskit_ibm_runtime import SamplerV2

            options = {} if self.seed is None else {"simulator": {"seed_simulator": self.seed}}
            return SamplerV2(mode=self.backend, options=options)
        from qiskit.primitives import StatevectorSampler

        return StatevectorSampler(default_shots=self.shots, seed=self.seed)

    def fidelities(self, circuit: QuantumCircuit, params_a: np.ndarray, params_b: np.ndarray,
                   initial_state: Any) -> np.ndarray:
        circuit = require_circuit(circuit, type(self).__name__)
        n = circuit.num_qubits
        parameters = list(circuit.parameters)
        columns = parameter_columns(parameters)
        a, b = sample_matrix(params_a, columns), sample_matrix(params_b, columns)
        if len(a) != len(b):
            raise ValueError(f"params_a and params_b must have the same number of rows, got {len(a)} and {len(b)}.")

        # |initial> -> U(a) -> U(b)^dagger -> back to |0...0> -> measure: P(0...0) = F.
        vec_a, vec_b = ParameterVector("a", len(parameters)), ParameterVector("b", len(parameters))
        prep = preparation(checked_state(initial_state, n), n)
        echo = QuantumCircuit(n)
        if prep is not None:
            echo.compose(prep, inplace=True)
        echo.compose(circuit.assign_parameters(dict(zip(parameters, vec_a))), inplace=True)
        echo.compose(circuit.assign_parameters(dict(zip(parameters, vec_b))).inverse(), inplace=True)
        if prep is not None:
            echo.compose(prep.inverse(), inplace=True)
        if self.sampler is not None or self.backend is not None:  # anything but the exact reference
            echo = portable(echo)
        echo.measure_all()
        device_circuit = for_device(echo, self.backend, self.optimization_level, self.seed)

        # Columns of [a | b] that feed each parameter of the (transpiled) circuit.
        pairs = np.concatenate([a[:, columns], b[:, columns]], axis=1)
        order = [p.index + (len(parameters) if p.vector.name == "b" else 0) for p in device_circuit.parameters]
        sampler = self._sampler()
        out = np.empty(len(a))
        for start in range(0, len(a), self.max_bindings):
            values = pairs[start:start + self.max_bindings][:, order]
            bits = sampler.run([(device_circuit, values)], shots=self.shots).result()[0].data.meas.array
            out[start:start + len(values)] = np.mean(~bits.any(axis=-1), axis=-1)  # share of all-zero shots
        return out
