"""Exact output statevectors of Qiskit circuits with qiskit.quantum_info.Statevector."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ansatz_search.backends.base import StateProvider

from ._common import checked_state, parameter_columns, require_circuit, sample_matrix


class QiskitStateProvider(StateProvider):
    """One sample at a time with Statevector: exact, but much slower than the NumPy backend.

    Statevectors exist only in simulation. On hardware, measure fidelities with
    QiskitFidelityProvider instead.
    """

    def states(self, circuit: QuantumCircuit, param_samples: np.ndarray, initial_state: Any) -> np.ndarray:
        circuit = require_circuit(circuit, type(self).__name__)
        parameters = list(circuit.parameters)
        columns = parameter_columns(parameters)
        samples = sample_matrix(param_samples, columns)
        start = Statevector(checked_state(initial_state, circuit.num_qubits))
        return np.array([start.evolve(circuit.assign_parameters(dict(zip(parameters, row[columns])))).data
                         for row in samples])
