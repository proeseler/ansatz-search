"""Samplers producing initial states for encodable metrics.

Samplers return plain complex amplitude arrays of length ``2 ** num_qubits``
rather than backend objects. Qiskit is used internally to construct states,
but it does not appear in the interface: an amplitude vector is backend-neutral
data, and Qiskit's array index convention (qubit 0 = least significant bit)
is the framework convention.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector, random_statevector

from .seeding import Reseedable

__all__ = [
    "StateSampler",
    "ZeroStatevectorSampler",
    "PlusStatevectorSampler",
    "CustomSampler",
    "HaarStatevectorSampler",
    "GaussianRyStateSampler",
    "GaussianRxRyRzStateSampler",
    "ZZFeatureMapSampler",
]

StateVec = NDArray[np.complex128]


def _as_amplitudes(state) -> StateVec:
    """Normalize any Qiskit/array state into a flat complex amplitude vector."""
    return np.asarray(state, dtype=complex).reshape(-1)


class StateSampler(Reseedable, ABC):
    @abstractmethod
    def __call__(self, num_qubits: int, num_states: int) -> Sequence[StateVec]:
        """Return `num_states` amplitude vectors of length 2 ** num_qubits."""

class ZeroStatevectorSampler(StateSampler):
    def __call__(self, num_qubits: int, num_states: int) -> Sequence[StateVec]:
        state = _as_amplitudes(Statevector.from_label('0' * num_qubits))
        return [state.copy() for _ in range(num_states)]

class PlusStatevectorSampler(StateSampler):
    def __call__(self, num_qubits: int, num_states: int) -> Sequence[StateVec]:
        state = _as_amplitudes(Statevector.from_label('+' * num_qubits))
        return [state.copy() for _ in range(num_states)]

class CustomSampler(StateSampler):
    def __init__(self, stv):
        super().__init__()
        self._stv = _as_amplitudes(stv)

    def __call__(self, num_qubits: int, num_states: int) -> Sequence[StateVec]:
        return [self._stv.copy() for _ in range(num_states)]

class HaarStatevectorSampler(StateSampler):
    def __init__(self, seed: int = None):
        # Draw the generator once: passing a fixed seed per draw would return
        # num_states identical states.
        self.rng = np.random.default_rng(seed)
        super().__init__()

    def __call__(self, num_qubits: int, num_states: int) -> Sequence[StateVec]:
        return [
            _as_amplitudes(random_statevector(2**num_qubits, seed=self.rng))
            for _ in range(num_states)
        ]


class GaussianRyStateSampler(StateSampler):
    def __init__(self, seed: int = None, mean: float = 0.0, std: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.mean = mean
        self.std = std
        super().__init__()

    def __call__(self, num_qubits: int, num_states: int) -> Sequence[StateVec]:
        zero_state = Statevector.from_label('0' * num_qubits)
        stv_vecs = []
        for _ in range(num_states):
            raw_angles = self.rng.normal(self.mean, self.std, size=num_qubits)
            angles = np.tanh(raw_angles)
            qc = QuantumCircuit(num_qubits)
            for qubit, angle in enumerate(angles):
                qc.ry(angle, qubit)
            stv_vecs.append(_as_amplitudes(zero_state.evolve(qc)))
        return stv_vecs
    
class GaussianRxRyRzStateSampler(StateSampler):
    def __init__(self, seed: int = None, mean: float = 0.0, std: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.mean = mean
        self.std = std
        super().__init__()

    def __call__(self, num_qubits: int, num_states: int) -> Sequence[StateVec]:
        zero_state = Statevector.from_label('0' * num_qubits)
        stv_vecs = []
        for _ in range(num_states):
            raw_angles = self.rng.normal(self.mean, self.std, size=3 * num_qubits)
            angles = np.tanh(raw_angles).reshape(num_qubits, 3)
            qc = QuantumCircuit(num_qubits)
            for qubit, (rx_angle, ry_angle, rz_angle) in enumerate(angles):
                qc.rx(rx_angle, qubit)
                qc.ry(ry_angle, qubit)
                qc.rz(rz_angle, qubit)
            stv_vecs.append(_as_amplitudes(zero_state.evolve(qc)))
        return stv_vecs

class ZZFeatureMapSampler(StateSampler):
    def __init__(self, seed: int = None):
        self.rng   = np.random.default_rng(seed)
        super().__init__()        


    def __call__(self, num_qubits: int, num_states: int) -> Sequence[StateVec]:
        feature_map = zz_feature_map(num_qubits, reps=2)
        sv = Statevector.from_label('0' * num_qubits)
        stv_vecs = []
        for _ in range(num_states):
            param_sample = self.rng.standard_normal(size=feature_map.num_parameters)
            qc = feature_map.assign_parameters(param_sample, inplace=False)
            stv_vecs.append(_as_amplitudes(sv.evolve(qc)))
        return stv_vecs
