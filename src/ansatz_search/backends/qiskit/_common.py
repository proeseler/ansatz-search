"""Helpers shared by the Qiskit providers: parameter order, observables, initial states, device transpilation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp

from ansatz_search.circuit.observables import pauli_sums


def require_circuit(circuit: Any, provider: str) -> QuantumCircuit:
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"{provider} needs a Qiskit QuantumCircuit (use QiskitCompiler), got {type(circuit).__name__}.")
    return circuit


def parameter_columns(parameters: Iterable) -> list[int]:
    """The `param_samples` column that feeds each parameter, in the given order.

    QiskitCompiler names framework parameter j `theta_j`, and Qiskit orders
    parameters by name (theta_10 before theta_2), so the columns come from the
    names. A circuit with other parameter names uses Qiskit's order.
    """
    parameters = list(parameters)
    suffixes = [p.name.removeprefix("theta_") for p in parameters]
    if all(p.name.startswith("theta_") and s.isdigit() for p, s in zip(parameters, suffixes)):
        return [int(s) for s in suffixes]
    return list(range(len(parameters)))


def sample_matrix(param_samples: Any, columns: list[int]) -> np.ndarray:
    samples = np.asarray(param_samples, dtype=float)
    needed = max(columns) + 1 if columns else 0
    if samples.ndim != 2 or samples.shape[1] < needed:
        raise ValueError(f"param_samples must be 2D with at least {needed} columns.")
    return samples


def checked_state(state: Any, num_qubits: int) -> np.ndarray:
    amplitudes = np.asarray(state, dtype=complex).reshape(-1)
    if amplitudes.size != 2 ** num_qubits:
        raise ValueError(f"Each initial state must have {2 ** num_qubits} amplitudes, got {amplitudes.size}.")
    if not np.isclose(np.vdot(amplitudes, amplitudes).real, 1.0):
        raise ValueError("Each initial state must be normalized.")
    return amplitudes


def preparation(amplitudes: np.ndarray, num_qubits: int) -> QuantumCircuit | None:
    """A circuit of plain gates preparing `amplitudes` from |0...0>, or None if that already is the state.

    StatePreparation is compiled to u/cx gates, which every primitive accepts
    (Aer's primitives reject the state_preparation instruction itself).
    """
    if np.isclose(abs(amplitudes[0]), 1.0):
        return None
    from qiskit import transpile

    prep = QuantumCircuit(num_qubits)
    prep.append(StatePreparation(amplitudes), range(num_qubits))
    return transpile(prep, basis_gates=["u", "cx"], optimization_level=0)


def pauli_ops(observables: Any, num_qubits: int) -> list[SparsePauliOp]:
    """Framework Pauli sums as SparsePauliOps (character i acts on qubit i; Qiskit labels run the other way)."""
    return [SparsePauliOp.from_list([(term.paulis[::-1], term.coeff) for term in terms])
            for terms in pauli_sums(observables, num_qubits)]


def portable(circuit: QuantumCircuit) -> QuantumCircuit:
    """`circuit` compiled to u and cx gates: same parameters, same unitary, accepted by every primitive.

    Used for any primitive other than Qiskit's exact reference ones. Aer's
    primitives reject gates such as ch, and Qiskit Aer (0.17) binds parameter
    values wrongly for some parameterized controlled rotations: their angles
    silently act as 0. Plain u/cx circuits avoid both.
    """
    from qiskit import transpile

    return transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)


def for_device(circuit: QuantumCircuit, backend: Any, optimization_level: int, seed: int | None = None) -> QuantumCircuit:
    """`circuit` transpiled to `backend`'s gates and qubit connectivity, or unchanged without a backend.

    `seed` fixes the transpiler's randomized qubit layout, so repeated runs use
    the same physical qubits (and therefore see the same device noise).
    """
    if backend is None:
        return circuit
    from qiskit.transpiler import generate_preset_pass_manager

    return generate_preset_pass_manager(backend=backend, optimization_level=optimization_level,
                                        seed_transpiler=seed).run(circuit)
