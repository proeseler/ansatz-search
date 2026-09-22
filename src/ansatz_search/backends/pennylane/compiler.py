"""Compile an AnsatzSpec into a function that queues PennyLane operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml

from ansatz_search.backends.base import Compiler
from ansatz_search.circuit.ansatz import AnsatzSpec
from ansatz_search.circuit.gates import GateName
from ansatz_search.circuit.observables import PauliTerm

_OPS = {
    GateName.X: qml.PauliX, GateName.Y: qml.PauliY, GateName.Z: qml.PauliZ,
    GateName.H: qml.Hadamard, GateName.S: qml.S, GateName.T: qml.T,
    GateName.RX: qml.RX, GateName.RY: qml.RY, GateName.RZ: qml.RZ, GateName.R1: qml.PhaseShift,
    GateName.CX: qml.CNOT, GateName.CY: qml.CY, GateName.CZ: qml.CZ, GateName.CH: qml.CH,
    GateName.SWAP: qml.SWAP,
    GateName.CRX: qml.CRX, GateName.CRY: qml.CRY, GateName.CRZ: qml.CRZ,
    GateName.CR1: qml.ControlledPhaseShift,
}


def wire(q: int, n: int) -> int:
    """Qubit q lives on wire n-1-q. PennyLane treats wire 0 as the most
    significant amplitude bit, so this makes qubit 0 the least significant bit,
    matching Qiskit and the NumPy backend."""
    return n - 1 - q


@dataclass(frozen=True)
class PennyLaneProgram:
    num_qubits: int
    num_params: int
    blocks: tuple[tuple[type, tuple[int, ...], int | None], ...]  # (op class, wires, param index)

    def apply(self, params, initial_state=None) -> None:
        """Queue the circuit inside a QNode. `params` may be broadcast: shape (B, P)."""
        if initial_state is not None:
            qml.StatePrep(initial_state, wires=range(self.num_qubits))
        for op, wires, k in self.blocks:
            if k is None:
                op(wires=wires)
            else:
                op(params[..., k], wires=wires)

    def observable(self, terms: tuple[PauliTerm, ...]):
        n = self.num_qubits
        sentence = qml.pauli.PauliSentence({
            qml.pauli.PauliWord({wire(q, n): p for q, p in enumerate(t.paulis) if p != "I"}): t.coeff
            for t in terms
        })
        return sentence.operation(wire_order=range(n))


class PennyLaneCompiler(Compiler):
    """Compile to a PennyLaneProgram (no JIT; PennyLane builds the tape per execution)."""

    # Same gates as the NumPy backend. PennyLane also has excitation gates, but
    # their convention would first have to be matched to the old framework's.
    supported_gates = frozenset(_OPS)

    def compile(self, spec: AnsatzSpec) -> PennyLaneProgram:
        self.validate(spec)
        n = spec.num_qubits
        blocks = tuple(
            (_OPS[b.op], tuple(wire(q, n) for q in b.qubits), b.params[0].index if b.params else None)
            for b in spec
        )
        return PennyLaneProgram(n, spec.num_params, blocks)
