"""Compile an AnsatzSpec into a function that queues PennyLane operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml

from ansatz_search.backends.base import Compiler
from ansatz_search.circuit.ansatz import AnsatzSpec, ParamRef
from ansatz_search.circuit.gates import GateName
from ansatz_search.circuit.observables import PauliTerm

_OPS = {
    GateName.X: qml.PauliX, GateName.Y: qml.PauliY, GateName.Z: qml.PauliZ,
    GateName.H: qml.Hadamard, GateName.S: qml.S, GateName.T: qml.T,
    GateName.SDG: qml.adjoint(qml.S), GateName.TDG: qml.adjoint(qml.T), GateName.SX: qml.SX,
    GateName.RX: qml.RX, GateName.RY: qml.RY, GateName.RZ: qml.RZ, GateName.R1: qml.PhaseShift,
    GateName.CX: qml.CNOT, GateName.CY: qml.CY, GateName.CZ: qml.CZ, GateName.CH: qml.CH,
    GateName.SWAP: qml.SWAP, GateName.ECR: qml.ECR,
    GateName.CRX: qml.CRX, GateName.CRY: qml.CRY, GateName.CRZ: qml.CRZ,
    GateName.CR1: qml.ControlledPhaseShift,
    GateName.RXX: qml.IsingXX, GateName.RYY: qml.IsingYY, GateName.RZZ: qml.IsingZZ,
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
    # (op class, wires, parameter index, constant): the angle is params[index] + constant,
    # or the constant alone (a fixed angle) without an index; neither for a gate without angles.
    blocks: tuple[tuple[type, tuple[int, ...], int | None, float | None], ...]

    def apply(self, params, initial_state=None) -> None:
        """Queue the circuit inside a QNode. `params` may be broadcast: shape (B, P)."""
        if initial_state is not None:
            qml.StatePrep(initial_state, wires=range(self.num_qubits))
        for op, wires, k, constant in self.blocks:
            if k is not None:
                op(params[..., k] + constant if constant else params[..., k], wires=wires)
            elif constant is not None:
                op(constant, wires=wires)
            else:
                op(wires=wires)

    def observable(self, terms: tuple[PauliTerm, ...]):
        n = self.num_qubits
        sentence = qml.pauli.PauliSentence({
            qml.pauli.PauliWord({wire(q, n): p for q, p in enumerate(t.paulis) if p != "I"}): t.coeff
            for t in terms
        })
        return sentence.operation(wire_order=range(n))


def _index_and_constant(block) -> tuple[int | None, float | None]:
    if not block.params:
        return None, None
    angle = block.params[0]
    return (angle.index, angle.offset) if isinstance(angle, ParamRef) else (None, angle)


class PennyLaneCompiler(Compiler):
    """Compile to a PennyLaneProgram (no JIT; PennyLane builds the tape per execution)."""

    # Same gates as the NumPy backend.
    supported_gates = frozenset(_OPS)

    def compile(self, spec: AnsatzSpec) -> PennyLaneProgram:
        self.validate(spec)
        n = spec.num_qubits
        blocks = tuple((_OPS[b.op], tuple(wire(q, n) for q in b.qubits), *_index_and_constant(b)) for b in spec)
        return PennyLaneProgram(n, spec.num_params, blocks)
