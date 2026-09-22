"""Compile an AnsatzSpec into a program for the batched NumPy simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ansatz_search.backends.base import Compiler
from ansatz_search.circuit.ansatz import AnsatzSpec
from ansatz_search.circuit.gates import GateName

from .simulator import CONTROLLED_ROTATIONS, FIXED_1Q, FIXED_2Q, ROTATIONS

PERMUTATION_GATES = frozenset({GateName.X, GateName.CX, GateName.SWAP})


@dataclass(frozen=True)
class Op:
    """One simulator instruction.

    Exactly one of `perm` (index permutation), `matrix` (fixed gate) or
    `param` (index into the parameter vector) is set.
    """

    gate: GateName
    qubits: tuple[int, ...]
    perm: np.ndarray | None = None
    matrix: np.ndarray | None = None
    param: int | None = None


@dataclass(frozen=True)
class NumpyProgram:
    num_qubits: int
    num_params: int
    ops: tuple[Op, ...]


def _permutation(gate: GateName, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """perm such that new_psi = psi[:, perm]; all three gates are involutions."""
    i = np.arange(2 ** n)
    if gate is GateName.X:
        return i ^ (1 << qubits[0])
    if gate is GateName.CX:
        control, target = qubits
        return np.where(i & (1 << control), i ^ (1 << target), i)
    a, b = qubits  # SWAP: exchange the two bits
    differ = ((i >> a) ^ (i >> b)) & 1
    return i ^ (differ * ((1 << a) | (1 << b)))


class NumpyCompiler(Compiler):
    """Compile to a NumpyProgram. No JIT step, so a new circuit per trial is cheap."""

    supported_gates = frozenset(
        PERMUTATION_GATES | set(FIXED_1Q) | set(FIXED_2Q) | ROTATIONS | set(CONTROLLED_ROTATIONS)
    )

    def compile(self, spec: AnsatzSpec) -> NumpyProgram:
        self.validate(spec)
        n = spec.num_qubits
        ops = []
        for block in spec:
            gate, qubits = block.op, block.qubits
            if gate in PERMUTATION_GATES:
                ops.append(Op(gate, qubits, perm=_permutation(gate, qubits, n)))
            elif gate in FIXED_1Q:
                ops.append(Op(gate, qubits, matrix=FIXED_1Q[gate]))
            elif gate in FIXED_2Q:
                ops.append(Op(gate, qubits, matrix=FIXED_2Q[gate]))
            else:
                ops.append(Op(gate, qubits, param=block.params[0].index))
        return NumpyProgram(n, spec.num_params, tuple(ops))
