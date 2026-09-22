"""Batched statevector simulation on (batch, 2**n) complex arrays.

Amplitude index convention: qubit q is bit ``1 << q`` of the index (qubit 0 is
the least significant bit), matching Qiskit, so its statevectors can be used
as initial states directly. A two-qubit matrix acting on qubits
``(a, b)`` uses the local basis index ``2 * bit(a) + bit(b)``: the first listed
qubit (e.g. the control) is the most significant, as in textbook matrices.
"""

from __future__ import annotations

import numpy as np

from ansatz_search.circuit.gates import GateName
from ansatz_search.circuit.observables import PauliTerm

_SQRT_HALF = 1 / np.sqrt(2)

# Gates without parameters, as fixed matrices. X, CX and SWAP are applied as
# index permutations instead (see compiler), which is cheaper.
FIXED_1Q = {
    GateName.Y: np.array([[0, -1j], [1j, 0]]),
    GateName.Z: np.diag([1, -1]).astype(complex),
    GateName.H: np.array([[1, 1], [1, -1]]) * _SQRT_HALF + 0j,
    GateName.S: np.diag([1, 1j]),
    GateName.T: np.diag([1, np.exp(1j * np.pi / 4)]),
}


def _controlled(U: np.ndarray) -> np.ndarray:
    out = np.zeros(U.shape[:-2] + (4, 4), dtype=complex)
    out[..., 0, 0] = out[..., 1, 1] = 1
    out[..., 2:, 2:] = U
    return out


FIXED_2Q = {
    GateName.CY: _controlled(FIXED_1Q[GateName.Y]),
    GateName.CZ: _controlled(FIXED_1Q[GateName.Z]),
    GateName.CH: _controlled(FIXED_1Q[GateName.H]),
}

ROTATIONS = frozenset({GateName.RX, GateName.RY, GateName.RZ, GateName.R1})
CONTROLLED_ROTATIONS = {
    GateName.CRX: GateName.RX, GateName.CRY: GateName.RY,
    GateName.CRZ: GateName.RZ, GateName.CR1: GateName.R1,
}


def _stack2x2(a, b, c, d) -> np.ndarray:
    return np.stack([np.stack([a, b], -1), np.stack([c, d], -1)], -2)


def rotation(gate: GateName, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched (U, dU/dtheta), each (B, 2, 2), for a single-qubit rotation."""
    h = np.asarray(theta, dtype=float) / 2
    c, s = np.cos(h) + 0j, np.sin(h) + 0j
    z = np.zeros_like(c)
    if gate is GateName.RX:
        U = _stack2x2(c, -1j * s, -1j * s, c)
        dU = 0.5 * _stack2x2(-s, -1j * c, -1j * c, -s)
    elif gate is GateName.RY:
        U = _stack2x2(c, -s, s, c)
        dU = 0.5 * _stack2x2(-s, -c, c, -s)
    elif gate is GateName.RZ:
        e = np.exp(-1j * h)
        U = _stack2x2(e, z, z, e.conj())
        dU = _stack2x2(-0.5j * e, z, z, 0.5j * e.conj())
    elif gate is GateName.R1:
        e = np.exp(2j * h)
        U = _stack2x2(z + 1, z, z, e)
        dU = _stack2x2(z, z, z, 1j * e)
    else:
        raise NotImplementedError(f"No rotation matrix for {gate.value!r}.")
    return U, dU


def parameterized(gate: GateName, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched (U, dU/dtheta) for any supported parameterized gate."""
    if gate in ROTATIONS:
        return rotation(gate, theta)
    U, dU = rotation(CONTROLLED_ROTATIONS[gate], theta)
    derivative = np.zeros(dU.shape[:-2] + (4, 4), dtype=complex)
    derivative[..., 2:, 2:] = dU
    return _controlled(U), derivative


def dagger(U: np.ndarray) -> np.ndarray:
    return np.conj(np.swapaxes(U, -1, -2))


def apply_matrix(psi: np.ndarray, U: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Apply a (B, 2**k, 2**k) or (2**k, 2**k) matrix to `qubits` of every state in the batch."""
    B = psi.shape[0]
    if len(qubits) == 1:
        (q,) = qubits
        # Split the index into (bits above q, bit q, bits below q); no copy needed.
        v = psi.reshape(B, 2 ** (n - 1 - q), 2, 2 ** q)
        out = (U if U.ndim == 2 else U[:, None]) @ v
        return out.reshape(B, -1)
    # General k-qubit case: bring the target axes to the end, in the order given.
    k = len(qubits)
    t = psi.reshape((B,) + (2,) * n)
    axes = [1 + (n - 1 - q) for q in qubits]
    t = np.moveaxis(t, axes, range(n + 1 - k, n + 1))
    shape = t.shape
    t = t.reshape(B, -1, 2 ** k) @ np.swapaxes(U, -1, -2)
    return np.moveaxis(t.reshape(shape), range(n + 1 - k, n + 1), axes).reshape(B, -1)


class PauliSumOperator:
    """Precomputed action of a Pauli sum on (batch, 2**n) states."""

    def __init__(self, terms: tuple[PauliTerm, ...], n: int):
        idx = np.arange(2 ** n)
        self._terms = []
        for term in terms:
            flip = sign = 0
            n_y = 0
            for q, p in enumerate(term.paulis):
                bit = 1 << q
                if p in "XY":
                    flip |= bit
                if p in "YZ":
                    sign |= bit
                n_y += p == "Y"
            src = idx ^ flip  # (P psi)[j] = phase(j ^ flip) * psi[j ^ flip]
            parity = np.array([bin(i).count("1") & 1 for i in (src & sign)])
            phase = term.coeff * (1j ** n_y) * (1 - 2 * parity)
            self._terms.append((None if flip == 0 else src, phase))

    def apply(self, psi: np.ndarray) -> np.ndarray:
        out = np.zeros_like(psi)
        for src, phase in self._terms:
            out += phase * (psi if src is None else psi[:, src])
        return out

    def expectation(self, psi: np.ndarray) -> np.ndarray:
        return np.real(np.sum(np.conj(psi) * self.apply(psi), axis=1))
