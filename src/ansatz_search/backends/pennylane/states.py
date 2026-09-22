"""Output statevectors from PennyLane's default.qubit, broadcast over all samples."""

from __future__ import annotations

from typing import Any

import numpy as np
import pennylane as qml

from ansatz_search.backends.base import StateProvider

from .compiler import PennyLaneProgram


class PennyLaneStateProvider(StateProvider):
    """Thanks to the compiler's wire mapping, PennyLane's amplitude order already
    matches the framework convention (qubit 0 = least significant bit)."""

    def __init__(self, max_batch_bytes: int = 2 ** 30):
        self.max_batch_bytes = max_batch_bytes

    def states(self, circuit: PennyLaneProgram, param_samples: np.ndarray, initial_state: Any) -> np.ndarray:
        if not isinstance(circuit, PennyLaneProgram):
            raise TypeError(f"PennyLaneStateProvider needs a PennyLaneProgram (use PennyLaneCompiler), got {type(circuit).__name__}.")
        samples = np.asarray(param_samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] < circuit.num_params:
            raise ValueError(f"param_samples must be 2D with at least {circuit.num_params} columns.")
        n, dim = circuit.num_qubits, 2 ** circuit.num_qubits
        state = np.asarray(initial_state, dtype=complex).reshape(-1)
        prep = None if np.isclose(abs(state[0]), 1.0) else state

        def body(params):
            circuit.apply(params, prep)
            return qml.state()

        qnode = qml.QNode(body, qml.device("default.qubit", wires=n))
        chunk = max(2, self.max_batch_bytes // (dim * 16 * 4))
        out = np.empty((len(samples), dim), dtype=complex)
        for start in range(0, len(samples), chunk):
            block = samples[start:start + chunk]
            padded = block if len(block) > 1 else np.repeat(block, 2, axis=0)  # avoid batch size 1
            out[start:start + len(block)] = np.asarray(qnode(padded))[:len(block)]
        return out
