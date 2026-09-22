"""Output statevectors from the batched NumPy simulator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ansatz_search.backends.base import StateProvider

from .compiler import NumpyProgram
from .gradients import statevectors


class NumpyStateProvider(StateProvider):
    """All samples in one batch (chunked to at most `max_batch_bytes` of states)."""

    def __init__(self, max_batch_bytes: int = 2 ** 30):
        self.max_batch_bytes = max_batch_bytes

    def states(self, circuit: NumpyProgram, param_samples: np.ndarray, initial_state: Any) -> np.ndarray:
        if not isinstance(circuit, NumpyProgram):
            raise TypeError(f"NumpyStateProvider needs a NumpyProgram (use NumpyCompiler), got {type(circuit).__name__}.")
        samples = np.asarray(param_samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] < circuit.num_params:
            raise ValueError(f"param_samples must be 2D with at least {circuit.num_params} columns.")
        dim = 2 ** circuit.num_qubits
        chunk = max(1, self.max_batch_bytes // (dim * 16 * 4))
        out = np.empty((len(samples), dim), dtype=complex)
        for start in range(0, len(samples), chunk):
            out[start:start + chunk] = statevectors(circuit, samples[start:start + chunk], initial_state)
        return out
