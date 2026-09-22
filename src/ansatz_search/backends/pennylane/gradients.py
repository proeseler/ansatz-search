"""Gradients through PennyLane devices."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ansatz_search.backends.base import GradientMethod, GradientProvider
from ansatz_search.circuit.observables import pauli_sums

from .compiler import PennyLaneProgram

_PL_DIFF = {
    GradientMethod.BACKPROP: "backprop",
    GradientMethod.ADJOINT: "adjoint",
    GradientMethod.PARAMETER_SHIFT: "parameter-shift",
    GradientMethod.FINITE_DIFFERENCE: "finite-diff",
}


class PennyLaneGradientProvider(GradientProvider):
    """Gradients for PennyLaneCompiler programs.

    method:
      - "auto" (default): "backprop" below `lightning_from` qubits, "adjoint"
        from there on. Measured for 32 parameters and 11,806 samples: backprop
        on default.qubit is fastest up to ~10 qubits but stalls from 12 qubits
        (PennyLane 0.45), where lightning.qubit adjoint is the fastest option.
      - "backprop": autodiff through default.qubit, all samples broadcast at once.
      - "adjoint": lightning.qubit adjoint, one execution per sample.
      - "parameter_shift", "finite_difference": default.qubit, one execution
        per sample (PennyLane cannot differentiate broadcast tapes with these).
        PennyLane applies the correct shift rules for controlled rotations and
        reused parameters.

    Observables are backend-neutral Pauli sums (see circuit.observables).
    """

    def __init__(
        self,
        method: GradientMethod | str = GradientMethod.AUTO,
        epsilon: float = 1e-6,
        max_batch_bytes: int = 2 ** 30,
        lightning_from: int = 12,
    ):
        self.method = GradientMethod(method)
        if self.method is not GradientMethod.AUTO and self.method not in _PL_DIFF:
            raise NotImplementedError(f"PennyLaneGradientProvider does not support {self.method.value!r}.")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")
        self.epsilon = epsilon
        self.max_batch_bytes = max_batch_bytes
        self.lightning_from = lightning_from

    def resolved_method(self, num_qubits: int) -> GradientMethod:
        if self.method is GradientMethod.AUTO:
            return GradientMethod.BACKPROP if num_qubits < self.lightning_from else GradientMethod.ADJOINT
        return self.method

    def gradients(self, circuit: PennyLaneProgram, observables: Any, param_samples: np.ndarray,
                  states: Sequence[Any]) -> np.ndarray:
        """Return shape (states, observables, parameter samples, parameters)."""
        if not isinstance(circuit, PennyLaneProgram):
            raise TypeError(f"PennyLaneGradientProvider needs a PennyLaneProgram (use PennyLaneCompiler), got {type(circuit).__name__}.")
        n, dim = circuit.num_qubits, 2 ** circuit.num_qubits
        samples = np.asarray(param_samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] < circuit.num_params:
            raise ValueError(f"param_samples must be 2D with at least {circuit.num_params} columns.")
        obs_ops = [circuit.observable(terms) for terms in pauli_sums(observables, n)]
        initial = [self._check_state(s, dim) for s in states]
        if not initial:
            raise ValueError("Provide at least one initial state.")

        method = self.resolved_method(n)
        grads = np.empty((len(initial), len(obs_ops)) + samples.shape)
        for s, state in enumerate(initial):
            prep = None if np.isclose(abs(state[0]), 1.0) else state  # skip StatePrep for |0...0>
            for o, obs in enumerate(obs_ops):
                qnode = self._qnode(circuit, obs, prep, method)
                if method is GradientMethod.BACKPROP:
                    grads[s, o] = self._broadcast(qnode, samples, circuit, n)
                else:
                    grads[s, o] = self._per_sample(qnode, samples)
        return grads

    def _qnode(self, program, obs, prep, method):
        device = "lightning.qubit" if method is GradientMethod.ADJOINT else "default.qubit"
        kwargs = {}
        if method is GradientMethod.FINITE_DIFFERENCE:
            kwargs["gradient_kwargs"] = {"h": self.epsilon, "approx_order": 2, "strategy": "center"}

        def body(params):
            program.apply(params, prep)
            return qml.expval(obs)

        return qml.QNode(body, qml.device(device, wires=program.num_qubits), diff_method=_PL_DIFF[method], **kwargs)

    @staticmethod
    def _per_sample(qnode, samples) -> np.ndarray:
        grad = qml.grad(qnode)
        return np.stack([np.asarray(grad(pnp.array(row, requires_grad=True))) for row in samples])

    def _broadcast(self, qnode, samples, program, n) -> np.ndarray:
        # Samples are independent, so the gradient of the summed batch output is
        # every sample's own gradient.
        grad = qml.grad(lambda x: pnp.sum(qnode(x)))
        # Backprop keeps the state after every gate: bound that memory per chunk.
        per_sample = (len(program.blocks) + 2) * 2 ** n * 16 * 3
        chunk = max(2, self.max_batch_bytes // per_sample)
        out = np.empty(samples.shape)
        for start in range(0, len(samples), chunk):
            block = samples[start:start + chunk]
            padded = block if len(block) > 1 else np.repeat(block, 2, axis=0)  # PennyLane mishandles batch size 1
            with warnings.catch_warnings():
                # autograd casts a complex intermediate to real in its backward
                # pass; the discarded imaginary part is zero (checked against
                # the NumPy adjoint in the tests).
                warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
                g = grad(pnp.array(padded, requires_grad=True))
            out[start:start + len(block)] = np.asarray(g)[:len(block)]
        return out

    @staticmethod
    def _check_state(state, dim: int) -> np.ndarray:
        amplitudes = np.asarray(state, dtype=complex).reshape(-1)
        if amplitudes.size != dim:
            raise ValueError(f"Each initial state must have {dim} amplitudes, got {amplitudes.size}.")
        if not np.isclose(np.vdot(amplitudes, amplitudes).real, 1.0):
            raise ValueError("Each initial state must be normalized.")
        return amplitudes
