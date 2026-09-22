"""Gradients on the batched NumPy simulator: adjoint (default), parameter shift, finite differences."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from ansatz_search.backends.base import GradientMethod, GradientProvider
from ansatz_search.circuit.observables import pauli_sums

from .compiler import NumpyProgram, Op
from .simulator import ROTATIONS, PauliSumOperator, apply_matrix, dagger, parameterized

_SUPPORTED = (GradientMethod.ADJOINT, GradientMethod.PARAMETER_SHIFT, GradientMethod.FINITE_DIFFERENCE)


def _apply(psi, op: Op, n: int, U=None):
    """Apply op (or, with U given, that matrix in place of the op's own)."""
    if op.perm is not None:
        return psi[:, op.perm]
    return apply_matrix(psi, op.matrix if U is None else U, op.qubits, n)


def statevectors(program: NumpyProgram, params: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    """Run the circuit for every parameter row; returns (B, 2**n) output states."""
    n = program.num_qubits
    psi = np.broadcast_to(np.asarray(initial_state, dtype=complex), (len(params), 2 ** n)).copy()
    for op in program.ops:
        U = None if op.param is None else parameterized(op.gate, params[:, op.param])[0]
        psi = _apply(psi, op, n, U)
    return psi


class NumpyGradientProvider(GradientProvider):
    """Exact gradients for NumpyCompiler programs, batched over all parameter samples.

    method:
      - "adjoint" (default): all gradients from one forward and one backward
        pass. Exact for every supported gate, including controlled rotations
        and reused parameters; cost independent of the number of parameters.
      - "parameter_shift": two circuit runs per parameter. Only valid for
        RX/RY/RZ/R1 with each parameter used once; raises otherwise.
      - "finite_difference": central differences, two runs per parameter.

    Observables are backend-neutral Pauli sums (see circuit.observables), e.g.
    ``{"ZZZZ": 1.0}``. Samples are processed in chunks of at most
    `max_batch_bytes` of state memory.
    """

    def __init__(
        self,
        method: GradientMethod | str = GradientMethod.ADJOINT,
        epsilon: float = 1e-6,
        max_batch_bytes: int = 2 ** 30,
    ):
        self.method = GradientMethod(method)
        if self.method not in _SUPPORTED:
            raise NotImplementedError(
                f"NumpyGradientProvider supports {[m.value for m in _SUPPORTED]}, not {self.method.value!r}."
            )
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")
        self.epsilon = epsilon
        self.max_batch_bytes = max_batch_bytes

    def gradients(self, circuit: NumpyProgram, observables: Any, param_samples: np.ndarray,
                  states: Sequence[Any]) -> np.ndarray:
        """Return shape (states, observables, parameter samples, parameters)."""
        if not isinstance(circuit, NumpyProgram):
            raise TypeError(f"NumpyGradientProvider needs a NumpyProgram (use NumpyCompiler), got {type(circuit).__name__}.")
        n, dim = circuit.num_qubits, 2 ** circuit.num_qubits
        samples = np.asarray(param_samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] < circuit.num_params:
            raise ValueError(f"param_samples must be 2D with at least {circuit.num_params} columns.")
        operators = [PauliSumOperator(terms, n) for terms in pauli_sums(observables, n)]
        initial = [self._check_state(s, dim) for s in states]
        if not initial:
            raise ValueError("Provide at least one initial state.")
        if self.method is GradientMethod.PARAMETER_SHIFT:
            self._check_parameter_shift(circuit)

        grads = np.empty((len(initial), len(operators)) + samples.shape)
        # Rough peak memory per sample: forward/backward states plus temporaries.
        per_sample = dim * 16 * (6 + 2 * len(operators))
        if self.method is not GradientMethod.ADJOINT:
            per_sample *= 2  # plus and minus shifts in one batch
        chunk = max(1, self.max_batch_bytes // per_sample)
        for s, state in enumerate(initial):
            for start in range(0, len(samples), chunk):
                block = samples[start:start + chunk]
                grads[s, :, start:start + len(block)] = self._chunk(circuit, block, state, operators)
        return grads

    def _chunk(self, program, theta, state, operators) -> np.ndarray:
        if self.method is GradientMethod.ADJOINT:
            return self._adjoint(program, theta, state, operators)
        shift, denom = (np.pi / 2, 2.0) if self.method is GradientMethod.PARAMETER_SHIFT else (self.epsilon, 2 * self.epsilon)
        out = np.zeros((len(operators),) + theta.shape)
        B = len(theta)
        for k in range(theta.shape[1]):
            shifted = np.concatenate([theta, theta])
            shifted[:B, k] += shift
            shifted[B:, k] -= shift
            psi = statevectors(program, shifted, state)
            for o, op in enumerate(operators):
                e = op.expectation(psi)
                out[o, :, k] = (e[:B] - e[B:]) / denom
        return out

    @staticmethod
    def _adjoint(program, theta, state, operators) -> np.ndarray:
        n = program.num_qubits
        phi = statevectors(program, theta, state)
        lams = [op.apply(phi) for op in operators]
        out = np.zeros((len(operators),) + theta.shape)
        for op in reversed(program.ops):
            if op.param is None:
                if op.perm is not None:          # X, CX, SWAP are their own inverses
                    phi = phi[:, op.perm]
                    lams = [lam[:, op.perm] for lam in lams]
                else:
                    Ud = dagger(op.matrix)
                    phi = apply_matrix(phi, Ud, op.qubits, n)
                    lams = [apply_matrix(lam, Ud, op.qubits, n) for lam in lams]
                continue
            U, dU = parameterized(op.gate, theta[:, op.param])
            Ud = dagger(U)
            phi = apply_matrix(phi, Ud, op.qubits, n)          # state before this gate
            mu = apply_matrix(phi, dU, op.qubits, n)            # dU/dtheta |phi>
            for o, lam in enumerate(lams):
                # += so a reused parameter accumulates every gate's contribution
                out[o, :, op.param] += 2 * np.real(np.sum(np.conj(lam) * mu, axis=1))
            lams = [apply_matrix(lam, Ud, op.qubits, n) for lam in lams]
        return out

    @staticmethod
    def _check_parameter_shift(program: NumpyProgram) -> None:
        uses = Counter(op.param for op in program.ops if op.param is not None)
        bad_gates = sorted({op.gate.value for op in program.ops if op.param is not None and op.gate not in ROTATIONS})
        reused = sorted(k for k, count in uses.items() if count > 1)
        if bad_gates or reused:
            raise ValueError(
                "The two-term parameter-shift rule is only exact for rx/ry/rz/r1 with each parameter "
                f"used once (found gates {bad_gates}, reused parameters {reused}). "
                "Use method='adjoint' or 'finite_difference'."
            )

    @staticmethod
    def _check_state(state, dim: int) -> np.ndarray:
        amplitudes = np.asarray(state, dtype=complex).reshape(-1)
        if amplitudes.size != dim:
            raise ValueError(f"Each initial state must have {dim} amplitudes, got {amplitudes.size}.")
        if not np.isclose(np.vdot(amplitudes, amplitudes).real, 1.0):
            raise ValueError("Each initial state must be normalized.")
        return amplitudes
