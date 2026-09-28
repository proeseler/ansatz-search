"""Parameter-shift gradients of Qiskit circuits through any Qiskit EstimatorV2: exact, simulated device, or hardware."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector

from ansatz_search.backends.base import GradientProvider

from ._common import (checked_state, for_device, parameter_columns, parameter_plus_constant, pauli_ops, portable,
                      preparation, require_circuit, sample_matrix)

_TWO_TERM = ((np.pi / 2, 0.5),)
_C_PLUS, _C_MINUS = (np.sqrt(2) + 1) / (4 * np.sqrt(2)), (np.sqrt(2) - 1) / (4 * np.sqrt(2))
# Controlled rotations have generator eigenvalues 0 and ±1/2 (two frequencies), which needs four terms.
_FOUR_TERM = ((np.pi / 2, _C_PLUS), (3 * np.pi / 2, -_C_MINUS))

# df/dtheta = sum of c * (f(theta + s) - f(theta - s)) over the (shift s, coefficient c) pairs of the gate.
SHIFT_RULES = {
    "rx": _TWO_TERM, "ry": _TWO_TERM, "rz": _TWO_TERM, "p": _TWO_TERM, "cp": _TWO_TERM,
    "rxx": _TWO_TERM, "ryy": _TWO_TERM, "rzz": _TWO_TERM,
    "crx": _FOUR_TERM, "cry": _FOUR_TERM, "crz": _FOUR_TERM,
}


def _split_parameters(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[int], list[tuple]]:
    """The circuit with its own parameter for every parameterized gate occurrence.

    Returns that circuit, the `param_samples` column each occurrence reads, and
    each occurrence's shift rule. Summing the per-occurrence derivatives of a
    reused parameter gives its exact gradient (chain rule).
    """
    columns = dict(zip(circuit.parameters, parameter_columns(circuit.parameters)))
    symbolic = [inst for inst in circuit.data if any(isinstance(p, ParameterExpression) for p in inst.operation.params)]
    phi = ParameterVector("phi", len(symbolic))
    split = circuit.copy_empty_like()
    occurrence_columns, rules = [], []
    for inst in circuit.data:
        op = inst.operation
        if not any(isinstance(p, ParameterExpression) for p in op.params):
            split.append(op, inst.qubits, inst.clbits)
            continue
        linear = parameter_plus_constant(op.params[0]) if len(op.params) == 1 else None
        if op.name not in SHIFT_RULES or linear is None:
            raise NotImplementedError(f"No parameter-shift rule for gate {op.name!r} with parameters {op.params}; "
                                      "each angle must be a parameter θ or θ + c.")
        parameter, offset = linear
        shifted = op.to_mutable()
        shifted.params = [phi[len(rules)] + offset if offset else phi[len(rules)]]
        split.append(shifted, inst.qubits, inst.clbits)
        occurrence_columns.append(columns[parameter])
        rules.append(SHIFT_RULES[op.name])
    return split, occurrence_columns, rules


class QiskitGradientProvider(GradientProvider):
    """Exact parameter-shift gradients through a Qiskit EstimatorV2.

    Every parameterized gate occurrence is shifted with its own rule (two terms
    for rx/ry/rz/p/cp/rxx/ryy/rzz, four for crx/cry/crz), so controlled
    rotations and reused parameters are exact. All shifted circuits for up to `max_bindings` parameter
    sets go to the estimator as one job.

    - No arguments: Qiskit's exact StatevectorEstimator (slow; for simulation the
      NumPy backend is much faster).
    - `backend=`: transpiles for that device and, without an explicit
      `estimator`, runs there with qiskit-ibm-runtime's EstimatorV2: a fake IBM
      device (local simulation with its noise) or real hardware. Needs the `ibm`
      extra. `precision` sets the target standard error per expectation value;
      `seed` fixes the transpiled qubit layout.

    On hardware each expectation value carries shot noise (variance about
    precision²), which adds to the variance of the gradients: a gradient variance
    measured there is biased upwards.
    """

    def __init__(
        self,
        estimator: Any = None,
        backend: Any = None,
        precision: float | None = None,
        optimization_level: int = 1,
        max_bindings: int = 50_000,
        seed: int | None = None,
    ):
        self.estimator = estimator
        self.backend = backend
        self.precision = precision
        self.optimization_level = optimization_level
        self.max_bindings = max_bindings
        self.seed = seed

    def _estimator(self):
        if self.estimator is not None:
            return self.estimator
        if self.backend is not None:
            from qiskit_ibm_runtime import EstimatorV2

            return EstimatorV2(mode=self.backend)
        from qiskit.primitives import StatevectorEstimator

        return StatevectorEstimator()

    def gradients(self, circuit: QuantumCircuit, observables: Any, param_samples: np.ndarray,
                  states: Sequence[Any]) -> np.ndarray:
        """Return shape (states, observables, parameter samples, parameters)."""
        circuit = require_circuit(circuit, type(self).__name__)
        n = circuit.num_qubits
        split, occurrence_columns, rules = _split_parameters(circuit)
        samples = sample_matrix(param_samples, occurrence_columns)
        ops = pauli_ops(observables, n)
        initial = [checked_state(state, n) for state in states]
        if not initial:
            raise ValueError("Provide at least one initial state.")

        grads = np.zeros((len(initial), len(ops)) + samples.shape)
        terms = [(k, shift, coeff) for k, rule in enumerate(rules) for shift, coeff in rule]
        if not terms:
            return grads
        estimator = self._estimator()
        run_options = {} if self.precision is None else {"precision": self.precision}
        chunk = max(1, self.max_bindings // (2 * len(terms)))
        occurrence_values = samples[:, occurrence_columns]  # (samples, occurrences)

        guarded = self.estimator is not None or self.backend is not None  # anything but the exact reference
        for s, amplitudes in enumerate(initial):
            prep = preparation(amplitudes, n)
            circuit_s = split if prep is None else prep.compose(split)
            if guarded:
                circuit_s = portable(circuit_s)
            device_circuit = for_device(circuit_s, self.backend, self.optimization_level, self.seed)
            order = [p.index for p in device_circuit.parameters]  # transpiled parameter order -> occurrence
            layout = device_circuit.layout
            device_ops = [[[op if layout is None else op.apply_layout(layout)]] for op in ops]  # (observables, 1, 1)
            for start in range(0, len(samples), chunk):
                base = occurrence_values[start:start + chunk]
                points = np.repeat(base[None], 2 * len(terms), axis=0)  # (2 * terms, samples, occurrences)
                for t, (k, shift, _) in enumerate(terms):
                    points[2 * t, :, k] += shift
                    points[2 * t + 1, :, k] -= shift
                job = estimator.run([(device_circuit, device_ops, points[..., order])], **run_options)
                evs = np.asarray(job.result()[0].data.evs)  # (observables, 2 * terms, samples)
                for t, (k, _, coeff) in enumerate(terms):
                    grads[s, :, start:start + chunk, occurrence_columns[k]] += coeff * (evs[:, 2 * t] - evs[:, 2 * t + 1])
        return grads
