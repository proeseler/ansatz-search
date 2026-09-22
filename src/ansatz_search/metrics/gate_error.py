"""Gate-count error proxy: the probability that at least one gate fails (arXiv:2603.14451, Eq. B1)."""

from __future__ import annotations

from typing import Sequence

from .base import Metric

_NON_GATES = ("barrier", "measure", "delay", "reset")


class GateError(Metric):
    """Heuristic circuit error probability ``1 - (1 - p1_err)^N1 * (1 - p2_err)^N2``.

    N1 and N2 are the numbers of one- and two-qubit gates. By default they are
    counted in the backend-neutral AnsatzSpec, exactly as the circuit is
    written, so the metric works with every backend. With `basis_gates`, the
    circuit is first transpiled to those gates with Qiskit (at
    `optimization_level`), e.g. to estimate the error on a device's native gates.
    """

    label = "Gate error"

    def __init__(
        self,
        p1_err: float = 0.001,
        p2_err: float = 0.01,
        basis_gates: Sequence[str] | None = None,
        optimization_level: int = 1,
    ):
        for name, p in (("p1_err", p1_err), ("p2_err", p2_err)):
            if not 0 <= p < 1:
                raise ValueError(f"{name} must be in [0, 1), got {p}.")
        self.p1_err = p1_err
        self.p2_err = p2_err
        self.basis_gates = basis_gates
        self.optimization_level = optimization_level

    @property
    def name(self) -> str:
        return "gate_error"

    def gate_counts(self, qc, spec=None) -> tuple[int, int]:
        """(one-qubit gates, two-qubit gates) of the circuit, transpiled first if `basis_gates` is set."""
        if self.basis_gates is None and spec is not None:
            sizes = [len(block.qubits) for block in spec]
        else:
            from qiskit import QuantumCircuit, transpile

            circuit = qc
            if spec is not None:
                from ansatz_search.backends.qiskit.compiler import compile_to_qiskit

                circuit = compile_to_qiskit(spec)
            elif not isinstance(circuit, QuantumCircuit):
                raise TypeError("GateError needs the circuit's AnsatzSpec (spec=) or a Qiskit QuantumCircuit.")
            if self.basis_gates is not None:
                circuit = transpile(circuit, basis_gates=list(self.basis_gates),
                                    optimization_level=self.optimization_level)
            sizes = [inst.operation.num_qubits for inst in circuit.data if inst.operation.name not in _NON_GATES]
        if any(size > 2 for size in sizes):
            raise NotImplementedError("GateError models one- and two-qubit gates only.")
        return sizes.count(1), sizes.count(2)

    def value(self, qc, spec=None) -> float:
        n1, n2 = self.gate_counts(qc, spec)
        return 1 - (1 - self.p1_err) ** n1 * (1 - self.p2_err) ** n2
