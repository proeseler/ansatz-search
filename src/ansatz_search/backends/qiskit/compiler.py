"""Compile backend-independent ansatz specifications into Qiskit circuits."""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from ansatz_search.backends.base import Compiler
from ansatz_search.circuit.ansatz import AnsatzSpec
from ansatz_search.circuit.gates import GateName, parse_gate_name


def compile_to_qiskit(spec: AnsatzSpec) -> QuantumCircuit:
    """Compile a backend-neutral ansatz specification to a Qiskit circuit."""

    qc = QuantumCircuit(spec.num_qubits)
    params = [Parameter(f"theta_{idx}") for idx in range(spec.num_params)]

    for block in spec:
        gate = parse_gate_name(block.op)
        qubits = list(block.qubits)
        values = [params[param.index] for param in block.params]

        if gate is GateName.X:
            qc.x(*qubits)
        elif gate is GateName.Y:
            qc.y(*qubits)
        elif gate is GateName.Z:
            qc.z(*qubits)
        elif gate is GateName.H:
            qc.h(*qubits)
        elif gate is GateName.S:
            qc.s(*qubits)
        elif gate is GateName.T:
            qc.t(*qubits)
        elif gate is GateName.RX:
            qc.rx(values[0], *qubits)
        elif gate is GateName.RY:
            qc.ry(values[0], *qubits)
        elif gate is GateName.RZ:
            qc.rz(values[0], *qubits)
        elif gate is GateName.R1:
            qc.p(values[0], *qubits)
        elif gate is GateName.CX:
            qc.cx(*qubits)
        elif gate is GateName.CY:
            qc.cy(*qubits)
        elif gate is GateName.CZ:
            qc.cz(*qubits)
        elif gate is GateName.CH:
            qc.ch(*qubits)
        elif gate is GateName.CRX:
            qc.crx(values[0], *qubits)
        elif gate is GateName.CRY:
            qc.cry(values[0], *qubits)
        elif gate is GateName.CRZ:
            qc.crz(values[0], *qubits)
        elif gate is GateName.CR1:
            qc.cp(values[0], *qubits)
        elif gate is GateName.SWAP:
            qc.swap(*qubits)
        else:
            raise NotImplementedError(f"Qiskit compilation is not implemented for gate '{gate.value}'.")

    return qc


class QiskitCompiler(Compiler):
    """Compile an AnsatzSpec to a parameterized Qiskit QuantumCircuit."""

    # Must match the gates compile_to_qiskit implements.
    supported_gates = frozenset({
        GateName.X, GateName.Y, GateName.Z, GateName.H, GateName.S, GateName.T,
        GateName.RX, GateName.RY, GateName.RZ, GateName.R1,
        GateName.CX, GateName.CY, GateName.CZ, GateName.CH, GateName.SWAP,
        GateName.CRX, GateName.CRY, GateName.CRZ, GateName.CR1,
    })

    def compile(self, spec: AnsatzSpec) -> QuantumCircuit:
        self.validate(spec)
        return compile_to_qiskit(spec)
