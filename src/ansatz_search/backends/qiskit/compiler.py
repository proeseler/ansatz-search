"""Compile backend-independent ansatz specifications into Qiskit circuits."""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression

from ansatz_search.backends.base import Compiler
from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef
from ansatz_search.circuit.gates import GateName, parse_gate_name

from ._common import parameter_columns, parameter_plus_constant


def _qiskit_angle(angle: ParamRef | float, params: list[Parameter]):
    if not isinstance(angle, ParamRef):
        return angle
    parameter = params[angle.index]
    return parameter + angle.offset if angle.offset else parameter


def compile_to_qiskit(spec: AnsatzSpec) -> QuantumCircuit:
    """Compile a backend-neutral ansatz specification to a Qiskit circuit."""

    qc = QuantumCircuit(spec.num_qubits)
    params = [Parameter(f"theta_{idx}") for idx in range(spec.num_params)]

    for block in spec:
        gate = parse_gate_name(block.op)
        qubits = list(block.qubits)
        values = [_qiskit_angle(p, params) for p in block.params]

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
        elif gate is GateName.SDG:
            qc.sdg(*qubits)
        elif gate is GateName.TDG:
            qc.tdg(*qubits)
        elif gate is GateName.SX:
            qc.sx(*qubits)
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
        elif gate is GateName.ECR:
            qc.ecr(*qubits)
        elif gate is GateName.RXX:
            qc.rxx(values[0], *qubits)
        elif gate is GateName.RYY:
            qc.ryy(values[0], *qubits)
        elif gate is GateName.RZZ:
            qc.rzz(values[0], *qubits)
        else:
            raise NotImplementedError(f"Qiskit compilation is not implemented for gate '{gate.value}'.")

    return qc


# u(θ, φ, λ) = rz(φ) ry(θ) rz(λ) up to a global phase, which no metric sees: the rotations in
# circuit order, each with the index of its angle in (θ, φ, λ).
_EULER_ROTATIONS = ((GateName.RZ, 2), (GateName.RY, 0), (GateName.RZ, 1))


def spec_from_qiskit(qc: QuantumCircuit) -> AnsatzSpec:
    """Convert a parameterized Qiskit circuit to an AnsatzSpec: the inverse of compile_to_qiskit.

    Every gate must be one QiskitCompiler supports, or u/u3, which becomes
    rz, ry, rz. Every angle must be a Parameter θ (reuse is fine), θ + c as in
    transpiled circuits, or a number, which becomes a fixed angle; barriers
    are skipped. Parameters named `theta_j` keep index j, others are numbered in
    Qiskit's order, so a metric draws the same samples for the spec as for the
    circuit itself.
    """
    parameters = list(qc.parameters)
    index = dict(zip(parameters, parameter_columns(parameters)))
    supported = sorted(gate.value for gate in QiskitCompiler.supported_gates)
    blocks = []
    for instruction in qc.data:
        op = instruction.operation
        if op.name == "barrier":
            continue
        euler = op.name in ("u", "u3")
        try:
            gate = parse_gate_name(op.name)
        except ValueError:
            gate = None
        if not euler and gate not in QiskitCompiler.supported_gates:
            raise ValueError(f"Gate '{op.name}' is not supported; transpile the circuit to these gates first, "
                             f"e.g. transpile(qc, basis_gates=['rx', 'ry', 'rz', 'cx']): {supported}.")
        refs = []
        for angle in op.params:
            linear = parameter_plus_constant(angle)
            if linear is not None:
                parameter, offset = linear
                refs.append(ParamRef(index[parameter], offset))
            elif isinstance(angle, ParameterExpression) and angle.parameters:
                raise ValueError(f"Gate '{op.name}' has the angle {angle}, but only a parameter θ, θ + c and fixed "
                                 "angles are supported, not other expressions of parameters.")
            else:
                refs.append(float(angle))
        qubits = tuple(qc.find_bit(q).index for q in instruction.qubits)
        if euler:
            blocks.extend(AnsatzBlock(rotation, qubits, (refs[i],)) for rotation, i in _EULER_ROTATIONS)
        else:
            blocks.append(AnsatzBlock(gate, qubits, tuple(refs)))
    return AnsatzSpec(qc.num_qubits, blocks)


class QiskitCompiler(Compiler):
    """Compile an AnsatzSpec to a parameterized Qiskit QuantumCircuit."""

    # Must match the gates compile_to_qiskit implements.
    supported_gates = frozenset({
        GateName.X, GateName.Y, GateName.Z, GateName.H, GateName.S, GateName.T,
        GateName.SDG, GateName.TDG, GateName.SX,
        GateName.RX, GateName.RY, GateName.RZ, GateName.R1,
        GateName.CX, GateName.CY, GateName.CZ, GateName.CH, GateName.SWAP, GateName.ECR,
        GateName.CRX, GateName.CRY, GateName.CRZ, GateName.CR1,
        GateName.RXX, GateName.RYY, GateName.RZZ,
    })

    def compile(self, spec: AnsatzSpec) -> QuantumCircuit:
        self.validate(spec)
        return compile_to_qiskit(spec)
