"""AnsatzSpec.from_qiskit inverts QiskitCompiler and keeps the circuit's parameters and qubits."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import U3Gate, efficient_su2

from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider, NumpyStateProvider
from ansatz_search.backends.qiskit import QiskitCompiler, QiskitGradientProvider, QiskitStateProvider
from ansatz_search.benchmarks import SIM2019
from ansatz_search.circuit.ansatz import AnsatzSpec, ParamRef
from ansatz_search.metrics import Expressibility

from _circuits import mixed_circuit, random_state


@pytest.mark.parametrize("spec", [mixed_circuit(), *(SIM2019[i]() for i in sorted(SIM2019))],
                         ids=["mixed", *(f"sim{i}" for i in sorted(SIM2019))])
def test_round_trip_through_qiskit(spec):
    assert AnsatzSpec.from_qiskit(QiskitCompiler()(spec)) == spec


def _own_circuit():
    """Two registers, a ParameterVector and named parameters in non-alphabetical use, one reused, fixed angles."""
    a, b = QuantumRegister(2, "a"), QuantumRegister(1, "b")
    theta, gamma, alpha = ParameterVector("θ", 3), Parameter("gamma"), Parameter("alpha")
    qc = QuantumCircuit(a, b)
    qc.h(a[0])
    qc.ry(gamma, b[0])
    qc.crz(theta[2], b[0], a[1])
    qc.barrier()
    qc.rx(alpha, a[1])
    qc.cp(theta[0], a[1], a[0])
    qc.rz(gamma, a[0])
    qc.swap(a[0], b[0])
    qc.ry(theta[1], a[0])
    qc.rx(0.4, b[0])
    qc.rzz(-1.1, a[0], a[1])
    return qc


@pytest.mark.parametrize("qc", [_own_circuit(), efficient_su2(3, reps=2)], ids=["own", "efficient_su2"])
def test_converted_circuit_has_the_same_states(qc):
    spec = AnsatzSpec.from_qiskit(qc)
    assert (spec.num_qubits, spec.num_params) == (qc.num_qubits, qc.num_parameters)
    theta = np.random.default_rng(3).uniform(0, 2 * np.pi, (4, qc.num_parameters))
    state = random_state(3)
    np.testing.assert_allclose(NumpyStateProvider().states(NumpyCompiler()(spec), theta, state),
                               QiskitStateProvider().states(qc, theta, state), atol=1e-10)


def test_u_gates_become_euler_rotations():
    a, b, c = Parameter("a"), Parameter("b"), Parameter("c")
    qc = QuantumCircuit(2)
    qc.u(a, b, c, 0)
    qc.cx(0, 1)
    qc.u(c, a, a, 1)
    qc.append(U3Gate(b, c, a), [0])
    spec = AnsatzSpec.from_qiskit(qc)
    assert [block.op.value for block in spec] == ["rz", "ry", "rz", "cx", "rz", "ry", "rz", "rz", "ry", "rz"]
    theta = np.random.default_rng(4).uniform(0, 2 * np.pi, (5, 3))
    ours = NumpyStateProvider().states(NumpyCompiler()(spec), theta, np.eye(4)[0])
    theirs = QiskitStateProvider().states(qc, theta, np.eye(4)[0])
    # Equal up to a global phase, which depends on the parameters.
    np.testing.assert_allclose(np.abs(np.sum(ours.conj() * theirs, axis=1)), 1, atol=1e-10)


@pytest.mark.parametrize("basis", [["rz", "sx", "x", "cz"], ["rz", "sx", "x", "ecr"]], ids=["heron", "eagle"])
def test_transpiled_circuits_convert_with_offsets(basis):
    qc = efficient_su2(3, reps=1)
    transpiled = transpile(qc, basis_gates=basis, optimization_level=1, seed_transpiler=1)
    spec = AnsatzSpec.from_qiskit(transpiled)
    assert any(isinstance(p, ParamRef) and p.offset for block in spec for p in block.params)  # rz(θ + π)
    theta = np.random.default_rng(5).uniform(0, 2 * np.pi, (4, qc.num_parameters))
    ours = NumpyStateProvider().states(NumpyCompiler()(spec), theta, np.eye(8)[0])
    theirs = QiskitStateProvider().states(qc, theta, np.eye(8)[0])
    # Transpiling keeps the state up to a global phase.
    np.testing.assert_allclose(np.abs(np.sum(ours.conj() * theirs, axis=1)), 1, atol=1e-10)
    # Parameter shift on the transpiled circuit itself handles the offsets too.
    grads = QiskitGradientProvider().gradients(transpiled, "ZZI", theta[:2], [np.eye(8)[0]])
    exact = NumpyGradientProvider().gradients(NumpyCompiler()(spec), "ZZI", theta[:2], [np.eye(8)[0]])
    np.testing.assert_allclose(grads, exact, atol=1e-8)


def test_a_metric_gives_the_same_value_for_the_circuit_and_its_spec():
    qc = _own_circuit()
    spec = AnsatzSpec.from_qiskit(qc)
    metric = Expressibility(num_qubits=3, num_of_param_samples=200, seed=5)
    assert metric.with_seed(5).value(NumpyCompiler()(spec), spec) == pytest.approx(metric.with_seed(5).value(qc))


@pytest.mark.parametrize("build, message", [
    (lambda qc: qc.iswap(0, 1), "Gate 'iswap' is not supported"),
    (lambda qc: qc.measure_all(), "Gate 'measure' is not supported"),
    (lambda qc: qc.rx(2 * Parameter("a"), 0), "expressions"),
    (lambda qc: qc.rx(Parameter("a") - Parameter("b"), 0), "expressions"),
], ids=["gate", "measure", "scaled", "two-parameters"])
def test_unsupported_circuits_are_rejected(build, message):
    qc = QuantumCircuit(2)
    build(qc)
    with pytest.raises(ValueError, match=message):
        AnsatzSpec.from_qiskit(qc)
