"""Batched NumPy backend: states against Qiskit, gradients against finite differences."""

import numpy as np
import pytest

from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider
from ansatz_search.backends.numpy.gradients import statevectors
from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef
from ansatz_search.circuit.gates import GATE_SHAPES

from _circuits import OBSERVABLES, mixed_circuit, random_state, rotation_circuit

qiskit = pytest.importorskip("qiskit")
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402
from ansatz_search.backends.qiskit.compiler import compile_to_qiskit  # noqa: E402

GATES = sorted(NumpyCompiler.supported_gates, key=lambda g: g.value)


@pytest.mark.parametrize("gate", GATES, ids=lambda g: g.value)
@pytest.mark.parametrize("placement", [0, 1])
def test_states_match_qiskit(gate, placement):
    nq, npar = GATE_SHAPES[gate]
    qubits = [(2,), (0,)][placement] if nq == 1 else [(2, 0), (0, 1)][placement]
    spec = AnsatzSpec(3, [AnsatzBlock(gate, qubits, tuple(ParamRef(i) for i in range(npar)))])
    circuit = compile_to_qiskit(spec)
    circuit = circuit.assign_parameters({p: 0.37 for p in circuit.parameters})
    state = random_state(3)
    expected = Statevector(state).evolve(circuit).data
    actual = statevectors(NumpyCompiler()(spec), np.full((1, max(npar, 1)), 0.37), state)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_pauli_convention_matches_qiskit():
    """Character i of a Pauli string acts on qubit i (Qiskit labels are reversed)."""
    from ansatz_search.backends.numpy.simulator import PauliSumOperator
    from ansatz_search.circuit.observables import pauli_sum

    state = random_state(3)
    for paulis in ["XIY", "ZYI", "IXZ", "YYX"]:
        mine = PauliSumOperator(pauli_sum({paulis: 0.7}, 3), 3).expectation(state[None])[0]
        theirs = Statevector(state).expectation_value(SparsePauliOp(paulis[::-1], 0.7)).real
        assert mine == pytest.approx(theirs, abs=1e-12)


def test_adjoint_matches_finite_differences():
    spec = mixed_circuit()
    theta = np.random.default_rng(0).normal(size=(6, spec.num_params))
    states = [random_state(3), np.eye(8)[0]]
    program = NumpyCompiler()(spec)
    adjoint = NumpyGradientProvider().gradients(program, OBSERVABLES, theta, states)
    fd = NumpyGradientProvider("finite_difference", epsilon=1e-6).gradients(program, OBSERVABLES, theta, states)
    assert adjoint.shape == (2, 2, 6, spec.num_params)
    np.testing.assert_allclose(adjoint, fd, atol=1e-7)


def test_adjoint_matches_parameter_shift_where_it_is_exact():
    spec = rotation_circuit()
    theta = np.random.default_rng(0).normal(size=(6, spec.num_params))
    program = NumpyCompiler()(spec)
    adjoint = NumpyGradientProvider().gradients(program, OBSERVABLES, theta, [random_state(3)])
    shift = NumpyGradientProvider("parameter_shift").gradients(program, OBSERVABLES, theta, [random_state(3)])
    np.testing.assert_allclose(adjoint, shift, atol=1e-12)


@pytest.mark.parametrize("blocks, reason", [
    ([AnsatzBlock("crx", (0, 1), (ParamRef(0),))], "crx"),
    ([AnsatzBlock("rx", (0,), (ParamRef(0),)), AnsatzBlock("ry", (1,), (ParamRef(0),))], "reused parameters \\[0\\]"),
])
def test_parameter_shift_refuses_where_it_would_be_wrong(blocks, reason):
    program = NumpyCompiler()(AnsatzSpec(2, blocks))
    with pytest.raises(ValueError, match=reason):
        NumpyGradientProvider("parameter_shift").gradients(program, "ZZ", np.zeros((2, 1)), [np.eye(4)[0]])


def test_chunking_does_not_change_results():
    spec = mixed_circuit()
    theta = np.random.default_rng(0).normal(size=(9, spec.num_params))
    program = NumpyCompiler()(spec)
    whole = NumpyGradientProvider().gradients(program, OBSERVABLES, theta, [random_state(3)])
    tiny = NumpyGradientProvider(max_batch_bytes=1).gradients(program, OBSERVABLES, theta, [random_state(3)])
    np.testing.assert_allclose(tiny, whole, atol=1e-13)


def test_extra_parameter_columns_get_zero_gradient():
    spec = rotation_circuit()
    theta = np.random.default_rng(0).normal(size=(3, spec.num_params + 2))
    g = NumpyGradientProvider().gradients(NumpyCompiler()(spec), "ZZZ", theta, [np.eye(8)[0]])
    assert g.shape == (1, 1, 3, spec.num_params + 2)
    assert np.all(g[..., spec.num_params:] == 0)


def test_rejects_foreign_circuits_and_unsupported_methods():
    with pytest.raises(TypeError, match="NumpyCompiler"):
        NumpyGradientProvider().gradients(object(), "Z", np.zeros((2, 1)), [[1, 0]])
    with pytest.raises(NotImplementedError, match="backprop"):
        NumpyGradientProvider("backprop")


def test_state_provider_matches_qiskit_and_chunks():
    from ansatz_search.backends.numpy import NumpyStateProvider

    spec = rotation_circuit()
    theta = np.random.default_rng(0).normal(size=(5, spec.num_params))
    state = random_state(3)
    program = NumpyCompiler()(spec)
    states = NumpyStateProvider().states(program, theta, state)
    circuit = compile_to_qiskit(spec)
    for row, psi in zip(theta, states):
        bound = circuit.assign_parameters(dict(zip(circuit.parameters, row)))
        np.testing.assert_allclose(psi, Statevector(state).evolve(bound).data, atol=1e-12)
    np.testing.assert_allclose(NumpyStateProvider(max_batch_bytes=1).states(program, theta, state), states, atol=1e-14)
