"""Meyer-Wallach entanglement against known states and across backends."""

import numpy as np
import pytest

from ansatz_search.backends.base import StateProvider
from ansatz_search.backends.numpy import NumpyCompiler, NumpyStateProvider
from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef
from ansatz_search.metrics.entanglement import Entanglement, meyer_wallach


def basis(n, *indices_and_amps):
    psi = np.zeros(2 ** n, dtype=complex)
    for i, a in indices_and_amps:
        psi[i] = a
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("state, n, q", [
    (basis(3, (0, 1)), 3, 0.0),                                  # |000>: product
    (np.kron([1, 1], np.kron([1, 1j], [1, 0])) / 2, 3, 0.0),     # product of superpositions
    (basis(2, (0, 1), (3, 1)), 2, 1.0),                          # Bell pair
    (basis(3, (0, 1), (7, 1)), 3, 1.0),                          # GHZ
    (basis(3, (1, 1), (2, 1), (4, 1)), 3, 8 / 9),                # W: each qubit rho = diag(2/3, 1/3)
])
def test_known_states(state, n, q):
    assert meyer_wallach(state[None], n)[0] == pytest.approx(q)


class HaarStates(StateProvider):
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def states(self, circuit, param_samples, initial_state):
        z = self.rng.normal(size=(len(param_samples), 16)) + 1j * self.rng.normal(size=(len(param_samples), 16))
        return z / np.linalg.norm(z, axis=1, keepdims=True)


def spec_with(blocks, n=4):
    return AnsatzSpec(n, blocks)


def metric(provider=None, **kwargs):
    kwargs.setdefault("num_of_param_samples", 2000)
    kwargs.setdefault("seed", 0)
    return Entanglement(num_qubits=4, state_provider=provider or NumpyStateProvider(), **kwargs)


def value(m, spec):
    return m.value(NumpyCompiler()(spec), spec)


def test_haar_random_states_reach_the_reference():
    m = metric(HaarStates(1))
    v = m.value(object(), spec_with([AnsatzBlock("ry", (0,), (ParamRef(0),))]))
    assert v == pytest.approx(m.reference, abs=0.01)   # (16 - 2) / (16 + 1)
    assert m.cost(v) < 0.02


def test_circuit_without_entangling_gates_costs_one():
    spec = spec_with([AnsatzBlock("ry", (q,), (ParamRef(q),)) for q in range(4)])
    m = metric()
    assert value(m, spec) == pytest.approx(0.0, abs=1e-12)
    assert m.cost(value(m, spec)) == pytest.approx(1.0)


def test_ghz_circuit_is_maximally_entangled():
    # h + cx chain gives a GHZ state; rz only adds a relative phase, which keeps Q = 1
    spec = spec_with([AnsatzBlock("h", (0,)), AnsatzBlock("rz", (0,), (ParamRef(0),)),
                      AnsatzBlock("cx", (0, 1)), AnsatzBlock("cx", (1, 2)), AnsatzBlock("cx", (2, 3))])
    m = metric(reference=1.0)
    assert value(m, spec) == pytest.approx(1.0)
    assert m.cost(value(m, spec)) == pytest.approx(0.0)


def test_reference_one_gives_one_minus_q():
    assert metric(reference=1.0).cost(0.3) == pytest.approx(0.7)


def test_matches_across_backends():
    pytest.importorskip("pennylane")
    from ansatz_search.backends.pennylane import PennyLaneCompiler, PennyLaneStateProvider

    spec = spec_with([AnsatzBlock("ry", (q,), (ParamRef(q),)) for q in range(4)]
                     + [AnsatzBlock("crx", (0, 1), (ParamRef(4),)), AnsatzBlock("cz", (1, 2)),
                        AnsatzBlock("cx", (2, 3)), AnsatzBlock("rz", (3,), (ParamRef(5),))])
    numpy_value = value(metric(), spec)
    pl_value = metric(PennyLaneStateProvider()).value(PennyLaneCompiler()(spec), spec)
    assert pl_value == pytest.approx(numpy_value, abs=1e-12)


def test_invalid_settings_are_rejected():
    # No state_provider is fine: the backend of the compiled circuit supplies one.
    assert Entanglement(num_qubits=4)._state_provider is None
    with pytest.raises(ValueError, match="reference must be in"):
        metric(reference=1.5)
    with pytest.raises(ValueError, match="single qubit"):
        Entanglement(num_qubits=1, state_provider=NumpyStateProvider())


def test_wrong_qubit_count_is_rejected():
    with pytest.raises(ValueError, match="4 qubits, got a 2-qubit"):
        value(metric(), AnsatzSpec(2, [AnsatzBlock("ry", (0,), (ParamRef(0),))]))
