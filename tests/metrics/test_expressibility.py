"""Expressibility against analytic Haar references and across backends."""

import numpy as np
import pytest

from ansatz_search.backends.base import StateProvider
from ansatz_search.backends.numpy import NumpyCompiler, NumpyStateProvider
from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef
from ansatz_search.metrics.expressibility import Expressibility

scipy_stats = pytest.importorskip("scipy.stats")


class HaarStates(StateProvider):
    """An ideal ansatz: every output state is Haar-random."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def states(self, circuit, param_samples, initial_state):
        z = self.rng.normal(size=(len(param_samples), 8)) + 1j * self.rng.normal(size=(len(param_samples), 8))
        return z / np.linalg.norm(z, axis=1, keepdims=True)


def layers(k):
    blocks, p = [], 0
    for _ in range(k):
        for q in range(3):
            blocks += [AnsatzBlock("ry", (q,), (ParamRef(p),)), AnsatzBlock("rz", (q,), (ParamRef(p + 1),))]
            p += 2
        blocks += [AnsatzBlock("cx", (0, 1)), AnsatzBlock("cx", (1, 2))]
    return AnsatzSpec(3, blocks)


def metric(provider=None, **kwargs):
    kwargs.setdefault("num_of_param_samples", 4000)
    kwargs.setdefault("seed", 0)
    return Expressibility(num_qubits=3, state_provider=provider or NumpyStateProvider(), **kwargs)


def value(m, spec):
    return m.value(NumpyCompiler()(spec), spec)


@pytest.mark.parametrize("epsilon", [0.0, 0.01])
def test_haar_distribution_matches_beta(epsilon):
    m = metric(epsilon=epsilon)
    D = 8
    regular = m.num_bins - 1 if epsilon else m.num_bins
    edges = np.linspace(0, m.t_max, regular + 1)
    # Differences of the survival function: CDF differences lose precision in the far tail.
    expected = -np.diff(scipy_stats.beta(1, D - 1).sf(edges))
    if epsilon:
        expected = np.append(expected, epsilon)
        assert scipy_stats.beta(1, D - 1).sf(m.t_max) == pytest.approx(epsilon)
    np.testing.assert_allclose(m.haar_distr, expected / expected.sum(), rtol=1e-9)


def test_haar_random_states_cost_almost_nothing():
    m = metric(HaarStates(1))
    v = m.value(object(), layers(1))
    assert m.noise_floor / 2 < v < 2 * m.noise_floor
    assert m.cost(v) < 1e-3


def test_idle_circuit_is_the_worst_case():
    idle = AnsatzSpec(3, [AnsatzBlock("rz", (0,), (ParamRef(0),))])  # only a phase: every fidelity is 1
    m = metric()
    assert value(m, idle) == pytest.approx(m.max_value)
    assert m.cost(value(m, idle)) == pytest.approx(1.0)


def test_more_layers_are_more_expressive():
    values = [value(metric(), layers(k)) for k in (1, 2, 4)]
    assert values[0] > values[1] > values[2]


def test_fidelities_rounded_above_one_are_still_counted():
    hist = metric()._histogram(np.array([1 + 1e-15, 0.5]))
    assert hist.sum() == pytest.approx(1.0) and hist[-1] == pytest.approx(0.5)


def test_cost_scales_agree_at_the_ends_and_log_spreads_the_middle():
    linear, log = metric(), metric(log_scale=True)
    for m in (linear, log):
        assert m.cost(m.noise_floor) == 0.0
        assert m.cost(m.max_value) == pytest.approx(1.0)
    middle = 20 * linear.noise_floor
    assert log.cost(middle) > 10 * linear.cost(middle)


def test_same_seed_is_reproducible():
    assert value(metric(), layers(2)) == value(metric(), layers(2))
    assert metric().noise_floor == metric().noise_floor


def test_matches_across_backends():
    pytest.importorskip("pennylane")
    from ansatz_search.backends.pennylane import PennyLaneCompiler, PennyLaneStateProvider

    spec = layers(2)
    numpy_value = value(metric(), spec)
    pl_value = metric(PennyLaneStateProvider()).value(PennyLaneCompiler()(spec), spec)
    assert pl_value == pytest.approx(numpy_value, abs=1e-12)


@pytest.mark.parametrize("kwargs, match", [
    ({"num_of_param_samples": 101}, "even"),
    ({"num_of_param_samples": 2}, "at least 4"),
    ({"epsilon": 1.0}, "epsilon"),
])
def test_invalid_settings_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        metric(**kwargs)


def test_state_provider_is_optional():
    # Resolved from the compiled circuit's backend (see tests/backends/test_default_providers.py).
    assert Expressibility(num_qubits=3)._state_provider is None


def test_wrong_qubit_count_is_rejected():
    spec = AnsatzSpec(2, [AnsatzBlock("ry", (0,), (ParamRef(0),))])
    with pytest.raises(ValueError, match="3 qubits, got a 2-qubit"):
        value(metric(), spec)
