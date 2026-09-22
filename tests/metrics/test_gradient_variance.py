"""Test variance reduction independently of backend gradient computation."""

import numpy as np
import pytest


@pytest.mark.parametrize("constant", [False, True], ids=["varying", "constant"])
def test_gradient_variance_reduction(constant):
    pytest.importorskip("qiskit", reason="GradientVariance currently depends on Qiskit")
    from ansatz_search.backends.base import GradientProvider
    from ansatz_search.metrics.gradient_variance import GradientVariance

    samples = np.array([[0.0], [np.pi / 2], [np.pi]])
    if constant:
        samples[:] = 0.3
    states = [[1, 0], [0, 1]]
    observables = ["Z", "X"]  # Labels consumed only by the test provider.
    theta = samples[:, 0]
    # Analytical gradients of RY(theta) for |0> and |1>, with Z and X.
    analytical = np.array([
        [-np.sin(theta), np.cos(theta)],
        [np.sin(theta), -np.cos(theta)],
    ])[..., None]

    class AnalyticalProvider(GradientProvider):
        def gradients(self, circuit, observables, param_samples, states):
            assert circuit is sentinel
            assert observables == ["Z", "X"]
            assert states == [[1, 0], [0, 1]]
            np.testing.assert_array_equal(param_samples, samples)
            return analytical.copy()

    sentinel = object()
    metric = GradientVariance(
        observables=observables,
        num_of_param_samples=len(samples),
        gradient_provider=AnalyticalProvider(),
    )
    # value_from_samples bypasses sampling so the expected input is deterministic.
    actual = metric.value_from_samples(sentinel, samples, states)
    # Z sample variance = 1/3, X sample variance = 1; average = 2/3.
    expected = 0.0 if constant else 2 / 3
    assert actual == pytest.approx(expected, abs=1e-12)


def _metric(**kwargs):
    pytest.importorskip("qiskit", reason="GradientVariance currently depends on Qiskit")
    from ansatz_search.backends.base import GradientProvider
    from ansatz_search.metrics.gradient_variance import GradientVariance

    class Unused(GradientProvider):
        def gradients(self, *args, **kwargs):
            raise AssertionError("not called")

    return GradientVariance(observables=None, num_of_param_samples=2, gradient_provider=Unused(), **kwargs)


@pytest.mark.parametrize("variance, max_variance, cost", [
    (0.0, None, 1.0),    # barren plateau: worst cost
    (0.25, None, 0.75),  # default max_variance 1.0 -> cost = 1 - variance
    (1.0, None, 0.0),    # maximal variance of clipped gradients
    (1.2, None, 0.0),    # ddof=1 can slightly exceed 1; clipped, never negative
    (0.25, 0.5, 0.5),    # custom normalization
])
def test_cost_normalizes_the_raw_variance(variance, max_variance, cost):
    kwargs = {} if max_variance is None else {"max_variance": max_variance}
    assert _metric(**kwargs).cost(variance) == pytest.approx(cost)


def test_more_trainable_ansatz_gets_a_lower_cost():
    metric = _metric()
    assert metric.cost(0.4) < metric.cost(0.01)


def test_compute_is_the_cost_of_the_value(monkeypatch):
    metric = _metric()
    monkeypatch.setattr(metric, "value", lambda qc, spec=None: 0.3)
    assert metric.compute(None) == pytest.approx(0.7)


@pytest.mark.parametrize("max_variance", [0.0, -0.1])
def test_max_variance_must_be_positive(max_variance):
    with pytest.raises(ValueError, match="max_variance must be positive"):
        _metric(max_variance=max_variance)
