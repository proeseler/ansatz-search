"""PennyLane backend: every gradient method against the NumPy adjoint."""

import numpy as np
import pytest

pytest.importorskip("pennylane")

from ansatz_search.backends.base import GradientMethod  # noqa: E402
from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider, NumpyStateProvider  # noqa: E402
from ansatz_search.backends.pennylane import PennyLaneCompiler, PennyLaneGradientProvider, PennyLaneStateProvider  # noqa: E402

from _circuits import OBSERVABLES, mixed_circuit, random_state  # noqa: E402


@pytest.fixture(scope="module")
def reference():
    spec = mixed_circuit()
    theta = np.random.default_rng(0).normal(size=(5, spec.num_params))
    states = [random_state(3), np.eye(8)[0]]
    grads = NumpyGradientProvider().gradients(NumpyCompiler()(spec), OBSERVABLES, theta, states)
    return spec, theta, states, grads


@pytest.mark.parametrize("method", ["auto", "backprop", "adjoint", "parameter_shift", "finite_difference"])
def test_matches_numpy_adjoint(reference, method):
    spec, theta, states, expected = reference
    grads = PennyLaneGradientProvider(method).gradients(PennyLaneCompiler()(spec), OBSERVABLES, theta, states)
    np.testing.assert_allclose(grads, expected, atol=1e-6 if method == "finite_difference" else 1e-9)


def test_single_sample_batch(reference):
    spec, theta, states, expected = reference
    grads = PennyLaneGradientProvider("backprop").gradients(PennyLaneCompiler()(spec), OBSERVABLES, theta[:1], states)
    np.testing.assert_allclose(grads, expected[:, :, :1], atol=1e-9)


def test_auto_switches_to_lightning_adjoint_at_twelve_qubits():
    provider = PennyLaneGradientProvider()
    assert provider.resolved_method(11) is GradientMethod.BACKPROP
    assert provider.resolved_method(12) is GradientMethod.ADJOINT


def test_rejects_foreign_circuits():
    with pytest.raises(TypeError, match="PennyLaneCompiler"):
        PennyLaneGradientProvider().gradients(object(), "Z", np.zeros((2, 1)), [[1, 0]])


@pytest.mark.parametrize("n_samples", [1, 5])
def test_state_provider_matches_numpy(reference, n_samples):
    spec, theta, states, _ = reference
    for state in states:
        mine = PennyLaneStateProvider().states(PennyLaneCompiler()(spec), theta[:n_samples], state)
        ref = NumpyStateProvider().states(NumpyCompiler()(spec), theta[:n_samples], state)
        np.testing.assert_allclose(mine, ref, atol=1e-12)
