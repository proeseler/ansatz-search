"""Qiskit providers agree with the exact NumPy backend; the device path runs on a simulated IBM device."""

import numpy as np
import pytest

pytest.importorskip("qiskit")

from ansatz_search.backends.base import default_provider  # noqa: E402
from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider, NumpyStateProvider  # noqa: E402
from ansatz_search.backends.qiskit import (  # noqa: E402
    QiskitCompiler, QiskitFidelityProvider, QiskitGradientProvider, QiskitStateProvider,
)
from ansatz_search.benchmarks import SIM2019  # noqa: E402
from ansatz_search.metrics import Expressibility, GradientVariance  # noqa: E402

from _circuits import OBSERVABLES, mixed_circuit, random_state, rotation_circuit  # noqa: E402


def _samples(spec, count, seed=7):
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, (count, spec.num_params))


def _zero(num_qubits):
    return np.eye(2 ** num_qubits)[0]


@pytest.mark.parametrize("spec, state", [
    (mixed_circuit(), random_state(3)),
    (SIM2019[6](), None),  # 28 parameters: Qiskit orders theta_10 before theta_2
], ids=["mixed", "28-parameters"])
def test_statevectors_match_numpy(spec, state):
    state = _zero(spec.num_qubits) if state is None else state
    theta = _samples(spec, 3)
    ours = QiskitStateProvider().states(QiskitCompiler()(spec), theta, state)
    exact = NumpyStateProvider().states(NumpyCompiler()(spec), theta, state)
    np.testing.assert_allclose(ours, exact, atol=1e-10)


def test_gradients_match_numpy_adjoint_with_controlled_rotations_and_reused_parameters():
    spec = mixed_circuit()  # crx/cry/crz, cr1, reused parameters, several observables and initial states
    theta, states = _samples(spec, 3), [_zero(3), random_state(3)]
    ours = QiskitGradientProvider().gradients(QiskitCompiler()(spec), OBSERVABLES, theta, states)
    exact = NumpyGradientProvider().gradients(NumpyCompiler()(spec), OBSERVABLES, theta, states)
    assert ours.shape == exact.shape
    np.testing.assert_allclose(ours, exact, atol=1e-8)


def test_gradients_with_more_than_ten_parameters():
    spec = SIM2019[6]()
    theta, zero = _samples(spec, 2), [_zero(4)]
    np.testing.assert_allclose(QiskitGradientProvider().gradients(QiskitCompiler()(spec), "ZZZZ", theta, zero),
                               NumpyGradientProvider().gradients(NumpyCompiler()(spec), "ZZZZ", theta, zero),
                               atol=1e-8)


def test_measured_fidelities_estimate_the_exact_overlap():
    spec, shots, state = mixed_circuit(), 20_000, random_state(3)
    a, b = _samples(spec, 4, seed=1), _samples(spec, 4, seed=2)
    program = QiskitCompiler()(spec)
    measured = QiskitFidelityProvider(shots=shots, seed=1).fidelities(program, a, b, state)
    psi_a, psi_b = (NumpyStateProvider().states(NumpyCompiler()(spec), p, state) for p in (a, b))
    exact = np.abs(np.sum(np.conj(psi_b) * psi_a, axis=1)) ** 2
    assert np.all(np.abs(measured - exact) < 5 * np.sqrt(exact * (1 - exact) / shots) + 1e-3)
    np.testing.assert_allclose(QiskitFidelityProvider(shots=500, seed=1).fidelities(program, a, a, state), 1.0)


def test_expressibility_from_measured_fidelities_tracks_the_statevector_value():
    spec = SIM2019[2]()
    kw = dict(num_qubits=4, num_of_param_samples=400, seed=5)
    exact = Expressibility(**kw).value(NumpyCompiler()(spec), spec)
    measured = Expressibility(**kw, fidelity_provider=QiskitFidelityProvider(shots=8000, seed=2))
    assert measured.value(QiskitCompiler()(spec), spec) == pytest.approx(exact, abs=0.05)


def test_qiskit_circuits_get_qiskit_providers_by_default():
    spec = rotation_circuit()
    program = QiskitCompiler()(spec)
    assert isinstance(default_provider(program, "gradient"), QiskitGradientProvider)
    assert isinstance(default_provider(program, "state"), QiskitStateProvider)
    kw = dict(observables="ZZZ", num_of_param_samples=6, seed=3)
    assert (GradientVariance(**kw).value(program, spec)
            == pytest.approx(GradientVariance(**kw).value(NumpyCompiler()(spec), spec), abs=1e-10))


def test_aer_primitives_are_guarded_against_their_controlled_rotation_bug():
    """Aer's parameter binding treats some controlled-rotation angles as 0; the providers decompose them first."""
    aer = pytest.importorskip("qiskit_aer.primitives")
    for spec, state in ((SIM2019[13](), _zero(4)), (mixed_circuit(), random_state(3))):
        program, n = QiskitCompiler()(spec), spec.num_qubits
        theta = _samples(spec, 3)
        observable = "Z" + "I" * (n - 1)
        exact = NumpyGradientProvider().gradients(NumpyCompiler()(spec), observable, theta, [state])
        got = QiskitGradientProvider(estimator=aer.EstimatorV2()).gradients(program, observable, theta, [state])
        np.testing.assert_allclose(got, exact, atol=1e-8)

        shots = 40_000
        a, b = _samples(spec, 4, seed=1), _samples(spec, 4, seed=2)
        psi_a, psi_b = (NumpyStateProvider().states(NumpyCompiler()(spec), p, state) for p in (a, b))
        exact_f = np.abs(np.sum(np.conj(psi_b) * psi_a, axis=1)) ** 2
        measured = QiskitFidelityProvider(sampler=aer.SamplerV2(seed=3), shots=shots).fidelities(program, a, b, state)
        assert np.all(np.abs(measured - exact_f) < 5 * np.sqrt(exact_f * (1 - exact_f) / shots) + 2e-3)


def test_simulated_ibm_device():
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    device, spec = FakeManilaV2(), rotation_circuit()
    program, theta = QiskitCompiler()(spec), _samples(spec, 5)
    same = QiskitFidelityProvider(backend=device, shots=1000, seed=1).fidelities(program, theta, theta, _zero(3))
    assert same.shape == (5,) and np.all(same > 0.7)  # identity, up to device noise
    grads = QiskitGradientProvider(backend=device, precision=0.05).gradients(program, "ZZZ", theta[:2], [_zero(3)])
    assert grads.shape == (1, 1, 2, spec.num_params) and np.all(np.isfinite(grads))
