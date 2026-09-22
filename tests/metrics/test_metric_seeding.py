"""A seeded metric must draw the same parameter samples on every construction."""

import numpy as np
import pytest

pytest.importorskip("qiskit", reason="SampledMetric currently depends on Qiskit")

from ansatz_search.metrics.base import SampledMetric


class _Recorder(SampledMetric):
    """Capture what the base class hands to value_from_samples."""

    @property
    def name(self):
        return "recorder"

    def value_from_samples(self, ansatz, param_samples, stv_vecs):
        self.seen = (np.asarray(param_samples), [np.asarray(s) for s in stv_vecs])
        return 0.0


class _Spec:
    num_qubits = 3
    num_params = 4


def _samples(**kwargs):
    metric = _Recorder(parameter_sampler="gaussian", num_of_param_samples=6, **kwargs)
    metric.compute(object(), _Spec())
    return metric.seen


def test_seeded_parameter_samples_are_reproducible():
    first, _ = _samples(seed=7)
    second, _ = _samples(seed=7)
    assert first.shape == (6, _Spec.num_params)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_samples():
    assert not np.array_equal(_samples(seed=7)[0], _samples(seed=8)[0])


def test_seed_reaches_a_string_aliased_state_sampler():
    first = _samples(seed=3, state_sampler="haar", num_of_state_samples=2)[1]
    second = _samples(seed=3, state_sampler="haar", num_of_state_samples=2)[1]
    assert len(first) == 2
    # Distinct states within one draw, identical across equally seeded metrics.
    assert not np.array_equal(first[0], first[1])
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_shape_falls_back_to_the_circuit_without_a_spec():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter

    qc = qiskit.QuantumCircuit(2)
    qc.ry(Parameter("a"), 0)
    metric = _Recorder(parameter_sampler="gaussian", num_of_param_samples=3, seed=1)
    metric.compute(qc)
    samples, states = metric.seen
    assert samples.shape == (3, 1)
    assert states[0].size == 2**2
