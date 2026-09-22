"""with_seed rebuilds a sampler, metric or cost function with the same arguments and another seed."""

import copy
import pickle

import numpy as np
import pytest

pytest.importorskip("qiskit", reason="SampledMetric currently depends on Qiskit")

from ansatz_search.backends.numpy import NumpyGradientProvider, NumpyStateProvider
from ansatz_search.cost_functions import HierarchicalCostFunction
from ansatz_search.metrics.complexity import Complexity
from ansatz_search.metrics.expressibility import Expressibility
from ansatz_search.metrics.gradient_variance import GradientVariance
from ansatz_search.metrics.trainability import Trainability
from ansatz_search.utils.parameter_sampler import GaussianSampler, UniformSampler
from ansatz_search.utils.statevector_sampler import HaarStatevectorSampler, ZeroStatevectorSampler


def _gradient_variance(**kwargs):
    return GradientVariance(observables={"ZZ": 1.0}, gradient_provider=NumpyGradientProvider(),
                            num_of_param_samples=6, **kwargs)


def test_sampler_copy_draws_like_a_fresh_sampler_and_keeps_its_settings():
    reseeded = UniformSampler(low=-1.0, high=1.0, seed=1).with_seed(7)
    assert (reseeded.low, reseeded.high) == (-1.0, 1.0)
    np.testing.assert_array_equal(reseeded(5, 3), UniformSampler(low=-1.0, high=1.0, seed=7)(5, 3))


def test_copy_does_not_depend_on_earlier_draws():
    used = GaussianSampler(seed=3)
    used(10, 8)
    np.testing.assert_array_equal(used.with_seed(7)(4, 2), GaussianSampler(seed=7)(4, 2))


def test_original_is_not_modified():
    sampler = GaussianSampler(seed=3)
    sampler.with_seed(7)
    np.testing.assert_array_equal(sampler(4, 2), GaussianSampler(seed=3)(4, 2))


def test_positional_arguments_and_classes_without_init():
    np.testing.assert_array_equal(UniformSampler(-1.0, 1.0, 1).with_seed(2)(3, 2),
                                  UniformSampler(-1.0, 1.0, 2)(3, 2))
    assert isinstance(ZeroStatevectorSampler().with_seed(2), ZeroStatevectorSampler)


def test_copied_and_pickled_objects_can_be_reseeded():
    sampler = UniformSampler(low=-1.0, high=1.0, seed=1)
    expected = UniformSampler(low=-1.0, high=1.0, seed=2)(3, 2)
    for clone in (copy.copy(sampler), copy.deepcopy(sampler), pickle.loads(pickle.dumps(sampler))):
        np.testing.assert_array_equal(clone.with_seed(2)(3, 2), expected)


def test_metric_reseeds_its_samplers_and_shares_its_provider():
    metric = _gradient_variance(parameter_sampler=UniformSampler(low=0.0, high=1.0, seed=1),
                                state_sampler="haar", seed=1, max_variance=0.1)
    reseeded = metric.with_seed(9)
    assert (reseeded.seed, reseeded.max_variance, metric.seed) == (9, 0.1, 1)
    assert reseeded._gradient_provider is metric._gradient_provider
    np.testing.assert_array_equal(reseeded.parameter_sampler(6, 4), UniformSampler(low=0.0, high=1.0, seed=9)(6, 4))
    np.testing.assert_array_equal(reseeded.state_sampler(4, 1)[0], HaarStatevectorSampler(seed=9)(4, 1)[0])


def test_expressibility_noise_floor_follows_the_new_seed():
    kwargs = dict(num_qubits=2, state_provider=NumpyStateProvider(), num_of_param_samples=100, floor_repetitions=5)
    metric = Expressibility(**kwargs, seed=1)
    floor = metric.noise_floor  # cached for seed 1
    assert metric.with_seed(2).noise_floor == Expressibility(**kwargs, seed=2).noise_floor
    assert metric.with_seed(2).noise_floor != floor


def test_trainability_reseeds_its_inner_gradient_variance():
    metric = Trainability(observables={"ZZ": 1.0}, gradient_provider=NumpyGradientProvider(),
                          num_of_param_samples=6, seed=1)
    assert metric.with_seed(4)._gradient_variance.seed == 4


def test_cost_function_reseeds_every_metric_in_both_views():
    gradient_variance, complexity = _gradient_variance(seed=1), Complexity(max_param=4)
    cost_fn = HierarchicalCostFunction({0: [gradient_variance], 1: [complexity]},
                                       thresholds={"gradient_variance": 0.05})
    reseeded = cost_fn.with_seed(8)
    new_gradient_variance, new_complexity = reseeded.metrics
    # evaluate reads metrics_by_priority and compute_metrics reads metrics: both must hold the copies.
    assert reseeded.metrics_by_priority == {0: [new_gradient_variance], 1: [new_complexity]}
    assert new_gradient_variance is not gradient_variance
    assert (new_gradient_variance.seed, gradient_variance.seed) == (8, 1)
    assert new_complexity.max_param == 4
    assert reseeded.thresholds == {"gradient_variance": 0.05}
