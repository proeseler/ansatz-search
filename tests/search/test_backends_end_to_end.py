"""A short search runs on the NumPy and PennyLane backends and both agree."""

import pytest

pytest.importorskip("optuna")

from ansatz_search.cost_functions import HierarchicalCostFunction  # noqa: E402
from ansatz_search import ansatz_search  # noqa: E402
from ansatz_search.metrics.gradient_variance import GradientVariance  # noqa: E402
from ansatz_search.search.base import SearchProblem  # noqa: E402
from ansatz_search.search.bayesian.bo_search import BayesianOptimizationSearch  # noqa: E402
from ansatz_search.search.bayesian.incremental_ansatz_builder import IncrementalAnsatzBuilder  # noqa: E402


def _backend(name):
    if name == "numpy":
        from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider
        return NumpyCompiler(), NumpyGradientProvider()
    pytest.importorskip("pennylane")
    from ansatz_search.backends.pennylane import PennyLaneCompiler, PennyLaneGradientProvider
    return PennyLaneCompiler(), PennyLaneGradientProvider()


def _run(name, tmp_path):
    compiler, provider = _backend(name)
    problem = SearchProblem(
        cost_fn=HierarchicalCostFunction({0: [
            GradientVariance(observables="ZZZ", gradient_provider=provider, num_of_param_samples=8, seed=0)
        ]}),
        num_qubits=3, max_param=6, max_depth=12, max_gates=12,
        allowed_gates=["h", "rx", "ry", "rz", "cx", "cz", "crx", "crz"],
    )
    algorithm = BayesianOptimizationSearch(
        builder=IncrementalAnsatzBuilder(), n_trials=4, study_name=name, data_dir=tmp_path / name, seed=0,
    )
    return ansatz_search(problem, algorithm, compiler, tmp_path / name)


@pytest.mark.parametrize("backend", ["numpy", "pennylane"])
def test_search_runs(tmp_path, backend):
    result = _run(backend, tmp_path)
    metrics = result.best_metrics["gradient_variance"]
    assert 0.0 <= metrics["cost"] <= 1.0
    assert metrics["value"] >= 0.0
    assert (tmp_path / backend / "config.yaml").exists()


def test_backends_find_the_same_result(tmp_path):
    # Same seeds -> same trials and parameter samples; exact gradients must agree.
    numpy_result, pl_result = _run("numpy", tmp_path), _run("pennylane", tmp_path)
    assert pl_result.best_ansatz == numpy_result.best_ansatz
    assert pl_result.best_value == pytest.approx(numpy_result.best_value, abs=1e-9)


def test_search_with_gradient_variance_and_expressibility(tmp_path):
    from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider, NumpyStateProvider
    from ansatz_search.metrics.expressibility import Expressibility

    problem = SearchProblem(
        cost_fn=HierarchicalCostFunction({
            0: [Expressibility(num_qubits=3, state_provider=NumpyStateProvider(), num_of_param_samples=200, seed=0)],
            1: [GradientVariance(observables="ZZZ", gradient_provider=NumpyGradientProvider(), num_of_param_samples=8, seed=0)],
        }),
        num_qubits=3, max_param=6, max_depth=12, max_gates=12,
        allowed_gates=["h", "rx", "ry", "rz", "cx", "cz", "crx", "crz"],
    )
    algorithm = BayesianOptimizationSearch(
        builder=IncrementalAnsatzBuilder(), n_trials=4, study_name="both", data_dir=tmp_path, seed=0,
    )
    result = ansatz_search(problem, algorithm, NumpyCompiler(), tmp_path)
    assert set(result.best_metrics) == {"expressibility", "gradient_variance"}
    for entry in result.best_metrics.values():
        assert 0.0 <= entry["cost"] <= 1.0
