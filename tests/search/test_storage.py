"""BayesianOptimizationSearch storage options."""

import pytest

pytest.importorskip("optuna")

from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider  # noqa: E402
from ansatz_search.circuit.gates import gate_spec  # noqa: E402
from ansatz_search.cost_functions import HierarchicalCostFunction  # noqa: E402
from ansatz_search.metrics.gradient_variance import GradientVariance  # noqa: E402
from ansatz_search.search.base import SearchProblem  # noqa: E402
from ansatz_search.search.bayesian.bo_search import BayesianOptimizationSearch  # noqa: E402
from ansatz_search.search.bayesian.incremental_ansatz_builder import IncrementalAnsatzBuilder  # noqa: E402

GATES = ["ry", "rz", "cx"]


def _search(tmp_path, storage, n_trials=3):
    problem = SearchProblem(
        cost_fn=HierarchicalCostFunction({0: [GradientVariance(
            observables="ZZ", gradient_provider=NumpyGradientProvider(), num_of_param_samples=4, seed=0)]}),
        num_qubits=2, max_param=4, max_depth=8, max_gates=8, allowed_gates=GATES,
    )
    algorithm = BayesianOptimizationSearch(
        builder=IncrementalAnsatzBuilder(), n_trials=n_trials, study_name="s",
        data_dir=tmp_path, seed=0, storage=storage,
    )
    return algorithm.run(problem, NumpyCompiler(), tuple(gate_spec(g) for g in GATES))


@pytest.mark.parametrize("storage, filename", [("journal", "s.log"), ("sqlite", "s.db")])
def test_persistent_storages_write_their_file(tmp_path, storage, filename):
    _search(tmp_path, storage)
    assert (tmp_path / filename).exists()


def test_memory_storage_writes_nothing(tmp_path):
    _search(tmp_path, "memory")
    assert not any(tmp_path.iterdir())


def test_journal_study_resumes(tmp_path):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    _search(tmp_path, "journal", n_trials=3)
    _search(tmp_path, "journal", n_trials=2)
    study = optuna.load_study(study_name="s", storage=JournalStorage(JournalFileBackend(str(tmp_path / "s.log"))))
    assert len(study.trials) == 5


def test_storages_give_the_same_result(tmp_path):
    results = [_search(tmp_path / s, s) for s in ("journal", "sqlite", "memory")]
    assert results[0].best_value == results[1].best_value == results[2].best_value


def test_unknown_storage_is_rejected():
    with pytest.raises(ValueError, match="storage must be one of"):
        BayesianOptimizationSearch(builder=None, n_trials=1, study_name="s", data_dir=".", storage="redis")
