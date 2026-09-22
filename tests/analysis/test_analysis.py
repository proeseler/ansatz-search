"""Reloading found circuits from a study, re-evaluating them, storing results."""

import pytest

pytest.importorskip("optuna")

from ansatz_search.analysis import (  # noqa: E402
    Evaluation, evaluate, evaluate_circuits, load_results, load_search, open_study, save_results, top_circuits,
    within_search_limits,
)
from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider  # noqa: E402
from ansatz_search.benchmarks import SIM2019  # noqa: E402
from ansatz_search.circuit.ansatz import AnsatzSpec  # noqa: E402
from ansatz_search import ansatz_search  # noqa: E402
from ansatz_search.cost_functions import HierarchicalCostFunction  # noqa: E402
from ansatz_search.metrics.complexity import Complexity  # noqa: E402
from ansatz_search.metrics.gradient_variance import GradientVariance  # noqa: E402
from ansatz_search.search.base import SearchProblem  # noqa: E402
from ansatz_search.search.bayesian.bo_search import BayesianOptimizationSearch  # noqa: E402
from ansatz_search.search.bayesian.incremental_ansatz_builder import IncrementalAnsatzBuilder  # noqa: E402

GATES = ["ry", "rz", "cx", "crx"]


def make_cost_function(seed, num_qubits=3):
    return HierarchicalCostFunction({0: [GradientVariance(
        observables="Z" * num_qubits, gradient_provider=NumpyGradientProvider(), num_of_param_samples=16, seed=seed)]})


def make_cost_function_4q(seed):
    """For the 4-qubit Sim et al. circuits."""
    return make_cost_function(seed, num_qubits=4)


def make_problem():
    return SearchProblem(cost_fn=make_cost_function(0), num_qubits=3, max_param=6, max_depth=10,
                         max_gates=10, allowed_gates=GATES)


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run")
    algorithm = BayesianOptimizationSearch(builder=IncrementalAnsatzBuilder(), n_trials=25, study_name="s",
                                           data_dir=run_dir, seed=0)
    result = ansatz_search(make_problem(), algorithm, NumpyCompiler(), run_dir)
    return run_dir, result


def test_spec_round_trips_through_dict():
    for build in SIM2019.values():
        spec = build()
        assert AnsatzSpec.from_dict(spec.to_dict()) == spec


def test_best_circuit_is_rebuilt_exactly(finished_run):
    run_dir, result = finished_run
    found = top_circuits(run_dir, NumpyCompiler(), make_problem(), k=3)
    assert found[0].spec == result.best_ansatz
    assert found[0].search_value == pytest.approx(result.best_value)


def test_top_circuits_are_sorted_distinct_and_limited(finished_run):
    run_dir, _ = finished_run
    found = top_circuits(run_dir, NumpyCompiler(), make_problem(), k=4)
    assert len(found) <= 4
    values = [f.search_value for f in found]
    assert values == sorted(values)
    assert len({tuple(f.spec.blocks) for f in found}) == len(found)


def test_first_trial_value_is_reproduced_by_its_rebuilt_circuit(finished_run):
    # The search shares one seeded cost function across trials, so its sampler keeps
    # drawing new samples; only trial 0 used the start of the seed-0 stream. Rebuilding
    # trial 0 (whether or not it was the best) must reproduce its recorded value exactly.
    run_dir, _ = finished_run
    first = next(f for f in top_circuits(run_dir, NumpyCompiler(), make_problem(), k=10 ** 6, unique=False)
                 if f.trial == 0)
    total, _ = make_cost_function(0).evaluate(NumpyCompiler()(first.spec), first.spec)
    assert total == pytest.approx(first.search_value, abs=1e-12)


def test_memory_runs_cannot_be_reopened(tmp_path):
    algorithm = BayesianOptimizationSearch(builder=IncrementalAnsatzBuilder(), n_trials=2, study_name="m",
                                           data_dir=tmp_path, seed=0, storage="memory")
    ansatz_search(make_problem(), algorithm, NumpyCompiler(), tmp_path)
    with pytest.raises(ValueError, match="memory"):
        open_study(tmp_path)


def test_evaluation_is_independent_of_order_and_includes_extras():
    circuits = {"a": SIM2019[1](), "b": SIM2019[2]()}
    cost_fn, extras = make_cost_function_4q(0), [Complexity()]
    forward = evaluate_circuits(circuits, cost_fn, NumpyCompiler(), [1, 2], extras, info={"a": {"x": 1}})
    backward = evaluate_circuits(dict(reversed(circuits.items())), cost_fn, NumpyCompiler(), [1, 2], extras)
    for label in circuits:
        assert forward[label]["objective"] == backward[label]["objective"]
    entry = forward["a"]
    assert entry["x"] == 1
    assert set(entry["metrics"]) == {"gradient_variance", "complexity"}
    assert len(entry["objective"]["runs"]) == 2
    assert AnsatzSpec.from_dict(entry["spec"]) == circuits["a"]


def test_objective_equals_the_cost_function_built_with_that_seed():
    spec = SIM2019[10]()
    entry = evaluate_circuits({"c": spec}, make_cost_function_4q(0), NumpyCompiler(), [7])["c"]
    total, _ = make_cost_function_4q(7).evaluate(NumpyCompiler()(spec), spec)
    assert entry["objective"]["mean"] == total and entry["objective"]["std"] == 0.0


def test_results_round_trip_with_settings(tmp_path):
    results = evaluate_circuits({"c": SIM2019[1]()}, make_cost_function_4q(0), NumpyCompiler(), [1])
    path = save_results(tmp_path / "out" / "eval.json", results, settings={"seeds": [1]})
    loaded = load_results(path)
    assert loaded["circuits"] == results
    assert loaded["settings"]["seeds"] == [1] and "created" in loaded["settings"]


def test_problem_is_read_back_from_the_run(finished_run):
    from ansatz_search.analysis import problem_from_config

    run_dir, _ = finished_run
    recorded, original = problem_from_config(run_dir), make_problem()
    for field in ("num_qubits", "max_param", "max_depth", "max_gates", "min_params", "prev_params", "topology"):
        assert getattr(recorded, field) == getattr(original, field)
    assert list(recorded.allowed_gates) == GATES
    # Replaying with the recorded problem gives the same circuits as with the original.
    assert ([f.spec for f in top_circuits(run_dir, NumpyCompiler(), k=5)]
            == [f.spec for f in top_circuits(run_dir, NumpyCompiler(), make_problem(), k=5)])


# --- SearchResult context and evaluate() ---

def test_search_result_records_what_evaluation_needs(finished_run):
    run_dir, result = finished_run
    assert isinstance(result.problem, SearchProblem) and isinstance(result.compiler, NumpyCompiler)
    assert isinstance(result.builder, IncrementalAnsatzBuilder)
    assert result.run_dir == str(run_dir) and result.study_dir == str(run_dir)


def test_evaluate_a_search_result(finished_run):
    run_dir, result = finished_run
    evaluation = evaluate(result, top=2, seeds=[1, 2], progress=False)
    found = top_circuits(run_dir, NumpyCompiler(), make_problem(), k=2)
    assert list(evaluation.circuits) == ["search #1", "search #2"]
    best = evaluation.circuits["search #1"]
    assert (best["source"], best["trial"], best["search_value"]) == ("search", found[0].trial, found[0].search_value)
    assert AnsatzSpec.from_dict(best["spec"]) == result.best_ansatz
    # The objective is the search's own cost function, reseeded.
    direct = evaluate_circuits({"x": result.best_ansatz}, make_cost_function(0), NumpyCompiler(), [1, 2])["x"]
    assert best["objective"] == direct["objective"]
    assert evaluation.settings["seeds"] == [1, 2] and evaluation.settings["run_dir"] == str(run_dir)


def test_seed_count_starts_at_1000_and_baselines_must_match_the_qubits(finished_run):
    _, result = finished_run
    assert evaluate(result, top=1, seeds=2, progress=False).settings["seeds"] == [1000, 1001]
    with pytest.raises(ValueError, match="qubits"):
        evaluate(result, baselines="sim2019", progress=False)
    with pytest.raises(ValueError, match="Unknown baselines"):
        evaluate(result, baselines="nope", progress=False)


def test_evaluate_given_circuits_with_sim2019_baselines():
    evaluation = evaluate({"mine": SIM2019[1]()}, cost_fn=make_cost_function_4q(0), compiler=NumpyCompiler(),
                          seeds=1, baselines="sim2019", progress=False)
    assert len(evaluation.circuits) == 1 + len(SIM2019)
    assert evaluation.circuits["mine"]["source"] == "given"
    assert evaluation.circuits["Sim 5"]["source"] == "sim2019"
    assert "within_search_limits" not in evaluation.circuits["Sim 5"]  # no search to compare with
    with pytest.raises(ValueError, match="cost_fn"):
        evaluate({"mine": SIM2019[1]()}, compiler=NumpyCompiler())


def test_within_search_limits():
    spec = SIM2019[1]()  # 8 parameters, 8 rotation gates, depth 2
    problem = SearchProblem(cost_fn=None, num_qubits=4, max_param=8, max_depth=2, max_gates=8, allowed_gates=["rx", "rz"])
    assert within_search_limits(spec, problem, NumpyCompiler())
    for change in (dict(max_param=7), dict(max_depth=1), dict(allowed_gates=["rx"]), dict(num_qubits=5)):
        tighter = SearchProblem(**{**dict(cost_fn=None, num_qubits=4, max_param=8, max_depth=2, max_gates=8,
                                          allowed_gates=["rx", "rz"]), **change})
        assert not within_search_limits(spec, tighter, NumpyCompiler()), change


def test_evaluation_table_and_round_trip(tmp_path):
    evaluation = evaluate({"a": SIM2019[1](), "b": SIM2019[2]()}, cost_fn=make_cost_function_4q(0),
                          compiler=NumpyCompiler(), seeds=[1], progress=False)
    table = str(evaluation)
    assert table == evaluation.table()
    ranked = [label for label, _ in evaluation.ranked()]
    assert table.index(f" {ranked[0]} ") < table.index(f" {ranked[1]} ")
    loaded = Evaluation.load(evaluation.save(tmp_path / "eval.json"))
    assert loaded.circuits == evaluation.circuits
    assert loaded.settings["seeds"] == [1] and "created" in loaded.settings


def test_memory_run_evaluates_only_its_best_circuit(tmp_path):
    algorithm = BayesianOptimizationSearch(builder=IncrementalAnsatzBuilder(), n_trials=3, study_name="m",
                                           data_dir=tmp_path, seed=0, storage="memory")
    result = ansatz_search(make_problem(), algorithm, NumpyCompiler(), tmp_path)
    evaluation = evaluate(result, top=1, seeds=[1], progress=False)
    best = evaluation.circuits["search #1"]
    assert best["trial"] is None and AnsatzSpec.from_dict(best["spec"]) == result.best_ansatz
    with pytest.raises(ValueError, match="top=1"):
        evaluate(result, top=2, progress=False)


def test_load_search_reproduces_the_evaluation_of_the_live_result(finished_run):
    run_dir, result = finished_run
    loaded = load_search(run_dir, NumpyCompiler(), make_cost_function(0))
    assert loaded.best_ansatz == result.best_ansatz
    assert loaded.best_value == pytest.approx(result.best_value)
    assert (evaluate(loaded, top=2, seeds=[3], progress=False).circuits
            == evaluate(result, top=2, seeds=[3], progress=False).circuits)
