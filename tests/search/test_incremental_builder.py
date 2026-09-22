"""Parameter allocation must leave no unused indices behind."""

import pytest

optuna = pytest.importorskip("optuna")

from ansatz_search.circuit.gates import gate_spec
from ansatz_search.search.base import SearchProblem
from ansatz_search.search.bayesian.incremental_ansatz_builder import IncrementalAnsatzBuilder


GATES = ["h", "rx", "ry", "rz", "cx", "cz", "crx", "crz"]
SPECS = tuple(gate_spec(name) for name in GATES)


def _problem(**overrides):
    kwargs = dict(
        cost_fn=None, num_qubits=4, max_param=12,
        max_depth=24, max_gates=24, allowed_gates=GATES,
    )
    kwargs.update(overrides)
    return SearchProblem(**kwargs)


def _sweep(problem, n_trials, seed=0):
    """Build many specs and return them, ignoring pruned trials."""
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    builder = IncrementalAnsatzBuilder()
    specs = []

    def objective(trial):
        specs.append(builder.build(trial, problem, SPECS))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return specs


def test_parameter_indices_are_dense():
    # A reserved-then-overwritten index would be counted by num_params while
    # appearing in no gate, contributing an identically-zero gradient.
    specs = _sweep(_problem(), n_trials=200)
    assert specs
    for spec in specs:
        used = {ref.index for block in spec.blocks for ref in block.params}
        assert used == set(range(spec.num_params))


def test_parameter_reuse_still_happens():
    """Guard the fix against trivially passing by never reusing a parameter."""
    specs = _sweep(_problem(), n_trials=200)
    reused = sum(
        sum(len(block.params) for block in spec.blocks) - spec.num_params
        for spec in specs
    )
    assert reused > 0


def test_no_reuse_gives_one_index_per_parameter_slot():
    specs = _sweep(_problem(prev_params=False), n_trials=50)
    assert specs
    for spec in specs:
        assert spec.num_params == sum(len(block.params) for block in spec.blocks)
