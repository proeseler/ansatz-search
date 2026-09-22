"""Reload the circuits a finished search found, straight from its Optuna study."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import optuna
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock
from optuna.trial import FixedTrial, TrialState

from ansatz_search.backends.base import Compiler
from ansatz_search.circuit.ansatz import AnsatzSpec
from ansatz_search.cost_functions.base import CostFunction
from ansatz_search.search.base import AnsatzBuilder, SearchProblem, SearchResult


@dataclass(frozen=True)
class FoundCircuit:
    trial: int | None   # Optuna trial number (None if the trials were not stored)
    search_value: float  # objective recorded during the search (one noisy estimate)
    spec: AnsatzSpec


def _config(run_dir) -> dict:
    return yaml.safe_load((Path(run_dir) / "config.yaml").read_text())


def problem_from_config(run_dir: str | os.PathLike) -> SearchProblem:
    """The search problem a run used, as recorded in its config.yaml.

    Contains everything the builder needs to replay trials (limits, gates,
    parameter settings); the cost function is not stored, so it is None.
    """
    config = _config(run_dir)
    return SearchProblem(
        cost_fn=None,
        num_qubits=config["num_qubits"],
        max_param=config["max_param"],
        max_depth=config["max_depth"],
        max_gates=config["max_gates"],
        allowed_gates=[gate[0] for gate in config["allowed_gates"]],
        min_params=config.get("min_params", 0),
        prev_params=config.get("prev_params", True),
        topology=config.get("topology"),
    )


def open_study(run_dir: str | os.PathLike, study_dir: str | os.PathLike | None = None) -> optuna.Study:
    """Open the study of a run from the config.yaml that ansatz_search wrote into `run_dir`.

    `study_dir` is where the study file lives (the search algorithm's data_dir);
    it defaults to `run_dir`.
    """
    run_dir = Path(run_dir)
    config = _config(run_dir)
    name, storage = config["study_name"], config.get("storage", "sqlite")
    study_dir = Path(study_dir) if study_dir is not None else run_dir
    if storage == "journal":
        path = str(study_dir / f"{name}.log")
        backend = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileSymlinkLock(path)))
    elif storage == "sqlite":
        backend = f"sqlite:///{study_dir / f'{name}.db'}"
    else:
        raise ValueError(f"Run used {storage!r} storage; nothing was persisted to reload.")
    return optuna.load_study(study_name=name, storage=backend)


def top_circuits(
    run_dir: str | os.PathLike,
    compiler: Compiler,
    problem: SearchProblem | None = None,
    builder: AnsatzBuilder | None = None,
    k: int = 5,
    unique: bool = True,
    study_dir: str | os.PathLike | None = None,
) -> list[FoundCircuit]:
    """The k best completed trials, rebuilt as AnsatzSpecs.

    Each circuit is reconstructed by replaying the builder with the trial's
    recorded decisions (optuna.trial.FixedTrial), so `compiler` and `builder`
    must match the search. `problem` defaults to the one recorded in the run's
    config.yaml, so a run is always replayed with the limits it was searched
    with. With `unique`, trials that produced the same circuit structure count
    once (the best of them is kept).
    """
    from ansatz_search.search.bayesian.incremental_ansatz_builder import IncrementalAnsatzBuilder

    study = open_study(run_dir, study_dir)
    problem = problem or problem_from_config(run_dir)
    builder = builder or IncrementalAnsatzBuilder()
    gate_specs = compiler.resolve_gates(problem.allowed_gates)
    trials = sorted(
        (t for t in study.trials if t.state is TrialState.COMPLETE and t.value is not None),
        key=lambda t: t.value,
    )
    found, seen = [], set()
    for t in trials:
        spec = builder.build(FixedTrial(t.params, t.number), problem, gate_specs)
        key = tuple(spec.blocks)
        if unique and key in seen:
            continue
        seen.add(key)
        found.append(FoundCircuit(t.number, float(t.value), spec))
        if len(found) == k:
            break
    return found


def load_search(
    run_dir: str | os.PathLike,
    compiler: Compiler | None = None,
    cost_fn: CostFunction | None = None,
    builder: AnsatzBuilder | None = None,
    study_dir: str | os.PathLike | None = None,
) -> SearchResult:
    """A finished run as a SearchResult, e.g. to evaluate it in a later session.

    Limits and gates come from the run's config.yaml, so the run is replayed
    exactly as it was searched. config.yaml does not store the cost function:
    pass the one to evaluate with as `cost_fn`. `compiler` (default: NumPy)
    and `builder` must match the search. `study_dir` defaults to `run_dir`.
    """
    from ansatz_search.backends.numpy import NumpyCompiler
    from ansatz_search.search.bayesian.incremental_ansatz_builder import IncrementalAnsatzBuilder

    compiler = NumpyCompiler() if compiler is None else compiler
    problem = replace(problem_from_config(run_dir), cost_fn=cost_fn)
    builder = builder or IncrementalAnsatzBuilder()
    best = top_circuits(run_dir, compiler, problem, builder, k=1, study_dir=study_dir)
    if not best:
        raise ValueError(f"The run in {run_dir} has no completed trials.")
    return SearchResult(
        best_ansatz=best[0].spec,
        best_value=best[0].search_value,
        best_metrics=None,  # not stored; evaluate() recomputes metrics
        best_circuit=compiler(best[0].spec),
        problem=problem,
        compiler=compiler,
        builder=builder,
        run_dir=str(run_dir),
        study_dir=str(study_dir if study_dir is not None else run_dir),
    )
