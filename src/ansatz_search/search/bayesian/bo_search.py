from __future__ import annotations

import os
from typing import Sequence

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock
from tqdm import tqdm

from ansatz_search.backends.base import Compiler
from ansatz_search.circuit.gates import GateSpec
from ansatz_search.search.base import AnsatzBuilder, SearchAlgorithm, SearchProblem, SearchResult

from .incremental_ansatz_builder import IncrementalAnsatzBuilder


STORAGES = ("journal", "sqlite", "memory")


class BayesianOptimizationSearch(SearchAlgorithm):
    """Optuna TPE search over ansatz structures.

    storage:
      - "journal" (default): append-only log `<data_dir>/<study_name>.log`.
        Much cheaper per suggest_* call than SQLite, resumable, and safe for
        several worker processes sharing one study (symlink lock, suitable
        for network filesystems).
      - "sqlite": `<data_dir>/<study_name>.db`; every suggest_* call commits a
        transaction, which dominated trial time in profiling.
      - "memory": fastest, but nothing is persisted: a crashed run is lost
        and cannot be resumed.

    `data_dir` defaults to the `fdata` folder given to `ansatz_search()`, next to config.yaml.
    `builder` defaults to an IncrementalAnsatzBuilder, which adds gates one by one.
    """

    def __init__(
        self,
        builder: AnsatzBuilder | None = None,
        n_trials: int = 100,
        study_name: str = "search",
        data_dir: str | None = None,
        stored_circuit: int = 100,
        allowed_rep: int = 10,
        seed: int | None = None,
        storage: str = "journal",
    ):
        super().__init__(seed=seed)
        if storage not in STORAGES:
            raise ValueError(f"storage must be one of {STORAGES}, got {storage!r}.")
        self.storage = storage
        self.builder = builder if builder is not None else IncrementalAnsatzBuilder()
        self.n_trials = n_trials
        self.study_name = study_name
        self.data_dir = data_dir
        self.stored_circuit = stored_circuit
        self.allowed_rep = allowed_rep

    def _optuna_storage(self):
        if self.storage == "memory":
            return None  # Optuna's in-memory storage
        if self.data_dir is None:
            raise ValueError("data_dir is not set: pass data_dir=, or run the search through ansatz_search(), "
                             "which stores the study in its fdata folder.")
        os.makedirs(self.data_dir, exist_ok=True)
        if self.storage == "sqlite":
            return f"sqlite:///{os.path.join(self.data_dir, f'{self.study_name}.db')}"
        path = os.path.join(self.data_dir, f"{self.study_name}.log")
        return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileSymlinkLock(path)))

    def run(self, problem: SearchProblem, compiler: Compiler, gate_specs: Sequence[GateSpec]) -> SearchResult:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)  # progress comes from the tqdm bar
        study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",  # metrics are costs: lower is always better
            sampler=sampler,
            storage=self._optuna_storage(),
            load_if_exists=True,
        )

        best = {"ansatz": None, "circuit": None, "value": None, "metrics": None}

        def objective(trial):
            spec = self.builder.build(trial, problem, gate_specs)
            circuit = compiler(spec)
            value, metrics = problem.cost_fn.evaluate(circuit, spec)

            if best["value"] is None or value < best["value"]:
                best["ansatz"] = spec
                best["circuit"] = circuit
                best["value"] = value
                best["metrics"] = metrics

            return value

        self._run_with_progress(study, objective)

        if best["ansatz"] is None:
            raise RuntimeError("Bayesian optimization finished without producing a valid ansatz.")

        return SearchResult(
            best_ansatz=best["ansatz"],
            best_value=best["value"],
            best_metrics=best["metrics"],
            best_circuit=best["circuit"],
            problem=problem,
            compiler=compiler,
            builder=self.builder,
            study_dir=None if self.storage == "memory" else str(self.data_dir),
        )

    def _run_with_progress(self, study, objective):
        bar = tqdm(total=self.n_trials, desc="Optuna", unit="trial")

        def callback(study, trial):
            bar.update(1)
            if trial.number % 50 == 0:
                try:
                    tqdm.write(
                        f"[{trial.number}] "
                        f"Best: {study.best_trial.value:.4f} "
                        f"Last: {trial.value:.4f}"
                    )
                except Exception:
                    pass

        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])
        bar.close()
