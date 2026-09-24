"""Re-evaluate circuits on fresh random samples and store the results.

During a search every metric is one noisy estimate, and the best trial is the
minimum over many of them, so its recorded value is optimistic. Evaluating the
found circuits (and baselines) again with independent seeds gives an unbiased
comparison:

    result = ansatz_search(problem, algorithm, compiler, fdata)
    evaluation = evaluate(result, top=5, seeds=5, baselines="sim2019")
    print(evaluation)
    evaluation.save("eval.json")
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from tqdm import tqdm

from ansatz_search.backends.base import Compiler
from ansatz_search.backends.numpy import NumpyCompiler
from ansatz_search.benchmarks import SIM2019
from ansatz_search.circuit.ansatz import AnsatzSpec
from ansatz_search.cost_functions.base import CostFunction
from ansatz_search.metrics.base import Metric
from ansatz_search.search.base import SearchProblem, SearchResult

from .runs import FoundCircuit, top_circuits

EVAL_SEED_START = 1000  # far from typical search seeds, so evaluation samples are independent

# Named reference circuit sets for `evaluate(..., baselines=name)`.
BASELINES = {
    "sim2019": lambda: {f"Sim {i}": SIM2019[i]() for i in sorted(SIM2019)},
}


def _summary(runs: Sequence[float]) -> dict:
    runs = [float(r) for r in runs]
    std = float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0
    return {"mean": float(np.mean(runs)), "std": std, "runs": runs}


def evaluate_circuits(
    circuits: Mapping[str, AnsatzSpec],
    cost_fn: CostFunction,
    compiler: Compiler,
    seeds: Sequence[int],
    extra_metrics: Sequence[Metric] = (),
    info: Mapping[str, dict] | None = None,
    progress: bool = False,
) -> dict[str, dict]:
    """Evaluate every circuit once per seed; return per-circuit summaries.

    For every circuit and seed, `cost_fn` and `extra_metrics` are rebuilt with
    `with_seed(seed)`: the "objective" is exactly what the search minimized,
    and a circuit's result does not depend on evaluation order. `extra_metrics`
    are reported but not part of the objective (e.g. entanglement, complexity).
    `info[label]` is merged into that circuit's entry (e.g. its origin).

    Each entry: {"spec", "objective": {mean, std, runs},
                 "metrics": {name: {"value": {...}, "cost": {...}}}, **info}.
    """
    results = {}
    bar = tqdm(total=len(circuits) * len(seeds), desc="Evaluating", unit="run", disable=not progress)
    for label, spec in circuits.items():
        program = compiler(spec)
        objective, per_metric = [], {}
        for seed in seeds:
            total, metrics = cost_fn.with_seed(seed).evaluate(program, spec)
            for metric in extra_metrics:
                metric = metric.with_seed(seed)
                value = metric.value(program, spec)
                metrics[metric.name] = {"value": value, "cost": metric.cost(value)}
            objective.append(total)
            for name, entry in metrics.items():
                per_metric.setdefault(name, {"value": [], "cost": []})
                per_metric[name]["value"].append(entry["value"])
                per_metric[name]["cost"].append(entry["cost"])
            bar.update(1)
        results[label] = {
            "spec": spec.to_dict(),
            "objective": _summary(objective),
            "metrics": {name: {k: _summary(v) for k, v in m.items()} for name, m in per_metric.items()},
            **(dict(info[label]) if info and label in info else {}),
        }
    bar.close()
    return results


def within_search_limits(spec: AnsatzSpec, problem: SearchProblem, compiler: Compiler) -> bool:
    """Could a search on `problem` have produced `spec`?

    Checks the qubit count, the parameter, gate and depth limits and the gate
    set; not parameter reuse or topology.
    """
    allowed = {gate.name for gate in compiler.resolve_gates(problem.allowed_gates)}
    return (spec.num_qubits == problem.num_qubits and spec.num_params <= problem.max_param
            and len(spec) <= problem.max_gates and spec.depth <= problem.max_depth
            and all(block.op in allowed for block in spec))


@dataclass
class Evaluation:
    """Re-evaluated circuits: per circuit the objective and every metric as mean, std and per-seed runs.

    `circuits` maps a label to the entry described in `evaluate_circuits`;
    `settings` records how the evaluation was run.
    """

    circuits: dict[str, dict]
    settings: dict = field(default_factory=dict)

    def ranked(self) -> list[tuple[str, dict]]:
        """(label, entry) pairs sorted by mean objective, best first."""
        return sorted(self.circuits.items(), key=lambda kv: kv[1]["objective"]["mean"])

    def table(self) -> str:
        """The ranked circuits as text: objective (mean ± std), value during the search, raw metric means."""
        names = list(dict.fromkeys(name for entry in self.circuits.values() for name in entry["metrics"]))
        lines = [f"{'circuit':>12} {'objective':>18} {'during search':>13} " + " ".join(f"{n:>18}" for n in names)]
        for label, entry in self.ranked():
            during = f"{entry['search_value']:.4f}" if entry.get("search_value") is not None else ""
            cells = " ".join(f"{entry['metrics'][n]['value']['mean']:>18.5g}" if n in entry["metrics"] else " " * 18
                             for n in names)
            mark = " *" if entry.get("within_search_limits") is False else "  "
            lines.append(f"{label + mark:>12} {entry['objective']['mean']:9.4f} ± {entry['objective']['std']:.4f} "
                         f"{during:>13} {cells}")
        if any(entry.get("within_search_limits") is False for entry in self.circuits.values()):
            lines.append("* outside the search limits or gate set: the search could not have found it.")
        lines.append("Metric columns are raw values, mean over seeds.")
        return "\n".join(lines)

    __str__ = table

    def save(self, path: str | os.PathLike) -> Path:
        """Write to JSON, adding the creation time and git commit to the settings."""
        return save_results(path, self.circuits, self.settings)

    @classmethod
    def load(cls, path: str | os.PathLike) -> Evaluation:
        data = load_results(path)
        return cls(circuits=data["circuits"], settings=data["settings"])

    def plot(self, **kwargs):
        """Shortcut for `analysis.plot(self, **kwargs)`; returns the matplotlib Figure."""
        from .plotting import plot

        return plot(self, **kwargs)


def _found_circuits(result: SearchResult, top: int) -> list[FoundCircuit]:
    """The `top` best distinct circuits of a search, replayed from its stored trials."""
    if result.run_dir is not None and result.study_dir is not None:
        return top_circuits(result.run_dir, result.compiler, result.problem, result.builder, k=top,
                            study_dir=result.study_dir)
    if top == 1:
        return [FoundCircuit(trial=None, search_value=float(result.best_value), spec=result.best_ansatz)]
    raise ValueError("The search's trials were not stored (e.g. storage='memory'), so only its best "
                     "circuit is known: use top=1.")


def _baseline_circuits(baselines: str | Mapping[str, AnsatzSpec] | None) -> dict[str, AnsatzSpec]:
    if baselines is None:
        return {}
    if isinstance(baselines, str):
        if baselines not in BASELINES:
            raise ValueError(f"Unknown baselines {baselines!r}; choose from {sorted(BASELINES)} or pass a mapping of circuits.")
        return BASELINES[baselines]()
    return dict(baselines)


def _plot_style(metric: Metric) -> dict:
    return {"label": metric.label or metric.name, "higher_is_better": bool(metric.higher_is_better),
            "log_axis": bool(metric.log_axis)}


def evaluate(
    circuits: SearchResult | Mapping[str, AnsatzSpec],
    *,
    top: int = 5,
    seeds: int | Sequence[int] = 5,
    baselines: str | Mapping[str, AnsatzSpec] | None = None,
    cost_fn: CostFunction | None = None,
    compiler: Compiler | None = None,
    extra_metrics: Sequence[Metric] = (),
    progress: bool = True,
) -> Evaluation:
    """Re-evaluate found circuits, and optionally baselines, on fresh seeds.

    `circuits` is a SearchResult, whose `top` best distinct circuits are
    replayed from the stored trials, or a mapping of label -> AnsatzSpec.
    `seeds` is a list of seeds or a count n, meaning seeds 1000 ... 1000 + n - 1.
    `baselines` adds reference circuits: "sim2019" for the 19 four-qubit
    circuits of Sim et al. (2019), or a mapping of your own. `cost_fn` and
    `compiler` default to the search's; for a mapping of circuits `cost_fn` is
    required and `compiler` defaults to NumPy.
    `extra_metrics` are reported but not part of the objective.

    For a SearchResult, each baseline records whether the search could have
    produced it (`within_search_limits`).
    """
    problem = result = None
    if isinstance(circuits, SearchResult):
        result, problem = circuits, circuits.problem
        if problem is None or result.compiler is None:
            raise ValueError("This SearchResult does not record its problem and compiler; "
                             "get it from ansatz_search() or load_search().")
        cost_fn = problem.cost_fn if cost_fn is None else cost_fn
        compiler = result.compiler if compiler is None else compiler
        to_evaluate, info = {}, {}
        for rank, found in enumerate(_found_circuits(result, top), start=1):
            label = f"search #{rank}"
            to_evaluate[label] = found.spec
            info[label] = {"source": "search", "trial": found.trial, "search_value": found.search_value}
        num_qubits = problem.num_qubits
    else:
        to_evaluate = dict(circuits)
        info = {label: {"source": "given"} for label in to_evaluate}
        num_qubits = next(iter(to_evaluate.values())).num_qubits if to_evaluate else None
    if cost_fn is None:
        raise ValueError("No cost function to evaluate with: pass cost_fn=.")
    compiler = NumpyCompiler() if compiler is None else compiler

    reference = _baseline_circuits(baselines)
    for label, spec in reference.items():
        if label in to_evaluate:
            raise ValueError(f"Baseline label {label!r} is already used by another circuit.")
        if num_qubits is not None and spec.num_qubits != num_qubits:
            raise ValueError(f"Baseline {label!r} has {spec.num_qubits} qubits, "
                             f"but the circuits being evaluated have {num_qubits}.")
        to_evaluate[label] = spec
        info[label] = {"source": baselines if isinstance(baselines, str) else "baseline"}
        if problem is not None:
            info[label]["within_search_limits"] = within_search_limits(spec, problem, compiler)

    seeds = list(range(EVAL_SEED_START, EVAL_SEED_START + seeds)) if isinstance(seeds, int) else [int(s) for s in seeds]
    if not seeds:
        raise ValueError("Provide at least one seed.")

    settings = {
        "seeds": seeds,
        "top": top if result is not None else None,
        "baselines": baselines if baselines is None or isinstance(baselines, str) else list(reference),
        "cost_fn": type(cost_fn).__name__,
        "compiler": getattr(compiler, "name", type(compiler).__name__),
        "extra_metrics": [metric.name for metric in extra_metrics],
        "run_dir": result.run_dir if result is not None else None,
        # How plot() shows each metric, stored so a loaded evaluation plots the same way.
        "metrics": {metric.name: _plot_style(metric) for metric in [*getattr(cost_fn, "metrics", ()), *extra_metrics]},
    }
    if problem is not None:
        settings["search_limits"] = {
            "num_qubits": problem.num_qubits, "max_param": problem.max_param, "max_gates": problem.max_gates,
            "max_depth": problem.max_depth, "allowed_gates": [getattr(g, "value", g) for g in problem.allowed_gates],
        }
    return Evaluation(evaluate_circuits(to_evaluate, cost_fn, compiler, seeds, extra_metrics, info, progress), settings)


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_results(path: str | os.PathLike, results: Mapping[str, dict], settings: Mapping | None = None) -> Path:
    """Write results with the settings and provenance needed to reproduce them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(path.parent),
        **dict(settings or {}),
    }
    path.write_text(json.dumps({"settings": meta, "circuits": dict(results)}, indent=2))
    return path


def load_results(path: str | os.PathLike) -> dict:
    """Read results written by save_results; circuit specs can be rebuilt with AnsatzSpec.from_dict."""
    return json.loads(Path(path).read_text())
