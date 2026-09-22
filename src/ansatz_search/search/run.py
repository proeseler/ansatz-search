"""Run a search: ansatz_search() runs an algorithm on a problem and stores the run's config.yaml."""

from __future__ import annotations
import os
import yaml

from typing import Sequence

from ansatz_search.backends.base import Compiler
from ansatz_search.backends.numpy import NumpyCompiler
from ansatz_search.circuit.gates import GateSpec
from ansatz_search.search.base import SearchAlgorithm, SearchProblem, SearchResult

def preprocess_config(config):
    """Recursively preprocess the config dictionary to make it YAML/JSON-serializable."""
    if isinstance(config, dict):
        return {key: preprocess_config(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [preprocess_config(item) for item in config]
    elif isinstance(config, (int, float, str, bool, type(None))):
        return config  # Already serializable
    elif hasattr(config, "__class__"):
        # Convert objects to string representations
        return f"{config.__class__.__name__}({config})"
    return str(config)  # Fallback to string conversion


def store_config(
    problem: SearchProblem,
    algorithm: SearchAlgorithm,
    compiler: Compiler,
    gate_specs: Sequence[GateSpec],
    path,
):
    """Save the configuration to a YAML file."""
    config = {
        "compiler": compiler.name,
        "cost_fn": problem.cost_fn.__class__.__name__,
        "thresholds": getattr(problem.cost_fn, "thresholds", {}),
        "metrics": [
            {
                "name": metric.name,
                "type": metric.__class__.__name__,
                # _init_args is with_seed's record of the constructor call, not a setting.
                "parameters": {k: v for k, v in vars(metric).items() if k != "_init_args"},
            }
            for metric in problem.cost_fn.metrics
        ],
        "num_qubits": problem.num_qubits,
        "max_param": problem.max_param,
        "max_depth": problem.max_depth,
        "max_gates": problem.max_gates,
        # Resolved specs rather than the raw names: aliases are normalized
        # and the recorded shape is exactly what the builder searched with.
        "allowed_gates": [
            [spec.name.value, spec.num_qubits, spec.num_params] for spec in gate_specs
        ],
        "min_params": problem.min_params,
        "prev_params": problem.prev_params,
        "topology": problem.topology,
        "data_dir": os.path.dirname(str(path)),
    }
    # Algorithm settings vary by implementation; record whatever is present.
    for field in ("study_name", "n_trials", "stored_circuit", "allowed_rep", "storage", "seed"):
        if hasattr(algorithm, field):
            config[field] = getattr(algorithm, field)

    config = preprocess_config(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def ansatz_search(
    problem: SearchProblem,
    algorithm: SearchAlgorithm,
    compiler: Compiler | None = None,
    fdata: str | os.PathLike | None = None,
) -> SearchResult:
    """Run `algorithm` on `problem`; config.yaml (and by default the study) goes into the folder `fdata`.

    `compiler` picks the backend and defaults to NumPy. Metrics created without
    a provider use the provider of that backend.
    """
    if fdata is None:
        raise TypeError("ansatz_search() needs fdata: the folder for config.yaml and the stored trials.")
    compiler = NumpyCompiler() if compiler is None else compiler
    # Resolve gate names first, so a misspelled or unsupported gate fails here,
    # before any config is written or a study is created.
    gate_specs = compiler.resolve_gates(problem.allowed_gates)
    # An algorithm without its own data directory stores its study next to config.yaml.
    if getattr(algorithm, "data_dir", "") is None:
        algorithm.data_dir = str(fdata)

    store_config(problem, algorithm, compiler, gate_specs, path=os.path.join(str(fdata), "config.yaml"))

    result = algorithm.run(problem, compiler, gate_specs)
    if isinstance(result, SearchResult):
        # Everything analysis.evaluate needs to replay and re-evaluate this run.
        result.run_dir = str(fdata)
        if result.problem is None:
            result.problem = problem
        if result.compiler is None:
            result.compiler = compiler
    return result
