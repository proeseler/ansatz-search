"""The public API is importable from short paths, without optional dependencies."""

import os
import subprocess
import sys
from pathlib import Path

import ansatz_search
import ansatz_search.cost_functions
import ansatz_search.metrics
import ansatz_search.search


def test_every_exported_name_exists():
    for module in (ansatz_search, ansatz_search.metrics, ansatz_search.search, ansatz_search.cost_functions):
        missing = [name for name in module.__all__ if not hasattr(module, name)]
        assert not missing, (module.__name__, missing)


def test_short_paths_are_the_same_objects_as_the_long_ones():
    from ansatz_search.analysis.evaluation import evaluate
    from ansatz_search.circuit.ansatz import AnsatzSpec
    from ansatz_search.search.run import ansatz_search as run
    from ansatz_search.metrics.expressibility import Expressibility
    from ansatz_search.search.bayesian.bo_search import BayesianOptimizationSearch

    assert ansatz_search.ansatz_search is run
    assert ansatz_search.evaluate is evaluate
    assert ansatz_search.AnsatzSpec is AnsatzSpec
    assert ansatz_search.metrics.Expressibility is Expressibility
    assert ansatz_search.BayesianOptimizationSearch is BayesianOptimizationSearch


def test_importing_the_package_does_not_load_optional_dependencies():
    code = "import sys, ansatz_search; print(sorted(m for m in ('pennylane', 'matplotlib') if m in sys.modules))"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "[]"
