"""Gate names are checked when a search is submitted, before any side effects."""

import pytest

pytest.importorskip("optuna")
pytest.importorskip("qiskit")

from ansatz_search.backends.qiskit.compiler import QiskitCompiler
from ansatz_search import ansatz_search
from ansatz_search.search.base import SearchAlgorithm, SearchProblem


class _Recording(SearchAlgorithm):
    def __init__(self):
        super().__init__()
        self.calls = []

    def run(self, problem, compiler, gate_specs):
        self.calls.append((compiler, gate_specs))
        return "ran"


class _NoMetrics:
    metrics = ()


def _problem(allowed_gates):
    return SearchProblem(
        cost_fn=_NoMetrics(), num_qubits=2, max_param=4,
        max_depth=8, max_gates=8, allowed_gates=allowed_gates,
    )


def test_bad_gate_fails_before_anything_is_written(tmp_path):
    algorithm = _Recording()
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="Unknown gate"):
        ansatz_search(_problem(["h", "cnot"]), algorithm, QiskitCompiler(), out)
    assert not out.exists()
    assert algorithm.calls == []


def test_resolved_specs_reach_the_algorithm_and_config(tmp_path):
    yaml = pytest.importorskip("yaml")
    algorithm = _Recording()
    compiler = QiskitCompiler()
    assert ansatz_search(_problem(["h", "p", "cx"]), algorithm, compiler, tmp_path) == "ran"

    (used_compiler, specs), = algorithm.calls
    assert used_compiler is compiler
    assert [s.name.value for s in specs] == ["h", "r1", "cx"]

    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config["compiler"] == "QiskitCompiler"
    assert config["allowed_gates"] == [["h", 1, 0], ["r1", 1, 1], ["cx", 2, 0]]
