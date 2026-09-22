"""Trainability reproduces the paper (arXiv:2603.14451): benchmark ratios and stored search costs."""

import json
from pathlib import Path

import numpy as np
import pytest

from ansatz_search import HierarchicalCostFunction
from ansatz_search.backends.numpy import NumpyCompiler
from ansatz_search.benchmarks import SIM2019
from ansatz_search.circuit.ansatz import AnsatzSpec
from ansatz_search.metrics import GateError, GradientVariance, Trainability

REFERENCE = json.loads((Path(__file__).parents[1] / "data" / "trainability_reference.json").read_text())


def _value(metric, spec):
    return metric.value(NumpyCompiler()(spec), spec)


@pytest.mark.parametrize("circuit", sorted(REFERENCE["benchmark"]["ratio"], key=int))
def test_benchmark_ratios_match_the_paper(circuit):
    """Sim et al. circuits with the paper's settings; the reference used 12,100 samples."""
    paper = REFERENCE["benchmark"]["ratio"][circuit]
    spec = SIM2019[int(circuit)]()
    ratio = _value(Trainability(observables="ZIII", num_of_param_samples=12100, seed=42), spec)
    if paper < 1e-9:  # Sim 9: no gradient signal at all
        assert ratio < 1e-9
    else:
        assert ratio == pytest.approx(paper, rel=0.025)  # sampling noise with 12,100 samples


def test_stored_search_costs_are_reproduced_with_the_paper_threshold():
    """The expr_train run (tau_BP = 2.5): threshold on the cost function gives Eq. (20)."""
    cost_fn = HierarchicalCostFunction({0: [Trainability(observables="ZIII", seed=1)]}, thresholds={"trainability": 2.5})
    for entry in REFERENCE["expr_train"]["circuits"]:
        spec = AnsatzSpec.from_dict(entry["spec"])
        _, results = cost_fn.evaluate(NumpyCompiler()(spec), spec)
        assert results["trainability"]["cost"] == pytest.approx(entry["stored_cost"], abs=0.01)


def test_value_is_gradient_variance_over_gate_error():
    spec = SIM2019[2]()
    gv = _value(GradientVariance("ZIII", parameter_sampler="uniform", num_of_param_samples=40, seed=3), spec)
    assert _value(Trainability(observables="ZIII", num_of_param_samples=40, seed=3), spec) == gv / _value(GateError(), spec)


def test_cost_and_the_paper_loss_via_a_cost_function_threshold():
    metric = Trainability(observables="ZIII", max_ratio=10)
    assert metric.higher_is_better
    assert [metric.cost(r) for r in (0, 5, 10, 20)] == [1.0, 0.5, 0.0, 0.0]
    fn = HierarchicalCostFunction({0: [metric]}, thresholds={"trainability": 2.5})
    for ratio in (0.0, 1.0, 2.4, 2.5, 4.0):
        assert fn._cost(metric, ratio) == pytest.approx(max(1 - ratio / 2.5, 0))


def test_threshold_beyond_max_ratio_is_rejected():
    with pytest.raises(ValueError, match="max_ratio"):
        HierarchicalCostFunction({0: [Trainability(observables="ZIII", max_ratio=10)]}, thresholds={"trainability": 12})
    # Exactly at the normalization is fine, also for other metrics.
    HierarchicalCostFunction({0: [Trainability(observables="ZIII", max_ratio=8)]}, thresholds={"trainability": 8})
    HierarchicalCostFunction({0: [GradientVariance("ZIII", max_variance=0.1)]}, thresholds={"gradient_variance": 0.1})


def test_expressibility_threshold_gives_the_paper_loss():
    """Eq. (8): max(log(Expr/tau) / log(Expr_max/tau), 0). The paper's 5,000 pairs are 10,000 samples."""
    from ansatz_search.metrics import Expressibility

    metric = Expressibility(num_qubits=4, num_of_param_samples=10000, log_scale=True, seed=42)
    fn = HierarchicalCostFunction({0: [metric]}, thresholds={"expressibility": 0.005})
    for value in (0.003, 0.005, 0.02, 0.3):
        expected = max(np.log(value / 0.005) / np.log(metric.max_value / 0.005), 0)
        assert fn._cost(metric, value) == pytest.approx(expected)
    # With 5,000 samples (2,500 pairs) the noise floor lies above 0.005, so that threshold is unreachable.
    with pytest.raises(ValueError, match="cost already is 0"):
        HierarchicalCostFunction({0: [Expressibility(num_qubits=4, num_of_param_samples=5000, seed=42)]},
                                 thresholds={"expressibility": 0.005})


def test_undefined_ratio_and_invalid_settings():
    spec = SIM2019[1]()
    with pytest.raises(ValueError, match="gate error is 0"):
        _value(Trainability(observables="ZIII", p1_err=0.0, p2_err=0.0, num_of_param_samples=4), spec)
    with pytest.raises(ValueError, match="max_ratio"):
        Trainability(observables="ZIII", max_ratio=0)


def test_gate_error_counts_gates_as_written_on_every_backend():
    spec = SIM2019[2]()  # 8 one-qubit rotations and 3 CX
    expected = 1 - 0.999 ** 8 * 0.99 ** 3
    assert _value(GateError(), spec) == pytest.approx(expected)
    qiskit = pytest.importorskip("qiskit")
    from ansatz_search.backends.qiskit import QiskitCompiler

    assert GateError().value(QiskitCompiler()(spec)) == pytest.approx(expected)  # no spec: counted on the circuit
    with pytest.raises(TypeError, match="spec"):
        GateError().value(NumpyCompiler()(spec))


def test_gate_error_can_count_after_transpiling():
    pytest.importorskip("qiskit")
    spec = SIM2019[4]()  # CRX gates decompose into several basis gates
    as_written = _value(GateError(), spec)
    transpiled = _value(GateError(basis_gates=("rz", "sx", "x", "cx")), spec)
    assert transpiled > as_written
    n1, n2 = GateError(basis_gates=("rz", "sx", "x", "cx")).gate_counts(None, spec)
    assert n2 >= 6 and transpiled == pytest.approx(1 - 0.999 ** n1 * 0.99 ** n2)
    with pytest.raises(ValueError, match="p2_err"):
        GateError(p2_err=1.0)
