"""Complexity from the AnsatzSpec, checked against the previous Qiskit-based definition."""

import numpy as np
import pytest

from ansatz_search.benchmarks import SIM2019
from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef
from ansatz_search.metrics.complexity import Complexity

qiskit = pytest.importorskip("qiskit")
from ansatz_search.backends.qiskit.compiler import compile_to_qiskit  # noqa: E402


def qiskit_complexity(spec):
    """The previous definition: len(qc.parameters) + qc.size() + qc.depth()."""
    qc = compile_to_qiskit(spec)
    return len(qc.parameters) + qc.size() + qc.depth()


@pytest.mark.parametrize("i", sorted(SIM2019))
def test_matches_qiskit_on_benchmark_circuits(i):
    spec = SIM2019[i]()
    assert Complexity().value(None, spec) == qiskit_complexity(spec)


def test_matches_qiskit_on_random_circuits_with_reused_parameters_and_fixed_angles():
    rng = np.random.default_rng(0)
    gates = [("ry", 1), ("rz", 1), ("h", 1), ("cx", 2), ("crx", 2), ("cz", 2)]
    for _ in range(50):
        blocks = []
        for _ in range(rng.integers(1, 15)):
            name, nq = gates[rng.integers(len(gates))]
            qubits = tuple(int(q) for q in rng.choice(4, size=nq, replace=False))
            angle = ParamRef(int(rng.integers(0, 5))) if rng.random() < 0.7 else float(rng.normal())
            params = (angle,) if name in ("ry", "rz", "crx") else ()
            blocks.append(AnsatzBlock(name, qubits, params))
        spec = AnsatzSpec(4, blocks)
        assert Complexity().value(None, spec) == qiskit_complexity(spec)


def test_counts_distinct_parameters_gates_and_depth():
    spec = AnsatzSpec(3, [
        AnsatzBlock("ry", (0,), (ParamRef(0),)), AnsatzBlock("ry", (1,), (ParamRef(0),)),  # reused, parallel
        AnsatzBlock("cx", (0, 1)), AnsatzBlock("rz", (2,), (ParamRef(1),)),               # rz runs alongside
    ])
    assert spec.depth == 2
    assert Complexity().value(None, spec) == 2 + 4 + 2


def test_cost_is_normalized_and_clipped():
    m = Complexity(max_param=2, max_gates=5, max_depth=3)
    assert m.cost(5) == pytest.approx(0.5)
    assert m.cost(0) == 0.0
    assert m.cost(25) == 1.0


def test_works_with_any_backend_program():
    from ansatz_search.backends.numpy import NumpyCompiler

    spec = SIM2019[1]()
    assert Complexity().value(NumpyCompiler()(spec), spec) == qiskit_complexity(spec)


def test_needs_the_spec_and_positive_limits():
    with pytest.raises(ValueError, match="pass `spec`"):
        Complexity().value(object())
    with pytest.raises(ValueError, match="must be positive"):
        Complexity(max_depth=0)
