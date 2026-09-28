"""Compilers resolve user-facing gate names into specs before a search starts."""

import pytest

from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef
from ansatz_search.circuit.gates import GATE_SHAPES, GateName, GateSpec


@pytest.fixture(params=["qiskit", "numpy", "pennylane"])
def compiler(request):
    if request.param == "qiskit":
        pytest.importorskip("qiskit")
        from ansatz_search.backends.qiskit.compiler import QiskitCompiler
        return QiskitCompiler()
    if request.param == "numpy":
        from ansatz_search.backends.numpy import NumpyCompiler
        return NumpyCompiler()
    pytest.importorskip("pennylane")
    from ansatz_search.backends.pennylane import PennyLaneCompiler
    return PennyLaneCompiler()


def test_resolves_names_to_canonical_specs(compiler):
    specs = compiler.resolve_gates(["h", "rx", "cx", "crz"])
    assert specs == (
        GateSpec(GateName.H, 1, 0),
        GateSpec(GateName.RX, 1, 1),
        GateSpec(GateName.CX, 2, 0),
        GateSpec(GateName.CRZ, 2, 1),
    )


def test_aliases_and_case_are_normalized(compiler):
    assert compiler.resolve_gates(["p", " CP ", "RX"]) == (
        GateSpec(GateName.R1, 1, 1),
        GateSpec(GateName.CR1, 2, 1),
        GateSpec(GateName.RX, 1, 1),
    )


def test_unknown_gate_is_a_value_error(compiler):
    with pytest.raises(ValueError, match="Unknown gate 'iswap'"):
        compiler.resolve_gates(["h", "iswap"])


def test_known_but_unsupported_gate_names_the_backend():
    from ansatz_search.backends.numpy import NumpyCompiler

    class HadamardOnly(NumpyCompiler):
        supported_gates = frozenset({GateName.H})

    with pytest.raises(NotImplementedError, match="rx.*HadamardOnly"):
        HadamardOnly().resolve_gates(["h", "rx"])


def test_bare_string_is_rejected(compiler):
    # Would otherwise be iterated as "r", "x" -- and "x" is a real gate.
    with pytest.raises(TypeError, match="sequence of gate names"):
        compiler.resolve_gates("rx")


def test_empty_gate_list_is_rejected(compiler):
    with pytest.raises(ValueError, match="at least one gate"):
        compiler.resolve_gates([])


def test_supported_gates_all_have_canonical_shapes(compiler):
    assert compiler.supported_gates <= set(GATE_SHAPES)


def test_compile_rejects_a_block_with_the_wrong_shape(compiler):
    with pytest.raises(ValueError, match="requires 1 qubits and 1 parameters"):
        compiler(AnsatzSpec(2, [AnsatzBlock("rx", (0, 1), (ParamRef(0),))]))


def test_compilers_support_the_same_gates(compiler):
    """Every backend offers the same gate set, so a search can switch backends freely."""
    pytest.importorskip("qiskit")
    from ansatz_search.backends.qiskit.compiler import QiskitCompiler
    assert compiler.supported_gates == QiskitCompiler.supported_gates


def test_qiskit_supported_gates_all_compile():
    """Guard supported_gates against drifting from compile_to_qiskit's if/elif chain."""
    pytest.importorskip("qiskit")
    from ansatz_search.backends.qiskit.compiler import QiskitCompiler
    compiler = QiskitCompiler()
    for gate in compiler.supported_gates:
        n_qubits, n_params = GATE_SHAPES[gate]
        block = AnsatzBlock(gate, tuple(range(n_qubits)), tuple(ParamRef(i) for i in range(n_params)))
        compiler(AnsatzSpec(n_qubits, [block]))
