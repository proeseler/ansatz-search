"""Backend-neutral Pauli observables."""

import pytest

from ansatz_search.circuit.observables import PauliTerm, pauli_sum, pauli_sums


def test_string_is_a_single_unit_term():
    assert pauli_sum("zzi", 3) == (PauliTerm(1.0, "ZZI"),)


def test_mapping_keeps_coefficients():
    assert pauli_sum({"XI": 0.5, "IZ": -2}, 2) == (PauliTerm(0.5, "XI"), PauliTerm(-2.0, "IZ"))


@pytest.mark.parametrize("bad", [{"ZZ": 1.0}, {"ZQZ": 1.0}])
def test_wrong_length_or_letter_is_rejected(bad):
    with pytest.raises(ValueError, match="3 characters from 'IXYZ'"):
        pauli_sum(bad, 3)


def test_complex_coefficient_must_be_real():
    with pytest.raises(ValueError, match="must be real"):
        pauli_sum({"Z": 1j}, 1)
    assert pauli_sum({"Z": 2 + 0j}, 1) == (PauliTerm(2.0, "Z"),)


@pytest.mark.parametrize("bad", [{}, 3.0])
def test_empty_or_non_mapping_is_rejected(bad):
    with pytest.raises(TypeError):
        pauli_sum(bad, 1)


def test_single_observable_and_sequence_are_normalized():
    assert len(pauli_sums("ZZ", 2)) == 1
    assert len(pauli_sums(["ZZ", {"XX": 1.0}], 2)) == 2
    with pytest.raises(ValueError, match="at least one observable"):
        pauli_sums([], 2)
