"""Backend-neutral Pauli observables.

A Pauli sum is written as a mapping from Pauli strings to real coefficients,
e.g. ``{"ZZII": 1.0, "XIXI": -0.5}``. Character ``i`` of a string acts on
qubit ``i`` (the leftmost character is qubit 0), and every string must have
exactly ``num_qubits`` characters. A bare string is shorthand for a single
term with coefficient 1. Backends convert this into their own operator type.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence, Union

PAULIS = frozenset("IXYZ")

PauliSumLike = Union[str, Mapping[str, float]]


class PauliTerm(NamedTuple):
    coeff: float
    paulis: str  # paulis[i] acts on qubit i


def pauli_sum(observable: PauliSumLike, num_qubits: int) -> tuple[PauliTerm, ...]:
    """Validate a Pauli sum and return its terms."""
    terms = {observable: 1.0} if isinstance(observable, str) else observable
    if not isinstance(terms, Mapping) or not terms:
        raise TypeError(
            "An observable must be a Pauli string or a non-empty mapping of Pauli "
            f"strings to coefficients, got {observable!r}."
        )
    parsed = []
    for paulis, coeff in terms.items():
        paulis = paulis.upper()
        if len(paulis) != num_qubits or not set(paulis) <= PAULIS:
            raise ValueError(
                f"Pauli string {paulis!r} must have {num_qubits} characters from 'IXYZ'."
            )
        if isinstance(coeff, complex) and coeff.imag != 0:
            raise ValueError(f"Coefficient of {paulis!r} must be real for a Hermitian observable.")
        parsed.append(PauliTerm(float(getattr(coeff, "real", coeff)), paulis))
    return tuple(parsed)


def pauli_sums(
    observables: PauliSumLike | Sequence[PauliSumLike], num_qubits: int
) -> tuple[tuple[PauliTerm, ...], ...]:
    """Normalize one observable or a sequence of them into a tuple of Pauli sums."""
    if isinstance(observables, (str, Mapping)):
        observables = [observables]
    sums = tuple(pauli_sum(obs, num_qubits) for obs in observables)
    if not sums:
        raise ValueError("Provide at least one observable.")
    return sums
