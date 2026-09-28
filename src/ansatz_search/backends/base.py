from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

from ansatz_search.circuit.ansatz import AnsatzSpec
from ansatz_search.circuit.gates import GateName, GateSpec, gate_spec


class GradientMethod(str, Enum):
    PARAMETER_SHIFT = "parameter_shift"
    FINITE_DIFFERENCE = "finite_difference"
    # All gradients from one forward and one backward pass (exact simulators only).
    ADJOINT = "adjoint"
    # Automatic differentiation through the simulation (exact simulators only).
    BACKPROP = "backprop"
    # Let the provider pick the fastest method it supports for the circuit size.
    AUTO = "auto"


class GradientProvider(ABC):
    """Backend capability for producing gradient samples."""

    @abstractmethod
    def gradients(
        self,
        circuit: Any,
        observables: Any,
        param_samples: np.ndarray,
        states: Sequence[Any],
    ) -> np.ndarray:
        """Return gradient samples in the shape expected by gradient-based metrics."""


class StateProvider(ABC):
    """Backend capability for producing output statevectors of a compiled circuit."""

    @abstractmethod
    def states(self, circuit: Any, param_samples: np.ndarray, initial_state: Any) -> np.ndarray:
        """Return shape (samples, 2**num_qubits): one output state per parameter row.

        Amplitudes use the framework convention: qubit 0 is the least significant
        bit of the index (as in Qiskit).
        """


class FidelityProvider(ABC):
    """Backend capability for measuring state fidelities directly, e.g. on quantum hardware.

    Hardware cannot return statevectors, but it can estimate how similar two
    output states are. Metrics that only need fidelities (expressibility) accept one.
    """

    @abstractmethod
    def fidelities(self, circuit: Any, params_a: np.ndarray, params_b: np.ndarray, initial_state: Any) -> np.ndarray:
        """Return shape (pairs,): |<psi(params_b[i]) | psi(params_a[i])>|^2, both prepared from `initial_state`."""


class Compiler(ABC):
    """Backend capability for turning an AnsatzSpec into an executable circuit.

    A compiler declares which logical gates its backend supports. It resolves
    the gate names a user allows into GateSpecs before a search starts, so an
    unsupported or misspelled gate fails at submission rather than mid-search.
    """

    supported_gates: frozenset[GateName] = frozenset()

    @property
    def name(self) -> str:
        return type(self).__name__

    def gate_spec(self, gate: str | GateName) -> GateSpec:
        """Return the spec of `gate` if this backend supports it."""
        spec = gate_spec(gate)
        if spec.name not in self.supported_gates:
            supported = sorted(g.value for g in self.supported_gates)
            raise NotImplementedError(
                f"Gate {spec.name.value!r} is not supported by {self.name}. "
                f"Supported gates: {supported}."
            )
        return spec

    def resolve_gates(self, gates: Iterable[str | GateName]) -> tuple[GateSpec, ...]:
        """Resolve user-facing gate names into the specs the search layer uses."""
        if isinstance(gates, str):
            # A bare string would otherwise be iterated character by character.
            raise TypeError(f"allowed_gates must be a sequence of gate names, got the string {gates!r}.")
        specs = tuple(self.gate_spec(gate) for gate in gates)
        if not specs:
            raise ValueError("allowed_gates must contain at least one gate.")
        return specs

    def validate(self, spec: AnsatzSpec) -> None:
        """Check that every block uses a supported gate with the right shape."""
        spec.validate()
        for block in spec:
            expected = self.gate_spec(block.op)
            if (block.num_qubits, block.num_params) != (expected.num_qubits, expected.num_params):
                raise ValueError(
                    f"Gate {block.op.value!r} requires {expected.num_qubits} qubits "
                    f"and {expected.num_params} parameters."
                )

    @abstractmethod
    def compile(self, spec: AnsatzSpec) -> Any:
        """Return the backend object for `spec`."""

    def __call__(self, spec: AnsatzSpec) -> Any:
        return self.compile(spec)


# Default providers per compiled-circuit type; each backend package registers its own on import.
_DEFAULT_PROVIDERS: dict[type, dict[str, type | None]] = {}


def register_providers(
    program_type: type,
    *,
    gradient: type[GradientProvider] | None = None,
    state: type[StateProvider] | None = None,
) -> None:
    """Use these providers for circuits of `program_type` when a metric is given none.

    A new backend calls this in its package `__init__`, next to its compiler.
    """
    _DEFAULT_PROVIDERS[program_type] = {"gradient": gradient, "state": state}


def _registered_provider(circuit: Any, kind: str) -> type | None:
    for cls in type(circuit).__mro__:
        provider = _DEFAULT_PROVIDERS.get(cls, {}).get(kind)
        if provider is not None:
            return provider
    return None


def default_provider(circuit: Any, kind: str) -> GradientProvider | StateProvider:
    """A new default provider of `kind` ("gradient" or "state") for a compiled circuit."""
    provider = _registered_provider(circuit, kind)
    if provider is None and any(cls.__module__.split(".")[0] == "qiskit" for cls in type(circuit).__mro__):
        # Other backends define their circuit type, so it exists only once they registered. A Qiskit
        # circuit can be built without importing the Qiskit backend, e.g. to score it directly.
        import ansatz_search.backends.qiskit  # noqa: F401

        provider = _registered_provider(circuit, kind)
    if provider is None:
        raise TypeError(f"No default {kind} provider for a {type(circuit).__name__}: "
                        f"pass {kind}_provider= to the metric.")
    return provider()

