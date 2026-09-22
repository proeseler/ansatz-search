"""Qiskit backend: Qiskit circuits, exact simulation with Qiskit's primitives, and IBM devices.

Pass `backend=` to QiskitGradientProvider or QiskitFidelityProvider to run on a
fake IBM device (local noisy simulation) or real hardware; this needs the
optional extra: pip install "ansatz-search[ibm]".
"""

from qiskit import QuantumCircuit

from ansatz_search.backends.base import register_providers

from .compiler import QiskitCompiler
from .fidelities import QiskitFidelityProvider
from .gradients import QiskitGradientProvider
from .states import QiskitStateProvider

# Statevectors exist only in simulation; fidelities on hardware are opt-in via QiskitFidelityProvider.
register_providers(QuantumCircuit, gradient=QiskitGradientProvider, state=QiskitStateProvider)

__all__ = ["QiskitCompiler", "QiskitFidelityProvider", "QiskitGradientProvider", "QiskitStateProvider"]
