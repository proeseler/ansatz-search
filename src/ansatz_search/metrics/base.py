from abc import ABC, abstractmethod
from typing import Dict, Sequence, Union, Type

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..utils.parameter_sampler import ParameterSampler, UniformSampler, GaussianSampler
from ..utils.statevector_sampler import (
    StateSampler,
    HaarStatevectorSampler,
    IrisZZFeatureMapSampler,
)
from ..utils.seeding import Reseedable

class Metric(Reseedable, ABC):
    """A property of an ansatz. A metric that samples takes a `seed` in `__init__`
    and creates all its randomness from it, so `with_seed` can rebuild it."""

    # How analysis.plot shows the raw value: panel title (default: `name`),
    # direction, and whether the axis is logarithmic.
    label: str | None = None
    higher_is_better: bool = False
    log_axis: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this metric (used as the key in your metrics dict)."""

    @abstractmethod
    def value(self, qc, spec=None) -> float:
        """Return the raw metric for a compiled circuit, in its natural units.

        This is the expensive part (simulation, sampling); compute it once and
        derive the cost from it with `cost`.

        `spec` is the backend-neutral AnsatzSpec the circuit was compiled from.
        It carries the qubit and parameter counts, which a compiled backend
        object may not expose.
        """

    def cost(self, value: float) -> float:
        """Map a raw value to a cost in [0, 1]: lower is always better.

        Searches only ever minimize. A metric whose raw value grows with quality
        (e.g. gradient variance) must override this to invert and normalize it,
        so cost functions can combine metrics without knowing their direction.
        The default is the identity, for metrics whose raw value already is such
        a cost (e.g. a gate error probability).
        """
        return value

    def compute(self, qc, spec=None) -> float:
        """Shorthand for ``cost(value(qc, spec))``."""
        return self.cost(self.value(qc, spec))


# Base for metrics estimated from random parameter samples (and optional input states)
class SampledMetric(Metric):
    _param_sampler_registry: Dict[str, Type[ParameterSampler]] = {
        'uniform': UniformSampler,
        'gaussian': GaussianSampler,
        # add new samplers here
    }

    _state_sampler_registry: Dict[str, Type[StateSampler]] = {
        'haar': HaarStatevectorSampler,
        'iris_z_feature_map': IrisZZFeatureMapSampler
        # add new samplers here
    }


    def __init__(self,
                parameter_sampler: Union[str, ParameterSampler],
                num_of_param_samples: int = 1,
                state_sampler: Union[str, StateSampler] = None,
                num_of_state_samples: int = 1,
                seed: int = None):
        """
        Base for any metric that (optionally) takes a circuit,
        samples it, or encodes from statevector.

        `seed` is applied to samplers this metric instantiates from a string
        alias. It has no effect on a sampler instance passed in directly --
        seed that one yourself.
        """
        self.seed = seed
        # If user passed a string, look it up
        if isinstance(parameter_sampler, str):
            try:
                p_sampler = self._param_sampler_registry[parameter_sampler]
            except KeyError:
                raise ValueError(f"Unknown sampler '{parameter_sampler}', "
                                 f"choose from {list(self._param_sampler_registry)}")
            parameter_sampler = p_sampler(seed=seed)
        self.parameter_sampler     = parameter_sampler
        self.num_of_param_samples = num_of_param_samples
        # Set states
        if isinstance(state_sampler, str):
            try:
                s_sampler = self._state_sampler_registry[state_sampler]
            except KeyError:
                raise ValueError(f"Unknown sampler '{state_sampler}', "
                                 f"choose from {list(self._state_sampler_registry)}")
            state_sampler = s_sampler(seed=seed)
        self.state_sampler = state_sampler
        self.num_of_states = num_of_state_samples
        super().__init__()
        
    @staticmethod
    def _circuit_shape(qc, spec) -> tuple[int, int]:
        """Resolve (num_qubits, num_params), preferring the backend-neutral spec."""
        if spec is not None:
            return spec.num_qubits, spec.num_params
        return qc.num_qubits, qc.num_parameters

    def value(self, qc, spec=None) -> float:
        return self.value_from_samples(qc, *self.sample_inputs(qc, spec))

    def sample_inputs(self, qc, spec=None):
        """Draw the (param_samples, stv_vecs) that value_from_samples receives."""
        num_qubits, num_params = self._circuit_shape(qc, spec)
        param_samples = self.parameter_sampler(self.num_of_param_samples, num_params)
        if self.state_sampler is None:
            stv_vecs = [np.asarray(Statevector.from_label('0' * num_qubits), dtype=complex)]
        else:
            stv_vecs = self.state_sampler(num_qubits, self.num_of_states)
        return param_samples, stv_vecs

    @abstractmethod
    def value_from_samples(self, ansatz: QuantumCircuit, param_samples: np.ndarray, stv_vecs: Sequence[Statevector]) -> float:
        """Given the already-sampled inputs, return the raw metric value."""