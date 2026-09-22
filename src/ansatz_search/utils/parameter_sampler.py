from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .seeding import Reseedable

__all__ = ["ParameterSampler", "UniformSampler", "GaussianSampler"]


class ParameterSampler(Reseedable, ABC):
    """Base interface for generating batches of ansatz parameters."""

    def __init__(self, seed: int = None) -> None:
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_shape(num_samples: int, num_parameters: int) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        if num_parameters < 1:
            raise ValueError("num_parameters must be at least 1.")

    @abstractmethod
    def __call__(self, num_samples: int, num_parameters: int) -> NDArray[np.float64]:
        """Return a `(num_samples, num_parameters)` array of sampled parameters."""


class UniformSampler(ParameterSampler):
    """Sample parameters independently from a uniform distribution."""

    def __init__(
        self,
        low: float = 0.0,
        high: float = 2 * np.pi,
        seed: int = None,
    ) -> None:
        super().__init__(seed=seed)
        if low >= high:
            raise ValueError("low must be smaller than high.")
        self.low = low
        self.high = high

    def __call__(self, num_samples: int, num_parameters: int) -> NDArray[np.float64]:
        self._validate_shape(num_samples, num_parameters)
        return self._rng.uniform(self.low, self.high, size=(num_samples, num_parameters))


class GaussianSampler(ParameterSampler):
    """Sample parameters independently from a normal distribution."""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = None,
    ) -> None:
        super().__init__(seed=seed)
        if std <= 0:
            raise ValueError("std must be positive.")
        self.mean = mean
        self.std = std

    def __call__(self, num_samples: int, num_parameters: int) -> NDArray[np.float64]:
        self._validate_shape(num_samples, num_parameters)
        return self._rng.normal(self.mean, self.std, size=(num_samples, num_parameters))
