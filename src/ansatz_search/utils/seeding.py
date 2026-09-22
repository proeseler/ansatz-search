"""Rebuild samplers, metrics and cost functions with another seed.

Every random draw comes from a generator created in a constructor from its
`seed` argument. A generator keeps its state, so an object that has already
been used gives different samples than a fresh one would. `with_seed` avoids
that: it calls the constructor again with the same arguments and a new seed,
so the copy starts with a fresh generator and without cached values. The
original is not modified.

Rule for new samplers, metrics and cost functions: take a `seed` argument in
`__init__` and create all randomness from it.
"""

from __future__ import annotations

import inspect
from typing import Any, TypeVar

T = TypeVar("T", bound="Reseedable")


class Reseedable:
    """Base class providing `with_seed`: the same object, constructed again with another seed."""

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        # Recorded so with_seed can call the constructor again. copy and pickle
        # call __new__ without arguments and then restore the original record.
        self._init_args = (args, kwargs)
        return self

    def with_seed(self: T, seed: int) -> T:
        """A new object built with the same constructor arguments and `seed`.

        Samplers, metrics and cost functions among the arguments (also inside
        lists, tuples and dicts) are rebuilt with `seed` too: a cost function
        reseeds its metrics, a metric reseeds a sampler passed to it. All other
        arguments, e.g. gradient or state providers, are shared with the copy.
        A class whose constructor takes no `seed` is rebuilt unchanged.
        """
        signature = inspect.signature(type(self).__init__)
        signature = signature.replace(parameters=list(signature.parameters.values())[1:])  # drop self
        args, kwargs = self._init_args
        bound = signature.bind(*args, **kwargs)
        bound.arguments = {name: _reseed(value, seed) for name, value in bound.arguments.items()}
        if "seed" in signature.parameters:
            bound.arguments["seed"] = seed
        return type(self)(*bound.args, **bound.kwargs)


def _reseed(value: Any, seed: int) -> Any:
    if isinstance(value, Reseedable):
        return value.with_seed(seed)
    if type(value) in (list, tuple):
        return type(value)(_reseed(item, seed) for item in value)
    if type(value) is dict:
        return {key: _reseed(item, seed) for key, item in value.items()}
    return value
