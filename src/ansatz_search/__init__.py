"""Ansatz search: find parameterized quantum circuits by searching over their structure.

    from ansatz_search import (BayesianOptimizationSearch, HierarchicalCostFunction, IncrementalAnsatzBuilder,
                               SearchProblem, ansatz_search, evaluate)
    from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider, NumpyStateProvider
    from ansatz_search.metrics import Expressibility, GradientVariance

Metrics live in `ansatz_search.metrics`. Backends are imported from
`ansatz_search.backends.<name>`, since some need optional dependencies.
"""

from importlib.metadata import PackageNotFoundError, version

from .analysis import Evaluation, evaluate, load_search, plot
from .search.run import ansatz_search
from .cost_functions import CostFunction, HierarchicalCostFunction
from .search import (
    AnsatzBuilder, BayesianOptimizationSearch, IncrementalAnsatzBuilder, SearchAlgorithm, SearchProblem, SearchResult,
)

try:
    __version__ = version("ansatz-search")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "unknown"

__all__ = [
    "ansatz_search", "SearchProblem", "SearchResult",
    "SearchAlgorithm", "BayesianOptimizationSearch", "AnsatzBuilder", "IncrementalAnsatzBuilder",
    "CostFunction", "HierarchicalCostFunction",
    "evaluate", "Evaluation", "load_search", "plot",
]
