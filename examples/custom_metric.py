"""Write your own metric, combine it with built-in ones by priority, and search with it.

A metric needs a `name`, a raw `value` for a compiled circuit, and a `cost` in [0, 1]
where lower is better. The one below counts two-qubit gates, a simple proxy for
hardware noise. It reads the backend-neutral AnsatzSpec, so it works with every backend.

    pip install "ansatz-search[plot]"
    python examples/custom_metric.py [--trials 100] [--out examples/runs/custom_metric]
"""

import argparse
from pathlib import Path

from ansatz_search import (BayesianOptimizationSearch, HierarchicalCostFunction, IncrementalAnsatzBuilder,
                           SearchProblem, ansatz_search, evaluate)
from ansatz_search.metrics import Expressibility, GradientVariance, Metric


class TwoQubitGates(Metric):
    """Number of two-qubit gates; cost = count / max_count, clipped to [0, 1]."""

    # How plots show this metric.
    label = "Two-qubit gates"
    higher_is_better = False

    # A metric that samples takes a `seed` argument here and creates all its
    # randomness from it, so evaluate() can rebuild it with fresh seeds.
    # This one is deterministic, so it needs none.
    def __init__(self, max_count: int = 10):
        self.max_count = max_count

    @property
    def name(self) -> str:
        return "two_qubit_gates"  # key in results, thresholds and plots

    def value(self, qc, spec=None) -> float:
        return float(sum(1 for block in spec if len(block.qubits) == 2))

    def cost(self, value: float) -> float:
        return min(value / self.max_count, 1.0)


parser = argparse.ArgumentParser(description="Search with a custom metric.")
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "custom_metric")
args = parser.parse_args()

# Priority 0 must be met first: a gradient variance of at least 0.05 (the threshold) costs 0.
# Only then does priority 1 count: expressive circuits with few two-qubit gates, equally weighted.
# Missing priority 0 always costs more than anything in priority 1.
cost_fn = HierarchicalCostFunction(
    metrics_by_priority={
        0: [GradientVariance(observables="ZZZZ", num_of_param_samples=1000, max_variance=0.1, seed=42)],
        1: [Expressibility(num_qubits=4, num_of_param_samples=1000, log_scale=True, seed=42),
            TwoQubitGates(max_count=10)],
    },
    thresholds={"gradient_variance": 0.05},
)
problem = SearchProblem(cost_fn=cost_fn, num_qubits=4, max_param=12, max_gates=24, max_depth=16,
                        allowed_gates=["rx", "ry", "rz", "cx", "crz"])
algorithm = BayesianOptimizationSearch(builder=IncrementalAnsatzBuilder(), n_trials=args.trials,
                                       study_name="custom_metric", seed=42)
result = ansatz_search(problem, algorithm, fdata=args.out)

evaluation = evaluate(result, top=3, seeds=3, baselines="sim2019")
print(evaluation)
evaluation.save(args.out / "eval.json")
evaluation.plot(out=args.out / "eval.png", title="Custom metric: trainable, expressive, few two-qubit gates")
print(f"saved {args.out / 'eval.json'} and {args.out / 'eval.png'}")
