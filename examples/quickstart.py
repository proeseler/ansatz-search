"""Quickstart: search for an ansatz, re-evaluate the best ones next to the Sim et al. (2019) circuits, plot.

    pip install "ansatz-search[plot]"
    python examples/quickstart.py [--trials 100] [--out examples/runs/quickstart]

NumPy backend (the default), 4 qubits; runs in about a minute.
"""

import argparse
from pathlib import Path

from ansatz_search import BayesianOptimizationSearch, HierarchicalCostFunction, SearchProblem, ansatz_search, evaluate
from ansatz_search.metrics import Expressibility, GradientVariance

parser = argparse.ArgumentParser(description="Search, evaluate and plot in one go.")
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "quickstart")
args = parser.parse_args()

# 1. What to search for: circuits that are trainable (high gradient variance) and expressive, equally weighted.
problem = SearchProblem(
    cost_fn=HierarchicalCostFunction({0: [
        GradientVariance(observables="ZZZZ", max_variance=0.1),  # typical variances are well below 1
        Expressibility(num_qubits=4, log_scale=True),             # the paper's Eq. 8
    ]}),
    num_qubits=4,
    allowed_gates=["rx", "ry", "rz", "cx", "crz"],
    max_param=12, max_gates=24, max_depth=16,
)

# 2. How to search: Bayesian optimization over circuits built gate by gate.
#    Pass compiler=... to ansatz_search to use another backend, e.g. PennyLaneCompiler().
result = ansatz_search(problem, BayesianOptimizationSearch(n_trials=args.trials, seed=42), fdata=args.out)

# 3. Re-evaluate the 5 best circuits on fresh seeds, next to the 19 Sim et al. (2019) circuits.
evaluation = evaluate(result, baselines="sim2019")
print(evaluation)
evaluation.save(args.out / "eval.json")
evaluation.plot(out=args.out / "eval.png", title="Quickstart: found ansatzes vs. Sim et al. (2019)")
print(f"saved {args.out / 'eval.json'} and {args.out / 'eval.png'}")
