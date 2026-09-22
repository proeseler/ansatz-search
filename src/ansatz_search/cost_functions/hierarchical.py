from typing import Dict, Mapping, Optional, Sequence, Tuple

from .base import CostFunction, MetricResults
from ansatz_search.metrics.base import Metric

AGGREGATES = ("sum", "mean")


class HierarchicalCostFunction(CostFunction):
    """
    Combine normalized metrics with explicit priority groups.

    `metrics_by_priority` maps integer priority -> sequence of Metric instances.
    Lower integer values represent higher priority. All metrics must be normalized
    to [0, 1] for fair hierarchical comparison.

    A group's score is the sum (default) or the mean of its metrics' costs.
    The objective is lexicographic: for the first group k whose score is above
    0, it is that score plus the highest possible score of every later group,
    and 0 once every group is met. Missing an earlier priority, even barely,
    therefore always costs more than anything in a later one. With "sum" this
    is Eqs. (28)-(29) of arXiv:2603.14451, e.g. ``L1 + L2 + 1`` for two metrics
    followed by a one-metric group.

    `weights` optionally maps metric names to positive weights (default 1): a
    group's score is then the weighted sum (or weighted mean) of its costs, and
    a later group's highest possible score is the sum of its weights, so the
    lexicographic order still holds. E.g. {"expressibility": 2} makes a gain
    in expressibility count twice as much as the same gain in trainability.
    """

    def __init__(
        self,
        metrics_by_priority: Dict[int, Sequence[Metric]],
        thresholds: Optional[Mapping[str, float]] = None,
        aggregate: str = "sum",
        weights: Optional[Mapping[str, float]] = None,
    ):
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}.")
        self.aggregate = aggregate
        self.metrics_by_priority = metrics_by_priority
        self.priority_list = sorted(self.metrics_by_priority.keys())

        flattened_metrics = []
        for priority in self.priority_list:
            group = self.metrics_by_priority[priority]
            flattened_metrics.extend(group)

        super().__init__(flattened_metrics, thresholds)

        self.weights = dict(weights or {})
        unknown = set(self.weights) - {metric.name for metric in flattened_metrics}
        if unknown:
            raise ValueError(f"Weights given for unknown metrics: {sorted(unknown)}.")
        if any(not weight > 0 for weight in self.weights.values()):
            raise ValueError(f"Weights must be positive, got {self.weights}.")

    def _weight(self, metric: Metric) -> float:
        return float(self.weights.get(metric.name, 1.0))

    def _score(self, group: Sequence[Metric], metrics_val: MetricResults) -> float:
        total = sum(self._weight(metric) * metrics_val[metric.name]["cost"] for metric in group)
        return total if self.aggregate == "sum" else total / sum(self._weight(metric) for metric in group)

    def _max_score(self, group: Sequence[Metric]) -> float:
        return sum(self._weight(metric) for metric in group) if self.aggregate == "sum" else 1.0

    def evaluate(self, circuit, spec=None) -> Tuple[float, MetricResults]:
        metrics_val = self.compute_metrics(circuit, spec)

        for k, priority in enumerate(self.priority_list):
            score = self._score(self.metrics_by_priority[priority], metrics_val)
            if score > 0:
                later = sum(self._max_score(self.metrics_by_priority[p]) for p in self.priority_list[k + 1:])
                return later + score, metrics_val

        return 0.0, metrics_val
