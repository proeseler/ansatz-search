from typing import Dict, Mapping, Optional, Sequence, Tuple

from .base import CostFunction, MetricResults
from ansatz_search.metrics.base import Metric


class WeightedHierarchicalCostFunction(CostFunction):
    """
    Combine normalized metrics with explicit priority groups.

    `metrics_by_priority` maps integer priority -> sequence of Metric instances.
    Lower integer values represent higher priority. All metrics must be normalized
    to [0, 1] for fair hierarchical comparison.
    """

    def __init__(
        self,
        metrics_by_priority: Dict[int, Sequence[Metric]],
        thresholds: Optional[Mapping[str, float]] = None,
    ):
        self.metrics_by_priority = metrics_by_priority
        self.priority_list = sorted(self.metrics_by_priority.keys())

        flattened_metrics = []
        for priority in self.priority_list:
            group = self.metrics_by_priority[priority]
            flattened_metrics.extend(group)

        super().__init__(flattened_metrics, thresholds)

    def evaluate(self, circuit, spec=None) -> Tuple[float, MetricResults]:
        metrics_val = self.compute_metrics(circuit, spec)

        group_scores = []
        for priority in self.priority_list:
            group = self.metrics_by_priority[priority]
            group_values = [metrics_val[metric.name]["cost"] for metric in group]
            group_scores.append(sum(group_values) / len(group_values))

        score = 0.0
        base = 2.0
        num_groups = len(group_scores)
        for index, group_score in enumerate(group_scores):
            weight = base ** (num_groups - index - 1)
            score += group_score * weight

        return score, metrics_val

