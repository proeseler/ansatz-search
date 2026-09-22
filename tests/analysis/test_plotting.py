"""plot(): one panel per metric, rows sorted by objective, plot styles stored with the evaluation."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ansatz_search.analysis import Evaluation, evaluate, plot  # noqa: E402
from ansatz_search.backends.numpy import NumpyCompiler, NumpyGradientProvider, NumpyStateProvider  # noqa: E402
from ansatz_search.benchmarks import SIM2019  # noqa: E402
from ansatz_search.cost_functions import HierarchicalCostFunction  # noqa: E402
from ansatz_search.metrics.complexity import Complexity  # noqa: E402
from ansatz_search.metrics.expressibility import Expressibility  # noqa: E402
from ansatz_search.metrics.gradient_variance import GradientVariance  # noqa: E402

TITLES = [
    "Objective · lower is better",
    "Gradient variance · higher is better",
    "Expressibility (KL divergence) · lower is better",
    "Complexity (params + gates + depth) · lower is better",
]


@pytest.fixture(scope="module")
def evaluation():
    cost_fn = HierarchicalCostFunction({0: [GradientVariance(
        observables="ZZZZ", gradient_provider=NumpyGradientProvider(), num_of_param_samples=16)]})
    extras = [Expressibility(num_qubits=4, state_provider=NumpyStateProvider(), num_of_param_samples=20,
                             floor_repetitions=3), Complexity()]
    return evaluate({"a": SIM2019[1](), "b": SIM2019[2](), "c": SIM2019[9]()}, cost_fn=cost_fn,
                    compiler=NumpyCompiler(), seeds=[1, 2], baselines={"ref": SIM2019[3]()},
                    extra_metrics=extras, progress=False)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _panels(fig):
    return [ax for ax in fig.axes if ax.get_visible()]


def _titles(fig):
    """Panel titles, with a wrapped direction joined back onto the label's line."""
    return [ax.get_title(loc="left").replace("\n", " · ") for ax in _panels(fig)]


def _synthetic(entries):
    """An Evaluation from (label, source, extra fields) without running any metric."""
    summary = {"mean": 0.5, "std": 0.1, "runs": [0.5]}
    circuits = {}
    for rank, (label, source, extra) in enumerate(entries):
        s = {**summary, "mean": 0.1 * (rank + 1)}
        circuits[label] = {"objective": s, "metrics": {"m": {"value": s, "cost": s}}, "source": source, **extra}
    return Evaluation(circuits, {"seeds": [1]})


def test_evaluate_stores_how_each_metric_is_plotted(evaluation):
    styles = evaluation.settings["metrics"]
    assert styles["expressibility"] == {"label": "Expressibility (KL divergence)", "higher_is_better": False,
                                        "log_axis": True}
    assert styles["gradient_variance"]["higher_is_better"] is True
    assert styles["complexity"]["label"] == Complexity.label


def test_one_panel_per_metric_with_direction_and_log_axis(evaluation, tmp_path):
    out = tmp_path / "plots" / "eval.png"
    fig = plot(evaluation, out=out)
    assert out.stat().st_size > 0
    assert _titles(fig) == TITLES
    assert [ax.get_xscale() for ax in _panels(fig)] == ["linear", "linear", "log", "linear"]


def test_rows_are_sorted_by_objective_best_on_top(evaluation):
    ax = _panels(plot(evaluation, metrics=[]))[0]
    by_height = sorted(zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]), reverse=True)
    assert [label for _, label in by_height] == [label for label, _ in evaluation.ranked()]


def test_legend_names_the_kinds_of_circuits(evaluation):
    assert [t.get_text() for t in plot(evaluation).legends[0].get_texts()] == ["evaluated circuit", "baseline"]
    search = _synthetic([
        ("search #1", "search", {"search_value": 0.05, "trial": 3}),
        ("Sim 1", "sim2019", {"within_search_limits": True}),
        ("Sim 2", "sim2019", {"within_search_limits": False}),
    ])
    assert [t.get_text() for t in plot(search).legends[0].get_texts()] == [
        "found by the search (re-evaluated)", "same circuit, value during the search",
        "Sim et al. (2019) baseline", "baseline outside the search limits",
    ]


def test_metric_selection_costs_and_invalid_arguments(evaluation):
    assert _titles(plot(evaluation, metrics=["complexity"])) == [TITLES[0], TITLES[3]]
    costs = plot(evaluation, values="cost")
    assert "Expressibility (KL divergence) cost · lower is better" in _titles(costs)
    assert {ax.get_xscale() for ax in _panels(costs)} == {"linear"}
    with pytest.raises(ValueError, match="available"):
        plot(evaluation, metrics=["nope"])
    with pytest.raises(ValueError, match="values"):
        plot(evaluation, values="raw")


def test_loaded_and_older_evaluations_plot(evaluation, tmp_path):
    loaded = Evaluation.load(evaluation.save(tmp_path / "eval.json"))
    assert _titles(loaded.plot()) == TITLES
    older = Evaluation(loaded.circuits, {"seeds": [1, 2]})  # saved before plot styles were recorded
    fig = older.plot()
    assert _titles(fig)[1:] == ["gradient_variance", "expressibility", "complexity"]  # no direction claimed
    assert {ax.get_xscale() for ax in _panels(fig)} == {"linear"}


def test_long_titles_move_the_direction_to_a_second_line_in_every_panel(evaluation):
    wrapped = [ax.get_title(loc="left") for ax in _panels(plot(evaluation))]
    assert wrapped[0] == "Objective\nlower is better"
    assert all("\n" in title for title in wrapped)
    short = [ax.get_title(loc="left") for ax in _panels(plot(evaluation, metrics=["gradient_variance"]))]
    assert short == ["Objective · lower is better", "Gradient variance · higher is better"]


def test_unsorted_rows_and_renamed_legend_entries(evaluation):
    fig = plot(evaluation, metrics=[], sort=False, legend={"found": "device", "baseline": "ideal"})
    ax = _panels(fig)[0]
    by_height = sorted(zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]), reverse=True)
    assert [label for _, label in by_height] == list(evaluation.circuits)
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["device", "ideal"]


def test_panels_wrap_after_four_per_row():
    summary = {"mean": 1.0, "std": 0.1, "runs": [1.0]}
    entry = {"objective": summary, "metrics": {f"m{i}": {"value": summary, "cost": summary} for i in range(5)},
             "source": "given"}
    panels = _panels(plot(Evaluation({"x": entry, "y": dict(entry)}, {})))
    assert len(panels) == 6
    assert sorted({ax.get_subplotspec().rowspan.start for ax in panels}) == [0, 1]
