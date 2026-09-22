"""Plot an Evaluation: one panel per metric, one row per circuit, mean ± std over seeds.

    evaluation = evaluate(result, baselines="sim2019")
    evaluation.plot(out="eval.png")   # same as plot(evaluation, out="eval.png")

matplotlib is optional: pip install "ansatz-search[plot]".
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .evaluation import Evaluation

# Chart tokens: light surface, ink, and the first three categorical slots of the
# reference data-viz palette (validated as a set). Text never wears a series color.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
EVALUATED, DURING_SEARCH, REFERENCE = "#2a78d6", "#eb6834", "#1baf7a"
PANELS_PER_ROW = 4
VALUES = ("value", "cost")


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('plot() needs matplotlib: pip install "ansatz-search[plot]"') from exc
    return plt


def _mean_std(summary) -> tuple[float, float]:
    return (summary["mean"], summary["std"]) if summary else (np.nan, np.nan)


def _panels(evaluation: Evaluation, metrics: Sequence[str] | None, values: str) -> list[tuple]:
    """(title, entry -> summary, log axis, is objective) for the objective and every metric."""
    available = list(dict.fromkeys(name for entry in evaluation.circuits.values() for name in entry["metrics"]))
    metrics = available if metrics is None else list(metrics)
    unknown = [name for name in metrics if name not in available]
    if unknown:
        raise ValueError(f"No results for metrics {unknown}; available: {available}.")
    # Evaluations saved before plot styles were recorded fall back to the name, a linear axis, no direction.
    styles = evaluation.settings.get("metrics", {})
    panels = [("Objective · lower is better", lambda entry: entry["objective"], False, True)]
    for name in metrics:
        style = styles.get(name, {})
        label = style.get("label", name)
        if values == "cost":  # every cost is in [0, 1] with 0 best
            title, log = f"{label} cost · lower is better", False
        else:
            better = style.get("higher_is_better")
            title = label if better is None else f"{label} · {'higher' if better else 'lower'} is better"
            log = style.get("log_axis", False)
        panels.append((title, lambda entry, n=name: entry["metrics"].get(n, {}).get(values), log, False))
    return panels


def _style_axis(ax, log: bool) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(which="both", colors=MUTED, labelcolor=INK_2, labelsize=8, length=3, width=0.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator

    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # 0.03, not 3×10⁻²
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


def _legend_handles(rows, primary, reference, outside, has_during, names: Mapping[str, str]):
    """Legend entries for the kinds of rows present; `names` overrides their labels."""
    from matplotlib.lines import Line2D

    handles = []
    if primary.any():
        searched = all(entry.get("source") == "search" for (_, entry), p in zip(rows, primary) if p)
        label = names.get("found", "found by the search (re-evaluated)" if searched else "evaluated circuit")
        handles.append(Line2D([], [], ls="", marker="o", ms=6.5, color=EVALUATED, mec=SURFACE, label=label))
    if has_during:
        handles.append(Line2D([], [], ls="", marker="D", ms=4.5, color=DURING_SEARCH, mec=SURFACE,
                              label=names.get("during_search", "same circuit, value during the search")))
    if (reference & ~outside).any() or (reference & outside).any():
        sim = all(entry.get("source") == "sim2019" for (_, entry), r in zip(rows, reference) if r)
        name = names.get("baseline", "Sim et al. (2019) baseline" if sim else "baseline")
        if (reference & ~outside).any():
            handles.append(Line2D([], [], ls="", marker="s", ms=5.5, color=REFERENCE, mec=SURFACE, label=name))
        if (reference & outside).any():
            handles.append(Line2D([], [], ls="", marker="s", ms=5.5, mfc=SURFACE, mec=REFERENCE, mew=1.4,
                                  label=names.get("outside_limits", "baseline outside the search limits")))
    return handles


def plot(
    evaluation: Evaluation,
    metrics: Sequence[str] | None = None,
    values: str = "value",
    title: str | None = None,
    out: str | os.PathLike | None = None,
    sort: bool = True,
    legend: Mapping[str, str] | None = None,
) -> Figure:
    """One panel for the objective and one per metric; one row per circuit, best objective on top.

    Each dot is the mean over the evaluation seeds with a ±1 std bar. Found
    circuits are blue dots, with the value the search recorded as an orange
    diamond in the objective panel; baselines are green squares, hollow if the
    search could not have produced them. `metrics` selects and orders the
    metric panels (default: all). `values="cost"` plots every metric's cost in
    [0, 1] instead of its raw value. Panels wrap after four per row.

    `sort=False` keeps the evaluation's own row order, e.g. to put related rows
    next to each other. `legend` renames legend entries by kind: "found",
    "during_search", "baseline", "outside_limits". The figure is saved to `out`
    if given, and returned either way.
    """
    if values not in VALUES:
        raise ValueError(f"values must be one of {VALUES}, got {values!r}.")
    if not evaluation.circuits:
        raise ValueError("Nothing to plot: the evaluation has no circuits.")
    plt = _pyplot()

    panels = _panels(evaluation, metrics, values)
    rows = evaluation.ranked() if sort else list(evaluation.circuits.items())
    n = len(rows)
    y = np.arange(n)[::-1].astype(float)
    primary = np.array([entry.get("source") in ("search", "given") for _, entry in rows])
    reference = ~primary
    outside = np.array([entry.get("within_search_limits") is False for _, entry in rows])
    during = np.array([np.nan if entry.get("search_value") is None else entry["search_value"] for _, entry in rows],
                      dtype=float)

    ncols = min(len(panels), PANELS_PER_ROW)
    nrows = math.ceil(len(panels) / ncols)
    header = 0.95  # inches reserved above the panels for title, subtitle and legend
    height = nrows * (0.26 * n + 1.1) + header
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols + 0.8, height),
                             sharey=True, squeeze=False, dpi=200, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - header / height))  # the layout arranges only the panels
    fig.patch.set_facecolor(SURFACE)
    flat = axes.ravel()
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    # A title wider than its panel would run off the figure: then every title puts its
    # direction on a second line, so the panel headers stay aligned.
    wrap = any(len(panel[0]) > 42 for panel in panels)

    for ax, (panel_title, get, log, is_objective) in zip(flat, panels):
        mean, std = np.array([_mean_std(get(entry)) for _, entry in rows], dtype=float).T
        log = log and np.any(mean > 0) and not np.any(mean <= 0)
        low = np.minimum(std, mean * 0.999) if log else std  # keep error bars above zero on a log axis
        ax.errorbar(mean, y, xerr=[low, std], fmt="none", ecolor=INK_2, elinewidth=1, capsize=0, zorder=2)
        inside = reference & ~outside
        ax.plot(mean[primary], y[primary], "o", ms=6.5, color=EVALUATED, mec=SURFACE, mew=1.3, zorder=3)
        ax.plot(mean[inside], y[inside], "s", ms=5.5, color=REFERENCE, mec=SURFACE, mew=1.2, zorder=3)
        ax.plot(mean[reference & outside], y[reference & outside], "s", ms=5.5, mfc=SURFACE, mec=REFERENCE,
                mew=1.4, zorder=3)
        if is_objective:
            ax.plot(during, y + 0.3, "D", ms=4.5, color=DURING_SEARCH, mec=SURFACE, mew=1, zorder=3)
        _style_axis(ax, log)
        ax.set_title(panel_title.replace(" · ", "\n") if wrap else panel_title, loc="left", color=INK, fontsize=9.5)

    flat[0].set_yticks(y, [label for label, _ in rows])
    flat[0].set_ylim(-0.7, n - 0.3)
    for ax in axes[:, 0]:
        for tick, is_primary in zip(ax.get_yticklabels(), primary):
            tick.set_color(INK if is_primary else INK_2)
            tick.set_fontweight("bold" if is_primary else "normal")

    seeds = evaluation.settings.get("seeds")
    subtitle = f"mean ± σ over {len(seeds)} seeds" if seeds else "mean ± σ"
    subtitle += (" · rows sorted by objective" if sort else "") + (" · metric costs, 0 is best" if values == "cost" else "")
    handles = _legend_handles(rows, primary, reference, outside, np.isfinite(during).any(), legend or {})
    fig.text(0.01, 1 - 0.12 / height, title or "Evaluated circuits", ha="left", va="top", color=INK,
             fontsize=12.5, fontweight="bold")
    fig.text(0.01, 1 - 0.42 / height, subtitle, ha="left", va="top", color=INK_2, fontsize=8.5)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.004, 1 - 0.58 / height),
               ncol=max(len(handles), 1), frameon=False, fontsize=8.5, labelcolor=INK, handletextpad=0.3,
               columnspacing=1.6)

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=SURFACE)
    return fig
