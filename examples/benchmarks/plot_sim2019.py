"""Plot expressibility and entanglement of the 19 Sim et al. (2019) circuits.

Computes both metrics with ansatz-search (mean and standard deviation over a few
seeds), overlays the values of the original implementation used for the paper
(arXiv:2603.14451, stored in tests/data/sim2019_reference.json),
prints the numbers as a table and saves the figure next to this script.

    python examples/benchmarks/plot_sim2019.py [--seeds 5] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from ansatz_search.backends.numpy import NumpyCompiler, NumpyStateProvider
from ansatz_search.benchmarks import SIM2019
from ansatz_search.metrics.entanglement import Entanglement
from ansatz_search.metrics.expressibility import Expressibility

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parents[1] / "tests" / "data" / "sim2019_reference.json"
NUM_SAMPLES = 10000
CIRCUITS = sorted(SIM2019)
# Order of the circuits along each panel's x-axis.
ORDER = {
    "expressibility": [9, 1, 2, 16, 3, 18, 10, 12, 15, 17, 4, 11, 7, 8, 19, 5, 13, 14, 6],
    "entanglement": [1, 7, 3, 16, 8, 5, 18, 17, 4, 10, 19, 13, 12, 14, 11, 6, 2, 15, 9],
}
for _kind, _order in ORDER.items():
    assert sorted(_order) == CIRCUITS, f"ORDER[{_kind!r}] must list every circuit once"

# Chart tokens (light surface, from the reference data-viz palette).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SERIES_CURRENT, SERIES_LEGACY = "#2a78d6", "#eb6834"  # categorical slots 1 and 2


def compute(seeds: list[int]) -> dict[str, np.ndarray]:
    """Array of shape (len(seeds), 19) per metric; column j is circuit j + 1."""
    out = {"expressibility": [], "entanglement": []}
    compiler, provider = NumpyCompiler(), NumpyStateProvider()
    for seed in seeds:
        row_e, row_q = [], []
        for i in CIRCUITS:
            spec = SIM2019[i]()
            program = compiler(spec)
            kwargs = dict(num_qubits=4, state_provider=provider, num_of_param_samples=NUM_SAMPLES, seed=seed)
            row_e.append(Expressibility(**kwargs).value(program, spec))
            row_q.append(Entanglement(**kwargs).value(program, spec))
        out["expressibility"].append(row_e)
        out["entanglement"].append(row_q)
    return {k: np.array(v) for k, v in out.items()}


def legacy_values() -> dict[str, np.ndarray] | None:
    if not REFERENCE.exists():
        return None
    circuits = json.loads(REFERENCE.read_text())["circuits"]
    return {
        kind: np.array([circuits[str(i)][kind]["legacy"] for i in CIRCUITS])
        for kind in ("expressibility", "entanglement")
    }


def print_table(values, legacy) -> None:
    e, q = values["expressibility"], values["entanglement"]
    header = f"{'circuit':>7} {'params':>6} | {'expressibility':>14} {'± σ':>8}"
    header += f" {'paper':>8}" if legacy else ""
    header += f" | {'entanglement':>12} {'± σ':>7}" + (f" {'paper':>7}" if legacy else "")
    print(header)
    for j, i in enumerate(CIRCUITS):
        line = f"{i:7d} {SIM2019[i]().num_params:6d} | {e[:, j].mean():14.5f} {e[:, j].std(ddof=1):8.5f}"
        line += f" {legacy['expressibility'][j]:8.5f}" if legacy else ""
        line += f" | {q[:, j].mean():12.4f} {q[:, j].std(ddof=1):7.4f}"
        line += f" {legacy['entanglement'][j]:7.4f}" if legacy else ""
        print(line)


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(which="both", colors=MUTED, labelcolor=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def dot_series(ax, x, mean, std, legacy):
    ax.errorbar(x, mean, yerr=std, fmt="none", ecolor=INK_2, elinewidth=1, capsize=0, zorder=2)
    ax.plot(x, mean, "o", ms=7, color=SERIES_CURRENT, mec=SURFACE, mew=1.5, zorder=3, label="ansatz-search (mean ± σ)")
    if legacy is not None:
        ax.plot(x + 0.28, legacy, "D", ms=5, color=SERIES_LEGACY, mec=SURFACE, mew=1.2, zorder=3,
                label="original implementation (paper)")


def label_point(ax, x, y, text, dx_points=0, dy_points=0, ha="center"):
    va = "bottom" if dy_points > 0 else "top" if dy_points < 0 else "center"
    ax.annotate(text, (x, y), xytext=(dx_points, dy_points), textcoords="offset points",
                ha=ha, va=va, fontsize=9, color=INK_2)


def plot(values, legacy, seeds, out: Path) -> None:
    x = np.arange(1, 20)
    fig, (ax_e, ax_q) = plt.subplots(2, 1, figsize=(10, 7.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    def panel(ax, kind):
        """Draw one metric with its circuits in ORDER[kind]; returns the per-position means."""
        cols = [i - 1 for i in ORDER[kind]]
        data = values[kind][:, cols]
        mean = data.mean(axis=0)
        dot_series(ax, x, mean, data.std(axis=0, ddof=1), None if legacy is None else legacy[kind][cols])
        ax.set_xticks(x, [str(i) for i in ORDER[kind]])
        ax.set_xlim(0.4, 19.7)
        ax.set_xlabel("Sim et al. (2019) circuit", color=INK_2, fontsize=9)
        style_axis(ax)
        return mean

    def label_extreme(ax, pos, y, text, dy_points):
        # Anchor the label so it runs into the plot, not over an axis edge.
        edge = "left" if pos < 3 else "right" if pos > len(x) - 4 else "center"
        if dy_points < 0 and edge != "center":  # below the point: put it beside instead
            label_point(ax, x[pos], y, text, dx_points=9 if edge == "left" else -9, ha=edge)
        else:
            label_point(ax, x[pos], y, text, dx_points={"left": -6, "right": 6, "center": 0}[edge],
                        dy_points=dy_points, ha=edge)

    e_mean = panel(ax_e, "expressibility")
    ax_e.set_yscale("log")
    # Plain decimals instead of 10^-k, so values can be read off directly.
    ax_e.yaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3, 1]))
    ax_e.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_e.yaxis.set_minor_locator(NullLocator())
    ax_e.set_ylim(0.002, 1.2)
    ax_e.set_ylabel("KL divergence to Haar\n(log scale)", color=INK_2, fontsize=9)
    ax_e.set_title("Expressibility  ·  lower is more expressive", loc="left", color=INK, fontsize=11)
    best = int(np.argmin(e_mean))
    label_extreme(ax_e, best, e_mean[best], "most expressive", -9)

    q_mean = panel(ax_q, "entanglement")
    ax_q.set_ylim(-0.04, 1.12)
    ax_q.set_ylabel("Meyer–Wallach Q", color=INK_2, fontsize=9)
    ax_q.set_title("Entanglement  ·  higher is more entangling", loc="left", color=INK, fontsize=11)
    top = int(np.argmax(q_mean))
    label_extreme(ax_q, top, q_mean[top], "most entangling", 7)

    handles, labels = ax_e.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK, handletextpad=0.3, columnspacing=1.4)
    fig.suptitle("Sim et al. (2019) benchmark circuits, 4 qubits", x=0.012, y=0.975, ha="left",
                 color=INK, fontsize=13, fontweight="bold")
    fig.text(0.012, 0.93, f"{NUM_SAMPLES:,} uniform parameter samples per circuit, "
             f"mean ± σ over {len(seeds)} seeds; numpy backend",
             ha="left", color=INK_2, fontsize=9)
    # Fixed margins: tight_layout reserves room for the figure legend and leaves a gap.
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.075, top=0.855, hspace=0.5)
    fig.savefig(out, facecolor=SURFACE)
    print(f"\nsaved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (42, 43, ...)")
    parser.add_argument("--out", type=Path, default=HERE / "sim2019_metrics.png")
    args = parser.parse_args()
    seeds = [42 + k for k in range(args.seeds)]
    values, legacy = compute(seeds), legacy_values()
    print_table(values, legacy)
    plot(values, legacy, seeds, args.out)


if __name__ == "__main__":
    main()
