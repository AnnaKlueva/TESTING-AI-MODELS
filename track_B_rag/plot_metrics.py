"""Column diagrams for RAG evaluation metrics with threshold-based coloring."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt

GREEN = "#2ca02c"
RED = "#d62728"
GREY = "#888888"

DEFAULT_THRESHOLD = 0.7


def bottlenecks(scenario: Mapping[str, float], threshold: float = DEFAULT_THRESHOLD) -> list[str]:
    """Return metric names that fall below the threshold."""
    return [name for name, value in scenario.items() if value < threshold]


def plot_metric_bars(
    scenario: Mapping[str, float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    title: str | None = None,
    figsize: tuple[float, float] = (8.5, 3.4),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Draw a bar chart for metric values; green if >= threshold, red otherwise."""
    names = list(scenario)
    vals = [scenario[n] for n in names]
    colors = [GREEN if v >= threshold else RED for v in vals]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(names, vals, color=colors)
    ax.axhline(threshold, color=GREY, ls="--", label=f"поріг {threshold}")
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    scenario = {
        "Hit pass-rate": 1.00,
        "MRR": 0.893,
        "Recall@K": 0.839,
        "NDCG@K": 0.812,
        "Precision@K": 0.571,
    }
    THRESH = 0.8

    plot_metric_bars(
        scenario,
        threshold=THRESH,
        title="Retrieval (k=2): Precision@K — червоний, решта — зелені",
    )
    print("Вузькі місця:", bottlenecks(scenario, THRESH))
