"""Generate publication-friendly figures and a readable benchmark report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#24372e"
GREEN = "#2c7054"
PAPER = "#faf8f1"
MUTED = "#72796d"


def write_report(summary: dict, actual: np.ndarray, predicted: np.ndarray, output: Path) -> None:
    selected = summary["selected_model"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    models = list(summary["test"])
    errors = [summary["test"][name]["mae"] for name in models]
    colors = [GREEN if name == selected else "#bcc5b5" for name in models]
    bars = axes[0].barh(models, errors, color=colors)
    axes[0].bar_label(bars, fmt="%.0f", padding=5)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(errors) * 1.22)
    axes[0].set_xlabel("Test MAE · puzzle-rating points (lower is better)")
    axes[0].set_title("01 / Does learning help?", loc="left", fontweight="bold")

    density = axes[1].hexbin(actual, predicted, gridsize=32, mincnt=1, cmap="Greens")
    low = min(actual.min(), predicted.min()) - 50
    high = max(actual.max(), predicted.max()) + 50
    axes[1].plot([low, high], [low, high], color=MUTED, linestyle="--", linewidth=1)
    axes[1].set(
        xlim=(low, high),
        ylim=(low, high),
        xlabel="Observed Lichess puzzle rating",
        ylabel="Position-only estimate",
    )
    axes[1].set_title("02 / Where estimates regress to the mean", loc="left", fontweight="bold")
    fig.colorbar(density, ax=axes[1], label="Puzzles")

    slices = summary["rating_slices"]
    centers = np.arange(len(slices))
    axes[2].bar(centers, [row["mae"] for row in slices], color=GREEN, width=0.65)
    axes[2].set_xticks(centers, [row["band"] for row in slices], rotation=30, ha="right")
    axes[2].set_ylabel("MAE · puzzle-rating points")
    axes[2].set_xlabel("Observed rating band")
    axes[2].set_title("03 / Average error hides difficult cases", loc="left", fontweight="bold")
    for i, row in enumerate(slices):
        axes[2].text(i, row["mae"] + 8, f"n={row['n']}", ha="center", fontsize=8)
    fig.suptitle(
        "PUZZLEPATH  /  POSITION-ONLY DIFFICULTY BENCHMARK",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"benchmark.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    important = summary["validation_permutation_importance"][:12]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.barh(
        [row["feature"] for row in reversed(important)],
        [row["mae_increase"] for row in reversed(important)],
        color=GREEN,
        xerr=[row["std"] for row in reversed(important)],
    )
    ax.set_xlabel("Validation MAE increase after shuffling one feature")
    ax.set_title("Which board features matter to this model?", loc="left", fontweight="bold")
    fig.savefig(output / "feature-importance.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    selected_metrics = summary["test"][selected]
    ci = summary["selected_test_mae_bootstrap_95"]
    interval = summary["interval"]
    lines = [
        "# Benchmark report",
        "",
        f"Run timestamp (UTC): {summary['created_utc']}",
        "",
        f"{summary['sample_rows']:,} public puzzles; "
        f"{summary['feature_count']} position-only features.",
        f"Selected **{selected}** using validation MAE, before test evaluation.",
        "",
        "| Model | Validation MAE | Test MAE | Test RMSE | Test R² | Within 200 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["test"].items():
        lines.append(
            f"| {name} | {summary['validation'][name]['mae']:.1f} | {result['mae']:.1f} "
            f"| {result['rmse']:.1f} | {result['r2']:.3f} | {result['within_200']:.1%} |"
        )
    lines += [
        "",
        f"Selected-model test MAE: **{selected_metrics['mae']:.1f}** rating points.",
        f"95% component-bootstrap interval for MAE: **{ci[0]:.1f}–{ci[1]:.1f}**.",
        "",
        "![Benchmark](benchmark.png)",
        "",
        "## Data boundaries",
        "",
        "| Partition | Puzzles | Connected components |",
        "| --- | ---: | ---: |",
    ]
    for name, counts in summary["partitions"].items():
        lines.append(f"| {name} | {counts['rows']:,} | {counts['components']:,} |")
    lines += [
        "",
        "Source games and identical solver positions do not cross partitions.",
        "The connected-component split includes transitive links. The fixed sample is a",
        "reservoir over an eligible export prefix; see `data/puzzles.manifest.json`.",
        "",
        "## Uncertainty",
        "",
        f"Nominal component coverage: {interval['nominal_coverage']:.0%}. Calibration uses",
        f"{interval['calibration_components']:,} component maximum absolute residuals.",
        f"The resulting radius is **±{interval['radius']:.0f}** rating points.",
        "",
        f"Observed test row coverage: **{interval['test_row_coverage']:.1%}**;",
        f"test component coverage: **{interval['test_component_coverage']:.1%}**.",
        "",
        "This is marginal coverage under exchangeability assumptions, not a guarantee",
        "for an individual position, a rating band, a different website, or a student.",
        "Wide intervals are an important result: board features leave much unexplained.",
        "",
        "## Error by observed rating",
        "",
        "| Rating band | Puzzles | MAE | Interval coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["rating_slices"]:
        lines.append(
            f"| {row['band']} | {row['n']} | {row['mae']:.1f} | {row['interval_coverage']:.1%} |"
        )
    lines += [
        "",
        "## Feature inspection",
        "",
        "![Feature importance](feature-importance.png)",
        "",
        "Permutation importance uses validation data (three repeats, at most 1,000 rows).",
        "The validation partition also selects the model. This plot is descriptive;",
        "correlated features share importance, and importance does not establish causation.",
        "",
        "## Limitations",
        "",
        *[f"- {limitation}" for limitation in summary["limitations"]],
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv sync --frozen --extra dev",
        "uv run puzzlepath train --data data/puzzles.csv --output outputs/reproduction",
        "uv run puzzlepath audit --data data/puzzles.csv --run outputs/reproduction",
        "```",
        "",
        f"Data SHA-256: `{summary['data_sha256']}`.",
        f"Model SHA-256: `{summary['model_sha256']}`.",
        "",
        "Timestamps and serialized model bytes may differ on a repeated run; compare",
        "splits, selected configuration, predictions, and metrics using the locked environment.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
