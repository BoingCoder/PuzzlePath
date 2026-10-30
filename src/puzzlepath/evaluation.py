"""Evaluate a frozen model on another source window without fitting anything."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from puzzlepath.data import load_puzzles, sha256
from puzzlepath.features import vectorize
from puzzlepath.model import component_bootstrap_mae, load_model, regression_metrics
from puzzlepath.splits import split_puzzles


def evaluate_external(data: Path, training_data: Path, run: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    previous = json.loads((run / "metrics.json").read_text())
    if sha256(training_data) != previous["data_sha256"]:
        raise ValueError("training snapshot does not match the frozen benchmark")
    if sha256(run / "model.joblib") != previous["model_sha256"]:
        raise ValueError("model hash does not match the frozen benchmark")
    model = load_model(run / "model.joblib")
    original = load_puzzles(training_data)
    outside = load_puzzles(data)
    original_ids = {p.id for p in original}
    unique_outside = [p for p in outside if p.id not in original_ids]
    if not unique_outside:
        raise ValueError("external sample has no new puzzle identities")
    combined = original + unique_outside
    _, components = split_puzzles(combined, previous["seed"])
    known_components = set(components[: len(original)])
    retained = [
        (p, group)
        for p, group in zip(unique_outside, components[len(original) :], strict=True)
        if group not in known_components
    ]
    if len(retained) < 20:
        raise ValueError("external evaluation needs at least 20 independent, non-overlapping rows")
    puzzles = [p for p, _ in retained]
    groups = [group for _, group in retained]
    features, names = vectorize([p.board() for p in puzzles])
    if names != model["feature_names"]:
        raise ValueError("model feature schema mismatch")
    with threadpool_limits(limits=2):
        predicted = model["estimator"].predict(features[:, model["columns"]])
    actual = np.asarray([p.rating for p in puzzles], dtype=float)
    original_splits, _ = split_puzzles(original, previous["seed"])
    median = float(
        np.median(
            [
                p.rating
                for p, split in zip(original, original_splits, strict=True)
                if split == "train"
            ]
        )
    )
    residuals = np.abs(actual - predicted)
    group_covered: dict[str, bool] = {}
    for group, covered in zip(groups, residuals <= model["radius"], strict=True):
        group_covered[group] = group_covered.get(group, True) and bool(covered)
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluation": "frozen model; no fitting, selection, or recalibration on this sample",
        "scope": "another bounded window of the same Lichess export; NOT a temporal holdout",
        "data_sha256": sha256(data),
        "training_data_sha256": previous["data_sha256"],
        "frozen_model_sha256": previous["model_sha256"],
        "input_rows": len(outside),
        "retained_rows": len(puzzles),
        "excluded_overlapping_rows": len(outside) - len(puzzles),
        "components": len(set(groups)),
        "selected_model": model["selected_model"],
        "model": regression_metrics(actual, predicted),
        "median_baseline": regression_metrics(actual, np.full(len(puzzles), median)),
        "model_mae_bootstrap_95": component_bootstrap_mae(residuals, groups, previous["seed"]),
        "interval_radius": model["radius"],
        "row_interval_coverage": float(np.mean(residuals <= model["radius"])),
        "component_interval_coverage": float(np.mean(list(group_covered.values()))),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["puzzle_id", "game_id", "component", "rating", "prediction"])
        for puzzle, group, prediction in zip(puzzles, groups, predicted, strict=True):
            writer.writerow([puzzle.id, puzzle.game_id, group, puzzle.rating, f"{prediction:.6f}"])
    ci = summary["model_mae_bootstrap_95"]
    (output / "report.md").write_text(
        "# Confirmation on a separate source window\n\n"
        f"Evaluation timestamp (UTC): {summary['created_utc']}\n\n"
        "The model, hyperparameter choice, feature definition, and calibration radius were frozen "
        "before this evaluation. Nothing is fitted or recalibrated here.\n\n"
        f"{summary['retained_rows']:,} of {summary['input_rows']:,} puzzles remain after excluding "
        "every component connected to the development sample by source game or solver position. "
        "This is another export window, **not a temporal or cross-platform holdout**.\n\n"
        "| Model | MAE | RMSE | R² |\n| --- | ---: | ---: | ---: |\n"
        f"| Training-median baseline | {summary['median_baseline']['mae']:.1f} "
        f"| {summary['median_baseline']['rmse']:.1f} | {summary['median_baseline']['r2']:.3f} |\n"
        f"| Frozen {model['selected_model']} | {summary['model']['mae']:.1f} "
        f"| {summary['model']['rmse']:.1f} | {summary['model']['r2']:.3f} |\n\n"
        f"Model MAE component-bootstrap 95% interval: {ci[0]:.1f}–{ci[1]:.1f}.\n\n"
        f"Observed interval coverage: {summary['row_interval_coverage']:.1%} of rows; "
        f"{summary['component_interval_coverage']:.1%} of complete components. "
        f"The unchanged radius is ±{model['radius']:.0f} rating points.\n\n"
        "This checks transfer to another part of the same source, not learner outcomes or "
        "difficulty on unrelated exercises. The sample still filters on play count and rating "
        "deviation, and similar positions can remain even after exact duplicates are excluded.\n\n"
        f"Data SHA-256: `{summary['data_sha256']}`.\n\n"
        f"Frozen model SHA-256: `{summary['frozen_model_sha256']}`.\n"
    )
    return summary
