"""Fit on train, select on validation, calibrate separately, then evaluate test."""

from __future__ import annotations

import csv
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path

import chess
import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from puzzlepath.data import load_puzzles, sha256
from puzzlepath.features import FEATURE_VERSION, MATERIAL_FEATURES, vectorize
from puzzlepath.splits import SPLIT_NAMES, split_puzzles


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.abs(actual - predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error": float(np.median(error)),
        "r2": float(r2_score(actual, predicted)),
        "within_200": float(np.mean(error <= 200)),
        "n": len(actual),
    }


def conformal_radius(
    residuals: np.ndarray, groups: list[str], alpha: float = 0.1
) -> tuple[float, int]:
    """Finite-sample quantile over per-component MAX absolute residuals."""
    if not 0 < alpha < 1 or len(residuals) != len(groups) or not len(groups):
        raise ValueError("require residuals, matching groups, and 0 < alpha < 1")
    maxima: dict[str, float] = {}
    for residual, group in zip(residuals, groups, strict=True):
        if not np.isfinite(residual) or residual < 0:
            raise ValueError("residuals must be finite and nonnegative")
        maxima[group] = max(maxima.get(group, 0.0), float(residual))
    scores = np.sort(list(maxima.values()))
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError("too few calibration components for a finite interval at this coverage")
    return float(scores[rank - 1]), len(scores)


def component_bootstrap_mae(
    residuals: np.ndarray, groups: list[str], seed: int, repeats: int = 1000
) -> list[float]:
    unique = sorted(set(groups))
    index = {group: i for i, group in enumerate(unique)}
    group_index = np.array([index[group] for group in groups])
    totals = np.bincount(group_index, weights=residuals)
    counts = np.bincount(group_index)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(unique), len(unique))
        estimates.append(totals[chosen].sum() / counts[chosen].sum())
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _fit_benchmark(data: Path, output: Path, seed: int, alpha: float) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}; choose a new run directory")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    print("Reading and validating puzzles...", flush=True)
    puzzles = load_puzzles(data)
    labels = np.asarray([p.rating for p in puzzles], dtype=float)
    print(f"Extracting position-only features for {len(puzzles):,} puzzles...", flush=True)
    features, names = vectorize([p.board() for p in puzzles])
    partitions, groups = split_puzzles(puzzles, seed)
    masks = {name: partitions == name for name in SPLIT_NAMES}
    if any(np.sum(mask) < 20 for mask in masks.values()):
        raise ValueError(
            "each partition needs at least 20 rows; use a larger or more diverse sample"
        )
    material_indices = [names.index(name) for name in MATERIAL_FEATURES]
    candidates = {
        "median": (DummyRegressor(strategy="median"), list(range(len(names)))),
        "material_ridge": (make_pipeline(StandardScaler(), Ridge(alpha=100)), material_indices),
        "position_ridge": (
            make_pipeline(StandardScaler(), Ridge(alpha=100)),
            list(range(len(names))),
        ),
    }
    for leaves, penalty in ((15, 2.0), (31, 10.0)):
        candidates[f"boosted_{leaves}"] = (
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.07,
                max_iter=180,
                max_leaf_nodes=leaves,
                min_samples_leaf=30,
                l2_regularization=penalty,
                early_stopping=False,
                random_state=seed,
            ),
            list(range(len(names))),
        )
    validation = {}
    for name, (estimator, columns) in candidates.items():
        print(f"Fitting {name}...", flush=True)
        estimator.fit(features[masks["train"]][:, columns], labels[masks["train"]])
        predicted = estimator.predict(features[masks["validation"]][:, columns])
        validation[name] = regression_metrics(labels[masks["validation"]], predicted)
    selected = min(validation, key=lambda name: (validation[name]["mae"], name))
    estimator, columns = candidates[selected]
    print(
        f"Validation selected {selected}; calibrating and evaluating held-out test...", flush=True
    )
    predictions = estimator.predict(features[:, columns])
    radius, calibration_groups = conformal_radius(
        np.abs(labels[masks["calibration"]] - predictions[masks["calibration"]]),
        [group for group, mask in zip(groups, masks["calibration"], strict=True) if mask],
        alpha,
    )
    test_results = {}
    for name, (candidate, candidate_columns) in candidates.items():
        test_predictions = candidate.predict(features[masks["test"]][:, candidate_columns])
        test_results[name] = regression_metrics(labels[masks["test"]], test_predictions)
    test_actual = labels[masks["test"]]
    test_predicted = predictions[masks["test"]]
    test_errors = np.abs(test_actual - test_predicted)
    test_groups = [group for group, keep in zip(groups, masks["test"], strict=True) if keep]
    intervals = test_errors <= radius
    group_covered: dict[str, bool] = {}
    for group, covered in zip(test_groups, intervals, strict=True):
        group_covered[group] = group_covered.get(group, True) and bool(covered)
    importance = permutation_importance(
        estimator,
        features[masks["validation"]][:, columns],
        labels[masks["validation"]],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=seed,
        max_samples=min(1000, int(np.sum(masks["validation"]))),
        n_jobs=1,
    )
    importances = sorted(
        (
            {"feature": names[column], "mae_increase": float(mean), "std": float(std)}
            for column, mean, std in zip(
                columns, importance.importances_mean, importance.importances_std, strict=True
            )
        ),
        key=lambda row: row["mae_increase"],
        reverse=True,
    )
    rating_slices = []
    for low, high in ((0, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 4001)):
        mask = (test_actual >= low) & (test_actual < high)
        if np.sum(mask) >= 2:
            rating_slices.append(
                {
                    "band": f"{low}–{high - 1}",
                    **regression_metrics(test_actual[mask], test_predicted[mask]),
                    "interval_coverage": float(np.mean(intervals[mask])),
                }
            )
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "data_sha256": sha256(data),
        "sample_rows": len(puzzles),
        "feature_version": FEATURE_VERSION,
        "feature_count": len(names),
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "chess": chess.__version__,
        },
        "partitions": {
            name: {
                "rows": int(np.sum(mask)),
                "components": len(
                    {group for group, keep in zip(groups, mask, strict=True) if keep}
                ),
            }
            for name, mask in masks.items()
        },
        "selected_model": selected,
        "selection_rule": "lowest validation MAE; all candidates fit on train only",
        "validation": validation,
        "test": test_results,
        "selected_test_mae_bootstrap_95": component_bootstrap_mae(test_errors, test_groups, seed),
        "interval": {
            "nominal_coverage": 1 - alpha,
            "radius": radius,
            "calibration_components": calibration_groups,
            "calibration_unit": "maximum absolute residual per connected component",
            "test_row_coverage": float(np.mean(intervals)),
            "test_component_coverage": float(np.mean(list(group_covered.values()))),
            "mean_width": 2 * radius,
            "assumption": "exchangeable components; not established for new sources or positions",
        },
        "rating_slices": rating_slices,
        "validation_permutation_importance": importances,
        "limitations": [
            "Bounded-prefix sample, filtered on play count and rating deviation.",
            "No temporal holdout or demonstrated generalization to an unrelated puzzle source.",
            "Static features do not search the solution or measure human thinking time.",
            "Intervals are marginal across components, not conditional on rating or learner.",
            "No measured teaching benefit or individual solve-probability calibration.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "estimator": estimator,
        "feature_names": names,
        "columns": columns,
        "feature_version": FEATURE_VERSION,
        "selected_model": selected,
        "radius": radius,
        "nominal_coverage": 1 - alpha,
        "data_sha256": summary["data_sha256"],
        "sklearn_version": sklearn.__version__,
        "piece_count_range": [
            float(np.min(features[masks["train"], names.index("piece_count")])),
            float(np.max(features[masks["train"], names.index("piece_count")])),
        ],
    }
    joblib.dump(bundle, output / "model.joblib", compress=3)
    summary["model_sha256"] = sha256(output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["puzzle_id", "game_id", "component", "partition", "rating", "prediction"])
        for puzzle, group, partition, predicted in zip(
            puzzles, groups, partitions, predictions, strict=True
        ):
            writer.writerow(
                [puzzle.id, puzzle.game_id, group, partition, puzzle.rating, f"{predicted:.6f}"]
            )
    from puzzlepath.report import write_report

    write_report(summary, test_actual, test_predicted, output)
    return summary


def train(data: Path, output: Path, seed: int = 42, alpha: float = 0.1) -> dict:
    # Bound BLAS/OpenMP to keep this CPU benchmark practical on shared machines.
    with threadpool_limits(limits=2):
        return _fit_benchmark(data, output, seed, alpha)


def load_model(path: Path) -> dict:
    # joblib is pickle-based. Only load an artifact from a source you trust.
    model = joblib.load(path)
    if model["feature_version"] != FEATURE_VERSION:
        raise ValueError("model feature version mismatch; retrain with this checkout")
    if model["sklearn_version"] != sklearn.__version__:
        raise ValueError("model scikit-learn version mismatch; run uv sync --frozen or retrain")
    return model


def predict_position(board: chess.Board, model: dict) -> dict:
    features, names = vectorize([board])
    if names != model["feature_names"]:
        raise ValueError("model feature schema does not match this checkout")
    with threadpool_limits(limits=2):
        estimate = float(model["estimator"].predict(features[:, model["columns"]])[0])
    low, high = model["piece_count_range"]
    warnings = [
        "Coarse puzzle-rating estimate; not a player rating, best move, or solve probability.",
        "Coverage on arbitrary new positions is unverified.",
    ]
    if not low <= chess.popcount(board.occupied) <= high:
        warnings.append("Piece count is outside the training range.")
    return {
        "solver_fen": board.fen(),
        "selected_model": model["selected_model"],
        "estimated_puzzle_rating": round(estimate),
        "interval": [max(0, round(estimate - model["radius"])), round(estimate + model["radius"])],
        "nominal_coverage": model["nominal_coverage"],
        "warnings": warnings,
    }
