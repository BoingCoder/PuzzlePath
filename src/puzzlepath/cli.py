"""Command-line entry points. All default outputs remain local."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import chess

from puzzlepath.data import fetch_sample, load_puzzles, sha256
from puzzlepath.evaluation import evaluate_external
from puzzlepath.model import load_model, predict_position, train
from puzzlepath.practice import export_practice
from puzzlepath.splits import split_puzzles


def audit_run(data: Path, run: Path) -> dict:
    metrics = json.loads((run / "metrics.json").read_text())
    if sha256(data) != metrics["data_sha256"]:
        raise ValueError("dataset hash differs from the benchmark")
    if sha256(run / "model.joblib") != metrics["model_sha256"]:
        raise ValueError("model hash differs from the benchmark")
    manifest_path = data.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["csv_sha256"] != metrics["data_sha256"]:
            raise ValueError("dataset manifest hash differs from the benchmark")
    puzzles = load_puzzles(data)
    partitions, groups = split_puzzles(puzzles, metrics["seed"])
    with (run / "predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    expected = {
        p.id: (p.game_id, group, partition, p.rating)
        for p, group, partition in zip(puzzles, groups, partitions, strict=True)
    }
    if len(predictions) != len(expected) or {p["puzzle_id"] for p in predictions} != set(expected):
        raise ValueError("prediction rows do not exactly cover the dataset")
    for row in predictions:
        actual = (row["game_id"], row["component"], row["partition"], int(row["rating"]))
        if actual != expected[row["puzzle_id"]]:
            raise ValueError(f"split/label mismatch for {row['puzzle_id']}")
    chosen = min(metrics["validation"], key=lambda n: (metrics["validation"][n]["mae"], n))
    if chosen != metrics["selected_model"]:
        raise ValueError("selected model is inconsistent with validation MAE")
    return {
        "status": "passed",
        "rows": len(puzzles),
        "data_sha256": metrics["data_sha256"],
        "model_sha256": metrics["model_sha256"],
        "partitions": metrics["partitions"],
        "checks": [
            "data and model hashes",
            "source manifest",
            "exact prediction coverage",
            "game/position component assignments",
            "labels",
            "validation selection",
        ],
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Position-only chess puzzle difficulty and practice sets"
    )
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download a bounded sample of the public CC0 export")
    fetch.add_argument("--output", type=Path, default=Path("data/puzzles.csv"))
    fetch.add_argument("--limit", type=int, default=20000)
    fetch.add_argument("--scan", type=int, default=150000)
    fetch.add_argument("--seed", type=int, default=42)
    fetch.add_argument("--min-plays", type=int, default=1000)
    fetch.add_argument("--max-deviation", type=int, default=100)
    fetch.add_argument(
        "--skip", type=int, default=0, help="Skip this many source rows before sampling"
    )

    fit = commands.add_parser(
        "train", help="Fit, validate, calibrate, and evaluate with disjoint groups"
    )
    fit.add_argument("--data", type=Path, default=Path("data/puzzles.csv"))
    fit.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--alpha", type=float, default=0.1, help="Interval miscoverage target")

    predict = commands.add_parser("predict", help="Estimate difficulty of a solver-ready FEN")
    predict.add_argument("--fen", required=True, help="Position the solver sees, with side to move")
    predict.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/benchmark/model.joblib"),
        help="Trusted local model only; joblib uses pickle",
    )

    practice = commands.add_parser(
        "practice", help="Create a printable PDF and separate PGN answers"
    )
    practice.add_argument("--data", type=Path, default=Path("data/puzzles.csv"))
    practice.add_argument("--output", type=Path, default=Path("outputs/practice"))
    practice.add_argument(
        "--target", type=int, default=1400, help="Target puzzle rating, not player rating"
    )
    practice.add_argument("--count", type=int, default=6)
    practice.add_argument("--window", type=int, default=350)
    practice.add_argument("--theme", help="Exact Lichess theme, e.g. fork or pin")
    practice.add_argument("--rank-by", choices=("observed", "predicted"), default="observed")
    practice.add_argument("--model", type=Path, default=Path("artifacts/benchmark/model.joblib"))

    audit = commands.add_parser(
        "audit", help="Check snapshot hashes, grouping, and benchmark identity"
    )
    audit.add_argument("--data", type=Path, default=Path("data/puzzles.csv"))
    audit.add_argument("--run", type=Path, default=Path("artifacts/benchmark"))

    evaluate = commands.add_parser("evaluate", help="Evaluate a frozen model on a separate CSV")
    evaluate.add_argument("--data", type=Path, default=Path("data/confirmation.csv"))
    evaluate.add_argument("--training-data", type=Path, default=Path("data/puzzles.csv"))
    evaluate.add_argument("--run", type=Path, default=Path("artifacts/benchmark"))
    evaluate.add_argument("--output", type=Path, default=Path("outputs/confirmation"))
    return root


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    try:
        match args.command:
            case "fetch":
                result = fetch_sample(
                    args.output,
                    limit=args.limit,
                    scan=args.scan,
                    seed=args.seed,
                    min_plays=args.min_plays,
                    max_deviation=args.max_deviation,
                    skip=args.skip,
                )
            case "train":
                summary = train(args.data, args.output, args.seed, args.alpha)
                result = {
                    "selected_model": summary["selected_model"],
                    "test": summary["test"][summary["selected_model"]],
                    "report": str(args.output / "report.md"),
                }
            case "predict":
                result = predict_position(chess.Board(args.fen), load_model(args.model))
            case "practice":
                session = export_practice(
                    load_puzzles(args.data),
                    args.output,
                    args.target,
                    args.count,
                    args.window,
                    args.theme,
                    args.rank_by,
                    args.model,
                )
                result = {
                    "puzzles": len(session["puzzles"]),
                    "rank_by": args.rank_by,
                    "worksheet": str(args.output / "practice.pdf"),
                    "answers": str(args.output / "answers.pgn"),
                }
            case "audit":
                result = audit_run(args.data, args.run)
            case "evaluate":
                result = evaluate_external(args.data, args.training_data, args.run, args.output)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"puzzlepath: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
