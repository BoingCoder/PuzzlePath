import csv
import io
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest

from puzzlepath.cli import audit_run, main
from puzzlepath.data import load_puzzles
from puzzlepath.evaluation import evaluate_external
from puzzlepath.features import vectorize
from puzzlepath.model import load_model, predict_position, train
from puzzlepath.practice import choose_puzzles, export_practice

DATA = Path(__file__).parents[1] / "data" / "puzzles.csv"


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    path = tmp_path_factory.mktemp("snapshot") / "puzzles.csv"
    with DATA.open(newline="") as source, path.open("w", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(next(reader) for _ in range(400))
    return path


@pytest.fixture(scope="module")
def fitted(small_data, tmp_path_factory):
    output = tmp_path_factory.mktemp("run") / "benchmark"
    summary = train(small_data, output)
    return output, summary


def test_training_export_audit_and_single_position_prediction_agree(small_data, fitted):
    output, summary = fitted
    assert audit_run(small_data, output)["status"] == "passed"
    model = load_model(output / "model.joblib")
    puzzle = load_puzzles(small_data)[0]
    result = predict_position(puzzle.board(), model)
    features, _ = vectorize([puzzle.board()])
    direct = model["estimator"].predict(features[:, model["columns"]])[0]
    assert result["estimated_puzzle_rating"] == round(direct)
    assert result["interval"][0] <= result["estimated_puzzle_rating"] <= result["interval"][1]
    expected = min(summary["validation"], key=lambda n: (summary["validation"][n]["mae"], n))
    assert summary["selected_model"] == expected
    assert np.isfinite(summary["test"][expected]["mae"])


def test_audit_detects_snapshot_change(small_data, fitted, tmp_path):
    output, _ = fitted
    changed = tmp_path / "changed.csv"
    changed.write_bytes(small_data.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dataset hash"):
        audit_run(changed, output)


def test_training_refuses_to_replace_an_existing_run(small_data, fitted):
    with pytest.raises(ValueError, match="not empty"):
        train(small_data, fitted[0])


def test_practice_uses_distinct_games_and_positions(small_data):
    puzzles = load_puzzles(small_data)
    chosen = choose_puzzles(puzzles, target=1400)
    assert len({p.game_id for p, _ in chosen}) == 6
    assert len({p.position_key() for p, _ in chosen}) == 6
    assert all(score == p.rating for p, score in chosen)
    assert [p.id for p, _ in chosen] == [p.id for p, _ in choose_puzzles(puzzles, target=1400)]


def test_pdf_and_pgn_start_at_the_solver_position(small_data, tmp_path):
    puzzles = load_puzzles(small_data)
    output = tmp_path / "practice"
    session = export_practice(puzzles, output, target=1400, count=2)
    assert (output / "practice.pdf").read_bytes().startswith(b"%PDF-")
    assert (output / "preview.png").read_bytes().startswith(b"\x89PNG")
    pgn = io.StringIO((output / "answers.pgn").read_text())
    for selected in session["puzzles"]:
        game = chess.pgn.read_game(pgn)
        assert not game.errors
        assert game.board().fen() == selected["solver_fen"]
        assert game.board().variation_san(list(game.mainline_moves())) == selected["solution_san"]
    assert chess.pgn.read_game(pgn) is None


def test_invalid_fen_has_a_clean_cli_error(capsys):
    assert main(["predict", "--fen", "not a FEN"]) == 2
    assert "puzzlepath:" in capsys.readouterr().err


def test_insufficient_practice_pool_is_an_error(small_data):
    with pytest.raises(ValueError, match="not enough distinct"):
        choose_puzzles(load_puzzles(small_data), target=1, window=0)


def test_external_evaluation_excludes_positions_seen_during_development(
    small_data, fitted, tmp_path
):
    with DATA.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [next(reader) for _ in range(460)]
    # A different identity and game still must not sneak an identical board across the boundary.
    overlap = {**rows[0], "PuzzleId": "fixtureOverlap", "GameUrl": "https://lichess.org/zzzzzzzz"}
    external = tmp_path / "external.csv"
    with external.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*rows[400:], overlap])
    model_before = (fitted[0] / "model.joblib").read_bytes()
    summary = evaluate_external(external, small_data, fitted[0], tmp_path / "evaluation")
    assert summary["excluded_overlapping_rows"] >= 1
    assert summary["retained_rows"] <= 60
    assert (fitted[0] / "model.joblib").read_bytes() == model_before


def test_a_development_sample_is_not_an_external_holdout(small_data, fitted, tmp_path):
    with pytest.raises(ValueError, match="no new puzzle identities"):
        evaluate_external(small_data, small_data, fitted[0], tmp_path / "evaluation")


def test_unknown_theme_has_a_clear_error_in_prediction_mode(small_data):
    with pytest.raises(ValueError, match="check --theme"):
        choose_puzzles(
            load_puzzles(small_data), target=1400, theme="not-a-theme", rank_by="predicted"
        )
