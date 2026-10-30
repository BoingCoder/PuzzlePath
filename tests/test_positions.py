import chess
import numpy as np
import pytest

from puzzlepath.data import parse_row
from puzzlepath.features import position_features, vectorize


@pytest.fixture
def puzzle_row():
    # Published example from database.lichess.org/#puzzles (CC0).
    return {
        "PuzzleId": "00sHx",
        "FEN": "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17",
        "Moves": "e8d7 a2e6 d7d8 f7f8",
        "Rating": "1760",
        "GameUrl": "https://lichess.org/yyznGmXs/black#34",
    }


def test_setup_move_is_not_part_of_the_solution(puzzle_row):
    puzzle = parse_row(puzzle_row)
    board = puzzle.board()
    assert board.turn == chess.WHITE
    assert board.king(chess.BLACK) == chess.D7
    assert puzzle.solution_san() == "18. Be6+ Kd8 19. Qf8#"


def test_features_cannot_see_rating_or_answer_metadata(puzzle_row):
    original = parse_row(puzzle_row)
    changed = parse_row({**puzzle_row, "Rating": "999", "Themes": "mateIn1 easy"})
    assert position_features(original.board()) == position_features(changed.board())


def test_color_mirroring_preserves_solver_relative_features(puzzle_row):
    board = parse_row(puzzle_row).board()
    assert position_features(board) == position_features(board.mirror())


def test_feature_extraction_does_not_mutate_board(puzzle_row):
    board = parse_row(puzzle_row).board()
    before = board.fen()
    vector, names = vectorize([board])
    assert board.fen() == before
    assert vector.shape == (1, len(names))
    assert np.isfinite(vector).all()


def test_illegal_setup_move_is_rejected(puzzle_row):
    with pytest.raises(ValueError):
        parse_row({**puzzle_row, "Moves": "e8e4 a2e6"})


def test_non_lichess_game_identity_is_rejected(puzzle_row):
    with pytest.raises(ValueError, match="game identity"):
        parse_row({**puzzle_row, "GameUrl": "https://example.com/yyznGmXs"})
