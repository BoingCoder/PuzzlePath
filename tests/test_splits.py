from dataclasses import replace

from puzzlepath.data import parse_row
from puzzlepath.splits import split_puzzles


def test_transitive_game_and_position_overlap_stays_together():
    base = parse_row(
        {
            "PuzzleId": "one",
            "FEN": "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17",
            "Moves": "e8d7 a2e6 d7d8 f7f8",
            "Rating": "1760",
            "GameUrl": "https://lichess.org/yyznGmXs/black#34",
        }
    )
    # A and B share a game; B and C share an identical position.
    different = replace(
        base, id="two", fen=base.board().mirror().fen(), moves=("a7e3", "d2e2", "e3f2", "e2e8")
    )
    # Grouping needs only a legal setup; use a separate reachable starting board.
    different = replace(
        different,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        moves=("e2e4", "e7e5"),
    )
    third = replace(different, id="three", game_id="abcdefgh")
    puzzles = [base, different, third]
    splits, groups = split_puzzles(puzzles)
    assert len(set(groups)) == 1
    assert len(set(splits)) == 1
    reversed_splits, _ = split_puzzles(list(reversed(puzzles)))
    assert list(reversed_splits) == list(reversed(splits))
