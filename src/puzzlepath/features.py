"""Engine-free features from the solver's position, without answer metadata."""

from __future__ import annotations

import chess
import numpy as np

FEATURE_VERSION = 2
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
MATERIAL_FEATURES = tuple(
    f"{side}_{chess.piece_name(piece)}s"
    for side in ("us", "them")
    for piece in range(chess.PAWN, chess.KING)
) + ("material_balance", "piece_count")


def _side_features(board: chess.Board, color: chess.Color, prefix: str) -> dict[str, float]:
    occupied = list(chess.scan_forward(board.occupied_co[color]))
    enemy = not color
    result = {
        f"{prefix}_{chess.piece_name(piece)}s": float(len(board.pieces(piece, color)))
        for piece in range(chess.PAWN, chess.KING)
    }
    non_king = [s for s in occupied if board.piece_type_at(s) != chess.KING]
    attacked = [s for s in non_king if board.is_attacked_by(enemy, s)]
    undefended = [s for s in attacked if not board.is_attacked_by(color, s)]
    king = board.king(color)
    ring = list(chess.scan_forward(chess.BB_KING_ATTACKS[king])) if king is not None else []
    pawns = list(board.pieces(chess.PAWN, color))
    files = [chess.square_file(square) for square in pawns]
    result.update(
        {
            f"{prefix}_material": float(
                sum(PIECE_VALUES.get(board.piece_type_at(s), 0) for s in non_king)
            ),
            f"{prefix}_attacked_pieces": float(len(attacked)),
            f"{prefix}_undefended_attacked": float(len(undefended)),
            f"{prefix}_undefended_value": float(
                sum(PIECE_VALUES[board.piece_type_at(s)] for s in undefended)
            ),
            f"{prefix}_pinned_pieces": float(sum(board.is_pinned(color, s) for s in non_king)),
            f"{prefix}_king_ring_pressure": float(
                sum(board.is_attacked_by(enemy, s) for s in ring)
            ),
            f"{prefix}_king_flights": float(
                sum(not board.piece_at(s) and not board.is_attacked_by(enemy, s) for s in ring)
            ),
            f"{prefix}_doubled_pawns": float(len(files) - len(set(files))),
            f"{prefix}_isolated_pawns": float(
                sum(f - 1 not in files and f + 1 not in files for f in files)
            ),
            f"{prefix}_advanced_pawns": float(
                sum(
                    (chess.square_rank(s) if color else 7 - chess.square_rank(s)) >= 5
                    for s in pawns
                )
            ),
            f"{prefix}_center_attacks": float(
                sum(
                    board.is_attacked_by(color, s) for s in (chess.D4, chess.E4, chess.D5, chess.E5)
                )
            ),
            f"{prefix}_attacked_squares": float(
                sum(board.is_attacked_by(color, s) for s in chess.SQUARES)
            ),
        }
    )
    return result


def position_features(board: chess.Board) -> dict[str, float]:
    """Accept a board only; labels, themes, IDs, and solution length cannot enter."""
    if not board.is_valid() or board.is_game_over(claim_draw=False):
        raise ValueError("expected a valid, non-terminal standard chess position")
    result = _side_features(board, board.turn, "us")
    result.update(_side_features(board, not board.turn, "them"))
    moves = list(board.legal_moves)
    captures = [move for move in moves if board.is_capture(move)]
    checking_moves = [move for move in moves if board.gives_check(move)]
    reply_counts = []
    safe_checks = 0
    for move in checking_moves:
        board.push(move)
        try:
            reply_counts.append(board.legal_moves.count())
            safe_checks += not board.is_attacked_by(board.turn, move.to_square)
        finally:
            board.pop()
    result.update(
        {
            "material_balance": result["us_material"] - result["them_material"],
            "piece_count": float(chess.popcount(board.occupied)),
            "in_check": float(board.is_check()),
            "legal_moves": float(len(moves)),
            "legal_captures": float(len(captures)),
            "legal_checks": float(len(checking_moves)),
            "mate_in_one_moves": float(sum(count == 0 for count in reply_counts)),
            "checks_with_one_reply": float(sum(count == 1 for count in reply_counts)),
            "minimum_check_replies": float(min(reply_counts, default=0)),
            "mean_check_replies": float(np.mean(reply_counts)) if reply_counts else 0.0,
            "safe_checking_destinations": float(safe_checks),
            "legal_promotions": float(sum(move.promotion is not None for move in moves)),
            "capture_value_max": float(
                max(
                    (PIECE_VALUES.get(board.piece_type_at(m.to_square), 1) for m in captures),
                    default=0,
                )
            ),
            "castling_rights_us": float(board.has_castling_rights(board.turn)),
            "castling_rights_them": float(board.has_castling_rights(not board.turn)),
            "pawn_fraction": float(chess.popcount(board.pawns) / chess.popcount(board.occupied)),
        }
    )
    return result


def vectorize(boards: list[chess.Board]) -> tuple[np.ndarray, list[str]]:
    rows = [position_features(board) for board in boards]
    if not rows:
        raise ValueError("at least one position is required")
    names = sorted(rows[0])
    return np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64), names
