"""Choose diverse exercises and export a student sheet plus separate solutions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chess
import chess.pgn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from threadpoolctl import threadpool_limits

from puzzlepath.data import Puzzle
from puzzlepath.features import vectorize
from puzzlepath.model import load_model

TEACHING_THEMES = (
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "deflection",
    "attraction",
    "backRankMate",
    "mate",
    "endgame",
    "promotion",
    "sacrifice",
)


def choose_puzzles(
    puzzles: list[Puzzle],
    target: int,
    count: int = 6,
    window: int = 350,
    theme: str | None = None,
    rank_by: str = "observed",
    model: dict | None = None,
) -> list[tuple[Puzzle, float]]:
    if count < 1 or count > 30 or window < 0 or not 0 < target <= 4000:
        raise ValueError("require count 1–30, window >= 0, and target 1–4000")
    if rank_by not in {"observed", "predicted"}:
        raise ValueError("rank_by must be observed or predicted")
    pool = [p for p in puzzles if theme is None or theme in p.themes]
    if not pool:
        raise ValueError("not enough distinct matching puzzles; check --theme")
    if rank_by == "predicted":
        if model is None:
            raise ValueError("prediction-based selection requires a trained model")
        features, names = vectorize([p.board() for p in pool])
        if names != model["feature_names"]:
            raise ValueError("model feature schema mismatch")
        with threadpool_limits(limits=2):
            scores = model["estimator"].predict(features[:, model["columns"]])
    else:
        scores = np.asarray([p.rating for p in pool], dtype=float)
    available = [
        (p, float(score))
        for p, score in zip(pool, scores, strict=True)
        if abs(score - target) <= window
    ]
    selected = []
    games, positions = set(), set()
    used_themes: dict[str, int] = {}
    for desired in np.linspace(target - min(150, window), target + min(150, window), count):
        candidates = [
            (p, score)
            for p, score in available
            if p.game_id not in games and p.position_key() not in positions
        ]
        if not candidates:
            raise ValueError(
                "not enough distinct matching puzzles; widen --window or lower --count"
            )

        def cost(
            item: tuple[Puzzle, float], desired_rating: float = float(desired)
        ) -> tuple[float, str]:
            puzzle, score = item
            main_theme = next((t for t in TEACHING_THEMES if t in puzzle.themes), "other")
            return abs(score - desired_rating) + 40 * used_themes.get(main_theme, 0), puzzle.id

        puzzle, score = min(candidates, key=cost)
        selected.append((puzzle, score))
        games.add(puzzle.game_id)
        positions.add(puzzle.position_key())
        main_theme = next((t for t in TEACHING_THEMES if t in puzzle.themes), "other")
        used_themes[main_theme] = used_themes.get(main_theme, 0) + 1
    return selected


def draw_board(ax: plt.Axes, board: chess.Board) -> None:
    for file in range(8):
        for rank in range(8):
            square = chess.square(file, rank)
            x, y = (file, rank) if board.turn else (7 - file, 7 - rank)
            color = "#f1ead7" if (file + rank) % 2 else "#769782"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="none"))
            piece = board.piece_at(square)
            if piece:
                ax.text(
                    x + 0.5,
                    y + 0.50,
                    piece.unicode_symbol(),
                    fontsize=22,
                    va="center",
                    ha="center",
                    fontfamily="DejaVu Sans",
                    color="#17251d",
                )
    ax.set(xlim=(0, 8), ylim=(0, 8), aspect="equal")
    ax.set_xticks(np.arange(8) + 0.5, list("abcdefgh" if board.turn else "hgfedcba"), fontsize=7)
    ax.set_yticks(np.arange(8) + 0.5, list("12345678" if board.turn else "87654321"), fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def export_practice(
    puzzles: list[Puzzle],
    output: Path,
    target: int,
    count: int = 6,
    window: int = 350,
    theme: str | None = None,
    rank_by: str = "observed",
    model_path: Path | None = None,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    model = load_model(model_path) if rank_by == "predicted" and model_path else None
    selected = choose_puzzles(puzzles, target, count, window, theme, rank_by, model)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#24372e"})
    with PdfPages(output / "practice.pdf") as pdf:
        for page_start in range(0, len(selected), 6):
            page = selected[page_start : page_start + 6]
            fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))
            fig.subplots_adjust(top=0.88, bottom=0.06, hspace=0.36, wspace=0.23)
            fig.text(0.10, 0.956, "PUZZLEPATH", fontsize=24, fontweight="bold")
            fig.text(0.10, 0.929, "CHESS PRACTICE  /  Find a move. Explain your idea.", fontsize=10)
            fig.text(
                0.10, 0.906, "Name ________________________    Date ______________", fontsize=10
            )
            for index, ax in enumerate(axes.flat):
                if index >= len(page):
                    ax.set_visible(False)
                    continue
                puzzle, _ = page[index]
                board = puzzle.board()
                draw_board(ax, board)
                side = "White" if board.turn else "Black"
                ax.set_title(
                    f"{page_start + index + 1:02d}   /   {side} to move",
                    loc="left",
                    fontsize=10,
                    pad=9,
                    fontweight="bold",
                )
                ax.text(
                    0,
                    -0.16,
                    "Move / idea ______________________",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            pdf.savefig(fig)
            if page_start == 0:
                fig.savefig(output / "preview.png", dpi=140)
            plt.close(fig)
        for page_start in range(0, len(selected), 10):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.1, 0.95, "ANSWER KEY", fontsize=24, fontweight="bold")
            fig.text(
                0.1, 0.918, f"Target {target} · ordered by {rank_by} puzzle rating", fontsize=10
            )
            fig.text(
                0.1,
                0.893,
                "Ratings are puzzle ratings, not student ability or solve probabilities.",
                fontsize=9,
            )
            for index, (puzzle, score) in enumerate(selected[page_start : page_start + 10]):
                y = 0.84 - index * 0.074
                line = f"{page_start + index + 1:02d}  /  {puzzle.id}   ·   Lichess {puzzle.rating}"
                if rank_by == "predicted":
                    line += f"   ·   estimate {score:.0f}"
                fig.text(0.1, y, line, fontsize=10, fontweight="bold")
                # Put the complete (possibly long) solution in PGN and session.json.
                solution = puzzle.solution_san()
                if len(solution) > 98:
                    solution = solution[:94] + " ..."
                fig.text(0.1, y - 0.022, solution, fontsize=9)
                fig.text(
                    0.1,
                    y - 0.043,
                    f"https://lichess.org/training/{puzzle.id}",
                    fontsize=8,
                    color="#57695c",
                )
            pdf.savefig(fig)
            plt.close(fig)
    games = []
    for puzzle, _ in selected:
        game = chess.pgn.Game()
        game.setup(puzzle.board())
        game.headers["Event"] = f"PuzzlePath / {puzzle.id}"
        game.headers["Site"] = f"https://lichess.org/training/{puzzle.id}"
        game.headers["PuzzleRating"] = str(puzzle.rating)
        node = game
        for move in puzzle.moves[1:]:
            node = node.add_variation(chess.Move.from_uci(move))
        games.append(str(game))
    (output / "answers.pgn").write_text("\n\n".join(games) + "\n")
    session = {
        "target_puzzle_rating": target,
        "rank_by": rank_by,
        "theme_filter": theme,
        "puzzles": [
            {
                "id": p.id,
                "solver_fen": p.board().fen(),
                "observed_rating": p.rating,
                "selection_score": round(score, 2),
                "themes": p.themes,
                "solution_san": p.solution_san(),
                "url": f"https://lichess.org/training/{p.id}",
            }
            for p, score in selected
        ],
        "worksheet_sha256": hashlib.sha256((output / "practice.pdf").read_bytes()).hexdigest(),
        "note": "No learner data or demonstrated educational outcome; known ratings preferred.",
    }
    (output / "session.json").write_text(json.dumps(session, indent=2) + "\n")
    return session
