"""Read public puzzle exports and retain a reproducible, bounded sample."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import chess
import zstandard

SOURCE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
REQUIRED_COLUMNS = {"PuzzleId", "FEN", "Moves", "Rating", "GameUrl"}


@dataclass(frozen=True)
class Puzzle:
    id: str
    fen: str
    moves: tuple[str, ...]
    rating: int
    themes: tuple[str, ...]
    game_id: str
    game_url: str
    plays: int = 0
    rating_deviation: int = 0

    def board(self) -> chess.Board:
        """Return the solver's position, AFTER the opponent's first move."""
        board = chess.Board(self.fen)
        if not board.is_valid():
            raise ValueError(f"{self.id}: invalid board")
        if len(self.moves) < 2:
            raise ValueError(f"{self.id}: missing setup move or solution")
        board.push_uci(self.moves[0])  # push_uci validates legality.
        if board.is_game_over(claim_draw=False):
            raise ValueError(f"{self.id}: setup move leaves no puzzle to solve")
        return board

    def position_key(self) -> str:
        # Move counters do not change the tactical position.
        return " ".join(self.board().fen().split()[:4])

    def solution_san(self) -> str:
        return self.board().variation_san([chess.Move.from_uci(m) for m in self.moves[1:]])


def parse_row(row: dict[str, str]) -> Puzzle:
    game = urlparse(row["GameUrl"])
    game_id = game.path.strip("/").split("/")[0]
    if game.hostname not in {"lichess.org", "www.lichess.org"} or not re.fullmatch(
        r"[A-Za-z0-9]{8}", game_id
    ):
        raise ValueError("missing or invalid source game identity")
    puzzle = Puzzle(
        id=row["PuzzleId"],
        fen=row["FEN"],
        moves=tuple(row["Moves"].split()),
        rating=int(row["Rating"]),
        themes=tuple(row.get("Themes", "").split()),
        game_id=game_id,
        game_url=row["GameUrl"],
        plays=int(row.get("NbPlays") or 0),
        rating_deviation=int(row.get("RatingDeviation") or 0),
    )
    if not re.fullmatch(r"[A-Za-z0-9]+", puzzle.id) or not 0 < puzzle.rating <= 4000:
        raise ValueError("invalid puzzle identity or rating")
    board = puzzle.board()
    # Validate the full answer for worksheet export, never for model features.
    for move in puzzle.moves[1:]:
        board.push_uci(move)
    return puzzle


def load_puzzles(path: Path) -> list[Puzzle]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {', '.join(sorted(REQUIRED_COLUMNS))}")
        puzzles = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            try:
                puzzle = parse_row(row)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
            if puzzle.id in seen:
                raise ValueError(f"{path}:{line}: duplicate puzzle ID {puzzle.id}")
            seen.add(puzzle.id)
            puzzles.append(puzzle)
    return puzzles


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_sample(
    output: Path,
    *,
    limit: int = 20000,
    scan: int = 150000,
    seed: int = 42,
    min_plays: int = 1000,
    max_deviation: int = 100,
) -> dict:
    """Reservoir-sample eligible rows from a bounded prefix of the remote export.

    This deliberately does not claim a uniform sample of the full database.
    The exact retained bytes, HTTP identity, and sampling boundary are recorded.
    """
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise ValueError(f"refusing to replace an existing snapshot: {output}")
    if limit < 1 or scan < limit or min_plays < 0 or max_deviation < 1:
        raise ValueError("require scan >= limit > 0, min_plays >= 0, max_deviation > 0")
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    counts: Counter = Counter()
    request = Request(SOURCE_URL, headers={"User-Agent": "PuzzlePath/0.1 research sampler"})
    with urlopen(request, timeout=60) as response:
        identity = {key: response.headers.get(key) for key in ("ETag", "Last-Modified")}
        with zstandard.ZstdDecompressor().stream_reader(response) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            fields = reader.fieldnames
            if not REQUIRED_COLUMNS.issubset(fields or []):
                raise ValueError("upstream CSV schema changed")
            for row in reader:
                counts["scanned"] += 1
                try:
                    if int(row.get("NbPlays") or 0) < min_plays:
                        counts["low_plays"] += 1
                    elif int(row.get("RatingDeviation") or 9999) > max_deviation:
                        counts["high_deviation"] += 1
                    else:
                        # Full validation is bounded to the retained sample below.
                        counts["eligible"] += 1
                        if len(selected) < limit:
                            selected.append(row)
                        else:
                            index = rng.randrange(counts["eligible"])
                            if index < limit:
                                selected[index] = row
                except (ValueError, TypeError):
                    counts["malformed"] += 1
                if counts["scanned"] >= scan:
                    break
    if len(selected) < limit:
        raise ValueError(f"only {len(selected)} eligible rows; increase --scan or lower --limit")
    selected.sort(key=lambda row: row["PuzzleId"])
    for row in selected:
        parse_row(row)
    if len({row["PuzzleId"] for row in selected}) != len(selected):
        raise ValueError("upstream sample contains duplicate puzzle IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(selected)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "source_url": SOURCE_URL,
        "source_documentation": "https://database.lichess.org/#puzzles",
        "license": "CC0-1.0",
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "http_identity": identity,
        "sampling": (
            "seeded reservoir over eligible rows in the scanned prefix; NOT full-export random"
        ),
        "seed": seed,
        "scan_limit": scan,
        "sample_size": len(selected),
        "min_plays": min_plays,
        "max_rating_deviation": max_deviation,
        "counts": dict(counts),
        "csv_sha256": sha256(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
