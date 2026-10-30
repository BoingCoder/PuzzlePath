"""Keep source games and identical solver positions in one partition."""

from __future__ import annotations

import hashlib

import numpy as np

from puzzlepath.data import Puzzle

SPLIT_NAMES = ("train", "validation", "calibration", "test")


def split_puzzles(puzzles: list[Puzzle], seed: int = 42) -> tuple[np.ndarray, list[str]]:
    if len({p.id for p in puzzles}) != len(puzzles):
        raise ValueError("puzzle IDs must be unique")
    parent = {p.id: p.id for p in puzzles}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        first, second = sorted((find(a), find(b)))
        parent[second] = first

    seen: dict[tuple[str, str], str] = {}
    for puzzle in puzzles:
        for key in (("game", puzzle.game_id), ("position", puzzle.position_key())):
            if key in seen:
                union(puzzle.id, seen[key])
            else:
                seen[key] = puzzle.id
    groups = [find(p.id) for p in puzzles]
    assignment = {}
    for group in sorted(set(groups)):
        digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        assignment[group] = (
            "train"
            if value < 0.60
            else "validation"
            if value < 0.75
            else "calibration"
            if value < 0.85
            else "test"
        )
    partitions = np.asarray([assignment[group] for group in groups])
    return partitions, groups
