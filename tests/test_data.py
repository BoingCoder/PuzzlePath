import csv
import io
import json
from pathlib import Path

import pytest
import zstandard

from puzzlepath.data import fetch_sample, load_puzzles, sha256

DATA = Path(__file__).parents[1] / "data" / "puzzles.csv"


def test_sample_is_offline_reproducible_and_does_not_replace_files(tmp_path, monkeypatch):
    with DATA.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [next(reader) for _ in range(6)]
    for index, row in enumerate(rows):
        row["NbPlays"] = "999" if index == 0 else "2000"
        row["RatingDeviation"] = "101" if index == 1 else "80"
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    compressed = zstandard.ZstdCompressor().compress(text.getvalue().encode())

    class Response(io.BytesIO):
        headers = {"ETag": "test-snapshot", "Last-Modified": "fixed-test-date"}

    monkeypatch.setattr("puzzlepath.data.urlopen", lambda *a, **kw: Response(compressed))
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    manifest = fetch_sample(first, limit=3, scan=6)
    fetch_sample(second, limit=3, scan=6)
    assert first.read_bytes() == second.read_bytes()
    assert manifest["counts"] == {"scanned": 6, "low_plays": 1, "high_deviation": 1, "eligible": 4}
    assert manifest["csv_sha256"] == sha256(first)
    assert len(load_puzzles(first)) == 3
    saved = json.loads(first.with_suffix(".manifest.json").read_text())
    assert saved["http_identity"]["ETag"] == "test-snapshot"
    with pytest.raises(ValueError, match="refusing to replace"):
        fetch_sample(first, limit=3, scan=6)


def test_csv_requires_a_known_schema(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("rating,fen\n1500,invalid\n")
    with pytest.raises(ValueError, match="CSV requires columns"):
        load_puzzles(path)
