# Contributing

Use Python 3.12 and the committed lockfile:

```bash
uv sync --frozen --extra dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
uv run puzzlepath audit
```

Tests use committed public data and do not download anything. The pipeline test
trains on a small fixture, exports a worksheet, parses its PGN, and checks that
single-position inference agrees with batch inference.

For model changes, keep the feature function limited to the solver's board.
Do not add rating, theme, solution, popularity, play count, or source identifiers
to the feature vector. Add a semantic test for a new chess feature and bump
`FEATURE_VERSION` when the schema changes.

Preserve previous results. Choose configurations on validation, calibrate on
calibration, and record which evaluation sets have already been viewed. The
committed confirmation sample has been consumed; a future revision cannot claim
it as a fresh untouched holdout.

Run a new experiment in a new directory:

```bash
uv run puzzlepath train --output outputs/my-experiment
uv run puzzlepath audit --run outputs/my-experiment
```

Use real commit timestamps and describe the actual change and evidence. Keep
student records and private source material outside the repository.
