# Public data snapshots

Both CSVs contain public chess positions from the
[Lichess puzzle export](https://database.lichess.org/#puzzles), released under
CC0-1.0. They contain no local student records, private notes, or solve histories.

- `puzzles.csv`: 20,000 eligible puzzles, sampled from the first 150,000 export
  rows. Train/validation/calibration/test assignments are recorded in the
  benchmark's `predictions.csv`.
- `confirmation.csv`: 4,000 eligible puzzles from a separate 50,000-row window
  after skipping 250,000 source rows. This sample evaluates the already-frozen
  version-2 model; it is not used for training or model selection.

Each adjacent `.manifest.json` records source URL, retrieval timestamp, HTTP
identity, filter counts, reservoir seed, sampling boundary, and exact CSV SHA-256.
The initial manifest predates the `skip_rows` option; its implied skip is zero.

The reservoir is uniform among eligible rows in its scanned window. These files
are **not** asserted to represent the full Lichess database uniformly. The sample
also excludes puzzles with fewer than 1,000 plays or rating deviation above 100.

The upstream URL changes over time. Reproduce published results with the committed
snapshots and `uv.lock`; fetching again is a new data-acquisition experiment.
The fetch command refuses to replace an existing snapshot.

```bash
uv run puzzlepath fetch --output outputs/new-snapshot.csv
```

Lichess ratings are puzzle Glicko ratings based on attempts, not player Elo or an
objective measure of thinking time. Source FENs precede the opponent's setup move;
the player sees the position after the first UCI move. See the
[official format notes](https://database.lichess.org/#puzzles).
