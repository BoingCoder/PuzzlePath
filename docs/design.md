# Design notes

## Question

How much of a puzzle's observed Lichess rating can be predicted from the position
shown to the solver, without an engine or access to the solution?

The motivation is chess practice and teaching: an unrated exercise needs a rough
starting level before someone can try it. A position estimate can be a triage
tool; it cannot measure a student's skill or replace observed solve results.

## Data contract

The public [Lichess puzzle export](https://database.lichess.org/#puzzles) is CC0.
Its FEN is **before the opponent's setup move**. Apply the first UCI move, then
extract features. The remaining moves belong only in the answer key.

The default sampler scans 150,000 rows, retains rows with at least 1,000 plays
and rating deviation at most 100, and uses a seeded reservoir to keep 20,000.
This is a random sample of that eligible prefix, not of the complete database.
The snapshot is committed with its SHA-256 and HTTP source identity. Re-running
the download later can change the data; reproducing the benchmark uses the
committed snapshot instead.

## Feature boundary

Only `chess.Board` enters `position_features`. Piece counts, material, mobility,
available checks/captures, attacked pieces, pawn structure, and king-ring pressure
are solver-relative. Color mirroring should produce the same features.

Rating, rating deviation, popularity, play count, source identity, opening labels,
theme tags, and solution moves/length never enter the model. Filtering on plays
and deviation is a data-quality selection and changes the population being tested.

## Evaluation plan

1. Join puzzles connected by the same source game OR the same normalized solver
   position. Transitive connections stay together.
2. Hash each connected component into train/validation/calibration/test, aiming
   for 60/15/10/15 percent of components. Counts need not be exact.
3. Fit a median baseline, material-only ridge, full-feature ridge, and two bounded
   gradient-boosting configurations on train only.
4. Choose the lowest validation MAE. Keep the selected training fit; do not refit
   it on calibration or test.
5. Use calibration component maximum absolute residuals for a conservative
   split-conformal radius, with the finite-sample order statistic.
6. Open the test partition for final metrics and error slices. Include a
   component-bootstrap interval for MAE and empirical interval coverage.

The interval's mathematical motivation assumes exchangeable components. That is
not established for a bounded export prefix or arbitrary new positions. Empirical
coverage on this snapshot is what the benchmark measures. Component maxima aim
to cover all sampled puzzles in a new exchangeable component, so intervals can
be wide. Calibration groups and test groups remain disjoint.

## Product decisions

- Use Python and a CLI to keep the experiment reproducible on a CPU laptop.
- Export PDF worksheets and PGN answer keys so the output is useful without an
  account or hosted service.
- Prefer observed ratings when making practice sets from already-rated puzzles.
  Offer prediction-based ordering as an explicit research option. There is no
  reason to replace known Lichess ratings just to use ML.
- Treat board features as a limited baseline. They do not search tactical lines,
  model individual solving histories, or establish improved teaching outcomes.

## Next questions

Engine-derived search features, a full-export sample, and a genuinely later
snapshot would each answer a different question. They need separate experiments;
none is represented by the first benchmark.
