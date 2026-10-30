# Model card

## Intended use

PuzzlePath estimates a rough Lichess-style puzzle difficulty from a valid,
solver-ready chess position. The intended use is experimentation with unrated
practice material. Use observed ratings when available. A teacher or solver
must assess whether a new exercise is suitable.

It does not produce best moves, individual solve probabilities, player ratings,
learning gains, or a validated curriculum. It is not suitable for assigning
high-stakes learner scores or claiming that a student has mastered a skill.

## Inputs and training

- Input: a standard chess position including whose turn it is.
- Features: 50 solver-relative numeric board features; version 2 adds five bounded
  tactical features from legal replies to checks. No chess engine is invoked.
- Excluded: the provided solution, its length, themes, ratings, rating deviation,
  popularity, plays, source identity, and opening labels.
- Label: the public Lichess puzzle rating in the retained export snapshot.
- Development data: 20,000 puzzles from an eligible 150,000-row prefix. Eligibility
  requires at least 1,000 plays and rating deviation no higher than 100.
- Exact sample: `data/puzzles.csv`; provenance: `data/puzzles.manifest.json`.
- Train/validation/calibration/test rows: 12,007 / 3,041 / 1,993 / 2,959.
- Groups: connected components under shared source game or identical normalized
  solver position. Games and positions do not cross the split boundary.

The chosen estimator is `HistGradientBoostingRegressor` with absolute-error loss,
180 iterations, learning rate 0.07, 15 leaves, minimum leaf size 30, regularization
2, and seed 42. It was selected from five candidates using validation MAE, and
remains fitted on train only. Internal early stopping is disabled.

## Evaluation history

The first 45-feature representation produced development-test MAE 308.2. The
50-feature revision produced 299.3. Because the first test result was viewed
before the feature revision, that comparison is exploratory. Both results are
retained in the [experiment record](experiments.md).

The revision was frozen in commit `1bd32f8`, before fetching a separate 4,000-row
sample from source rows 250,001–300,000. Excluding one component overlapping the
development sample left 3,999 puzzles. Frozen-model confirmation MAE was **296.8**
versus **316.4** for the training-median baseline; RMSE was 370.2 and R² was 0.106.
The model MAE component-bootstrap 95% interval was 289.9–303.4.

This is transfer within one export snapshot, not a temporal holdout. Puzzles
outside the sampled windows, new websites, and human-written exercises have not
been evaluated. Exact position and game exclusion does not eliminate near-duplicate
tactical patterns.

## Uncertainty

The interval uses the finite-sample split-conformal order statistic over maximum
absolute residuals per calibration component. It has nominal 90% component
coverage under exchangeability of components. This assumption is not established
for arbitrary new positions.

The resulting radius is approximately ±614 puzzle-rating points. The frozen
radius covered 90.9% of confirmation rows/components. This marginal coverage
must not be read as conditional coverage for each rating, theme, or learner.
Development rating slices show large errors at the easy and difficult extremes.

## Limitations

- Most rating variation remains unexplained. The representation does not perform
  multi-move tactical search or model human perception and attention.
- Estimates regress toward the middle of the sampled rating distribution.
- Play-count and rating-deviation filters select established puzzles; cold-start
  exercises may behave differently.
- Prefix/window sampling is bounded and is not asserted to be representative of
  the complete puzzle population.
- Lichess ratings change with solve attempts and are platform-specific. They are
  not interchangeable with a player's game rating.
- Permutation importance is descriptive and is computed on validation data, which
  also selected the model. Correlation affects the apparent importance of features.
- No classroom evaluation, learner study, user adoption, or improved chess result
  has been demonstrated.

## Reproducibility and artifact use

The data and model hashes appear in the generated metric files. `puzzlepath audit`
checks dataset identity, model bytes, prediction coverage, component assignments,
labels, and validation selection. Use `uv.lock` to match the fitted model's
scikit-learn version.

Joblib uses pickle, so loading an untrusted model can execute code. Use a trusted
checkout or regenerate the artifact with `puzzlepath train`. No user positions or
solve histories are sent to a remote service by inference or worksheet export.
