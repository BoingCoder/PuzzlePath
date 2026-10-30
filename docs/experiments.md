# Experiment record

These runs and repository commits record actual implementation and evaluation.
Run timestamps are UTC and come from the generated metrics files.

## 1. Board-statistics baseline

- Run: `2026-10-30T02:59:24Z`
- Code snapshot: `14c09d6`
- Inputs: 45 features covering material, mobility, captures/checks, pawn structure,
  attacked pieces, and king-ring pressure.
- Validation-selected model: `boosted_15`.
- Validation MAE: 311.49; development-test MAE: 308.18.
- Test median-baseline MAE: 320.52.
- Artifacts: [board-baseline report](../artifacts/board-baseline/report.md).

The improvement was small. Counts of available checks did not distinguish a
position with an immediate forced mate from one with many harmless checks.
The calibration interval was also wide. A coarse feature representation leaves
much of tactical difficulty unobserved.

To reproduce the original baseline, use the original commit in a separate clone
or worktree. The archived version-1 model expects the version-1 feature schema;
the current CLI intentionally refuses to mix it with version-2 features.

## 2. Bounded tactical features

- Run: `2026-10-30T03:02:29Z`
- Add five position-derived features: immediate mating moves, checks with one
  legal reply, minimum and mean legal replies to checks, and checking
  destinations not attacked by the opponent.
- These require legal-move generation one reply deep, not Stockfish or solution
  moves. The supplied answer, tags, and rating remain outside the feature API.
- Validation MAE improved to 303.47 with the same two boosting configurations.
- Validation again selected `boosted_15`; development-test MAE was 299.27.
- Artifacts: [current benchmark](../artifacts/benchmark/report.md).

The development test had already been viewed during the first run. Reusing it
for a revised representation makes the second test result exploratory. The next
step is to freeze this model and evaluate a separately acquired export window.
Do not interpret the revision's development-test score as a new untouched test.

## Confirmation protocol

Freeze the version-2 features, model choice, model bytes, and calibration radius.
Then acquire a 4,000-puzzle sample from source rows 250,001–300,000, disjoint from
the original scanned prefix. Exclude any connected component sharing a game or
exact solver position with the development sample. Evaluate once without fitting
or recalibrating. Preserve its source manifest, predictions, and report.

This checks another part of the same export, not future puzzles, a new platform,
individual learners, or improved classroom outcomes.

## 3. Frozen-model confirmation

- Freeze commit: `1bd32f8`, preceding confirmation-data acquisition.
- Model SHA-256:
  `282947948e3737b5563ef972d531868d713757da4c9c0aea201c6f22e6aa6a2d`.
- Confirmation sample acquired at `2026-10-30T03:03:53Z`; the HTTP export identity
  matches the development snapshot.
- Evaluation: `2026-10-30T03:05:02Z`.
- Retained: 3,999 of 4,000 rows after one overlap exclusion.
- Frozen-model MAE: 296.83; training-median baseline MAE: 316.42.
- MAE component-bootstrap 95% interval: 289.94–303.44.
- Observed interval coverage: 90.87%, with the unchanged radius of ±614.03.
- Artifacts: [confirmation report](../artifacts/confirmation/report.md).

The gain transfers modestly to this other source window. The model remains a
coarse baseline with substantial error. No further features or hyperparameters
were selected against the confirmation result. A future iteration needs a fresh
evaluation sample if it is to make another untouched-holdout claim.
