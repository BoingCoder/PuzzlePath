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
