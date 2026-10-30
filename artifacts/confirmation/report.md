# Confirmation on a separate source window

Evaluation timestamp (UTC): 2026-10-30T03:05:02.113549+00:00

The model, hyperparameter choice, feature definition, and calibration radius were frozen before this evaluation. Nothing is fitted or recalibrated here.

3,999 of 4,000 puzzles remain after excluding every component connected to the development sample by source game or solver position. This is another export window, **not a temporal or cross-platform holdout**.

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Training-median baseline | 316.4 | 392.9 | -0.007 |
| Frozen boosted_15 | 296.8 | 370.2 | 0.106 |

Model MAE component-bootstrap 95% interval: 289.9–303.4.

Observed interval coverage: 90.9% of rows; 90.9% of complete components. The unchanged radius is ±614 rating points.

This checks transfer to another part of the same source, not learner outcomes or difficulty on unrelated exercises. The sample still filters on play count and rating deviation, and similar positions can remain even after exact duplicates are excluded.

Data SHA-256: `572b2972064ea3766d55c11805a692770314459df536e7caf5eeab2320fa268c`.

Frozen model SHA-256: `282947948e3737b5563ef972d531868d713757da4c9c0aea201c6f22e6aa6a2d`.
