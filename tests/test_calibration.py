import numpy as np
import pytest

from puzzlepath.model import component_bootstrap_mae, conformal_radius


def test_conformal_uses_the_finite_sample_order_statistic():
    # n=9 and alpha=.2 -> ceil((9+1)*.8)=8, the eighth score.
    radius, count = conformal_radius(np.arange(1, 10), list("abcdefghi"), alpha=0.2)
    assert radius == 8
    assert count == 9


def test_calibration_components_use_their_worst_residual():
    radius, count = conformal_radius(np.array([1, 9, 2, 3]), ["a", "a", "b", "c"], alpha=0.5)
    assert radius == 3
    assert count == 3


def test_small_calibration_sample_cannot_claim_finite_high_coverage():
    with pytest.raises(ValueError, match="too few calibration"):
        conformal_radius(np.arange(8), list("abcdefgh"), alpha=0.1)


@pytest.mark.parametrize("scores", [np.array([np.nan]), np.array([-1.0])])
def test_invalid_residuals_are_rejected(scores):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        conformal_radius(scores, ["a"], alpha=0.5)


def test_component_bootstrap_handles_unequal_component_sizes():
    ci = component_bootstrap_mae(np.array([10, 10, 20]), ["a", "a", "b"], seed=42)
    assert ci == [10.0, 20.0]
