"""Calibration application and selection helpers."""

import numpy as np

from src.predict import calibrate
from src.train import apply_calibration, select_calibration


def test_apply_calibration_identity_when_none():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(apply_calibration(p, None), p)


def test_apply_calibration_maps_through_table():
    cal = {"x": [0.0, 1.0], "y": [0.0, 0.5]}  # halve every probability
    assert np.allclose(apply_calibration(np.array([0.4, 0.8]), cal), [0.2, 0.4])


def test_predict_calibrate_respects_meta():
    assert np.allclose(calibrate(np.array([0.3]), {}), [0.3])  # no key -> identity
    cal = {"x": [0.0, 1.0], "y": [0.0, 0.5]}
    assert np.allclose(calibrate(np.array([0.6]), {"calibration": cal}), [0.3])


def test_select_calibration_prefers_none_for_calibrated_scores():
    # already-calibrated scores: p equals the empirical rate -> "none" should win
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)
    assert select_calibration(p, y) is None
