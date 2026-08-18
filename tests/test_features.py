"""
tests/test_features.py — Tests for pipeline/characterize/features.py
"""

import numpy as np
import pytest

from pipeline.characterize.generator import LightCurveGenerator, LightCurveParams
from pipeline.characterize.inversion import invert
from pipeline.characterize.features import (
    extract_features,
    N_FEATURES,
    FEATURE_NAMES,
    _SHAPE_HINT_ENCODING,
)

CADENCE_S = 0.1
N_SAMPLES = 256


def _make_pair(shape: str, rotation_rate_hz: float = 0.5, snr: float = 30.0, seed: int = 0):
    """Return (lc, InversionResult) for a given shape."""
    gen = LightCurveGenerator()
    params = LightCurveParams(
        size_class="medium",
        shape=shape,
        rotation_rate_hz=rotation_rate_hz,
        phase_offset=0.0,
        albedo=0.15,
        snr=snr,
        n_samples=N_SAMPLES,
        cadence_s=CADENCE_S,
        seed=seed,
    )
    lc = gen.generate(params)
    inv = invert(lc, cadence_s=CADENCE_S)
    return lc, inv


# ── Shape and dtype ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", ["sphere", "flat_plate", "cylinder", "tumbling"])
def test_output_shape(shape):
    lc, inv = _make_pair(shape)
    features = extract_features(lc, inv)
    assert features.shape == (N_FEATURES,), (
        f"Expected ({N_FEATURES},), got {features.shape} for shape={shape}"
    )


@pytest.mark.parametrize("shape", ["sphere", "flat_plate", "cylinder", "tumbling"])
def test_dtype_float64(shape):
    lc, inv = _make_pair(shape)
    features = extract_features(lc, inv)
    assert features.dtype == np.float64


# ── No NaN or Inf ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", ["sphere", "flat_plate", "cylinder", "tumbling"])
def test_no_nan(shape):
    lc, inv = _make_pair(shape)
    features = extract_features(lc, inv)
    assert not np.any(np.isnan(features)), f"NaN in features for shape={shape}"


@pytest.mark.parametrize("shape", ["sphere", "flat_plate", "cylinder", "tumbling"])
def test_no_inf(shape):
    lc, inv = _make_pair(shape)
    features = extract_features(lc, inv)
    assert not np.any(np.isinf(features)), f"Inf in features for shape={shape}"


# ── Feature name alignment ────────────────────────────────────────────────────

def test_feature_names_count():
    assert len(FEATURE_NAMES) == N_FEATURES


# ── Specific feature values ───────────────────────────────────────────────────

def test_rotation_rate_matches_inversion():
    lc, inv = _make_pair("flat_plate", rotation_rate_hz=1.0)
    features = extract_features(lc, inv)
    # Index 0 should equal inv.rotation_rate_hz
    assert features[0] == inv.rotation_rate_hz


def test_amplitude_matches_inversion():
    lc, inv = _make_pair("tumbling", rotation_rate_hz=0.5)
    features = extract_features(lc, inv)
    assert features[2] == inv.amplitude


def test_shape_hint_encoded_correctly():
    for shape in ("sphere", "flat_plate", "cylinder", "tumbling"):
        lc, inv = _make_pair(shape, snr=200.0)
        features = extract_features(lc, inv)
        expected_enc = float(_SHAPE_HINT_ENCODING[inv.shape_hint])
        assert features[9] == expected_enc, (
            f"shape={shape}: expected enc={expected_enc}, got {features[9]}"
        )


def test_mean_feature_in_unit_range():
    """Generator normalises to [0,1] so mean should be in (0,1)."""
    for shape in ("sphere", "flat_plate", "tumbling"):
        lc, inv = _make_pair(shape)
        features = extract_features(lc, inv)
        mean_val = features[10]
        assert 0.0 <= mean_val <= 1.0, f"Mean {mean_val:.4f} outside [0,1] for shape={shape}"


def test_std_non_negative():
    for shape in ("sphere", "flat_plate", "cylinder"):
        lc, inv = _make_pair(shape)
        features = extract_features(lc, inv)
        assert features[11] >= 0.0, f"Negative std for shape={shape}"


def test_peak_count_non_negative():
    lc, inv = _make_pair("flat_plate")
    features = extract_features(lc, inv)
    # peak_count is at index 15 (after log_rotation_rate_hz was added at index 10)
    peak_idx = FEATURE_NAMES.index("peak_count")
    assert features[peak_idx] >= 0.0


def test_pct_range_non_negative():
    for shape in ("sphere", "flat_plate", "tumbling"):
        lc, inv = _make_pair(shape)
        features = extract_features(lc, inv)
        assert features[16] >= 0.0, f"Negative pct range for shape={shape}"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_constant_curve_no_crash():
    """A constant light curve (all ones) should not crash."""
    lc = np.ones(N_SAMPLES)
    inv = invert(lc, cadence_s=CADENCE_S)
    features = extract_features(lc, inv)
    assert features.shape == (N_FEATURES,)
    assert not np.any(np.isnan(features))


def test_empty_curve_raises():
    from pipeline.characterize.inversion import InversionResult
    # Empty array should raise before we even get to inversion
    with pytest.raises((ValueError, Exception)):
        lc = np.array([])
        inv = invert(np.ones(8))  # dummy inv
        extract_features(lc, inv)


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_deterministic():
    """Same (lc, inv) always produces identical features."""
    lc, inv = _make_pair("cylinder", seed=77)
    f1 = extract_features(lc, inv)
    f2 = extract_features(lc, inv)
    np.testing.assert_array_equal(f1, f2)
