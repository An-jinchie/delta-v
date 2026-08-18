"""
tests/test_risk_map.py — Tests for pipeline/map/risk_map.py
"""

import time

import pandas as pd
import pytest

from pipeline.map.density import ALTITUDE_BANDS, compute_density
from pipeline.map.tle_fetcher import TLEFetcher
from pipeline.map.risk_map import RiskDensityMap, _BAND_LABELS

FALLBACK_PATH = "data/tle_snapshot_fallback.csv"


def _make_density_df() -> pd.DataFrame:
    """Load the fallback TLE snapshot and compute density."""
    fetcher = TLEFetcher(fallback_path=FALLBACK_PATH)
    lines = fetcher._load_fallback()
    return compute_density(lines)


def _make_detection(band: str, confidence: float = 0.8,
                    age_seconds: float = 0.0) -> dict:
    return {
        "altitude_band_km": band,
        "confidence": confidence,
        "timestamp": time.time() - age_seconds,
        "size_class": "medium",
    }


# ── Output contract ───────────────────────────────────────────────────────────

def test_compute_returns_all_bands():
    density_df = _make_density_df()
    rmap = RiskDensityMap()
    result = rmap.compute(density_df)
    assert len(result) == len(ALTITUDE_BANDS)


def test_compute_required_columns():
    density_df = _make_density_df()
    rmap = RiskDensityMap()
    result = rmap.compute(density_df)
    required = {
        "altitude_band_km", "band_label", "band_lo_km", "band_hi_km",
        "tracked_object_count", "density_per_km",
        "detection_count", "detection_confidence_weighted",
        "composite_risk_density", "last_updated",
    }
    assert required.issubset(set(result.columns))


def test_no_nan_values():
    density_df = _make_density_df()
    rmap = RiskDensityMap()
    result = rmap.compute(density_df)
    assert not result.isnull().any().any()


def test_composite_risk_density_range():
    """composite_risk_density must be in [0, 1]."""
    density_df = _make_density_df()
    rmap = RiskDensityMap()
    result = rmap.compute(density_df)
    assert (result["composite_risk_density"] >= 0.0).all()
    assert (result["composite_risk_density"] <= 1.0).all()


# ── Detection accumulation ────────────────────────────────────────────────────

def test_update_adds_detections():
    rmap = RiskDensityMap()
    assert rmap.detection_count == 0
    rmap.update([_make_detection("400-600")])
    assert rmap.detection_count == 1
    rmap.update([_make_detection("600-800"), _make_detection("400-600")])
    assert rmap.detection_count == 3


def test_clear_removes_detections():
    rmap = RiskDensityMap()
    rmap.update([_make_detection("400-600")] * 5)
    rmap.clear()
    assert rmap.detection_count == 0


def test_detection_increases_risk_for_band():
    """Adding detections to a band should increase its composite_risk_density."""
    density_df = _make_density_df()
    rmap_baseline = RiskDensityMap()
    baseline = rmap_baseline.compute(density_df)

    rmap_with_detections = RiskDensityMap()
    rmap_with_detections.update([_make_detection("400-600", confidence=1.0)] * 20)
    with_detections = rmap_with_detections.compute(density_df)

    base_score = baseline.loc[
        baseline["altitude_band_km"] == "400-600", "composite_risk_density"
    ].iloc[0]
    det_score = with_detections.loc[
        with_detections["altitude_band_km"] == "400-600", "composite_risk_density"
    ].iloc[0]

    assert det_score > base_score, (
        f"Detections should increase risk: before={base_score:.4f}, after={det_score:.4f}"
    )


def test_higher_density_band_scores_higher_baseline():
    """The band with most TLE objects should have highest composite_risk baseline."""
    density_df = _make_density_df()
    rmap = RiskDensityMap()
    result = rmap.compute(density_df)

    # Find the band with most objects
    top_density_band = density_df.sort_values("density_per_km", ascending=False).iloc[0]
    top_risk_band    = result.sort_values("composite_risk_density", ascending=False).iloc[0]

    assert top_density_band["altitude_band_km"] == top_risk_band["altitude_band_km"], (
        f"Highest density band ({top_density_band['altitude_band_km']}) should be "
        f"highest risk, but got ({top_risk_band['altitude_band_km']})"
    )


# ── Recency decay ─────────────────────────────────────────────────────────────

def test_old_detections_have_less_weight():
    """Detections from 14 days ago should contribute less than fresh ones."""
    density_df = _make_density_df()

    rmap_fresh = RiskDensityMap(decay_half_life_days=7.0)
    rmap_fresh.update([_make_detection("400-600", confidence=1.0, age_seconds=0)])
    fresh = rmap_fresh.compute(density_df)

    rmap_old = RiskDensityMap(decay_half_life_days=7.0)
    rmap_old.update([
        _make_detection("400-600", confidence=1.0, age_seconds=14 * 86400)
    ])
    old = rmap_old.compute(density_df)

    fresh_score = fresh.loc[fresh["altitude_band_km"] == "400-600",
                            "detection_confidence_weighted"].iloc[0]
    old_score   = old.loc[old["altitude_band_km"] == "400-600",
                          "detection_confidence_weighted"].iloc[0]

    assert fresh_score > old_score, (
        f"Fresh detection should weigh more: fresh={fresh_score:.4f}, old={old_score:.4f}"
    )


def test_zero_detections_gives_zero_detection_weight():
    density_df = _make_density_df()
    rmap = RiskDensityMap()
    result = rmap.compute(density_df)
    assert (result["detection_confidence_weighted"] == 0.0).all()
    assert (result["detection_count"] == 0).all()


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_unknown_band_label_ignored():
    """Detections for a non-existent band should be silently ignored."""
    density_df = _make_density_df()
    rmap = RiskDensityMap()
    rmap.update([{"altitude_band_km": "9999-9999", "confidence": 1.0}])
    result = rmap.compute(density_df)
    assert not result.isnull().any().any()


def test_detection_count_column_correct():
    density_df = _make_density_df()
    rmap = RiskDensityMap()
    rmap.update([_make_detection("400-600")] * 3)
    rmap.update([_make_detection("600-800")] * 2)
    result = rmap.compute(density_df)
    count_400 = result.loc[result["altitude_band_km"] == "400-600", "detection_count"].iloc[0]
    count_600 = result.loc[result["altitude_band_km"] == "600-800", "detection_count"].iloc[0]
    assert count_400 == 3
    assert count_600 == 2
