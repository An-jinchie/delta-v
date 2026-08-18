"""
tests/test_scorer.py — Tests for pipeline/prioritize/scorer.py
"""

import math
import os
import tempfile

import pandas as pd
import pytest

from pipeline.map.density import ALTITUDE_BANDS, compute_density
from pipeline.map.tle_fetcher import TLEFetcher
from pipeline.map.risk_map import RiskDensityMap
from pipeline.prioritize.scorer import (
    PriorityScorer,
    hohmann_dv_ms,
    severity_index,
    TIER_HIGH_MIN,
    TIER_MONITOR_MIN,
    H_REF_KM_DEFAULT,
    GM,
    R_EARTH,
)

FALLBACK_PATH = "data/tle_snapshot_fallback.csv"


def _make_risk_df() -> pd.DataFrame:
    """Build a realistic risk DataFrame from the fallback TLE snapshot."""
    fetcher = TLEFetcher(fallback_path=FALLBACK_PATH)
    lines = fetcher._load_fallback()
    density_df = compute_density(lines)
    rmap = RiskDensityMap()
    return rmap.compute(density_df)


# ── Hohmann delta-v formula ───────────────────────────────────────────────────

def test_hohmann_dv_reference_orbit_to_itself_is_zero():
    """Transferring from ref to same altitude should give ~0 Hohmann dv."""
    dv = hohmann_dv_ms(400.0, 400.0, plane_change_deg=0.0)
    assert abs(dv["dv_hohmann_ms"]) < 1e-6


def test_hohmann_dv_increases_with_altitude():
    """Higher target altitudes from a fixed reference should cost more delta-v."""
    dvs = []
    for alt in [500, 700, 900, 1100, 1300, 1500, 1800]:
        dv = hohmann_dv_ms(H_REF_KM_DEFAULT, alt, plane_change_deg=0.0)
        dvs.append(dv["dv_hohmann_ms"])
    # Each successive altitude should cost more
    for i in range(len(dvs) - 1):
        assert dvs[i] < dvs[i + 1], (
            f"dv[{i}]={dvs[i]:.2f} should be < dv[{i+1}]={dvs[i+1]:.2f}"
        )


def test_hohmann_dv_reference_400_to_600():
    """400→600 km Hohmann: textbook value is ~47 m/s, plane change adds ~11 m/s."""
    dv = hohmann_dv_ms(400.0, 600.0)
    # Total should be in a physically realistic range
    assert 50.0 < dv["dv_hohmann_ms"] < 200.0, (
        f"dv_hohmann={dv['dv_hohmann_ms']:.1f} m/s outside expected range"
    )
    assert dv["dv_total_ms"] > dv["dv_hohmann_ms"]


def test_hohmann_dv_no_nan():
    for alt in [300, 500, 800, 1100, 1500, 1900]:
        dv = hohmann_dv_ms(H_REF_KM_DEFAULT, alt)
        assert math.isfinite(dv["dv_hohmann_ms"])
        assert math.isfinite(dv["dv_plane_ms"])
        assert math.isfinite(dv["dv_total_ms"])


def test_plane_change_zero_gives_no_plane_dv():
    dv = hohmann_dv_ms(400.0, 800.0, plane_change_deg=0.0)
    assert abs(dv["dv_plane_ms"]) < 1e-6


def test_plane_change_adds_positive_dv():
    dv_no_pc  = hohmann_dv_ms(400.0, 800.0, plane_change_deg=0.0)
    dv_with_pc = hohmann_dv_ms(400.0, 800.0, plane_change_deg=5.0)
    assert dv_with_pc["dv_total_ms"] > dv_no_pc["dv_total_ms"]


# ── Severity index ────────────────────────────────────────────────────────────

def test_severity_index_ordering():
    """Large > medium > small."""
    sev_small  = severity_index("small")
    sev_medium = severity_index("medium")
    sev_large  = severity_index("large")
    assert sev_small < sev_medium < sev_large, (
        f"Expected small<medium<large: {sev_small:.4f} {sev_medium:.4f} {sev_large:.4f}"
    )


def test_severity_index_large_equals_one():
    """Large is the maximum — normalised to 1.0."""
    assert abs(severity_index("large") - 1.0) < 1e-9


def test_severity_index_positive():
    for size in ("small", "medium", "large"):
        assert severity_index(size) > 0.0


# ── PriorityScorer output contract ────────────────────────────────────────────

def test_score_returns_all_bands():
    risk_df = _make_risk_df()
    scorer = PriorityScorer()
    result = scorer.score(risk_df)
    assert len(result) == len(ALTITUDE_BANDS)


def test_score_required_columns():
    risk_df = _make_risk_df()
    scorer = PriorityScorer()
    result = scorer.score(risk_df)
    required = {
        "rank", "band_label", "altitude_km_mid",
        "composite_risk_density", "severity_index",
        "dv_hohmann_ms", "dv_plane_ms", "dv_total_ms",
        "priority_score", "tier", "explanation_text",
    }
    assert required.issubset(set(result.columns))


def test_score_no_nan():
    risk_df = _make_risk_df()
    scorer = PriorityScorer()
    result = scorer.score(risk_df)
    for col in ["priority_score", "dv_total_ms", "composite_risk_density"]:
        assert not result[col].isnull().any(), f"NaN in column {col}"


def test_score_priority_in_unit_range():
    risk_df = _make_risk_df()
    scorer = PriorityScorer()
    result = scorer.score(risk_df)
    assert (result["priority_score"] >= 0.0).all()
    assert (result["priority_score"] <= 1.0).all()


def test_score_sorted_descending():
    risk_df = _make_risk_df()
    scorer = PriorityScorer()
    result = scorer.score(risk_df)
    scores = result["priority_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_score_rank_starts_at_one():
    risk_df = _make_risk_df()
    result = PriorityScorer().score(risk_df)
    assert result["rank"].iloc[0] == 1
    assert list(result["rank"]) == list(range(1, len(result) + 1))


def test_tiers_valid_values():
    risk_df = _make_risk_df()
    result = PriorityScorer().score(risk_df)
    valid = {"HIGH-PRIORITY", "MONITOR", "LOW"}
    assert set(result["tier"].unique()).issubset(valid)


def test_all_tiers_assigned():
    """Every row must have a tier assigned (no empty strings or NaN)."""
    risk_df = _make_risk_df()
    result = PriorityScorer().score(risk_df)
    assert not result["tier"].isnull().any()
    assert not (result["tier"] == "").any()


def test_dv_increases_with_altitude_above_reference():
    """dv_total_ms should increase monotonically for bands ABOVE the reference orbit.

    Note: the 200-400 km band midpoint (300 km) is BELOW the 400 km reference orbit.
    A downward transfer also costs delta-v, so dv is not monotonically increasing
    across the full range — only above the reference orbit.
    """
    risk_df = _make_risk_df()
    result = PriorityScorer(h_ref_km=H_REF_KM_DEFAULT).score(risk_df)
    # Filter to bands above the reference orbit (midpoint > 400 km)
    above_ref = result[result["altitude_km_mid"] > H_REF_KM_DEFAULT].sort_values("altitude_km_mid")
    dvs = above_ref["dv_total_ms"].tolist()
    assert len(dvs) >= 2, "Need at least 2 bands above reference for this test"
    for i in range(len(dvs) - 1):
        assert dvs[i] <= dvs[i + 1], (
            f"dv should increase above ref orbit: "
            f"dvs[{i}]={dvs[i]:.1f} > dvs[{i+1}]={dvs[i+1]:.1f}"
        )


def test_explanation_text_contains_tier():
    risk_df = _make_risk_df()
    result = PriorityScorer().score(risk_df)
    for _, row in result.iterrows():
        assert row["tier"] in row["explanation_text"], (
            f"Tier '{row['tier']}' not found in explanation: {row['explanation_text']}"
        )


# ── Export functions ──────────────────────────────────────────────────────────

def test_export_csv_creates_file():
    risk_df = _make_risk_df()
    result = PriorityScorer().score(risk_df)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "priority.csv")
        PriorityScorer.export_csv(result, path)
        assert os.path.exists(path)
        df_reload = pd.read_csv(path)
        assert len(df_reload) == len(result)
        assert "dv_total_ms" in df_reload.columns


def test_export_json_creates_file():
    risk_df = _make_risk_df()
    result = PriorityScorer().score(risk_df)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "priority.json")
        PriorityScorer.export_json(result, path)
        assert os.path.exists(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert len(data) == len(result)
        assert "dv_total_ms" in data[0]
