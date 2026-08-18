"""
tests/test_density.py — Tests for pipeline/map/density.py

Tests use the committed fallback TLE snapshot so they run offline without
any network access.  The fallback is loaded via the TLEFetcher fallback path.
"""

import os
import tempfile

import pandas as pd
import pytest

from pipeline.map.density import (
    ALTITUDE_BANDS,
    compute_density,
    _extract_tle_pairs,
    _is_tle_line1,
    _is_tle_line2,
)
from pipeline.map.tle_fetcher import TLEFetcher, _parse_tle_lines

FALLBACK_PATH = "data/tle_snapshot_fallback.csv"

# Minimal synthetic TLE data — two real-format TLE lines for ISS
# (NORAD ID 25544 — used only as format validation, epoch may be stale)
_ISS_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00001234  00000-0  12345-4 0  9991"
_ISS_TLE_LINE2 = "2 25544  51.6400 208.9163 0001234  86.9745 273.1572 15.49990000123456"

# A minimal known-good TLE pair for a LEO object at ~400 km
_LEO_LINE1 = "1 00001U 60001A   24001.00000000  .00000000  00000-0  00000-0 0  9991"
_LEO_LINE2 = "2 00001  65.0000 180.0000 0010000  90.0000 270.0000 15.32000000012345"


# ── TLE parsing helpers ───────────────────────────────────────────────────────

def test_is_tle_line1_valid():
    assert _is_tle_line1(_ISS_TLE_LINE1)


def test_is_tle_line1_rejects_line2():
    assert not _is_tle_line1(_ISS_TLE_LINE2)


def test_is_tle_line2_valid():
    assert _is_tle_line2(_ISS_TLE_LINE2)


def test_is_tle_line2_rejects_line1():
    assert not _is_tle_line2(_ISS_TLE_LINE1)


def test_extract_tle_pairs_basic():
    lines = [_ISS_TLE_LINE1, _ISS_TLE_LINE2]
    pairs = _extract_tle_pairs(lines)
    assert len(pairs) == 1
    assert pairs[0][0] == _ISS_TLE_LINE1
    assert pairs[0][1] == _ISS_TLE_LINE2


def test_extract_tle_pairs_with_name_line():
    lines = ["ISS (ZARYA)", _ISS_TLE_LINE1, _ISS_TLE_LINE2]
    pairs = _extract_tle_pairs(lines)
    assert len(pairs) == 1


def test_extract_tle_pairs_multiple():
    lines = [
        _ISS_TLE_LINE1, _ISS_TLE_LINE2,
        "ANOTHER SAT",
        _ISS_TLE_LINE1, _ISS_TLE_LINE2,
    ]
    pairs = _extract_tle_pairs(lines)
    assert len(pairs) == 2


def test_parse_tle_lines_filters_comments():
    raw = "# This is a comment\n" + _ISS_TLE_LINE1 + "\n" + _ISS_TLE_LINE2
    lines = _parse_tle_lines(raw)
    assert not any(l.startswith("#") for l in lines)
    assert len(lines) == 2


def test_parse_tle_lines_strips_whitespace():
    raw = "  " + _ISS_TLE_LINE1 + "  \n  " + _ISS_TLE_LINE2 + "  "
    lines = _parse_tle_lines(raw)
    assert lines[0] == _ISS_TLE_LINE1
    assert lines[1] == _ISS_TLE_LINE2


# ── compute_density output contract ──────────────────────────────────────────

def test_density_output_columns():
    """compute_density always returns the required columns."""
    # Use minimal TLE pair (even if it errors, we get empty bands)
    lines = [_ISS_TLE_LINE1, _ISS_TLE_LINE2]
    df = compute_density(lines)
    required = {"altitude_band_km", "band_label", "band_lo_km", "band_hi_km",
                "object_count", "band_width_km", "density_per_km"}
    assert required.issubset(set(df.columns))


def test_density_all_bands_present():
    """All 8 altitude bands must appear even with zero objects."""
    df = compute_density([])
    assert len(df) == len(ALTITUDE_BANDS)


def test_density_zero_input_gives_zero_counts():
    df = compute_density([])
    assert (df["object_count"] == 0).all()
    assert (df["density_per_km"] == 0.0).all()


def test_density_no_negative_counts():
    lines = [_ISS_TLE_LINE1, _ISS_TLE_LINE2]
    df = compute_density(lines)
    assert (df["object_count"] >= 0).all()


def test_density_no_nan():
    df = compute_density([])
    assert not df.isnull().any().any()


def test_density_per_km_formula():
    """density_per_km must equal object_count / band_width_km for all rows."""
    df = compute_density([_ISS_TLE_LINE1, _ISS_TLE_LINE2])
    for _, row in df.iterrows():
        expected = row["object_count"] / row["band_width_km"]
        assert abs(row["density_per_km"] - expected) < 1e-9, (
            f"Band {row['altitude_band_km']}: density mismatch"
        )


def test_density_band_widths_correct():
    df = compute_density([])
    for _, row in df.iterrows():
        expected_width = row["band_hi_km"] - row["band_lo_km"]
        assert row["band_width_km"] == expected_width


# ── Fallback snapshot ─────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(FALLBACK_PATH),
    reason="Fallback TLE snapshot not yet committed (run fetch first)",
)
def test_fallback_snapshot_loads():
    """Fallback snapshot must load and contain real TLE pairs."""
    fetcher = TLEFetcher(fallback_path=FALLBACK_PATH)
    lines = fetcher._load_fallback()
    pairs = _extract_tle_pairs(lines)
    assert len(pairs) > 100, f"Expected >100 TLE pairs in fallback, got {len(pairs)}"


@pytest.mark.skipif(
    not os.path.exists(FALLBACK_PATH),
    reason="Fallback TLE snapshot not yet committed",
)
def test_density_from_fallback_structure():
    """Running compute_density on the fallback should produce a valid DataFrame."""
    fetcher = TLEFetcher(fallback_path=FALLBACK_PATH)
    lines = fetcher._load_fallback()
    df = compute_density(lines)

    assert len(df) == len(ALTITUDE_BANDS)
    assert (df["object_count"] >= 0).all()
    assert not df.isnull().any().any()
    # At least some bands should have objects
    assert df["object_count"].sum() > 0, "No objects found in any LEO band from fallback"


@pytest.mark.skipif(
    not os.path.exists(FALLBACK_PATH),
    reason="Fallback TLE snapshot not yet committed",
)
def test_density_from_fallback_known_bands():
    """ISS-altitude band (400-600 km) should have a non-zero count."""
    fetcher = TLEFetcher(fallback_path=FALLBACK_PATH)
    lines = fetcher._load_fallback()
    df = compute_density(lines)
    iss_band = df[df["altitude_band_km"] == "400-600"]
    assert len(iss_band) == 1
    assert iss_band.iloc[0]["object_count"] > 0, (
        "Expected objects in 400-600 km band (ISS altitude)"
    )


# ── TLEFetcher cache mechanics (offline) ─────────────────────────────────────

def test_fetcher_uses_fallback_when_no_cache(tmp_path):
    """With no cache and no network, fetcher should use fallback."""
    if not os.path.exists(FALLBACK_PATH):
        pytest.skip("Fallback not available")
    fetcher = TLEFetcher(
        cache_path=str(tmp_path / "cache.txt"),
        cache_meta_path=str(tmp_path / "meta.json"),
        fallback_path=FALLBACK_PATH,
    )
    # cache_is_fresh will return False (no cache file)
    # We test _load_fallback directly to avoid network call
    lines = fetcher._load_fallback()
    assert len(lines) > 0


def test_fetcher_cache_freshness_false_for_new_instance(tmp_path):
    fetcher = TLEFetcher(
        cache_path=str(tmp_path / "cache.txt"),
        cache_meta_path=str(tmp_path / "meta.json"),
    )
    assert not fetcher._cache_is_fresh()
