"""
tests/test_tle_staleness.py — Tests for automatic TLE fallback regeneration.

Covers:
  - _fallback_age_days() correctly reads the # Generated: header
  - _refresh_fallback_if_stale() skips regeneration when snapshot is fresh
  - _refresh_fallback_if_stale() regenerates when snapshot is stale
  - generate_fallback_snapshot() produces a valid file with ≥1 object
  - The regenerated file passes the density pipeline (objects in LEO bands)
"""

import datetime
import os
import textwrap

import pytest

from pipeline.map.tle_fetcher import TLEFetcher, FALLBACK_MAX_AGE_DAYS, _GENERATED_RE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_snapshot_header(path: str, generated_at: datetime.datetime) -> None:
    """Write a minimal snapshot file with only a header (no TLE data)."""
    ts = generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    content = textwrap.dedent(f"""\
        # SYNTHETIC TLE SNAPSHOT — generated for offline demo and testing
        # Objects distributed across all LEO altitude bands (200–2000 km)
        # Generated: {ts}
        # Format matches CelesTrak TLE output. NOT real tracking data.
        #
    """)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ── _GENERATED_RE pattern ─────────────────────────────────────────────────────

def test_generated_re_matches_header_line():
    line = "# Generated: 2026-08-18T04:17:18Z"
    m = _GENERATED_RE.search(line)
    assert m is not None
    assert m.group(1) == "2026-08-18T04:17:18Z"


def test_generated_re_no_match_on_tle_line():
    line = "1 25544U 98067A   24001.50000000  .00001234  00000-0  12345-4 0  9991"
    assert _GENERATED_RE.search(line) is None


# ── _fallback_age_days ────────────────────────────────────────────────────────

def test_fallback_age_days_fresh(tmp_path):
    """A snapshot generated 1 hour ago should report ~0.04 days."""
    path = str(tmp_path / "snap.csv")
    now = datetime.datetime.now(datetime.timezone.utc)
    one_hour_ago = now - datetime.timedelta(hours=1)
    _write_snapshot_header(path, one_hour_ago)

    fetcher = TLEFetcher(fallback_path=path)
    age = fetcher._fallback_age_days()

    assert 0.03 < age < 0.1, f"Expected ~0.04 days, got {age}"


def test_fallback_age_days_stale(tmp_path):
    """A snapshot generated 10 days ago should report ~10 days."""
    path = str(tmp_path / "snap.csv")
    now = datetime.datetime.now(datetime.timezone.utc)
    ten_days_ago = now - datetime.timedelta(days=10)
    _write_snapshot_header(path, ten_days_ago)

    fetcher = TLEFetcher(fallback_path=path)
    age = fetcher._fallback_age_days()

    assert 9.9 < age < 10.1, f"Expected ~10 days, got {age}"


def test_fallback_age_days_missing_file(tmp_path):
    """A missing snapshot should return infinity."""
    path = str(tmp_path / "nonexistent.csv")
    fetcher = TLEFetcher(fallback_path=path)
    age = fetcher._fallback_age_days()
    assert age == float("inf")


def test_fallback_age_days_no_header(tmp_path):
    """A file with no # Generated: line should return infinity."""
    path = str(tmp_path / "snap.csv")
    with open(path, "w") as f:
        f.write("# Some comment\n# No timestamp here\n")
    fetcher = TLEFetcher(fallback_path=path)
    age = fetcher._fallback_age_days()
    assert age == float("inf")


# ── _refresh_fallback_if_stale ────────────────────────────────────────────────

def test_refresh_skipped_when_fresh(tmp_path, capsys):
    """No regeneration should happen when the snapshot is < 7 days old."""
    path = str(tmp_path / "snap.csv")
    now = datetime.datetime.now(datetime.timezone.utc)
    _write_snapshot_header(path, now - datetime.timedelta(hours=6))

    fetcher = TLEFetcher(fallback_path=path)
    original_mtime = os.path.getmtime(path)

    fetcher._refresh_fallback_if_stale()

    # File should not have been modified
    assert os.path.getmtime(path) == original_mtime
    captured = capsys.readouterr()
    assert "regenerating" not in captured.out


def test_refresh_triggered_when_stale(tmp_path):
    """Regeneration should fire when snapshot is older than FALLBACK_MAX_AGE_DAYS."""
    path = str(tmp_path / "snap.csv")
    now = datetime.datetime.now(datetime.timezone.utc)
    stale_ts = now - datetime.timedelta(days=FALLBACK_MAX_AGE_DAYS + 1)
    _write_snapshot_header(path, stale_ts)

    fetcher = TLEFetcher(fallback_path=path)
    fetcher._refresh_fallback_if_stale()

    # After refresh the snapshot should be fresh (age < 1 day)
    new_age = fetcher._fallback_age_days()
    assert new_age < 1.0, f"After regeneration, age should be < 1 day, got {new_age}"

    # The new file should contain actual TLE data (not just a header)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    non_comment_lines = [
        l for l in content.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    assert len(non_comment_lines) > 0, "Regenerated file contains no TLE data"


# ── generate_fallback_snapshot ────────────────────────────────────────────────

def test_generate_fallback_snapshot_produces_valid_file(tmp_path):
    """generate_fallback_snapshot() must produce a non-empty, parseable file."""
    import sys
    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from generate_tle_fallback import generate_fallback_snapshot

    out_path = str(tmp_path / "snap.csv")
    count = generate_fallback_snapshot(out_path, verbose=False)

    assert count > 0, "Expected at least one valid TLE object"
    assert os.path.exists(out_path)

    with open(out_path, encoding="utf-8") as f:
        content = f.read()

    # Header must have a fresh # Generated: timestamp
    m = _GENERATED_RE.search(content)
    assert m is not None, "Missing # Generated: header in output file"

    # Must have parseable TLE lines
    from pipeline.map.tle_fetcher import _parse_tle_lines
    lines = _parse_tle_lines(content)
    assert len(lines) > 0


def test_generate_fallback_snapshot_leos_in_bands(tmp_path):
    """Objects in the generated snapshot must propagate into valid LEO bands."""
    import sys
    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from generate_tle_fallback import generate_fallback_snapshot

    out_path = str(tmp_path / "snap.csv")
    generate_fallback_snapshot(out_path, verbose=False)

    from pipeline.map.tle_fetcher import _parse_tle_lines
    from pipeline.map.density import compute_density

    with open(out_path, encoding="utf-8") as f:
        raw = f.read()

    lines = _parse_tle_lines(raw)
    df = compute_density(lines)

    total = int(df["object_count"].sum())
    assert total > 0, (
        f"Expected >0 objects in LEO bands after regeneration, got {total}. "
        "TLE epoch may still be misformatted."
    )
    # ISS-altitude band must be populated
    iss = df[df["altitude_band_km"] == "400-600"]
    assert iss.iloc[0]["object_count"] > 0, "400-600 km band empty after regeneration"


# ── Integration: _load_fallback auto-regen ────────────────────────────────────

def test_load_fallback_auto_regenerates_stale_snapshot(tmp_path):
    """_load_fallback() must silently fix a stale snapshot and return valid lines."""
    path = str(tmp_path / "snap.csv")
    now = datetime.datetime.now(datetime.timezone.utc)
    # Write a header-only stale file (beyond the age limit)
    _write_snapshot_header(path, now - datetime.timedelta(days=FALLBACK_MAX_AGE_DAYS + 2))

    fetcher = TLEFetcher(fallback_path=path)
    lines = fetcher._load_fallback()

    # Must return actual TLE content, not just nothing
    assert len(lines) > 0, "_load_fallback returned empty list after auto-regen"

    # Snapshot on disk should now be fresh
    new_age = fetcher._fallback_age_days()
    assert new_age < 1.0
