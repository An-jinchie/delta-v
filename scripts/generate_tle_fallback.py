"""
scripts/generate_tle_fallback.py — Generates a synthetic TLE fallback snapshot.

Creates syntactically valid TLE pairs using fresh epochs so SGP4 can propagate
them without divergence. Objects are distributed across all LEO altitude bands.

The core logic lives in generate_fallback_snapshot(), which is importable and
called automatically by TLEFetcher when the committed snapshot is stale (>7 days).

Run directly from the project root to regenerate manually:
    python scripts/generate_tle_fallback.py
"""

from __future__ import annotations

import datetime
import math
import os
import sys
from typing import List, Tuple

import numpy as np
from sgp4.api import Satrec, jday

# ── Physical constants ────────────────────────────────────────────────────────
R_EARTH = 6371.0        # km
MU      = 398600.4418   # km³/s²

# ── Default output path (relative to project root) ───────────────────────────
DEFAULT_OUT_PATH = "data/tle_snapshot_fallback.csv"

# Bands: (target_altitude_km, count, inclination_deg)
# Distribution roughly reflects the real LEO population.
_BAND_CONFIG: List[Tuple[float, int, float]] = [
    (300,  15, 65.0),    # 200–400 km: decaying / reentry
    (420,  80, 51.6),    # 400–600 km: ISS-family
    (540, 120, 97.6),    # 400–600 km: Starlink shell 1
    (580,  40, 53.0),    # 400–600 km: mixed
    (650,  30, 98.0),    # 600–800 km: SSO
    (720,  50, 74.0),    # 600–800 km: mixed
    (780,  25, 86.4),    # 600–800 km: near-polar
    (870,  20, 99.0),    # 800–1000 km: SSO
    (960,  15, 56.0),    # 800–1000 km
    (1100, 12, 63.0),    # 1000–1200 km
    (1300,  8, 87.0),    # 1200–1400 km
    (1500,  6, 90.0),    # 1400–1600 km
    (1800,  5, 65.0),    # 1600–2000 km
]


# ── Pure functions (no module-level side effects) ─────────────────────────────

def _checksum(line: str) -> int:
    """TLE modulo-10 checksum."""
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line) % 10


def _mean_motion_rev_per_day(alt_km: float) -> float:
    """Circular-orbit mean motion in revolutions per day."""
    a = R_EARTH + alt_km
    return math.sqrt(MU / a**3) * 86400.0 / (2 * math.pi)


def _make_tle(
    norad: int,
    alt_km: float,
    inc_deg: float,
    raan_deg: float,
    epoch_yr: int,
    epoch_day: float,
    jd_now: float,
    fr_now: float,
) -> Tuple[str, str] | None:
    """Build and SGP4-verify one TLE pair for the given parameters.

    Returns (line1, line2) if SGP4 propagation succeeds, or None if it fails.

    Parameters
    ----------
    epoch_yr   : 2-digit year (e.g. 26 for 2026)
    epoch_day  : fractional day-of-year (e.g. 230.17)
    jd_now, fr_now : Julian date at the epoch (for SGP4 verification)
    """
    mm  = _mean_motion_rev_per_day(alt_km)
    cat = str(norad).zfill(5)

    # ── Line 1 (69 chars) ─────────────────────────────────────────────────────
    # Col layout: line_no(1) sp(1) NORAD(5) U(1) sp(1) desig(8) sp(1) epoch(14)
    #             + fixed_fields(36) + checksum(1) = 69 total
    desig     = f"{epoch_yr:02d}{cat}A"           # 8 chars, e.g. "2690001A"
    epoch_str = f"{epoch_yr:02d}{epoch_day:012.8f}"  # 14 chars, e.g. "26230.17611111"

    l1 = f"1 {cat}U {desig:<8s} {epoch_str}  .00001764  00000-0  38792-4 0  9990"
    l1 = l1[:-1] + str(_checksum(l1[:-1]))

    # ── Line 2 (69 chars) ─────────────────────────────────────────────────────
    ecc_str = "0007000"   # eccentricity 0.0007 (near-circular)
    arg_p   = 17.6667     # argument of perigee (degrees)
    mean_a  = 270.0       # mean anomaly (degrees)
    rev_no  = 20248       # revolution number

    l2 = (
        f"2 {cat} {inc_deg:8.4f} {raan_deg:8.4f} {ecc_str}"
        f" {arg_p:8.4f} {mean_a:8.4f} {mm:11.8f}{rev_no:5d}0"
    )
    l2 = l2[:-1] + str(_checksum(l2[:-1]))

    # ── SGP4 verification ─────────────────────────────────────────────────────
    sat = Satrec.twoline2rv(l1, l2)
    e, r, _ = sat.sgp4(jd_now, fr_now)
    if e != 0:
        return None
    computed_alt = float(np.linalg.norm(r)) - R_EARTH
    if not np.isfinite(computed_alt):
        return None

    return l1, l2


def generate_fallback_snapshot(
    out_path: str = DEFAULT_OUT_PATH,
    verbose: bool = True,
) -> int:
    """Generate a fresh synthetic TLE fallback snapshot and write it to disk.

    Uses the current UTC time as the epoch so SGP4 propagation stays accurate.
    All generated TLEs are SGP4-verified before being written.

    Parameters
    ----------
    out_path : str
        Destination file path (created with parent directories if needed).
    verbose : bool
        Print progress and summary.

    Returns
    -------
    int
        Number of valid TLE objects written.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    jd_now, fr_now = jday(
        now.year, now.month, now.day,
        now.hour, now.minute, now.second,
    )
    epoch_yr  = now.year % 100
    epoch_day = (
        now.timetuple().tm_yday
        + (now.hour * 3600 + now.minute * 60 + now.second) / 86400.0
    )

    header = [
        "# SYNTHETIC TLE SNAPSHOT — generated for offline demo and testing",
        "# Objects distributed across all LEO altitude bands (200–2000 km)",
        f"# Generated: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "# Format matches CelesTrak TLE output. NOT real tracking data.",
        "#",
    ]
    lines: List[str] = list(header)

    norad  = 90001
    total  = 0
    errors = 0

    for alt_km, count, inc_deg in _BAND_CONFIG:
        for j in range(count):
            raan = (j * 360.0 / count) % 360.0
            result = _make_tle(
                norad, alt_km, inc_deg, raan,
                epoch_yr, epoch_day, jd_now, fr_now,
            )
            if result is None:
                errors += 1
            else:
                l1, l2 = result
                lines.append(f"SYNTH-{norad:05d}")
                lines.append(l1)
                lines.append(l2)
                total += 1
            norad += 1

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if verbose:
        print(f"[generate_tle_fallback] {total} valid TLE objects written "
              f"({errors} skipped) → {out_path}")

    return total


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """Regenerate the snapshot and print a verification summary."""
    # Ensure project root is on the path when run directly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    total = generate_fallback_snapshot(DEFAULT_OUT_PATH, verbose=True)

    # Verification: parse and propagate to confirm object counts per band
    from pipeline.map.tle_fetcher import _parse_tle_lines
    from pipeline.map.density import _extract_tle_pairs, compute_density

    with open(DEFAULT_OUT_PATH, encoding="utf-8") as f:
        raw = f.read()

    parsed = _parse_tle_lines(raw)
    pairs  = _extract_tle_pairs(parsed)
    print(f"[generate_tle_fallback] Verified: {len(pairs)} TLE pairs parseable by density module")

    df = compute_density(parsed)
    print("\nAltitude band distribution:")
    for _, row in df.iterrows():
        bar = "#" * (row["object_count"] // 3)
        print(f"  {row['band_label']:15s} {row['object_count']:4d}  {bar}")
    print(f"\nTotal in LEO bands: {df['object_count'].sum()}")


if __name__ == "__main__":
    main()
