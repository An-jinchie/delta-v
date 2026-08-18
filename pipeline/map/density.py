"""
pipeline/map/density.py — SGP4 Propagation + Altitude Band Density

Propagates each TLE object to the current UTC epoch using SGP4,
computes its altitude above the WGS-84 reference sphere, and bins
objects into LEO altitude bands.

Altitude bands
--------------
200–400 km, 400–600 km, 600–800 km, 800–1000 km,
1000–1200 km, 1200–1400 km, 1400–1600 km, 1600–2000 km

Objects below 200 km or above 2000 km are excluded (not LEO of interest).

Altitude computation
--------------------
SGP4 returns position in Earth-centred inertial (ECI) coordinates in km.
Altitude above the reference sphere:

    altitude_km = |r| - R_EARTH_KM

where R_EARTH_KM = 6371.0 (mean Earth radius).  This is a simplified
spherical approximation — adequate for altitude-band binning.

Output DataFrame columns
------------------------
    altitude_band_km   : str   — e.g. "200-400"
    band_label         : str   — e.g. "200–400 km"
    band_lo_km         : float — lower edge of band
    band_hi_km         : float — upper edge of band
    object_count       : int   — number of objects in this band at epoch
    band_width_km      : float — hi - lo
    density_per_km     : float — object_count / band_width_km
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

# ── Constants ──────────────────────────────────────────────────────────────────
R_EARTH_KM = 6371.0   # mean Earth radius in km (spherical approximation)

# LEO altitude bands (lower_km, upper_km)
ALTITUDE_BANDS: List[Tuple[int, int]] = [
    (200,  400),
    (400,  600),
    (600,  800),
    (800,  1000),
    (1000, 1200),
    (1200, 1400),
    (1400, 1600),
    (1600, 2000),
]

ALT_MIN_KM = ALTITUDE_BANDS[0][0]
ALT_MAX_KM = ALTITUDE_BANDS[-1][1]


def compute_density(tle_lines: List[str]) -> pd.DataFrame:
    """Propagate TLE objects to current UTC and compute altitude-band density.

    Parameters
    ----------
    tle_lines : list[str]
        Raw TLE lines as returned by TLEFetcher.fetch().
        May include name lines (non-TLE lines that don't start with 1 or 2).

    Returns
    -------
    pd.DataFrame with columns:
        altitude_band_km, band_label, band_lo_km, band_hi_km,
        object_count, band_width_km, density_per_km
    All bands are always present (even if count = 0).
    """
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day,
                  now.hour, now.minute, now.second + now.microsecond / 1e6)

    # Parse TLE pairs from the line list
    tle_pairs = _extract_tle_pairs(tle_lines)

    # Propagate each object and collect altitudes
    altitudes_km: List[float] = []
    errors = 0

    for line1, line2 in tle_pairs:
        try:
            sat = Satrec.twoline2rv(line1, line2)
            e, r, _ = sat.sgp4(jd, fr)
            if e != 0:
                # SGP4 error codes: 1=mean eccentricity, 2=mean motion, etc.
                errors += 1
                continue
            alt = float(np.linalg.norm(r)) - R_EARTH_KM
            if ALT_MIN_KM <= alt <= ALT_MAX_KM:
                altitudes_km.append(alt)
        except Exception:
            errors += 1
            continue

    if errors > 0:
        print(f"[density] {errors} TLE propagation errors (skipped)")

    # Build band edges for pd.cut
    band_edges = [lo for lo, hi in ALTITUDE_BANDS] + [ALTITUDE_BANDS[-1][1]]
    band_labels = [f"{lo}-{hi}" for lo, hi in ALTITUDE_BANDS]

    if altitudes_km:
        alt_series = pd.Series(altitudes_km)
        binned = pd.cut(
            alt_series,
            bins=band_edges,
            labels=band_labels,
            right=True,
            include_lowest=True,
        )
        counts = binned.value_counts().reindex(band_labels, fill_value=0)
    else:
        counts = pd.Series(0, index=band_labels)

    # Build output DataFrame — all bands always present
    rows = []
    for (lo, hi), label in zip(ALTITUDE_BANDS, band_labels):
        width = hi - lo
        count = int(counts[label])
        rows.append({
            "altitude_band_km": label,
            "band_label":       f"{lo}\u2013{hi} km",
            "band_lo_km":       float(lo),
            "band_hi_km":       float(hi),
            "object_count":     count,
            "band_width_km":    float(width),
            "density_per_km":   count / width,
        })

    df = pd.DataFrame(rows)
    print(
        f"[density] Propagated {len(tle_pairs)} TLE pairs, "
        f"{len(altitudes_km)} in LEO bands, "
        f"epoch {now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_tle_pairs(lines: List[str]) -> List[Tuple[str, str]]:
    """Extract (line1, line2) TLE pairs from a list of raw lines.

    Handles three-line format (name + line1 + line2) and two-line format.
    A line is a TLE line 1 if it starts with "1 " and has length 69–71.
    A line is a TLE line 2 if it starts with "2 " and has length 69–71.
    """
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_tle_line1(line):
            if i + 1 < len(lines) and _is_tle_line2(lines[i + 1]):
                pairs.append((line, lines[i + 1]))
                i += 2
                continue
        i += 1
    return pairs


def _is_tle_line1(line: str) -> bool:
    return (
        len(line) >= 69
        and line[0] == "1"
        and line[1] == " "
    )


def _is_tle_line2(line: str) -> bool:
    return (
        len(line) >= 69
        and line[0] == "2"
        and line[1] == " "
    )
