"""
pipeline/prioritize/scorer.py — Hohmann Delta-V Costing + Risk Tier Assignment

Computes the delta-v cost to reach each LEO altitude band from a reference
orbit, weighs it against the risk removed, and assigns priority tiers.

THIS IS REGION-LEVEL COSTING ONLY.
Stage 1 gives size/shape/rotation — not a precise state vector.
Stage 2 works at region level. Object-level delta-v is not supportable
by the underlying data and is not computed here.

Delta-V Computation
-------------------
We use a two-burn Hohmann transfer from a reference circular orbit (default 400 km)
to the midpoint of each altitude band, plus a worst-case plane-change component.

Reference orbit:    a1 = R_EARTH + h_ref  (km)
Target orbit:       a2 = R_EARTH + h_mid  (km)  where h_mid = (band_lo + band_hi) / 2
Transfer orbit:     a_t = (a1 + a2) / 2

First burn (departure):
    dv1 = sqrt(GM/a1) * (sqrt(2*a2 / (a1+a2)) - 1)  [km/s]

Second burn (arrival):
    dv2 = sqrt(GM/a2) * (1 - sqrt(2*a1 / (a1+a2)))  [km/s]

Hohmann delta-v:    dv_hohmann = |dv1| + |dv2|  → converted to m/s

Plane change (at apoapsis of transfer orbit, worst case 5°):
    v_apoapsis = sqrt(GM * (2/a2 - 1/a_t))          [km/s]
    dv_plane   = 2 * v_apoapsis * sin(delta_i / 2)  [km/s]
    (converted to m/s)

Total:  dv_total = dv_hohmann + dv_plane  [m/s]

Severity Index
--------------
    severity = size_weight * kinetic_energy_proxy
    size_weight: small=1.0, medium=2.5, large=5.0
    kinetic_energy_proxy = size_estimate_m^2 * V_REL^2  (proportional to KE)
    V_REL = 7500 m/s (mean LEO relative velocity, constant, documented)

Priority Score and Tiers
------------------------
    priority_score = composite_risk_density * severity / dv_total_ms
    (normalised to [0, 1])

    HIGH-PRIORITY : priority_score_norm >= 0.60
    MONITOR       : 0.30 <= priority_score_norm < 0.60
    LOW           : priority_score_norm < 0.30

Constants (all documented)
--------------------------
    GM         = 398600.4418 km³/s²  (Earth gravitational parameter)
    R_EARTH    = 6371.0 km           (mean Earth radius)
    V_REL      = 7500 m/s            (mean LEO relative velocity)
    h_ref      = 400 km (default)    (reference orbit altitude)
    delta_i    = 5.0°  (default)     (worst-case plane change)
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

# ── Physical constants ────────────────────────────────────────────────────────
GM        = 398600.4418    # km³/s²  — Earth gravitational parameter
R_EARTH   = 6371.0         # km      — mean Earth radius
V_REL_MS  = 7500.0         # m/s     — mean LEO relative velocity (debris vs. active)

# ── Severity weights by size class ───────────────────────────────────────────
_SIZE_WEIGHTS = {"small": 1.0, "medium": 2.5, "large": 5.0}

# ── Size class to midpoint size estimate (m) ─────────────────────────────────
_SIZE_TO_M = {"small": 0.02, "medium": 0.05, "large": 0.085}

# ── Tier thresholds (applied to normalised priority score) ───────────────────
TIER_HIGH_MIN    = 0.60
TIER_MONITOR_MIN = 0.30

# ── Default priority parameters ──────────────────────────────────────────────
H_REF_KM_DEFAULT      = 400.0    # km — reference orbit altitude
PLANE_CHANGE_DEG_DEFAULT = 5.0   # degrees — worst-case plane change assumption


def hohmann_dv_ms(
    h_ref_km: float,
    h_target_km: float,
    plane_change_deg: float = PLANE_CHANGE_DEG_DEFAULT,
) -> dict:
    """Compute Hohmann transfer + plane-change delta-v in m/s.

    Parameters
    ----------
    h_ref_km : float
        Altitude of the reference (departure) circular orbit in km.
    h_target_km : float
        Altitude of the target circular orbit in km.
    plane_change_deg : float
        Plane-change angle in degrees applied at apoapsis.

    Returns
    -------
    dict with keys:
        dv_hohmann_ms   : float — Hohmann transfer delta-v (m/s)
        dv_plane_ms     : float — Plane-change delta-v (m/s)
        dv_total_ms     : float — Total delta-v (m/s)
        a1_km, a2_km, a_t_km : float — semi-major axes (km)
    """
    a1 = R_EARTH + h_ref_km
    a2 = R_EARTH + h_target_km
    a_t = (a1 + a2) / 2.0

    # Hohmann burns (km/s)
    dv1 = math.sqrt(GM / a1) * (math.sqrt(2 * a2 / (a1 + a2)) - 1.0)
    dv2 = math.sqrt(GM / a2) * (1.0 - math.sqrt(2 * a1 / (a1 + a2)))
    dv_hohmann_kms = abs(dv1) + abs(dv2)

    # Plane change at apoapsis (km/s)
    v_apoapsis = math.sqrt(GM * (2.0 / a2 - 1.0 / a_t))
    delta_i_rad = math.radians(plane_change_deg)
    dv_plane_kms = 2.0 * v_apoapsis * math.sin(delta_i_rad / 2.0)

    dv_total_kms = dv_hohmann_kms + dv_plane_kms

    return {
        "dv_hohmann_ms": dv_hohmann_kms * 1000.0,
        "dv_plane_ms":   dv_plane_kms   * 1000.0,
        "dv_total_ms":   dv_total_kms   * 1000.0,
        "a1_km":  a1,
        "a2_km":  a2,
        "a_t_km": a_t,
    }


def severity_index(size_class: str = "medium") -> float:
    """Compute a dimensionless severity index for a given size class.

    severity = size_weight * (size_estimate_m^2 * V_REL_MS^2)
    Normalised to size_class "large" = 1.0.

    The relative velocity V_REL_MS is constant (7500 m/s mean LEO, documented).
    """
    size_m  = _SIZE_TO_M.get(size_class, _SIZE_TO_M["medium"])
    weight  = _SIZE_WEIGHTS.get(size_class, _SIZE_WEIGHTS["medium"])
    ke_proxy = (size_m ** 2) * (V_REL_MS ** 2)
    severity = weight * ke_proxy
    # Normalise by the max possible (large)
    max_severity = _SIZE_WEIGHTS["large"] * (_SIZE_TO_M["large"] ** 2) * (V_REL_MS ** 2)
    return severity / max_severity


class PriorityScorer:
    """Scores altitude bands by cost-weighted risk and assigns priority tiers.

    This operates on altitude-band DataFrames only.
    Object-level delta-v is not computed — see module docstring.

    Usage
    -----
    scorer = PriorityScorer()
    priority_df = scorer.score(risk_df)
    PriorityScorer.export_csv(priority_df, "output.csv")
    PriorityScorer.export_json(priority_df, "output.json")
    """

    def __init__(
        self,
        h_ref_km: float = H_REF_KM_DEFAULT,
        plane_change_deg: float = PLANE_CHANGE_DEG_DEFAULT,
        dominant_size_class: str = "medium",
    ):
        self.h_ref_km          = h_ref_km
        self.plane_change_deg  = plane_change_deg
        self.dominant_size_class = dominant_size_class

    def score(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        """Compute priority scores and assign tiers for each altitude band.

        Parameters
        ----------
        risk_df : pd.DataFrame
            Output of RiskDensityMap.compute().
            Required columns: altitude_band_km, band_label, band_lo_km, band_hi_km,
                              composite_risk_density.

        Returns
        -------
        pd.DataFrame sorted descending by priority_score, with columns:
            rank, band_label, altitude_km_mid, composite_risk_density,
            severity_index, dv_hohmann_ms, dv_plane_ms, dv_total_ms,
            priority_score, tier, explanation_text
        """
        sev = severity_index(self.dominant_size_class)
        rows = []

        for _, row in risk_df.iterrows():
            h_mid = (row["band_lo_km"] + row["band_hi_km"]) / 2.0
            dv = hohmann_dv_ms(self.h_ref_km, h_mid, self.plane_change_deg)

            # Raw priority = risk * severity / delta-v
            # Add small epsilon to prevent division-by-zero
            raw = (row["composite_risk_density"] * sev) / (dv["dv_total_ms"] + 1e-6)

            rows.append({
                "altitude_band_km":      row["altitude_band_km"],
                "band_label":            row["band_label"],
                "altitude_km_mid":       h_mid,
                "composite_risk_density": row["composite_risk_density"],
                "severity_index":        sev,
                "dv_hohmann_ms":         dv["dv_hohmann_ms"],
                "dv_plane_ms":           dv["dv_plane_ms"],
                "dv_total_ms":           dv["dv_total_ms"],
                "raw_priority":          raw,
            })

        df = pd.DataFrame(rows)

        # Normalise priority to [0, 1]
        max_raw = df["raw_priority"].max()
        if max_raw > 0:
            df["priority_score"] = df["raw_priority"] / max_raw
        else:
            df["priority_score"] = 0.0

        # Assign tiers
        df["tier"] = df["priority_score"].apply(_assign_tier)

        # Sort descending by priority
        df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))

        # Plain-English explanation per band
        df["explanation_text"] = df.apply(_make_explanation, axis=1)

        # Drop the raw column from output
        df = df.drop(columns=["raw_priority"])

        return df

    @staticmethod
    def export_csv(df: pd.DataFrame, path: str) -> None:
        """Export the priority DataFrame to CSV with delta-v cost on every entry."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def export_json(df: pd.DataFrame, path: str) -> None:
        """Export the priority DataFrame to JSON with delta-v cost on every entry."""
        import os, json
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        df.to_json(path, orient="records", indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assign_tier(score: float) -> str:
    if score >= TIER_HIGH_MIN:
        return "HIGH-PRIORITY"
    elif score >= TIER_MONITOR_MIN:
        return "MONITOR"
    else:
        return "LOW"


def _make_explanation(row) -> str:
    tier   = row["tier"]
    band   = row["band_label"]
    dv     = row["dv_total_ms"]
    risk   = row["composite_risk_density"]
    ps     = row["priority_score"]
    return (
        f"{tier} — {band}: risk density {risk:.3f}, "
        f"priority score {ps:.3f}, "
        f"requires ~{dv:.0f} m/s total delta-v "
        f"(Hohmann {row['dv_hohmann_ms']:.0f} + plane-change {row['dv_plane_ms']:.0f} m/s)."
    )
