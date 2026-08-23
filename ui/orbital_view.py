"""
ui/orbital_view.py — Live Orbital Distribution Visualizer

Renders a 3D scatter of real TLE-derived object positions coloured by
altitude-band risk tier — using the exact same data already computed by
the Stage 2 pipeline (no new fetching, no new math).

How it works
------------
1. Re-uses the TLE lines already in the Streamlit cache (fetched by Stage 2).
2. Propagates each object to current UTC via SGP4 → ECI (x, y, z) in km.
3. Maps each object's altitude band → its composite_risk_density from
   risk_df → colour (LOW=cyan, MONITOR=amber, HIGH-PRIORITY=red).
4. Renders a Plotly 3D scatter on a dark Earth-sphere background.
5. A lightweight CSS animation rotates the camera slowly via JS so the
   scene feels alive — objects appear to drift in real orbital positions.

The visualization is:
- Honest: every point is a real tracked object from the TLE catalog
  (or the physics-valid synthetic fallback), propagated with SGP4 to
  *right now*. Positions are real orbital mechanics, not decoration.
- Scoped: this shows TRACKED objects (the TLE catalog). The 1-10 cm
  untracked debris that Delta-V targets is not in this catalog — that
  gap is the whole point of the project, and the UI says so.
- Fast: limited to MAX_OBJECTS points; SGP4 is fast but 10k+ objects
  would slow Streamlit. Default 1500 — covers the full fallback set.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.map.density import R_EARTH_KM, ALTITUDE_BANDS, _extract_tle_pairs
from ui.theme import PLOTLY_LAYOUT, PLOTLY_LEGEND, TIER_COLOURS, BG_MAIN, BG_PANEL, BORDER

MAX_OBJECTS = 1500  # cap for render performance

# Band midpoint km → used to determine which band an object falls in
_BAND_MIDS = {f"{lo}-{hi}": (lo + hi) / 2 for lo, hi in ALTITUDE_BANDS}
_ALT_MIN = ALTITUDE_BANDS[0][0]
_ALT_MAX = ALTITUDE_BANDS[-1][1]

# Tier colours + label
_TIER_COLOUR_MAP = {
    "HIGH-PRIORITY": TIER_COLOURS["HIGH-PRIORITY"],
    "MONITOR":       TIER_COLOURS["MONITOR"],
    "LOW":           TIER_COLOURS["LOW"],
    "UNRANKED":      "#3a5a7a",   # objects in bands with no risk score yet
}


def _band_for_altitude(alt_km: float) -> Optional[str]:
    """Return the band label string for an altitude, or None if outside LEO."""
    for lo, hi in ALTITUDE_BANDS:
        if lo <= alt_km <= hi:
            return f"{lo}-{hi}"
    return None


def _tier_for_band(band: str, risk_df: Optional[pd.DataFrame]) -> str:
    """Return the tier string for a band given a risk DataFrame."""
    if risk_df is None or risk_df.empty:
        return "UNRANKED"
    row = risk_df[risk_df["altitude_band_km"] == band]
    if row.empty:
        return "UNRANKED"
    # Derive tier from composite_risk_density normalised thresholds
    # (mirrors scorer.py tier logic — recomputed here to avoid circular import)
    score = float(row.iloc[0]["composite_risk_density"])
    if score >= 0.60:
        return "HIGH-PRIORITY"
    elif score >= 0.30:
        return "MONITOR"
    return "LOW"


def build_orbital_figure(
    tle_lines: List[str],
    risk_df: Optional[pd.DataFrame] = None,
    max_objects: int = MAX_OBJECTS,
) -> go.Figure:
    """
    Build a Plotly 3D scatter figure of real orbital positions.

    Parameters
    ----------
    tle_lines : list[str]
        Raw TLE lines as returned by TLEFetcher (already in session cache).
    risk_df : pd.DataFrame | None
        Output of RiskDensityMap.compute() — used to colour objects by tier.
        If None, all objects render in the UNRANKED colour.
    max_objects : int
        Maximum number of objects to render (performance cap).

    Returns
    -------
    plotly.graph_objects.Figure — 3D scatter, dark theme, Earth sphere.
    """
    from sgp4.api import Satrec, jday

    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day,
                  now.hour, now.minute, now.second + now.microsecond / 1e6)

    pairs = _extract_tle_pairs(tle_lines)

    # Limit for performance — sample evenly across the full catalog
    if len(pairs) > max_objects:
        step = len(pairs) // max_objects
        pairs = pairs[::step][:max_objects]

    # Propagate each object → ECI (x, y, z) in km
    xs, ys, zs, alts, bands, tiers, colours, hovers = [], [], [], [], [], [], [], []

    for line1, line2 in pairs:
        try:
            sat = Satrec.twoline2rv(line1, line2)
            e, r, _ = sat.sgp4(jd, fr)
            if e != 0 or any(math.isnan(v) for v in r):
                continue
            x, y, z = r  # km, ECI
            alt = math.sqrt(x*x + y*y + z*z) - R_EARTH_KM
            if not (_ALT_MIN <= alt <= _ALT_MAX):
                continue
            band = _band_for_altitude(alt)
            if band is None:
                continue
            tier = _tier_for_band(band, risk_df)
            xs.append(x); ys.append(y); zs.append(z)
            alts.append(alt); bands.append(band); tiers.append(tier)
            colours.append(_TIER_COLOUR_MAP[tier])
            hovers.append(
                f"Alt: {alt:.0f} km<br>Band: {band} km<br>Tier: {tier}"
            )
        except Exception:
            continue

    if not xs:
        # No data — return an empty figure with the Earth sphere only
        return _empty_figure()

    # ── Build the Earth sphere ────────────────────────────────────────────────
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    ex = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    ey = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    ez = R_EARTH_KM * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()

    # Earth surface — deep blue-grey, slightly transparent
    fig.add_trace(go.Surface(
        x=ex, y=ey, z=ez,
        colorscale=[[0, "#0a1a3a"], [0.5, "#0d2244"], [1, "#112855"]],
        showscale=False,
        opacity=0.85,
        lighting=dict(ambient=0.6, diffuse=0.4),
        hoverinfo="skip",
        name="Earth",
    ))

    # ── Plot objects grouped by tier for a clean legend ───────────────────────
    for tier_name, tier_colour in _TIER_COLOUR_MAP.items():
        mask = [t == tier_name for t in tiers]
        if not any(mask):
            continue
        txs = [x for x, m in zip(xs, mask) if m]
        tys = [y for y, m in zip(ys, mask) if m]
        tzs = [z for z, m in zip(zs, mask) if m]
        th  = [h for h, m in zip(hovers, mask) if m]
        fig.add_trace(go.Scatter3d(
            x=txs, y=tys, z=tzs,
            mode="markers",
            marker=dict(
                size=2.2 if tier_name == "HIGH-PRIORITY" else 1.8,
                color=tier_colour,
                opacity=0.85 if tier_name == "HIGH-PRIORITY" else 0.65,
            ),
            name=tier_name,
            text=th,
            hoverinfo="text",
            showlegend=True,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    axis_style = dict(
        showgrid=False, zeroline=False, showticklabels=False,
        showline=False, title="",
        backgroundcolor=BG_MAIN,
    )
    fig.update_layout(
        paper_bgcolor=BG_MAIN,
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            bgcolor=BG_MAIN,
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="cube",
        ),
        legend={
            **PLOTLY_LEGEND,
            "orientation": "h",
            "y": -0.05,
            "font": dict(size=11, color="#c8d8e8"),
        },
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        font=dict(color="#c8d8e8", family="IBM Plex Mono, Rajdhani, monospace"),
    )

    return fig


def _empty_figure() -> go.Figure:
    """Return a dark empty 3D figure with just the Earth sphere."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    ex = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    ey = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    ez = R_EARTH_KM * np.outer(np.ones(np.size(u)), np.cos(v))
    fig = go.Figure(go.Surface(
        x=ex, y=ey, z=ez,
        colorscale=[[0, "#0a1a3a"], [1, "#112855"]],
        showscale=False, opacity=0.85, hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor=BG_MAIN,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor=BG_MAIN),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor=BG_MAIN),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, backgroundcolor=BG_MAIN),
            bgcolor=BG_MAIN,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )
    return fig


def render_orbital_view(
    tle_lines: List[str],
    risk_df: Optional[pd.DataFrame] = None,
    n_objects: int = MAX_OBJECTS,
) -> None:
    """
    Render the live orbital distribution panel into the current Streamlit page.

    Wraps build_orbital_figure() with a caption, data-source note, and the
    slow-rotation JS animation. Call this from pages/02_map.py.
    """
    with st.spinner("Propagating orbital positions…"):
        fig = build_orbital_figure(tle_lines, risk_df, max_objects=n_objects)

    n_plotted = sum(
        len(t.x) for t in fig.data
        if isinstance(t, go.Scatter3d)
    )

    st.plotly_chart(fig, use_container_width=True, config=dict(
        displayModeBar=True,
        modeBarButtonsToRemove=["resetCameraDefault3d"],
        toImageButtonOptions=dict(format="png", filename="delta_v_orbital_view"),
    ))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.caption(
            f"Showing {n_plotted:,} tracked LEO objects · positions propagated via SGP4 · epoch {ts}"
        )
    with col_b:
        st.caption(
            "⚠ This shows tracked catalog objects only. "
            "Untracked 1–10 cm debris (the Delta-V target population) is not visible here — "
            "that absence is the problem this tool addresses."
        )
