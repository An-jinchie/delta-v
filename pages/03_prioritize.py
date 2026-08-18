"""
pages/03_prioritize.py — Stage 3: Delta-V Costed Prioritization

User workflow:
  1. Takes the risk-density map from Stage 2 (or computes one on demand)
  2. Applies Hohmann transfer + plane-change delta-v costing per altitude band
  3. Computes priority score: risk_density × severity ÷ delta_v
  4. Assigns tiers: HIGH-PRIORITY / MONITOR / LOW
  5. IBM Granite writes a mission brief from the computed numbers
  6. Export full results as CSV or JSON — delta-v on every entry

SCOPING NOTE (shown in UI):
  Delta-v costing is at the REGION/ALTITUDE-BAND LEVEL ONLY.
  Stage 1 gives size/shape/rotation — not a state vector.
  Object-level delta-v is not supported by the underlying data.

Math is deterministic. No model guesses any value.
"""

import io
import json
import math

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from ai.granite import write_mission_brief, granite_status
from pipeline.map.density import compute_density, ALTITUDE_BANDS
from pipeline.map.risk_map import RiskDensityMap
from pipeline.map.tle_fetcher import TLEFetcher
from pipeline.prioritize.scorer import (
    PriorityScorer,
    hohmann_dv_ms,
    severity_index,
    TIER_HIGH_MIN,
    TIER_MONITOR_MIN,
    H_REF_KM_DEFAULT,
    PLANE_CHANGE_DEG_DEFAULT,
    GM,
    R_EARTH,
    V_REL_MS,
)

st.set_page_config(page_title="Delta-V · Prioritize", page_icon="🛰️", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Stage 3 · Prioritize**")
    st.caption(
        "Region-level delta-v costing only. "
        "Stage 1 gives no state vector — object-level delta-v is not supportable."
    )
    gs = granite_status()
    if gs["available"]:
        st.success(f"Granite: {gs['status_text']}", icon="🤖")
    else:
        st.info(f"Granite: {gs['status_text']}", icon="🤖")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("Stage 3 · Prioritize")
st.caption(
    "Each high-risk altitude band is **costed** with Hohmann transfer + plane-change "
    "delta-v, then ranked by `risk × severity ÷ delta_v`. "
    "Outputs tiers (HIGH-PRIORITY / MONITOR / LOW) with delta-v on every entry."
)

st.error(
    "**Scoping constraint:** Delta-v costing is at the **region / altitude-band level only** — "
    "not per individual object. Stage 1 produces size/shape/rotation (no state vector). "
    "Stage 2 works at regional level. Object-level delta-v is not supported by the underlying data.",
    icon="🚀",
)

st.divider()

# ── Scorer configuration ──────────────────────────────────────────────────────
st.subheader("1 · Scoring Parameters")

cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

with cfg_col1:
    h_ref = st.number_input(
        "Reference orbit altitude (km)",
        min_value=200.0, max_value=800.0, value=float(H_REF_KM_DEFAULT), step=10.0,
        help="Departure orbit altitude. Default: 400 km (near-ISS).",
    )

with cfg_col2:
    plane_change_deg = st.slider(
        "Plane change (degrees)",
        min_value=0.0, max_value=15.0, value=float(PLANE_CHANGE_DEG_DEFAULT), step=0.5,
        help="Worst-case plane change angle assumed. Increases total delta-v.",
    )

with cfg_col3:
    dominant_size = st.selectbox(
        "Dominant debris size class",
        ["small", "medium", "large"],
        index=1,
        help="Size class used for severity index calculation across all bands.",
    )

run_btn = st.button("⚡ Compute Priority Scores", type="primary", use_container_width=True)

st.divider()

# ── Get or build risk map ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading TLE data…")
def _get_fetcher():
    return TLEFetcher()


@st.cache_data(show_spinner="Computing density…", ttl=3600)
def _get_density():
    fetcher = _get_fetcher()
    tle_lines = fetcher.fetch()
    return compute_density(tle_lines)


def _get_risk_df():
    """Return risk_df from session state, or compute a fresh one."""
    cached = st.session_state.get("risk_map_df")
    if cached is not None and not cached.empty:
        return cached, "from Stage 2 session"

    # Compute a fresh map with no injected detections
    try:
        density_df = _get_density()
    except Exception as e:
        st.error(f"Could not load TLE data: {e}")
        st.stop()

    risk_map = RiskDensityMap()
    risk_df = risk_map.compute(density_df)
    return risk_df, "fresh (no detections injected)"


# ── Run scoring ───────────────────────────────────────────────────────────────
if run_btn or st.session_state.get("priority_df") is None:
    risk_df, risk_source = _get_risk_df()
    scorer = PriorityScorer(
        h_ref_km=h_ref,
        plane_change_deg=plane_change_deg,
        dominant_size_class=dominant_size,
    )
    priority_df = scorer.score(risk_df)
    st.session_state["priority_df"] = priority_df
    st.session_state["granite_brief"] = None   # invalidate cached brief
    st.toast(f"Scored {len(priority_df)} altitude bands.", icon="✅")
else:
    risk_df, risk_source = _get_risk_df()
    priority_df = st.session_state["priority_df"]

if priority_df is None or priority_df.empty:
    st.info("Click '⚡ Compute Priority Scores' to run.", icon="👆")
    st.stop()

st.subheader("2 · Priority Results")
st.caption(f"Risk map source: **{risk_source}**")

# ── Tier summary cards ────────────────────────────────────────────────────────
high_df    = priority_df[priority_df["tier"] == "HIGH-PRIORITY"]
monitor_df = priority_df[priority_df["tier"] == "MONITOR"]
low_df     = priority_df[priority_df["tier"] == "LOW"]

card1, card2, card3 = st.columns(3)
with card1:
    st.metric(
        "🔴 HIGH-PRIORITY",
        len(high_df),
        help=f"Priority score ≥ {TIER_HIGH_MIN}",
    )
with card2:
    st.metric(
        "🟡 MONITOR",
        len(monitor_df),
        help=f"Priority score {TIER_MONITOR_MIN}–{TIER_HIGH_MIN}",
    )
with card3:
    st.metric(
        "🔵 LOW",
        len(low_df),
        help=f"Priority score < {TIER_MONITOR_MIN}",
    )

# ── Priority vs delta-v scatter ───────────────────────────────────────────────
_TIER_COLOURS = {
    "HIGH-PRIORITY": "#e05555",
    "MONITOR":       "#f5a623",
    "LOW":           "#3b82d4",
}
plot_df = priority_df.copy()
plot_df["colour"] = plot_df["tier"].map(_TIER_COLOURS)
plot_df["hover"] = plot_df.apply(
    lambda r: (
        f"<b>{r['band_label']}</b><br>"
        f"Tier: {r['tier']}<br>"
        f"Priority score: {r['priority_score']:.4f}<br>"
        f"Δv total: {r['dv_total_ms']:.0f} m/s<br>"
        f"Risk density: {r['composite_risk_density']:.4f}<br>"
        f"Severity: {r['severity_index']:.4f}"
    ),
    axis=1,
)

fig_scatter = go.Figure()
for tier, colour in _TIER_COLOURS.items():
    sub = plot_df[plot_df["tier"] == tier]
    if sub.empty:
        continue
    fig_scatter.add_trace(go.Scatter(
        x=sub["dv_total_ms"],
        y=sub["priority_score"],
        mode="markers+text",
        name=tier,
        marker=dict(color=colour, size=14, symbol="circle"),
        text=sub["band_label"],
        textposition="top center",
        hovertext=sub["hover"],
        hoverinfo="text",
    ))

fig_scatter.update_layout(
    title="Priority score vs. delta-v cost per altitude band",
    xaxis_title="Total delta-v (m/s)  ←  cheaper",
    yaxis_title="Priority score  ↑  higher risk",
    height=400,
    margin=dict(l=40, r=20, t=60, b=60),
    plot_bgcolor="#f7f8fa",
    paper_bgcolor="#ffffff",
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Ranked table ──────────────────────────────────────────────────────────────
_TIER_ICONS = {"HIGH-PRIORITY": "🔴", "MONITOR": "🟡", "LOW": "🔵"}

display_df = priority_df[[
    "rank", "band_label", "tier", "priority_score",
    "composite_risk_density", "severity_index",
    "dv_hohmann_ms", "dv_plane_ms", "dv_total_ms",
]].copy()
display_df["tier"] = display_df["tier"].map(lambda t: f"{_TIER_ICONS.get(t, '')} {t}")
display_df.columns = [
    "Rank", "Band", "Tier", "Priority score",
    "Risk density", "Severity",
    "Δv Hohmann (m/s)", "Δv plane-chg (m/s)", "Δv total (m/s)",
]
for col in ["Priority score", "Risk density", "Severity"]:
    display_df[col] = display_df[col].map("{:.4f}".format)
for col in ["Δv Hohmann (m/s)", "Δv plane-chg (m/s)", "Δv total (m/s)"]:
    display_df[col] = display_df[col].map("{:.1f}".format)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Inspectable formulas ──────────────────────────────────────────────────────
with st.expander("🔍 Inspect: Delta-v formulas + current parameters"):
    sev = severity_index(dominant_size)
    st.markdown(f"""
**Delta-v computation (region-level only — not per object):**

```
a1  = R_EARTH + h_ref        = {R_EARTH} + {h_ref} = {R_EARTH + h_ref:.1f} km
a2  = R_EARTH + h_mid        (midpoint of each altitude band)
a_t = (a1 + a2) / 2

dv1 = √(GM/a1) × (√(2·a2/(a1+a2)) − 1)         [km/s]
dv2 = √(GM/a2) × (1 − √(2·a1/(a1+a2)))          [km/s]
dv_hohmann = |dv1| + |dv2|                        → m/s

v_apoapsis = √(GM × (2/a2 − 1/a_t))              [km/s]
dv_plane   = 2 × v_apoapsis × sin(Δi / 2)        [km/s, Δi = {plane_change_deg}°]

dv_total   = dv_hohmann + dv_plane                [m/s]
```

**Priority score formula:**
```
priority_score = composite_risk_density × severity_index / dv_total_ms
               (normalised to [0,1] across all bands)
```

**Severity index (dominant size: {dominant_size}):**
```
severity = size_weight × (size_m² × V_REL²) / max_possible
         = {sev:.6f}
(size_weight: small=1.0, medium=2.5, large=5.0  ·  V_REL = {V_REL_MS} m/s)
```

**Tier thresholds:**
- HIGH-PRIORITY: priority_score ≥ {TIER_HIGH_MIN}
- MONITOR: {TIER_MONITOR_MIN} ≤ priority_score < {TIER_HIGH_MIN}
- LOW: priority_score < {TIER_MONITOR_MIN}

**Constants:**  GM = {GM} km³/s²  ·  R_EARTH = {R_EARTH} km  ·  V_REL = {V_REL_MS} m/s
""")

with st.expander("🔍 Inspect: Full priority DataFrame (raw numbers)"):
    st.dataframe(priority_df, use_container_width=True, hide_index=True)

st.divider()

# ── Per-band delta-v bar chart ────────────────────────────────────────────────
st.subheader("3 · Delta-V Cost Breakdown by Band")

dv_fig = go.Figure()
dv_fig.add_trace(go.Bar(
    name="Hohmann Δv",
    x=priority_df["band_label"],
    y=priority_df["dv_hohmann_ms"],
    marker_color="#7c5cd8",
))
dv_fig.add_trace(go.Bar(
    name=f"Plane-change Δv ({plane_change_deg}°)",
    x=priority_df["band_label"],
    y=priority_df["dv_plane_ms"],
    marker_color="#f5a623",
))
dv_fig.update_layout(
    barmode="stack",
    title=f"Delta-v cost per band (from {h_ref:.0f} km reference orbit)",
    xaxis_title="Altitude band",
    yaxis_title="Delta-v (m/s)",
    height=320,
    margin=dict(l=40, r=20, t=50, b=60),
    plot_bgcolor="#f7f8fa",
    paper_bgcolor="#ffffff",
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(dv_fig, use_container_width=True)
st.caption(
    "Note: the 200–400 km band midpoint (300 km) is **below** the 400 km reference orbit — "
    "that's a downward Hohmann transfer, which also costs delta-v. "
    "Costs are not monotonically increasing below the reference altitude."
)

st.divider()

# ── IBM Granite mission brief ─────────────────────────────────────────────────
st.subheader("4 · Mission Brief (IBM Granite)")

if st.session_state.get("granite_brief") is None:
    with st.spinner("Generating mission brief…"):
        brief = write_mission_brief(priority_df)
        st.session_state["granite_brief"] = brief
else:
    brief = st.session_state["granite_brief"]

gs = granite_status()
source_label = "IBM Granite" if gs["available"] else "Fallback text (Granite unavailable)"
st.info(brief, icon="🤖")
st.caption(
    f"Source: **{source_label}** · "
    "Granite receives the computed priority scores, tiers, and delta-v costs above. "
    "It does not produce or guess any numerical values."
)

if st.button("🔄 Regenerate mission brief"):
    st.session_state["granite_brief"] = None
    st.rerun()

st.divider()

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("5 · Export")
st.caption("Download the full priority table with delta-v costs on every entry.")

export_col1, export_col2 = st.columns(2)

with export_col1:
    csv_buf = io.StringIO()
    # Write header comment for data transparency
    csv_buf.write("# DELTA-V PRIORITY EXPORT — region-level delta-v costing only, not per-object\n")
    priority_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇ Download CSV",
        data=csv_buf.getvalue(),
        file_name="delta_v_priority.csv",
        mime="text/csv",
        use_container_width=True,
    )

with export_col2:
    json_str = priority_df.to_json(orient="records", indent=2)
    st.download_button(
        "⬇ Download JSON",
        data=json_str,
        file_name="delta_v_priority.json",
        mime="application/json",
        use_container_width=True,
    )

with st.expander("📋 Preview export (first 4 rows)"):
    preview = priority_df.head(4)[[
        "rank", "band_label", "tier", "priority_score",
        "dv_hohmann_ms", "dv_plane_ms", "dv_total_ms",
        "composite_risk_density", "explanation_text",
    ]]
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.caption(
        "Every row has delta-v cost attached — built to be used, not just viewed. "
        "Explanations are plain-text summaries of the computed numbers."
    )

st.divider()
st.caption(
    "**Data transparency:** All delta-v figures are computed from Hohmann transfer orbital mechanics "
    f"(GM = {GM} km³/s², R_EARTH = {R_EARTH} km). Reference orbit: {h_ref:.0f} km. "
    f"Plane-change assumption: {plane_change_deg}° worst case. "
    "Severity uses size midpoints and mean LEO relative velocity (7500 m/s). "
    "All computations are deterministic — no AI-generated numbers."
)
