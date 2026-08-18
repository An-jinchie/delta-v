"""
pages/02_map.py — Stage 2: LEO Risk-Density Map

User workflow:
  1. Load TLE data (from cache / fallback CSV — CelesTrak unavailable in some environments)
  2. Propagate objects via SGP4 → altitude-band density
  3. Optionally inject simulated characterised detections from Stage 1
  4. Compute composite risk-density map (recency + confidence weighted)
  5. Visualise as a bar chart — every score fully inspectable
  6. IBM Granite writes a situation report from the computed numbers

IMPORTANT SCOPING NOTE (shown in UI):
  This is a grid-based accumulator — NOT a Bayesian filter or particle filter.
  It does not track individual objects. It gives a regional risk picture only.

Data transparency:
  TLE data sourced from CelesTrak (public, no registration) or synthetic fallback.
  Both are clearly labeled in the UI.
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.granite import write_situation_report, granite_status
from pipeline.map.density import compute_density
from pipeline.map.risk_map import RiskDensityMap
from pipeline.map.tle_fetcher import TLEFetcher

st.set_page_config(page_title="Delta-V · Map", page_icon="🛰️", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Stage 2 · Map**")
    st.caption(
        "Grid-based accumulator across LEO altitude bands. "
        "**NOT** a Bayesian filter — does not track individual objects."
    )
    gs = granite_status()
    if gs["available"]:
        st.success(f"Granite: {gs['status_text']}", icon="🤖")
    else:
        st.info(f"Granite: {gs['status_text']}", icon="🤖")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("Stage 2 · Map")
st.caption(
    "Characterised detections combine with TLE-tracked object density into a "
    "**grid-based probabilistic risk-density map** across LEO altitude bands."
)

st.warning(
    "**Scope note:** This is a grid-based accumulator — NOT a Bayesian filter or "
    "particle filter. It does not track individual objects. It provides a live, "
    "evolving regional risk picture, updated with each new detection batch.",
    icon="⚠️",
)

st.divider()

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading TLE data…")
def _get_fetcher():
    return TLEFetcher()


@st.cache_data(show_spinner="Propagating TLE objects…", ttl=3600)
def _compute_density_cached():
    fetcher = _get_fetcher()
    tle_lines = fetcher.fetch()
    source = "CelesTrak (live)" if len(tle_lines) > 100 else "synthetic fallback snapshot"
    density_df = compute_density(tle_lines)
    return density_df, source, len(tle_lines)


# ── TLE loading section ───────────────────────────────────────────────────────
st.subheader("1 · TLE Data")

load_col, status_col = st.columns([2, 3])

with load_col:
    force_reload = st.button("🔄 Reload TLE data", help="Bypass cache and re-fetch")

if force_reload:
    st.cache_data.clear()

try:
    density_df, tle_source, tle_line_count = _compute_density_cached()
    tle_ok = True
except Exception as e:
    st.error(f"TLE load failed: {e}")
    tle_ok = False
    density_df = None

if tle_ok:
    with status_col:
        total_obj = int(density_df["object_count"].sum())
        st.success(
            f"✓ TLE data loaded · **{total_obj} objects** across 8 LEO bands · Source: **{tle_source}**",
            icon="🛰️",
        )

    if "celestrak" not in tle_source.lower():
        st.info(
            "⚠️ **Using synthetic TLE fallback** — CelesTrak is not reachable from this environment. "
            "The fallback contains 426 physics-valid synthetic objects with fresh epoch dates. "
            "Real results will differ once network access is available.",
            icon="⚠️",
        )

st.divider()

# ── Detection injection ───────────────────────────────────────────────────────
st.subheader("2 · Inject Characterised Detections")
st.caption(
    "Simulate detection batches to see how characterised debris updates the risk map. "
    "In a live deployment, detections would flow from Stage 1 automatically."
)

det_col1, det_col2 = st.columns([1, 2])

with det_col1:
    n_detections = st.slider("Number of synthetic detections", 0, 50, 10)
    dominant_band = st.selectbox(
        "Concentrate in band",
        ["400-600", "600-800", "800-1000", "200-400", "1000-1200",
         "1200-1400", "1400-1600", "1600-2000"],
        index=0,
        help="Most detections will land in this band to show the map updating.",
    )
    det_size = st.selectbox("Detection size class", ["small", "medium", "large"], index=1)
    det_seed = st.number_input("Detection seed", value=7, step=1)

    inject_btn = st.button("💉 Inject detections", type="primary", use_container_width=True)
    clear_btn  = st.button("🗑 Clear all detections", use_container_width=True)

_BAND_LABELS_ALL = ["200-400","400-600","600-800","800-1000",
                    "1000-1200","1200-1400","1400-1600","1600-2000"]

if "risk_map_obj" not in st.session_state:
    st.session_state["risk_map_obj"] = RiskDensityMap()

risk_map: RiskDensityMap = st.session_state["risk_map_obj"]

if clear_btn:
    risk_map.clear()
    st.session_state["risk_map_df"] = None
    st.session_state["granite_sitrep"] = None
    st.toast("Detections cleared.", icon="🗑")

if inject_btn and n_detections > 0:
    rng = np.random.default_rng(int(det_seed))
    detections = []
    now_ts = time.time()
    for i in range(n_detections):
        # 70% go to dominant band, 30% spread randomly
        band = dominant_band if rng.random() < 0.7 else rng.choice(_BAND_LABELS_ALL)
        age_hours = float(rng.uniform(0, 24))
        detections.append({
            "altitude_band_km": band,
            "confidence": float(rng.uniform(0.5, 1.0)),
            "timestamp": now_ts - age_hours * 3600,
            "size_class": det_size,
        })
    risk_map.update(detections)
    st.session_state["granite_sitrep"] = None  # invalidate cached sitrep
    st.toast(f"Injected {n_detections} detections.", icon="💉")

# Also pull any real characterization from Stage 1
char_result = st.session_state.get("characterize_result")
if char_result is not None and inject_btn:
    inv = char_result.get("inversion_result")
    size_cls = char_result.get("size_class", "medium")
    conf = char_result.get("size_confidence", 0.5)
    # Map size → most likely band heuristic (just for demo wiring)
    _SIZE_BAND = {"small": "400-600", "medium": "600-800", "large": "800-1000"}
    real_band = _SIZE_BAND.get(size_cls, "600-800")
    risk_map.update([{
        "altitude_band_km": real_band,
        "confidence": conf,
        "timestamp": time.time(),
        "size_class": size_cls,
    }])

with det_col2:
    det_count = risk_map.detection_count
    st.metric("Total detections in map", det_count)
    if det_count > 0:
        st.caption(f"Decay half-life: {risk_map.decay_half_life_days} days · "
                   f"Weights: TLE density {risk_map.w1:.0%}, detections {risk_map.w2:.0%}")

st.divider()

# ── Compute risk map ──────────────────────────────────────────────────────────
if not tle_ok or density_df is None:
    st.error("Cannot compute risk map without TLE data.")
    st.stop()

st.subheader("3 · Composite Risk-Density Map")

risk_df = risk_map.compute(density_df)
st.session_state["risk_map_df"] = risk_df
st.session_state["tle_last_updated"] = datetime.now(timezone.utc).isoformat()
st.session_state["tle_object_count"] = int(density_df["object_count"].sum())

# ── Bar chart ─────────────────────────────────────────────────────────────────
_TIER_COLOUR = {
    "HIGH-PRIORITY": "#e05555",
    "MONITOR":       "#f5a623",
    "LOW":           "#3b82d4",
}

bands     = risk_df["band_label"].tolist()
tle_norm  = risk_df["density_per_km"] / max(risk_df["density_per_km"].max(), 1e-12)
det_norm  = (risk_df["detection_confidence_weighted"] /
             max(risk_df["detection_confidence_weighted"].max(), 1e-12))
composite = risk_df["composite_risk_density"].tolist()

fig = go.Figure()
fig.add_trace(go.Bar(
    name="TLE density (norm)",
    x=bands,
    y=tle_norm.tolist(),
    marker_color="#7c5cd8",
    opacity=0.6,
))
fig.add_trace(go.Bar(
    name="Detection weight (norm)",
    x=bands,
    y=det_norm.tolist(),
    marker_color="#f5a623",
    opacity=0.7,
))
fig.add_trace(go.Scatter(
    name="Composite risk density",
    x=bands,
    y=composite,
    mode="lines+markers",
    line=dict(color="#e05555", width=2),
    marker=dict(size=8),
))
fig.update_layout(
    barmode="overlay",
    title="LEO Risk-Density Map — composite score by altitude band",
    xaxis_title="Altitude band",
    yaxis_title="Normalised score",
    height=360,
    legend=dict(orientation="h", y=1.1),
    margin=dict(l=40, r=20, t=60, b=60),
    plot_bgcolor="#f7f8fa",
    paper_bgcolor="#ffffff",
)
st.plotly_chart(fig, use_container_width=True)

# ── Data table ────────────────────────────────────────────────────────────────
display_df = risk_df[[
    "band_label", "tracked_object_count", "density_per_km",
    "detection_count", "detection_confidence_weighted", "composite_risk_density",
]].copy()
display_df.columns = [
    "Band", "Tracked objects", "Density/km",
    "Detections", "Det. weight (recency)", "Composite risk",
]
display_df["Composite risk"] = display_df["Composite risk"].map("{:.4f}".format)
display_df["Density/km"]     = display_df["Density/km"].map("{:.4f}".format)
display_df["Det. weight (recency)"] = display_df["Det. weight (recency)"].map("{:.3f}".format)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Inspectable formulas ──────────────────────────────────────────────────────
with st.expander("🔍 Inspect: How the composite score is computed"):
    st.markdown(f"""
**Algorithm (grid-based accumulator — NOT Bayesian):**

For each altitude band **b**:

```
detection_weight(b) = Σ  confidence_i × exp(−age_days_i / half_life)

composite_risk_density(b) = w1 × density_norm(b) + w2 × detection_weight_norm(b)
```

**Current parameters:**
- TLE density weight (w1): `{risk_map.w1}`
- Detection weight (w2): `{risk_map.w2}`
- Recency decay half-life: `{risk_map.decay_half_life_days}` days
- Total detections stored: `{risk_map.detection_count}`

**Normalisation:** each component is divided by its maximum across all bands before combining.
This keeps the composite in [0, 1] regardless of absolute magnitudes.

**Why not Bayesian?**  
A particle filter or full Bayesian tracker would require individual object state vectors.
Stage 1 produces size/shape/rotation — not a precise position/velocity.
The grid-based approach is consistent with what the data actually provides.
""")
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

st.divider()

# ── IBM Granite situation report ──────────────────────────────────────────────
st.subheader("4 · Situation Report (IBM Granite)")

if st.session_state.get("granite_sitrep") is None:
    with st.spinner("Generating situation report…"):
        sitrep = write_situation_report(risk_df)
        st.session_state["granite_sitrep"] = sitrep
else:
    sitrep = st.session_state["granite_sitrep"]

gs = granite_status()
source_label = "IBM Granite" if gs["available"] else "Fallback text (Granite unavailable)"
st.info(sitrep, icon="🤖")
st.caption(
    f"Source: **{source_label}** · "
    "Granite receives the computed risk densities above and translates them to plain English. "
    "It does not produce or guess any numbers."
)

if st.button("🔄 Regenerate situation report"):
    st.session_state["granite_sitrep"] = None
    st.rerun()

st.divider()
st.caption(
    "**Proceed to Stage 3 →** The composite risk-density map above is passed directly "
    "to the Prioritize stage, where each band is costed with Hohmann delta-v and ranked."
)
