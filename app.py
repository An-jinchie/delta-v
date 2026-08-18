"""
app.py — Delta-V Streamlit entry point and home dashboard.

Session state keys used across all pages (initialised on first load):
    st.session_state.characterize_result   : dict | None  — latest characterization output
    st.session_state.risk_map_df           : pd.DataFrame | None — latest risk map
    st.session_state.priority_df           : pd.DataFrame | None — latest priority scores
    st.session_state.granite_sitrep        : str | None  — cached situation report text
    st.session_state.granite_brief         : str | None  — cached mission brief text
    st.session_state.granite_explain       : str | None  — cached characterization explanation
    st.session_state.tle_last_updated      : str | None  — ISO timestamp of last TLE fetch
    st.session_state.tle_object_count      : int | None  — number of TLE objects loaded
"""

import streamlit as st
from config import get_config, granite_available

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delta-V",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation (safe no-ops if keys already exist) ─────────
_defaults = {
    "characterize_result": None,
    "risk_map_df": None,
    "priority_df": None,
    "granite_sitrep": None,
    "granite_brief": None,
    "granite_explain": None,
    "tle_last_updated": None,
    "tle_object_count": None,
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Config ───────────────────────────────────────────────────────────────────
cfg = get_config()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛰️ Delta-V")
    st.caption("Space debris characterization, risk mapping, and delta-v-costed prioritization.")
    st.divider()
    st.page_link("app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/01_characterize.py", label="Stage 1 — Characterize")
    st.page_link("pages/02_map.py", label="Stage 2 — Map")
    st.page_link("pages/03_prioritize.py", label="Stage 3 — Prioritize")
    st.divider()

    # Granite status indicator
    if granite_available(cfg):
        st.success("IBM Granite ✓ connected", icon="🤖")
    else:
        st.info("IBM Granite — using fallback text\n\nSet WATSONX_API_KEY to enable.", icon="🤖")

# ── Home page ─────────────────────────────────────────────────────────────────
st.title("🛰️ Delta-V")
st.subheader("Physics-grounded debris characterization, risk mapping, and delta-v-costed prioritization for LEO.")

st.markdown("""
> **The problem:** Small space debris (1mm–10cm) is too small for the U.S. Space Surveillance
> Network to individually track, yet capable of mission-ending damage at orbital velocities.
> Less than 1% of debris in this danger category is currently tracked.
> Operators who most need this picture — university cubesat teams, early-stage startups,
> independent researchers — are locked out of official conjunction data.
> And a risk score alone is not a decision: it doesn't say what it would cost to act on,
> or which threat deserves attention first.
""")

st.divider()

# ── Pipeline flow ─────────────────────────────────────────────────────────────
col1, arrow1, col2, arrow2, col3 = st.columns([4, 1, 4, 1, 4])

with col1:
    st.markdown("### 1 · Characterize")
    st.markdown(
        "Given a debris light curve, estimate **size, shape, and rotation state** "
        "using Fourier decomposition and amplitude-based inversion. "
        "ML refines on top of those physical estimates."
    )

with arrow1:
    st.markdown("<div style='font-size:2rem;text-align:center;margin-top:2rem'>→</div>",
                unsafe_allow_html=True)

with col2:
    st.markdown("### 2 · Map")
    st.markdown(
        "Characterized detections feed a **grid-based risk-density map** across LEO "
        "altitude bands, weighted by recency and confidence. "
        "Updated with each new detection batch."
    )

with arrow2:
    st.markdown("<div style='font-size:2rem;text-align:center;margin-top:2rem'>→</div>",
                unsafe_allow_html=True)

with col3:
    st.markdown("### 3 · Prioritize")
    st.markdown(
        "Each high-risk region is **costed using Hohmann transfer delta-v** "
        "and ranked by `risk × severity ÷ delta-v`. "
        "Outputs tiers (HIGH-PRIORITY / MONITOR / LOW) with delta-v on every entry."
    )

st.divider()

# ── Summary dashboard ─────────────────────────────────────────────────────────
st.markdown("### Pipeline Status")

m1, m2, m3, m4 = st.columns(4)

risk_df = st.session_state.risk_map_df
priority_df = st.session_state.priority_df
char_result = st.session_state.characterize_result

with m1:
    if risk_df is not None and not risk_df.empty:
        top_risk = risk_df.sort_values("composite_risk_density", ascending=False).iloc[0]
        st.metric("Top Risk Band", top_risk["band_label"])
    else:
        st.metric("Top Risk Band", "—")

with m2:
    if priority_df is not None and not priority_df.empty:
        top = priority_df.iloc[0]
        st.metric("Top Priority Band", top["band_label"], delta=top["tier"])
    else:
        st.metric("Top Priority Band", "—")

with m3:
    if priority_df is not None and not priority_df.empty:
        dv = priority_df.iloc[0]["dv_total_ms"]
        st.metric("Delta-V (top band)", f"{dv:.0f} m/s")
    else:
        st.metric("Delta-V (top band)", "—")

with m4:
    if granite_available(cfg):
        st.metric("IBM Granite", "Connected")
    else:
        st.metric("IBM Granite", "Fallback mode")

st.divider()

# ── Run full pipeline demo ────────────────────────────────────────────────────
st.markdown("### Run Full Pipeline Demo")
st.caption(
    "Runs all three stages end-to-end using a synthetic demo light curve "
    "and the TLE fallback snapshot. Results populate the status cards above."
)

if st.button("▶ Run Full Pipeline Demo", type="primary"):
    import time as _time
    import numpy as _np

    progress = st.progress(0, text="Stage 1 — generating demo light curve…")

    # ── Stage 1: Characterize ────────────────────────────────────────────────
    try:
        from pipeline.characterize.generator import LightCurveGenerator, LightCurveParams
        from pipeline.characterize.inversion import invert
        from pipeline.characterize.features import extract_features

        params = LightCurveParams(
            size_class="medium", shape="flat_plate",
            rotation_rate_hz=0.65, phase_offset=0.0,
            albedo=0.15, snr=20.0, n_samples=256, cadence_s=0.1, seed=42,
        )
        lc = LightCurveGenerator().generate(params)
        inv = invert(lc, cadence_s=0.1)

        try:
            import os
            from pipeline.characterize.model import CharacterizeModel
            MODEL_PATH = "data/models/characterize_model.pkl"
            if os.path.exists(MODEL_PATH):
                model = CharacterizeModel.load(MODEL_PATH)
            else:
                raise FileNotFoundError("model not found")
            prediction = model.predict(lc, inv)
        except Exception:
            prediction = {
                "size_class": inv.size_class, "shape": inv.shape_hint,
                "size_confidence": 0.0, "shape_confidence": 0.0,
                "inversion_result": inv, "features": extract_features(lc, inv),
            }

        st.session_state["characterize_result"] = prediction
        st.session_state["char_lc"] = lc
        st.session_state["char_inv"] = inv
        st.session_state["char_prediction"] = prediction
        st.session_state["granite_explain"] = None
        progress.progress(33, text="Stage 2 — computing risk-density map…")
    except Exception as exc:
        st.error(f"Stage 1 failed: {exc}")

    # ── Stage 2: Map ─────────────────────────────────────────────────────────
    try:
        from pipeline.map.tle_fetcher import TLEFetcher
        from pipeline.map.density import compute_density
        from pipeline.map.risk_map import RiskDensityMap

        tle_lines = TLEFetcher().fetch()
        density_df = compute_density(tle_lines)

        risk_map = RiskDensityMap()
        # Inject a demo detection from the Stage 1 result
        risk_map.update([{
            "altitude_band_km": "600-800",
            "confidence": 0.85,
            "timestamp": _time.time(),
            "size_class": prediction.get("size_class", "medium"),
        }])
        risk_df = risk_map.compute(density_df)
        st.session_state["risk_map_obj"] = risk_map
        st.session_state["risk_map_df"] = risk_df
        st.session_state["tle_object_count"] = int(density_df["object_count"].sum())
        st.session_state["granite_sitrep"] = None
        progress.progress(66, text="Stage 3 — computing priority scores…")
    except Exception as exc:
        st.error(f"Stage 2 failed: {exc}")
        risk_df = None

    # ── Stage 3: Prioritize ──────────────────────────────────────────────────
    try:
        from pipeline.prioritize.scorer import PriorityScorer
        if risk_df is not None:
            scorer = PriorityScorer()
            priority_df_new = scorer.score(risk_df)
            st.session_state["priority_df"] = priority_df_new
            st.session_state["granite_brief"] = None
    except Exception as exc:
        st.error(f"Stage 3 failed: {exc}")

    progress.progress(100, text="Done!")
    st.success(
        "✅ Full pipeline demo complete. "
        "Status cards above are now populated. "
        "Navigate to each stage page to explore results.",
        icon="🛰️",
    )
    st.rerun()

st.divider()
st.caption(
    "⚠️ **Data transparency:** Light-curve training data is physics-based synthetic. "
    "TLE orbital data sourced from CelesTrak (public, no registration). "
    "Characterization validated against Mini-MegaTORTORA real light-curve data. "
    "All synthetic data is clearly labeled throughout the app."
)
