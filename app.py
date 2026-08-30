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
from ui.theme import apply_theme, SYM_TARGET, SYM_GRID, SYM_VECTOR, SYM_ORBIT, SYM_ONLINE, SYM_OFFLINE
from ui.starfield import inject_starfield

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

apply_theme()
inject_starfield()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:monospace;font-size:1.1rem;font-weight:600;"
        "color:#00d4ff;letter-spacing:0.08em;text-transform:uppercase;"
        "padding:0.5rem 0 0.25rem'>△ DELTA-V</div>",
        unsafe_allow_html=True,
    )
    st.caption("Space debris characterization, risk mapping, and delta-v-costed prioritization.")
    st.divider()
    st.page_link("app.py",                    label=f"{SYM_ORBIT}  Mission Home")
    st.page_link("pages/01_characterize.py",  label=f"{SYM_TARGET}  Stage 1 — Characterize")
    st.page_link("pages/02_map.py",           label=f"{SYM_GRID}  Stage 2 — Map")
    st.page_link("pages/03_prioritize.py",    label=f"{SYM_VECTOR}  Stage 3 — Prioritize")
    st.divider()

    if granite_available(cfg):
        st.markdown(
            f"<span style='color:#2a9d6a;font-family:monospace;font-size:0.75rem'>"
            f"{SYM_ONLINE} IBM Granite — connected</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<span style='color:#5a7a9a;font-family:monospace;font-size:0.75rem'>"
            f"{SYM_OFFLINE} IBM Granite — fallback mode<br>"
            f"<span style='font-size:0.68rem'>Set WATSONX_API_KEY to enable.</span></span>",
            unsafe_allow_html=True,
        )

# ── Home page header ──────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-top:0.5rem;border-bottom:none;padding-bottom:0'>△ DELTA-V</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#8ecfea;font-family:monospace;font-size:0.88rem;"
    "letter-spacing:0.04em;margin-top:-0.5rem;margin-bottom:1rem'>"
    "PHYSICS-GROUNDED DEBRIS CHARACTERIZATION · RISK MAPPING · DELTA-V-COSTED PRIORITIZATION FOR LEO"
    "</p>",
    unsafe_allow_html=True,
)

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

# ── Pipeline stage cards ───────────────────────────────────────────────────────
# Rendered as a single HTML block so the arrow columns can use flexbox
# vertical-centering — st.columns arrows with fixed margin-top drift on
# different screen sizes and card heights.
st.markdown("""
<div class="pipeline-row">
  <div class="stage-card">
    <div class="stage-num">Stage 01</div>
    <div class="stage-title">◎ Characterize</div>
    <div class="stage-body">
      Given a debris light curve, estimate <strong>size, shape, and rotation state</strong>
      using Fourier decomposition and amplitude-based inversion.
      ML refines on top of those physical estimates — math is always primary.
    </div>
  </div>
  <div class="pipe-arrow">⟶</div>
  <div class="stage-card">
    <div class="stage-num">Stage 02</div>
    <div class="stage-title">⊞ Map</div>
    <div class="stage-body">
      Characterised detections feed a <strong>grid-based risk-density map</strong>
      across LEO altitude bands, weighted by recency and confidence.
      Updated with each new detection batch. Not a Bayesian filter.
    </div>
  </div>
  <div class="pipe-arrow">⟶</div>
  <div class="stage-card">
    <div class="stage-num">Stage 03</div>
    <div class="stage-title">△ Prioritize</div>
    <div class="stage-body">
      Each high-risk region is <strong>costed using Hohmann transfer delta-v</strong>
      and ranked by <code>risk × severity ÷ delta-v</code>.
      Outputs tiers with delta-v on every entry. Exports CSV/JSON.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Summary dashboard ─────────────────────────────────────────────────────────
st.markdown(
    "<h3 style='margin-bottom:0.75rem'>⊞ PIPELINE STATUS</h3>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

risk_df     = st.session_state.risk_map_df
priority_df = st.session_state.priority_df

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
        st.metric("IBM Granite", "Fallback")

st.divider()

# ── Run full pipeline demo ────────────────────────────────────────────────────
st.markdown(
    "<h3 style='margin-bottom:0.25rem'>▶ RUN FULL PIPELINE DEMO</h3>",
    unsafe_allow_html=True,
)
st.caption(
    "Runs all three stages end-to-end using a synthetic demo light curve "
    "and the TLE fallback snapshot. Results populate the status cards above."
)

if st.button("▶ Execute Pipeline Demo", type="primary"):
    import time as _time

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

    progress.progress(100, text="Done.")
    st.success(
        "Pipeline demo complete. Status cards above are now populated. "
        "Navigate to each stage page to explore results.",
    )
    st.rerun()

st.divider()
st.caption(
    "◎ Data transparency: light-curve training data is physics-based synthetic. "
    "TLE orbital data sourced from CelesTrak (public, no registration). "
    "Characterization validated against Mini-MegaTORTORA real light-curve data. "
    "All synthetic data is clearly labeled throughout the app."
)
