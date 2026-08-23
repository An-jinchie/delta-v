"""
pages/01_characterize.py — Stage 1: Light-Curve Characterization

User workflow:
  1. Generate a synthetic light curve (or upload CSV) with configurable parameters
  2. Fourier inversion runs deterministically — produces rotation rate, amplitude, shape hint
  3. ML classifier refines size/shape classification on top of those physical features
  4. IBM Granite (or fallback) translates results to a plain-language analyst brief
  5. Every result is fully inspectable — click expanders to see the raw numbers + formulas

Data transparency: all curves generated here are SYNTHETIC (physics-based).
Results are labeled accordingly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.granite import explain_characterization, granite_status
from ui.theme import (apply_theme, PLOTLY_LAYOUT, ACCENT_CYAN, ACCENT_RED,
                      BG_PANEL, BG_MAIN, BORDER, TEXT_MAIN,
                      SYM_TARGET, SYM_SIGNAL, SYM_INSPECT, SYM_ORBIT, SYM_ONLINE, SYM_OFFLINE)
from ui.starfield import inject_starfield
from pipeline.characterize.generator import LightCurveGenerator, LightCurveParams
from pipeline.characterize.inversion import invert
from pipeline.characterize.features import FEATURE_NAMES, extract_features

st.set_page_config(page_title="Delta-V · Characterize", page_icon="🛰️", layout="wide")
apply_theme()
inject_starfield()

# ── Sidebar note ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<span style='font-family:monospace;font-size:0.75rem;color:#00d4ff'>"
        f"{SYM_TARGET} STAGE 1 · CHARACTERIZE</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Fourier decomposition + amplitude estimation is the primary method. "
        "ML is secondary — it refines, never replaces, the math."
    )
    gs = granite_status()
    sym = SYM_ONLINE if gs["available"] else SYM_OFFLINE
    colour = "#2a9d6a" if gs["available"] else "#5a7a9a"
    st.markdown(
        f"<span style='color:{colour};font-family:monospace;font-size:0.73rem'>"
        f"{sym} Granite: {gs['status_text']}</span>",
        unsafe_allow_html=True,
    )

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"<h1>{SYM_TARGET} CHARACTERIZE</h1>", unsafe_allow_html=True)
st.caption(
    "Given a debris light curve (brightness over time as an object tumbles), estimate "
    "size, shape, and rotation state. Math is primary. ML is secondary. All outputs are inspectable."
)
st.info(
    f"{SYM_TARGET} Data transparency: light curves generated on this page are physics-based "
    "synthetic data. Results are order-of-magnitude estimates — not precision measurements.",
)

st.divider()


# ── Helper: load or lazy-train model ─────────────────────────────────────────

@st.cache_resource(show_spinner="Loading characterization model…")
def _load_model():
    """Load the trained CharacterizeModel, training it if not found."""
    import os
    MODEL_PATH = "data/models/characterize_model.pkl"
    from pipeline.characterize.model import CharacterizeModel

    if os.path.exists(MODEL_PATH):
        try:
            return CharacterizeModel.load(MODEL_PATH)
        except Exception:
            pass

    # Model not found — train a small in-memory model (500 curves, quick)
    st.toast("Model not found — training a quick in-memory model (500 curves)…", icon="⚙️")
    import csv, io, tempfile
    from pipeline.characterize.generator import generate_dataset
    from pipeline.characterize.inversion import invert as _inv

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "lc.csv")
        generate_dataset(n_samples=500, output_path=csv_path, seed=99)

        # Read, skipping the leading # SYNTHETIC DATA comment line
        with open(csv_path, newline="", encoding="utf-8") as f:
            lines = [ln for ln in f if not ln.startswith("#")]

    reader = csv.DictReader(io.StringIO("".join(lines)))
    feats, sz_labels, sh_labels = [], [], []
    for row in reader:
        time_cols = [k for k in row.keys() if k.startswith("t_")]
        lc = np.array([float(row[c]) for c in time_cols])
        try:
            inv_r = _inv(lc)
            fv = extract_features(lc, inv_r)
            feats.append(fv)
            sz_labels.append(row["size_class"])
            sh_labels.append(row["shape"])
        except Exception:
            continue

    X = np.array(feats)
    model = CharacterizeModel()
    model.train(X, np.array(sz_labels), np.array(sh_labels))
    return model


# ── Curve generation controls ─────────────────────────────────────────────────
st.subheader(f"{SYM_SIGNAL} Configure Light Curve")

col_params, col_preview = st.columns([1, 2])

with col_params:
    shape = st.selectbox(
        "Object shape",
        ["flat_plate", "tumbling", "cylinder", "sphere"],
        help="Flat plate: single dominant frequency. Tumbling: two-axis rotation. "
             "Cylinder: partial modulation. Sphere: no modulation (pure noise).",
    )
    size_class = st.selectbox(
        "Size class",
        ["small", "medium", "large"],
        index=1,
        help="small ≈ 1–3 cm, medium ≈ 3–7 cm, large ≈ 7–10 cm",
    )
    _ROT_RANGES = {"small": (1.0, 2.4), "medium": (0.3, 1.0), "large": (0.02, 0.3)}
    rot_lo, rot_hi = _ROT_RANGES[size_class]
    rotation_hz = st.slider(
        "Rotation rate (Hz)",
        min_value=float(rot_lo),
        max_value=float(rot_hi),
        value=float((rot_lo + rot_hi) / 2),
        step=float((rot_hi - rot_lo) / 40),
        help="Physical rotation frequency. For flat_plate/cylinder the FFT peak appears at 2× this value.",
    )
    snr = st.slider("Signal-to-noise ratio", min_value=5.0, max_value=40.0, value=20.0, step=1.0,
                    help="Higher SNR = cleaner curve. Real MMT observations ≈ 8–15.")
    seed = st.number_input("Random seed", value=42, step=1)

    generate_btn = st.button(f"{SYM_TARGET} Generate & Analyse", type="primary", use_container_width=True)


# ── Run pipeline on button press ──────────────────────────────────────────────
if generate_btn or "char_lc" not in st.session_state:
    params = LightCurveParams(
        size_class=size_class,
        shape=shape,
        rotation_rate_hz=rotation_hz,
        phase_offset=0.0,
        albedo=0.15,
        snr=snr,
        n_samples=256,
        cadence_s=0.1,
        seed=int(seed),
    )
    gen = LightCurveGenerator()
    lc = gen.generate(params)
    inv = invert(lc, cadence_s=0.1)

    try:
        model = _load_model()
        prediction = model.predict(lc, inv)
    except Exception as e:
        prediction = {
            "size_class": inv.size_class,
            "shape": inv.shape_hint,
            "size_confidence": 0.0,
            "shape_confidence": 0.0,
            "inversion_result": inv,
            "features": extract_features(lc, inv),
        }

    st.session_state["char_lc"] = lc
    st.session_state["char_inv"] = inv
    st.session_state["char_prediction"] = prediction
    st.session_state["char_params"] = params
    st.session_state["characterize_result"] = prediction
    # Clear cached Granite text so a new explanation is generated
    st.session_state["granite_explain"] = None


lc         = st.session_state.get("char_lc")
inv        = st.session_state.get("char_inv")
prediction = st.session_state.get("char_prediction")
params_used = st.session_state.get("char_params")

if lc is None:
    st.stop()

# ── Light curve plot ──────────────────────────────────────────────────────────
with col_preview:
    cadence = 0.1
    t_axis = np.arange(len(lc)) * cadence
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_axis, y=lc,
        mode="lines",
        line=dict(color=ACCENT_CYAN, width=1.5),
        name="Brightness",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Synthetic light curve — {shape} / {size_class} / SNR {snr:.0f}", font_size=12),
        xaxis_title="Time (s)",
        yaxis_title="Normalised brightness",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⚠️ SYNTHETIC DATA — physics-based generator. Not real telescope observations.")

st.divider()

# ── Results ───────────────────────────────────────────────────────────────────
st.subheader(f"{SYM_TARGET} Inversion Results")
st.caption(
    "**Primary method:** Fourier decomposition + amplitude estimation. "
    "**Secondary:** ML classifier refines size/shape. "
    "All numbers are computed deterministically — no model guesses any value."
)

res_col1, res_col2, res_col3, res_col4 = st.columns(4)

with res_col1:
    st.metric(
        "Size class",
        f"{prediction['size_class'].upper()}",
        delta=f"ML conf {prediction['size_confidence']*100:.0f}%",
    )
    st.caption(f"Inversion heuristic: **{inv.size_class}** · {inv.size_estimate_m*100:.1f} cm")

with res_col2:
    st.metric(
        "Shape",
        prediction["shape"].replace("_", " ").title(),
        delta=f"ML conf {prediction['shape_confidence']*100:.0f}%",
    )
    st.caption(f"Inversion hint: **{inv.shape_hint.replace('_', ' ')}**")

with res_col3:
    st.metric(
        "Rotation rate (FFT)",
        f"{inv.rotation_rate_hz:.4f} Hz",
        delta=f"±{inv.rotation_rate_uncertainty_hz:.4f} Hz",
    )
    period = 1.0 / inv.rotation_rate_hz if inv.rotation_rate_hz > 0 else 0
    st.caption(f"Apparent period: {period:.2f} s · For |cos| shapes, physical rate = FFT peak ÷ 2")

with res_col4:
    st.metric("SNR estimate", f"{inv.snr_estimate:.1f}")
    reliability = "reliable" if inv.snr_estimate > 10 else "marginal — treat with caution"
    st.caption(f"Signal quality: **{reliability}**")

# ── FFT power spectrum ────────────────────────────────────────────────────────
st.subheader(f"{SYM_SIGNAL} FFT Power Spectrum")

fft_full  = np.fft.rfft(lc)
fft_power = np.abs(fft_full)
freqs     = np.fft.rfftfreq(len(lc), d=0.1)

fig_fft = go.Figure()
fig_fft.add_trace(go.Bar(
    x=freqs[1:],
    y=fft_power[1:],
    marker_color=ACCENT_CYAN,
    marker_opacity=0.75,
    name="AC power",
))
fig_fft.add_vline(
    x=inv.rotation_rate_hz,
    line_dash="dash",
    line_color=ACCENT_RED,
    annotation_text=f"Dominant: {inv.rotation_rate_hz:.4f} Hz",
    annotation_position="top right",
    annotation_font_color=ACCENT_RED,
)
fig_fft.update_layout(
    **PLOTLY_LAYOUT,
    title="FFT power spectrum (AC bins only — DC removed)",
    xaxis_title="Frequency (Hz)",
    yaxis_title="|FFT| amplitude",
    height=260,
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig_fft, use_container_width=True)

# ── Inspectable raw numbers ───────────────────────────────────────────────────
with st.expander(f"{SYM_INSPECT} Inspect: Inversion raw numbers + formulas"):
    st.markdown("""
**Formulas used (deterministic — no model weights):**

| Quantity | Formula |
|---|---|
| Rotation rate | `argmax(|FFT[1:N/2]|) × (1 / (N × cadence_s))` |
| Rate uncertainty | `0.5 / (N × cadence_s)` = half FFT bin width |
| Amplitude | `(max(lc) − min(lc)) / mean(lc)` |
| Size heuristic | amp < 0.10 → small · 0.10–0.40 → medium · ≥ 0.40 → large |
| Shape (dom_frac) | `dominant_AC_bin / total_AC_power` |
""")
    raw = {
        "rotation_rate_hz":              inv.rotation_rate_hz,
        "rotation_rate_uncertainty_hz":  inv.rotation_rate_uncertainty_hz,
        "amplitude":                     inv.amplitude,
        "size_class (inversion)":        inv.size_class,
        "size_estimate_m":               inv.size_estimate_m,
        "shape_hint (inversion)":        inv.shape_hint,
        "snr_estimate":                  inv.snr_estimate,
        "n_samples":                     inv.n_samples,
        "cadence_s":                     inv.cadence_s,
        "fourier_coefficients":          inv.fourier_coefficients.tolist(),
    }
    st.json(raw)

with st.expander(f"{SYM_INSPECT} Inspect: ML feature vector (18 features)"):
    feat_vec = prediction["features"]
    feat_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Value": [f"{v:.6f}" for v in feat_vec],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
    st.caption(
        "The ML classifier receives these 18 features — not the raw light curve. "
        "Every feature is physically derived or statistically descriptive. "
        "No feature is a black-box embedding."
    )

st.divider()

# ── IBM Granite analyst brief ─────────────────────────────────────────────────
st.subheader("◉ Analyst Brief — IBM Granite")

if st.session_state.get("granite_explain") is None:
    with st.spinner("Generating analyst brief…"):
        brief = explain_characterization(prediction)
        st.session_state["granite_explain"] = brief
else:
    brief = st.session_state["granite_explain"]

gs = granite_status()
source_label = "IBM Granite" if gs["available"] else "Fallback text (Granite unavailable)"
st.info(brief, icon="🤖")
st.caption(
    f"Source: **{source_label}** · "
    "Granite receives the computed numbers above and translates them to plain English. "
    "It does not produce or guess any values."
)

if st.button(f"{SYM_ORBIT} Regenerate brief", help="Re-run Granite / fallback with the same data"):
    st.session_state["granite_explain"] = None
    st.rerun()

st.divider()
st.caption(
    "**Accuracy note:** ML classifier achieves ~83.8% size accuracy and ~95.2% shape accuracy "
    "on synthetic validation data. Real light curves are noisier — accuracy will degrade. "
    "All results are order-of-magnitude estimates. See README for validation methodology."
)
