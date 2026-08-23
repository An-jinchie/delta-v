"""
ui/theme.py — Shared space-ops visual theme for Delta-V.

Inject with:
    from ui.theme import apply_theme
    apply_theme()

Design language:
  - Deep-space dark palette with cyan telemetry accent
  - JetBrains Mono for all data readouts and labels
  - Themed SVG/Unicode symbols instead of emoji
  - Mission-board tier colours: alert-red / amber / muted-green
  - Scanline texture on cards for CRT / instrument-panel feel
"""

import streamlit as st

# ── Palette ───────────────────────────────────────────────────────────────────
TIER_COLOURS = {
    "HIGH-PRIORITY": "#ff4b4b",
    "MONITOR":       "#f5a623",
    "LOW":           "#2a9d6a",
}

ACCENT_CYAN   = "#00d4ff"
ACCENT_AMBER  = "#f5a623"
ACCENT_RED    = "#ff4b4b"
ACCENT_GREEN  = "#2a9d6a"
ACCENT_PURPLE = "#7b5ea7"

BG_MAIN       = "#0a0e1a"
BG_PANEL      = "#0f1629"
BG_CARD       = "#111827"
BORDER        = "#1e2d4a"
TEXT_MAIN     = "#c8d8e8"
TEXT_MUTED    = "#5a7a9a"

# ── Themed symbol constants (use these instead of emoji) ──────────────────────
# Unicode characters that read as space-instrument notation
SYM_TARGET    = "◎"   # targeting reticle  — Stage 1 Characterize
SYM_GRID      = "⊞"   # grid / map         — Stage 2 Map
SYM_VECTOR    = "△"   # delta / vector     — Stage 3 Prioritize / delta-v
SYM_ALERT     = "▲"   # alert triangle     — HIGH-PRIORITY
SYM_MONITOR   = "◈"   # watch/monitor      — MONITOR
SYM_NOMINAL   = "●"   # nominal dot        — LOW
SYM_INSPECT   = "⊕"   # crosshair expand   — inspect expanders
SYM_DATA      = "≡"   # data lines         — data tables
SYM_SIGNAL    = "∿"   # waveform           — light curve
SYM_ORBIT     = "⟳"   # orbit refresh      — reload / regenerate
SYM_DOWNLOAD  = "↓"   # download arrow
SYM_ONLINE    = "◉"   # filled circle      — connected/online
SYM_OFFLINE   = "○"   # empty circle       — fallback/offline
SYM_INJECT    = "⊳"   # forward feed       — inject detections
SYM_CLEAR     = "⊘"   # null / clear

# ── Plotly layout defaults ────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor  = BG_PANEL,
    paper_bgcolor = BG_MAIN,
    font          = dict(color=TEXT_MAIN, family="JetBrains Mono, Consolas, monospace"),
    xaxis         = dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MAIN),
    yaxis         = dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_MAIN),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(color=TEXT_MAIN)),
)


def apply_theme() -> None:
    """Inject Delta-V CSS into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ── Fonts ───────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Space+Grotesk:wght@300;400;500;600&display=swap');

/* ── CSS custom properties ───────────────────────────────────────────────── */
:root {
  --bg:        #0a0e1a;
  --bg-panel:  #0f1629;
  --bg-card:   #111827;
  --border:    #1e2d4a;
  --border-hi: #2a4a6a;
  --cyan:      #00d4ff;
  --amber:     #f5a623;
  --red:       #ff4b4b;
  --green:     #2a9d6a;
  --purple:    #7b5ea7;
  --text:      #c8d8e8;
  --muted:     #5a7a9a;
  --mono:      'JetBrains Mono', Consolas, 'Courier New', monospace;
  --sans:      'Space Grotesk', system-ui, sans-serif;
}

/* ── Root background ────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  color: var(--text);
}

/* Make the main block container have a transparent bg so starfield shows */
[data-testid="block-container"] {
  background: transparent !important;
}

[data-testid="stMain"] {
  background: transparent !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: rgba(6, 9, 20, 0.92) !important;
  border-right: 1px solid var(--border);
  backdrop-filter: blur(8px);
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Top bar ────────────────────────────────────────────────────────────── */
[data-testid="stHeader"] {
  background-color: rgba(6, 9, 20, 0.85) !important;
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(6px);
}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1 {
  font-family: var(--mono) !important;
  color: var(--cyan) !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-weight: 600;
  font-size: 1.6rem !important;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4em;
  margin-bottom: 0.6em;
  /* Subtle cyan glow */
  text-shadow: 0 0 18px rgba(0, 212, 255, 0.25);
}

h2 {
  font-family: var(--mono) !important;
  color: #8ecfea !important;
  letter-spacing: 0.03em;
  font-size: 1.15rem !important;
  font-weight: 500;
}

h3 {
  font-family: var(--mono) !important;
  color: #7ab8d8 !important;
  letter-spacing: 0.02em;
  font-size: 0.98rem !important;
  font-weight: 400;
  text-transform: uppercase;
}

p, li, div {
  font-family: var(--sans);
  line-height: 1.65;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: linear-gradient(135deg, var(--bg-panel) 0%, rgba(15,22,41,0.6) 100%);
  border: 1px solid var(--border);
  border-top: 2px solid rgba(0,212,255,0.3);
  border-radius: 4px;
  padding: 0.8rem 1rem;
  position: relative;
  overflow: hidden;
}

/* Subtle scanline effect on cards */
[data-testid="metric-container"]::after {
  content: '';
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 3px,
    rgba(0,212,255,0.015) 3px,
    rgba(0,212,255,0.015) 4px
  );
  pointer-events: none;
}

[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.68rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

[data-testid="stMetricValue"] {
  color: var(--cyan) !important;
  font-family: var(--mono) !important;
  font-size: 1.55rem !important;
  font-weight: 600;
  text-shadow: 0 0 12px rgba(0,212,255,0.3);
}

[data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
}

/* ── DataFrames ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 4px;
}

[data-testid="stDataFrame"] table {
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background-color: var(--bg-panel);
  border: 1px solid var(--border) !important;
  border-radius: 4px;
  margin-bottom: 0.5rem;
}

[data-testid="stExpander"] summary {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

[data-testid="stExpander"] summary:hover {
  color: var(--cyan) !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
  font-family: var(--mono) !important;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  font-size: 0.74rem !important;
  border-radius: 3px !important;
  transition: all 0.15s ease;
}

[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, #0090b0 0%, #00d4ff 100%) !important;
  color: #030b12 !important;
  border: none !important;
  font-weight: 600 !important;
  box-shadow: 0 0 14px rgba(0,212,255,0.25);
}

[data-testid="stButton"] > button[kind="primary"]:hover {
  box-shadow: 0 0 22px rgba(0,212,255,0.45) !important;
  transform: translateY(-1px);
}

[data-testid="stButton"] > button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--muted) !important;
}

[data-testid="stButton"] > button[kind="secondary"]:hover {
  border-color: var(--cyan) !important;
  color: var(--cyan) !important;
}

/* ── Download buttons ───────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
  font-family: var(--mono) !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 0.74rem !important;
  border-radius: 3px !important;
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  transition: all 0.15s ease;
}

[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--cyan) !important;
  color: var(--cyan) !important;
}

/* ── Alert / info / warning / success boxes ─────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 3px !important;
  font-family: var(--sans) !important;
  font-size: 0.84rem !important;
  border-left-width: 3px !important;
  backdrop-filter: blur(4px);
}

/* info → cyan-tinted */
[data-testid="stAlert"][data-baseweb*="info"],
div[data-testid="stAlert"]:has(svg[data-testid*="info"]) {
  background: rgba(0,30,60,0.7) !important;
  border-left-color: var(--cyan) !important;
}

/* ── Code / JSON ────────────────────────────────────────────────────────── */
pre, code, [data-testid="stJson"] {
  font-family: var(--mono) !important;
  background-color: #06090f !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px;
  font-size: 0.78rem !important;
  color: #a8c8e8 !important;
}

/* ── Form labels ────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label,
[data-testid="stRadio"] label {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  color: var(--muted) !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ── Captions ───────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"],
small {
  color: var(--muted) !important;
  font-size: 0.73rem !important;
  font-family: var(--sans) !important;
}

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr {
  border-color: var(--border) !important;
  margin: 1.2rem 0;
}

/* ── Blockquote (problem statement) ─────────────────────────────────────── */
blockquote {
  border-left: 3px solid var(--cyan) !important;
  background: rgba(0,212,255,0.04) !important;
  padding: 0.75rem 1rem !important;
  margin: 0.5rem 0 1rem !important;
  border-radius: 0 3px 3px 0;
  color: var(--text) !important;
  font-family: var(--sans) !important;
  font-size: 0.9rem;
}

/* ── Toast notifications ─────────────────────────────────────────────────── */
[data-testid="stToast"] {
  background: rgba(15,22,41,0.95) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  backdrop-filter: blur(8px);
}

/* ── Tier badge classes ──────────────────────────────────────────────────── */
.tier-high {
  display: inline-block;
  background: rgba(255,75,75,0.12);
  color: var(--red);
  border: 1px solid rgba(255,75,75,0.5);
  border-radius: 2px;
  padding: 1px 8px;
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  font-weight: 600;
  text-transform: uppercase;
}

.tier-monitor {
  display: inline-block;
  background: rgba(245,166,35,0.10);
  color: var(--amber);
  border: 1px solid rgba(245,166,35,0.4);
  border-radius: 2px;
  padding: 1px 8px;
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  font-weight: 600;
  text-transform: uppercase;
}

.tier-low {
  display: inline-block;
  background: rgba(42,157,106,0.10);
  color: var(--green);
  border: 1px solid rgba(42,157,106,0.35);
  border-radius: 2px;
  padding: 1px 8px;
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

/* ── Pipeline stage cards ────────────────────────────────────────────────── */
.stage-card {
  background: linear-gradient(160deg, rgba(15,22,41,0.85) 0%, rgba(8,12,24,0.9) 100%);
  border: 1px solid var(--border);
  border-top: 2px solid rgba(0,212,255,0.4);
  border-radius: 4px;
  padding: 1.25rem 1.5rem;
  backdrop-filter: blur(6px);
  position: relative;
  overflow: hidden;
}

.stage-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  opacity: 0.4;
}

.stage-card .stage-num {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--cyan);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 0.3rem;
  opacity: 0.7;
}

.stage-card .stage-title {
  font-family: var(--mono);
  font-size: 1.0rem;
  color: var(--cyan);
  font-weight: 600;
  letter-spacing: 0.04em;
  margin-bottom: 0.6rem;
}

.stage-card .stage-body {
  font-family: var(--sans);
  font-size: 0.84rem;
  color: var(--text);
  line-height: 1.6;
}

/* ── Pipeline arrow ──────────────────────────────────────────────────────── */
.pipe-arrow {
  font-size: 1.4rem;
  color: var(--border-hi);
  text-align: center;
  margin-top: 2.5rem;
  font-family: var(--mono);
}

/* ── Orbital ring decoration on page titles ─────────────────────────────── */
.page-header {
  position: relative;
  padding-left: 1rem;
}

.page-header::before {
  content: '';
  position: absolute;
  left: 0; top: 50%;
  width: 4px; height: 70%;
  background: var(--cyan);
  transform: translateY(-50%);
  border-radius: 2px;
  box-shadow: 0 0 8px rgba(0,212,255,0.5);
}

/* ── Symbol styling ──────────────────────────────────────────────────────── */
.sym {
  display: inline-block;
  color: var(--cyan);
  font-family: var(--mono);
  margin-right: 0.4em;
  font-size: 0.9em;
  vertical-align: middle;
}

.sym-amber { color: var(--amber); }
.sym-red   { color: var(--red);   }
.sym-green { color: var(--green); }
.sym-muted { color: var(--muted); }

/* ── Data readout rows (key: value pairs) ────────────────────────────────── */
.readout-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.2rem 0;
  border-bottom: 1px solid rgba(30,45,74,0.5);
  font-family: var(--mono);
  font-size: 0.78rem;
}

.readout-key   { color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.readout-value { color: var(--cyan);  font-weight: 600; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hi); }

/* ── Selection ───────────────────────────────────────────────────────────── */
::selection { background: rgba(0,212,255,0.2); color: var(--text); }
</style>
"""
