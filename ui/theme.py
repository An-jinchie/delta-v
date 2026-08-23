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
# Base layout applied to every chart via **PLOTLY_LAYOUT.
# Only contains properties that are safe to unpack alongside any per-chart overrides.
# - xaxis/yaxis use _shorthand_ keys so they merge cleanly with xaxis_title etc.
# - legend is NOT included here; each chart passes its own legend= kwarg.
PLOTLY_LAYOUT = dict(
    plot_bgcolor         = BG_PANEL,
    paper_bgcolor        = BG_MAIN,
    font                 = dict(color=TEXT_MAIN, family="Rajdhani, Jura, system-ui, sans-serif"),
    xaxis_gridcolor      = BORDER,
    xaxis_zerolinecolor  = BORDER,
    xaxis_color          = TEXT_MAIN,
    yaxis_gridcolor      = BORDER,
    yaxis_zerolinecolor  = BORDER,
    yaxis_color          = TEXT_MAIN,
)

# Legend defaults — merge into per-chart legend= kwarg when needed
PLOTLY_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(color=TEXT_MAIN))


def apply_theme() -> None:
    """Inject Delta-V CSS into the current Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ── Fonts ───────────────────────────────────────────────────────────────── */
/* Orbitron — dramatic sci-fi HUD titles (H1 only, used sparingly)
   Rajdhani — condensed, sharp, mission-control labels & subheadings
   Jura     — clean, minimal futuristic body text                      */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Jura:wght@300;400;500;600&display=swap');

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
  --text:      #d4e4f4;
  --muted:     #7a9ab8;
  /* Font stack: Orbitron for big titles, Rajdhani for UI chrome,
     Jura for readable body text */
  --font-title:  'Orbitron', 'Rajdhani', sans-serif;
  --font-ui:     'Rajdhani', 'Jura', system-ui, sans-serif;
  --font-body:   'Jura', 'Rajdhani', system-ui, sans-serif;
  --mono:        'Rajdhani', Consolas, monospace;
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

/* ── Sidebar — more opaque, distinct from main canvas ───────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,
    rgba(5,8,18,0.97) 0%,
    rgba(8,12,26,0.97) 60%,
    rgba(6,10,22,0.97) 100%) !important;
  border-right: 1px solid rgba(0,212,255,0.18);
  backdrop-filter: blur(14px);
  box-shadow: 2px 0 20px rgba(0,0,0,0.5);
}

/* Sidebar text — plain white for maximum legibility */
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] li { color: #ffffff !important; }

/* ── Top bar ────────────────────────────────────────────────────────────── */
[data-testid="stHeader"] {
  background-color: rgba(5, 8, 18, 0.92) !important;
  border-bottom: 1px solid rgba(0,212,255,0.15);
  backdrop-filter: blur(10px);
}

/* ── Main content area — semi-transparent backdrop for legibility ────────── */
[data-testid="block-container"] > div {
  background: rgba(8, 12, 24, 0.55) !important;
  border-radius: 6px;
  backdrop-filter: blur(4px);
}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1 {
  font-family: var(--font-title) !important;
  color: var(--cyan) !important;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  font-weight: 700;
  font-size: 1.85rem !important;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4em;
  margin-bottom: 0.6em;
  text-shadow: 0 0 22px rgba(0, 212, 255, 0.35);
}

h2 {
  font-family: var(--font-ui) !important;
  color: #a0d4f0 !important;
  letter-spacing: 0.06em;
  font-size: 1.25rem !important;
  font-weight: 600;
  text-transform: uppercase;
}

h3 {
  font-family: var(--font-ui) !important;
  color: #88bcd8 !important;
  letter-spacing: 0.05em;
  font-size: 1.05rem !important;
  font-weight: 600;
  text-transform: uppercase;
}

p, li {
  font-family: var(--font-body);
  font-size: 0.97rem;
  line-height: 1.7;
  color: var(--text);
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
  font-family: var(--font-ui) !important;
  font-size: 0.80rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

[data-testid="stMetricValue"] {
  color: var(--cyan) !important;
  font-family: 'Orbitron', var(--font-ui) !important;
  font-size: 1.6rem !important;
  font-weight: 700;
  text-shadow: 0 0 14px rgba(0,212,255,0.35);
}

[data-testid="stMetricDelta"] {
  font-family: var(--font-ui) !important;
  font-size: 0.78rem !important;
}

/* ── DataFrames ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 4px;
}

[data-testid="stDataFrame"] table {
  font-family: var(--font-ui) !important;
  font-size: 0.82rem !important;
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
  font-family: var(--font-ui) !important;
  font-size: 0.83rem !important;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

[data-testid="stExpander"] summary:hover {
  color: var(--cyan) !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
  font-family: var(--font-ui) !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.82rem !important;
  font-weight: 600;
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
  font-family: var(--font-ui) !important;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  font-size: 0.82rem !important;
  font-weight: 600;
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
  font-family: var(--font-body) !important;
  font-size: 0.90rem !important;
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
  font-family: 'Rajdhani', Consolas, monospace !important;
  background-color: #06090f !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px;
  font-size: 0.84rem !important;
  color: #b8d8f0 !important;
}

/* ── Form labels ────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label,
[data-testid="stRadio"] label {
  font-family: var(--font-ui) !important;
  font-size: 0.82rem !important;
  color: var(--muted) !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 500;
}

/* ── Captions ───────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"],
small {
  color: var(--muted) !important;
  font-size: 0.82rem !important;
  font-family: var(--font-body) !important;
}

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr {
  border-color: var(--border) !important;
  margin: 1.2rem 0;
}

/* ── Blockquote (problem statement) ─────────────────────────────────────── */
blockquote {
  border-left: 3px solid var(--cyan) !important;
  background: rgba(0,212,255,0.05) !important;
  padding: 0.85rem 1.1rem !important;
  margin: 0.5rem 0 1rem !important;
  border-radius: 0 3px 3px 0;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
  font-size: 1.0rem;
  line-height: 1.7;
}

/* ── Toast notifications ─────────────────────────────────────────────────── */
[data-testid="stToast"] {
  background: rgba(15,22,41,0.95) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--font-ui) !important;
  font-size: 0.84rem !important;
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
  font-family: var(--font-ui);
  font-size: 0.72rem;
  color: var(--cyan);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-bottom: 0.3rem;
  opacity: 0.75;
}

.stage-card .stage-title {
  font-family: 'Orbitron', var(--font-ui);
  font-size: 1.05rem;
  color: var(--cyan);
  font-weight: 700;
  letter-spacing: 0.06em;
  margin-bottom: 0.65rem;
  text-shadow: 0 0 10px rgba(0,212,255,0.25);
}

.stage-card .stage-body {
  font-family: var(--font-body);
  font-size: 0.92rem;
  color: var(--text);
  line-height: 1.65;
}

/* ── Pipeline arrow ──────────────────────────────────────────────────────── */
.pipe-arrow {
  font-size: 1.6rem;
  color: var(--border-hi);
  text-align: center;
  margin-top: 2.5rem;
  font-family: var(--font-ui);
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
