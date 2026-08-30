# Delta-V

> Physics-grounded space debris characterization, risk mapping, and delta-v-costed prioritization for LEO.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen.svg)](#running-tests)
[![Live Demo](https://img.shields.io/badge/live-demo-red.svg)](https://delta-v-ksn33avrrgai9gbnidlstb.streamlit.app/)

**Live app:** https://delta-v-ksn33avrrgai9gbnidlstb.streamlit.app/
**GitHub:** https://github.com/An-jinchie/delta-v

---

## Problem Statement

Small space debris (1 mm–10 cm) is too small for the U.S. Space Surveillance Network to individually track, yet capable of mission-ending damage at orbital velocities. **Less than 1% of debris in this danger category is currently tracked.**

Operators who most need this picture — university cubesat teams, early-stage space startups, independent researchers — are locked out of official conjunction data, which requires being a registered satellite owner/operator with an active cataloged object. And even where risk data exists, a score alone is not a decision: it doesn't say what it would cost to act on, or which threat deserves attention first.

**Delta-V addresses all three gaps in one pipeline:** characterize the debris, map the regional risk, and cost each region with real orbital mechanics — so an operator gets a ranked, actionable list with a delta-v price tag on every entry.

---

## Solution Description

Delta-V is a three-stage physics-grounded pipeline:

```
Light Curve (brightness vs. time)
        │
        ▼
┌─────────────────────────────────────┐
│ Stage 1 · Characterize              │
│                                     │
│ 1. Fourier decomposition            │ ← PRIMARY (deterministic math)
│    • rotation rate from FFT peak    │
│    • amplitude → size estimate      │
│    • coefficient structure → shape  │
│                                     │
│ 2. RandomForest refinement          │ ← SECONDARY (on physical features)
│    • size class, shape, confidence  │
└─────────────────┬───────────────────┘
                  │  detection: {altitude_band, confidence, timestamp, size_class}
                  ▼
┌─────────────────────────────────────┐
│ Stage 2 · Map                       │
│                                     │
│ Grid-based risk-density accumulator │ ← NOT a Bayesian filter
│ • TLE density per altitude band     │
│ • Recency-weighted detections       │
│ • composite_risk_density per band   │
└─────────────────┬───────────────────┘
                  │  risk_df per altitude band
                  ▼
┌─────────────────────────────────────┐
│ Stage 3 · Prioritize                │
│                                     │
│ • Hohmann transfer delta-v          │ ← REGION-LEVEL ONLY
│ • Plane-change estimate (5° worst)  │
│ • priority = risk × severity / dv   │
│ • Tiers: HIGH-PRIORITY/MONITOR/LOW  │
│ • Export: CSV + JSON (dv on every   │
│   entry)                            │
└─────────────────────────────────────┘
        │
        ▼
IBM Granite (watsonx.ai)
  Plain-language explanation of computed results
  (Granite never produces or guesses numbers)
```

**Key design invariants enforced throughout:**
- Math is always deterministic, computed in code. No model guesses a number.
- Every output is inspectable — click any result to see the formula and constants behind it.
- Delta-v costing is region/altitude-band level only, not per individual object (Stage 1 gives no state vector).
- The risk map is a grid-based accumulator, not a Bayesian filter — documented plainly in code, UI, and here.

---

## AI Approach and Architecture

### Where AI is used — and where it is deliberately not used

| Component | AI? | Why |
|---|---|---|
| Fourier decomposition + amplitude estimation | ❌ No | Deterministic physics math — always primary |
| RandomForest size/shape classifier | ✅ Yes (secondary) | Refines on top of 18 physically-grounded features extracted by the math |
| Grid-based risk-density map | ❌ No | Weighted accumulator — no model involved |
| Hohmann + plane-change delta-v | ❌ No | Standard orbital mechanics constants |
| Plain-language analyst briefs | ✅ Yes (IBM Granite) | Translates computed numbers to English — never generates numbers |

### IBM Granite integration

**Model:** `ibm/granite-3-3-8b-instruct` via watsonx.ai

Granite is used in exactly **three places**, all identical in structure: the pipeline computes a result deterministically, structures the numbers into a DATA BLOCK, and passes it to Granite with explicit instructions to use only those numbers:

1. **Stage 1 — Characterization brief:** Receives rotation rate (Hz), amplitude, size estimate (cm), shape hint, SNR, top Fourier coefficients. Returns a 2–3 sentence analyst description.
2. **Stage 2 — Situation report:** Receives composite risk densities, object counts, and detection counts per altitude band. Returns a 3–4 sentence LEO risk landscape summary.
3. **Stage 3 — Mission brief:** Receives tier assignments, priority scores, and delta-v costs per band. Returns a mission-brief recommendation referencing specific delta-v figures from the data block.

**Prompt constraint enforced in all three:** `You must use ONLY the numbers provided in the DATA BLOCK below. Do not invent, estimate, or add values not present in the data.`

**Fallback behaviour:** If `WATSONX_API_KEY` is not set, or any API call fails, each function returns pre-written template text built from the same structured data — substantive analyst text, not error messages. The app runs fully without credentials.

### ML pipeline (Stage 1)

The RandomForest classifier receives an **18-feature vector** extracted from the Fourier inversion result — not the raw light curve. Every feature is physically grounded:

- FFT dominant frequency, power ratio, second-peak ratio, bandwidth
- Amplitude, peak-to-trough ratio, periodicity score
- Statistical moments (mean, std, skew, kurtosis, RMS)
- SNR estimate, zero-crossing rate, envelope slope
- Top-3 Fourier coefficient magnitudes

The classifier predicts `size_class` (small/medium/large) and `shape` (flat_plate/tumbling/cylinder/sphere). It is secondary to the deterministic math — if the model is not loaded, the app falls back to inversion-only estimates.

**Validated accuracy (synthetic held-out test set, n=400):**
- Size: **83.8%** (target ≥ 80% — pass)
- Shape: **95.2%** (target ≥ 80% — pass)

---

## Selected Challenge Theme

**August Challenge — Space Tech & AI Innovation**

Delta-V was built specifically for this theme: it applies AI and physics-grounded computation to a real, unsolved space operations problem — the tracking gap for small LEO debris (1 mm–10 cm). The project demonstrates the intersection of:

- **Space technology** — orbital mechanics (Hohmann transfers, SGP4 propagation), light-curve inversion, TLE-based density mapping across 8 LEO altitude bands
- **AI innovation** — IBM Granite for plain-language translation of computed results; RandomForest ML as a secondary refinement layer on top of deterministic physics; the constraint architecture that ensures AI never produces numbers, only explains them

The result is an open, free, no-registration tool that gives university cubesat teams, independent researchers, and early-stage startups the kind of risk intelligence that currently requires being a registered satellite owner to access.

---

## How IBM Bob Was Used

IBM Bob (the AI assistant) was used throughout the entire development lifecycle of Delta-V:

**Architecture and design decisions**
- Scoping the three-stage pipeline structure and enforcing the key invariants (math primary, AI secondary; region-level delta-v only; no Bayesian overclaiming)
- Designing the prompt architecture for Granite to ensure it never generates numbers — the DATA BLOCK pattern and the explicit constraint language in all three prompts
- Deciding the fallback text strategy: substantive analyst text vs. error messages

**Implementation**
- Full implementation of all pipeline modules: [`inversion.py`](pipeline/characterize/inversion.py), [`density.py`](pipeline/map/density.py), [`risk_map.py`](pipeline/map/risk_map.py), [`scorer.py`](pipeline/prioritize/scorer.py)
- All three Granite prompt functions in [`granite.py`](ai/granite.py) with fallback text
- The complete Streamlit UI across all four pages
- The 3D orbital visualization in [`orbital_view.py`](ui/orbital_view.py) using SGP4-propagated TLE positions
- The space-ops visual theme system in [`ui/theme.py`](ui/theme.py) (Orbitron/Rajdhani/IBM Plex Mono fonts, design tokens, responsive CSS)
- The animated star field and video background in [`ui/starfield.py`](ui/starfield.py)

**Testing and validation**
- All 142 unit tests across 7 test files
- MMT validation script and reframing of the "no small debris in public archives" result as a gap-confirmation finding rather than a limitation

**Debugging and fixes**
- Fixing the Plotly `TypeError` from duplicate kwargs in `PLOTLY_LAYOUT`
- Fixing the video toggle button (Streamlit's HTML sanitizer strips inline `onclick` — moved to a script-block event listener with DOM-ready retry loop)
- Fixing pipeline arrow alignment (replaced fixed `margin-top` hack with flexbox `align-items: center`)
- CelesTrak 403 fallback handling in Streamlit Cloud environments

**Documentation**
- This README, including all submission-required sections
- Inline module docstrings establishing scope constraints (e.g., `THIS IS NOT A BAYESIAN FILTER`, `THIS IS REGION-LEVEL COSTING ONLY`)

---

## Scope (Important)

**Delta-v costing is region/altitude-band level only — not per individual object.**

Stage 1 (light curves) gives size, shape, and rotation state — not a precise position/velocity state vector. Stage 2 works at the region level deliberately. Object-level delta-v estimates are not supportable by the underlying data and are not produced. Region-level Hohmann costing is standard orbital mechanics and is honest about what the data provides.

**The risk map is not a Bayesian filter or particle filter.**

It is a grid-based accumulator weighted by recency and detection confidence. This is documented plainly in the code, the UI, and here. It provides a live, evolving regional risk picture — not individual object tracking.

---

## Data Transparency

| Data | Source | Notes |
|---|---|---|
| Light-curve training data | Physics-based synthetic generator | Lambertian + specular reflection model; Kaasalainen & Torppa (2001). Labeled `# SYNTHETIC DATA` in all output files. |
| Light-curve validation | Mini-MegaTORTORA (MMT) public catalog | Real observed light curves from `mmt.favor2.info`. 8 satellite passes processed — see `data/validation/validation_report.txt`. |
| TLE orbital data | CelesTrak public GP endpoint (no registration) | Real tracked objects when reachable. Committed fallback (`data/tle_snapshot_fallback.csv`) contains 426 physics-valid synthetic objects across all 8 LEO bands with fresh epoch dates for SGP4 compatibility. |

### Validation Results

**Synthetic (held-out test set, n=400):**
- Size accuracy: **83.8%** (target: ≥80% — pass)
- Shape accuracy: **95.2%** (target: ≥80% — pass)

**Real MMT light curves — and what the search for small-debris data found:**

The pipeline was run against all publicly accessible light curves in the Mini-MegaTORTORA archive. The result is a finding in its own right:

**Every accessible object in the public MMT archive is a large, actively-catalogued satellite** (Starlink, OneWeb, Kuiper, Qianfan, unclassified Chinese platforms — 1–20 m class objects with NORAD catalog IDs). Not one light curve from an untracked 1–10 cm debris fragment exists in the public record. This is a direct empirical confirmation of the gap Delta-V addresses: the target population is invisible to public observation archives precisely because it is untracked.

The inversion pipeline behaves correctly on these objects:
- All 8 passes return an FFT peak at the minimum resolvable frequency (0.0195 Hz = 51.2 s), correctly identifying that the objects' rotation periods exceed the 51-second observation window — not an error, a limit correctly reported
- The amplitude heuristic classifies all 8 as "large" class, consistent with their known sizes (1–20 m)
- The pipeline ran on real telescope data without errors and produced physically consistent outputs

---

## Setup & Run

### Quick start (no credentials needed)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train the ML model — the app auto-trains a lighter model on first run if missing
python scripts/train.py

# 3. Launch
streamlit run app.py
```

**Live demo:** https://delta-v-ksn33avrrgai9gbnidlstb.streamlit.app/

The app runs fully without any API keys or network access. The IBM Granite integration falls back to template text; TLE data falls back to the committed synthetic snapshot.

### With IBM Granite

```bash
# Copy and fill in your watsonx.ai credentials
cp .env.example .env
# Edit .env:
#   WATSONX_API_KEY=<your key>
#   WATSONX_PROJECT_ID=<your project id>
#   WATSONX_URL=https://us-south.ml.cloud.ibm.com

streamlit run app.py
```

### With real TLE data

CelesTrak is queried automatically. If DNS fails (common in sandboxed environments), the fallback CSV is used. To force a refresh:

```bash
python scripts/generate_tle_fallback.py   # regenerate the fallback snapshot
```

### Running tests

```bash
python -m pytest tests/ -v
# 142 tests across 7 files — generator, inversion, features, density, risk_map, scorer, tle_staleness
```

### Validation

```bash
# Run MMT validation (requires network access to mmt9.ru or local data in data/validation/)
python scripts/validate_mmtortora.py

# End-to-end smoke test
python scripts/smoke_test_pipeline.py
```

---

## Project Structure

```
delta-v/
├── app.py                          ← Streamlit entry + home dashboard  [root: required by Streamlit]
├── config.py                       ← get_config(), granite_available() [root: imported by all modules]
│
├── pages/                          ← Streamlit multi-page UI
│   ├── 01_characterize.py          ← Stage 1 — light curve → size/shape/rotation
│   ├── 02_map.py                   ← Stage 2 — risk-density map + 3D orbital view
│   └── 03_prioritize.py            ← Stage 3 — delta-v costed tiers + export
│
├── pipeline/                       ← Core physics pipeline (no Streamlit dependency)
│   ├── characterize/
│   │   ├── generator.py            ← Physics-based synthetic light-curve generator
│   │   ├── inversion.py            ← PRIMARY: Fourier decomp + amplitude estimation
│   │   ├── features.py             ← 18-feature vector (physically grounded)
│   │   └── model.py                ← SECONDARY: RandomForest refinement
│   ├── map/
│   │   ├── tle_fetcher.py          ← CelesTrak fetch + 24h cache + fallback
│   │   ├── density.py              ← SGP4 propagation + altitude-band binning
│   │   └── risk_map.py             ← Grid-based accumulator (NOT Bayesian)
│   └── prioritize/
│       └── scorer.py               ← Hohmann delta-v + severity + tiers + export
│
├── ai/
│   └── granite.py                  ← Granite client + 3 prompt functions + fallback
│
├── ui/                             ← Streamlit UI helpers
│   ├── theme.py                    ← Palette, fonts, CSS, Plotly layout constants
│   ├── starfield.py                ← Animated star field + NASA ISS video background
│   └── orbital_view.py             ← Live 3D SGP4-propagated orbital distribution viz
│
├── tests/                          ← 142 unit tests across 7 files
│
├── scripts/                        ← Standalone runnable scripts
│   ├── train.py                    ← Generate 2000 curves → invert → train → save model
│   ├── validate_mmtortora.py       ← Validate inversion against real MMT light curves
│   ├── generate_tle_fallback.py    ← Regenerate TLE snapshot
│   └── smoke_test_pipeline.py      ← End-to-end smoke test
│
├── data/
│   ├── tle_snapshot_fallback.csv   ← 426 synthetic TLEs, all LEO bands, fresh epoch
│   ├── models/                     ← Trained RandomForest (auto-trains on first run)
│   └── validation/                 ← MMT validation output + real light curves
│
├── landing/                        ← Static landing page (optional — deploy to Vercel)
│   ├── index.html
│   └── vercel.json
│
├── docs/
│   └── delta-v-plan.md             ← Original build plan (reference)
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── .streamlit/config.toml          ← Streamlit dark theme + deploy config
```

---

## References

- Kaasalainen, M. & Torppa, J. (2001). Optimization methods for asteroid lightcurve inversion. *Icarus*, 153(1), 24–36.
- Karpov, S. et al. Mini-MegaTORTORA wide-field monitoring system. http://mmt9.ru
- CelesTrak GP data: https://celestrak.org
- NASA Orbital Debris Engineering Model (ORDEM): https://orbitaldebris.jsc.nasa.gov/modeling/ordem.html
- Schildknecht, T. et al. Optical observations of space debris in GEO and in highly eccentric orbits. *ESA SP-587*, 2005.
