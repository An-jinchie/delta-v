# Delta-V

> Physics-grounded space debris characterization, risk mapping, and delta-v-costed prioritization for LEO.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen.svg)](#running-tests)
[![Live Demo](https://img.shields.io/badge/live-demo-red.svg)](https://delta-v-ksn33avrrgai9gbnidlstb.streamlit.app/)

---

## The Problem

Small space debris (1 mm–10 cm) is too small for the U.S. Space Surveillance Network to individually track, yet capable of mission-ending damage at orbital velocities. **Less than 1% of debris in this danger category is currently tracked.**

Operators who most need this picture — university cubesat teams, early-stage space startups, independent researchers — are locked out of official conjunction data, which requires being a registered satellite owner/operator with an active cataloged object. And even where risk data exists, a score alone is not a decision: it doesn't say what it would cost to act on, or which threat deserves attention first.

---

## Pipeline

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

The pipeline was run against all publicly accessible light curves in the Mini-MegaTORTORA archive. The result is a finding in its own right, not just a limitation:

**Every accessible object in the public MMT archive is a large, actively-catalogued satellite** (Starlink, OneWeb, Kuiper, Qianfan, unclassified Chinese platforms — 1–20 m class objects with NORAD catalog IDs). Not one light curve from an untracked 1–10 cm debris fragment exists in the public record. This is a direct empirical confirmation of the gap Delta-V addresses: the target population is invisible to public observation archives precisely because it is untracked. There is no public ground-truth dataset to validate against because that dataset would require the infrastructure Delta-V is designed to help build.

The inversion pipeline behaves correctly on these objects:
- All 8 passes return an FFT peak at the minimum resolvable frequency (0.0195 Hz = 51.2 s), correctly identifying that the objects' rotation periods exceed the 51-second observation window — not an error, a limit correctly reported
- The amplitude heuristic classifies all 8 as "large" class, consistent with their known sizes (1–20 m)
- The pipeline ran on real telescope data without errors and produced physically consistent outputs

**Accuracy on the target population (small untracked debris, 1–10 cm):** Cannot be measured from public data — the target population is, by definition, untracked and unarchived. Accuracy figures (83.8% size / 95.2% shape) are reported on physics-grounded synthetic data, calibrated to the physical properties of small debris. See `data/validation/validation_report.txt` for the full per-curve analysis.

---

## IBM Technology

**IBM Granite (`ibm/granite-3-3-8b-instruct`)** via watsonx.ai is used in exactly three places:

1. **Characterize** — Translates inversion + ML results into a 2–3 sentence analyst brief
2. **Map** — Writes a situation report on the current regional risk landscape
3. **Prioritize** — Writes a mission-brief recommendation for top-tier bands including delta-v context

**Granite never produces or guesses numbers.** Every risk score, delta-v figure, and priority ranking is computed deterministically by the pipeline. Granite receives only the computed numbers in its prompt and translates them into plain English. If no credentials are set, the app falls back to substantive template text built from the same computed data — not error messages.

---

## Setup & Run

### Quick start (no credentials needed)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train the ML model — the app auto-trains a lighter model on first run if missing
python train.py

# 3. Launch
streamlit run app.py
```

**Live demo:** [https://delta-v-ksn33avrrgai9gbnidlstb.streamlit.app/](https://delta-v-ksn33avrrgai9gbnidlstb.streamlit.app/)

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

### Validation

```bash
# Run MMT validation (requires network access to mmt9.ru or local data in data/validation/)
python validate_mmtortora.py

# Run all unit tests
python -m pytest tests/ -v
```

---

## Project Structure

```
delta-v/
├── app.py                          ← Streamlit entry + home dashboard
├── pages/
│   ├── 01_characterize.py          ← Stage 1 UI (light curve → size/shape/rotation)
│   ├── 02_map.py                   ← Stage 2 UI (risk-density map + situation report)
│   └── 03_prioritize.py            ← Stage 3 UI (delta-v costed tiers + export)
├── pipeline/
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
├── ai/
│   └── granite.py                  ← Granite client + 3 prompt functions + fallback
├── data/
│   ├── tle_snapshot_fallback.csv   ← 426 synthetic TLEs, all LEO bands, fresh epoch
│   ├── models/                     ← Trained RandomForest (gitignored after train.py)
│   └── validation/                 ← MMT validation output (populated by validate_mmtortora.py)
├── tests/                          ← 131 passing unit tests
├── scripts/
│   ├── generate_tle_fallback.py    ← Regenerate TLE snapshot
│   └── smoke_test_pipeline.py      ← End-to-end smoke test
├── train.py                        ← Generate 2000 curves → invert → train → save model
├── validate_mmtortora.py           ← Validate inversion against real MMT light curves
├── config.py                       ← get_config(), granite_available()
├── requirements.txt
├── .env.example
└── .streamlit/config.toml          ← Streamlit theme + deploy config
```

---

## References

- Kaasalainen, M. & Torppa, J. (2001). Optimization methods for asteroid lightcurve inversion. *Icarus*, 153(1), 24–36.
- Karpov, S. et al. Mini-MegaTORTORA wide-field monitoring system. http://mmt9.ru
- CelesTrak GP data: https://celestrak.org
- NASA Orbital Debris Engineering Model (ORDEM): https://orbitaldebris.jsc.nasa.gov/modeling/ordem.html
- Schildknecht, T. et al. Optical observations of space debris in GEO and in highly eccentric orbits. *ESA SP-587*, 2005.
