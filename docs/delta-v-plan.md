# Delta-V — Build Plan (v2 — Final Spec)

## Top-Level Overview

**Project:** Delta-V — Space debris characterization, risk mapping, and remediation
prioritization for the IBM AI Builders Challenge (August).

**The Problem:** Small debris (1mm–10cm) is too small for the U.S. Space Surveillance Network to
individually track, yet capable of mission-ending damage at orbital velocities. Less than 1% of
debris in this danger category is currently tracked. University cubesat teams, early-stage
startups, and independent researchers are locked out of official conjunction data (requires being a
registered operator with an active cataloged object). And even where risk data exists, a score alone
is not a decision — it doesn't say what it would cost to act on, or which threat deserves priority.

**Three-Stage Pipeline:**

1. **Characterize** — Given a debris light curve (brightness vs. time), estimate size, shape, and
   rotation state using deterministic light-curve inversion math (Fourier decomposition +
   amplitude-based estimation) as the primary method. ML is used only to refine/classify on top of
   those extracted physical features — never as a black-box replacement for the math.
   Validated against real published light-curve data from the Mini-MegaTORTORA database. Accuracy
   is reported honestly, including where it degrades on noisy real-world data.

2. **Map** — Characterized detections combine into a grid-based probabilistic risk-density map
   across LEO altitude/region bands, recomputed with each new batch of detections, weighted by
   recency and confidence. Explicitly NOT a full Bayesian or particle filter — documented as such.
   A live, evolving risk picture, not a static one-off report.

3. **Prioritize & Act** — Each high-risk *region* (not individual object) is costed using standard
   orbital mechanics (Hohmann transfer + plane-change delta-v). Prioritization formula:
   `risk_density × severity ÷ delta_v_cost`, sorted into tiers: HIGH-PRIORITY / MONITOR / LOW.
   Exports as a structured CSV/JSON with delta-v cost attached to every entry.

**Scoping Constraint (hard):** Delta-v costing is computed at the *region/altitude-band level only*.
Stage 1 gives size/shape/rotation — not a precise state vector. Stage 2 works at region level
deliberately. Object-level delta-v estimates are not supportable by the underlying data and will not
be produced.

**Design Principles:**
- Math is always deterministic, computed in code. AI translates structured results into
  plain-language explanations only — no model ever guesses a risk number or delta-v figure.
- Every output is inspectable — any risk score, priority tier, or delta-v estimate links back to
  the real numbers behind it.
- CelesTrak public GP endpoint (no registration) for tracked-object context.
- Mini-MegaTORTORA for light-curve validation.
- README states plainly which results come from real data vs. physics-grounded synthetic data.
- Final prioritized list exports as CSV/JSON with delta-v cost on every entry.

**IBM Technology Integration:**
- **IBM Granite (granite-3-3-8b-instruct)** via watsonx.ai Inference API: generates natural
  language explanations, situation reports, and mission-brief recommendations from structured
  pipeline outputs. Granite never produces numbers — it translates numbers already computed.
- watsonx.ai free Lite tier (no credit card required)
- Graceful fallback to pre-written template text if credentials are not provided

**Stack:**
- Language: Python 3.11
- UI: Streamlit (deployed free on Streamlit Cloud)
- Math: numpy, scipy (FFT, Fourier decomposition, Kepler/Hohmann mechanics)
- ML: scikit-learn RandomForestClassifier (on physically-extracted features, not raw curves)
- Orbital data: CelesTrak GP endpoint (no auth); sgp4 for TLE propagation
- IBM AI: ibm-watsonx-ai SDK → watsonx.ai granite-3-3-8b-instruct
- Visualization: Plotly (interactive charts)
- Key libs: sgp4, pandas, numpy, scipy, python-dotenv, joblib

---

## Realism Flags (Updated)

| Risk Item | Status | Resolution |
|---|---|---|
| Real light-curve data availability | ✅ Addressed | Mini-MegaTORTORA is public. Validation section uses it and reports accuracy honestly including noise degradation. |
| Object-level delta-v costing | ✅ Corrected | Removed from scope. Region/altitude-band costing only, using Hohmann + plane-change. Documented in README. |
| "Bayesian risk map" overclaim | ✅ Corrected | Explicitly grid-based, recency+confidence weighted. README states it is NOT a particle filter. |
| ML as primary characterization method | ✅ Corrected | Math (Fourier decomp, amplitude estimation) is primary. ML refines on top of extracted features. |
| watsonx.ai free tier limits | ✅ Managed | Granite used only for explanation text. Short prompts, ≤300 tokens. Graceful fallback. |
| Space-Track.org auth | ✅ Managed | CelesTrak is default (no auth). Space-Track optional via .env. |
| Real-time TLE streaming | ✅ Scoped | Snapshot-on-demand with Refresh button. Not live streaming. |
| 3D globe visualization | 🔵 Stretch | 2D altitude-band Plotly charts primary. 3D globe stretch goal only. |
| PyTorch / deep learning | ✅ Managed | scikit-learn RandomForest on physically-extracted features. Fast, explainable, no GPU. |
| Synthetic vs. real data honesty | ✅ Enforced | README and UI both label synthetic data. Validation section distinguishes. |

---

## Project Structure

```
delta-v/
├── app.py                            # Streamlit entry point + home dashboard
├── train.py                          # One-time: extract features → train → save model
├── validate_mmtortora.py             # Validation script: run characterizer on MMT data, report accuracy
├── config.py                         # Env loader, typed config, feature flags
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── cache/                        # TLE snapshots (gitignored; fallback committed separately)
│   ├── synthetic/                    # Generated light-curve CSVs + impact evidence (gitignored)
│   ├── models/                       # Saved sklearn model artifacts (gitignored)
│   ├── validation/                   # Mini-MegaTORTORA sample curves for validation
│   └── tle_snapshot_fallback.csv     # Committed fallback TLE snapshot for offline/demo
├── pipeline/
│   ├── __init__.py
│   ├── characterize/
│   │   ├── __init__.py
│   │   ├── generator.py              # Synthetic light-curve generator (physics-based)
│   │   ├── inversion.py              # PRIMARY: Fourier decomp + amplitude-based physical estimation
│   │   ├── features.py               # Feature extraction from inversion output (feeds ML)
│   │   └── model.py                  # RandomForest classifier: train/save/load/predict (secondary)
│   ├── map/
│   │   ├── __init__.py
│   │   ├── tle_fetcher.py            # CelesTrak + Space-Track TLE ingestion with caching
│   │   ├── density.py                # SGP4 propagation + altitude binning
│   │   └── risk_map.py               # Grid-based risk-density map: recency+confidence weighted
│   └── prioritize/
│       ├── __init__.py
│       └── scorer.py                 # Hohmann+plane-change delta-v → risk×severity÷dv → tiers
├── ai/
│   ├── __init__.py
│   └── granite.py                    # watsonx.ai Granite client: explain/report/brief
├── pages/
│   ├── 01_characterize.py            # Stage 1 page
│   ├── 02_map.py                     # Stage 2 page
│   └── 03_prioritize.py              # Stage 3 page + CSV/JSON export
└── tests/
    ├── test_generator.py
    ├── test_inversion.py
    ├── test_features.py
    ├── test_density.py
    ├── test_risk_map.py
    └── test_scorer.py
```

---

## Week 1 — Core Pipeline (No UI)

### Sub-Task 1 — Project Scaffold

**Status:** `[ ] pending`

**Intent:**
Establish the full project structure, dependency manifest, configuration system, and README
skeleton. Every subsequent sub-task builds on this foundation.

**Expected Outcomes:**
- All directories and `__init__.py` files exist
- `requirements.txt` lists all pinned dependencies
- `.env.example` documents all environment variables
- `config.py` loads and exposes typed config with documented defaults
- `README.md` skeleton includes project name, description, pipeline overview, data transparency
  disclaimer, and explicit scope notes (not Bayesian, not object-level delta-v)
- `app.py` runs without error and shows a placeholder Delta-V title page

**Todo List:**
1. Create full directory tree with placeholder `__init__.py` files
2. Write `requirements.txt`:
   streamlit, scikit-learn, numpy, pandas, plotly, requests, python-dotenv, sgp4, scipy,
   joblib, ibm-watsonx-ai, pytest
3. Write `.env.example`:
   ```
   DATA_SOURCE=celestrak            # celestrak | spacetrack
   SPACETRACK_USER=
   SPACETRACK_PASS=
   WATSONX_API_KEY=
   WATSONX_PROJECT_ID=
   WATSONX_URL=https://us-south.ml.cloud.ibm.com
   ```
4. Write `config.py` — uses `python-dotenv`, exposes `get_config() -> dict`
5. Write `README.md` skeleton with: title, problem statement, pipeline diagram (ASCII),
   explicit scope section (grid-based not Bayesian, region-level not object-level delta-v),
   data transparency section (synthetic light curves labeled, real TLE sourced from CelesTrak,
   validation against Mini-MegaTORTORA)
6. Write `app.py` — Streamlit shell with sidebar navigation and placeholder home page

**Relevant Context:** This is the foundation.

---

### Sub-Task 2 — Synthetic Light-Curve Generator

**Status:** `[ ] pending`

**Intent:**
Implement a physics-based synthetic light-curve generator. This provides training data for the ML
classifier and a known-ground-truth baseline for validating the inversion math against a controlled
signal before testing on noisy real data.

**Expected Outcomes:**
- `pipeline/characterize/generator.py` generates a brightness time-series (numpy array) given:
  size class (small/medium/large), shape (sphere/flat_plate/cylinder/tumbling),
  rotation rate, phase angle, albedo, SNR, and random seed
- Output is reproducible given the same seed
- `generate_dataset(n_samples, output_path, seed)` writes a labeled CSV with columns:
  `[t_0...t_N (brightness values), size_class, shape, rotation_rate_hz]`
- Physical model documented in code comments (simplified Lambertian + specular reflection,
  referencing Kaasalainen & Torppa 2001)
- `tests/test_generator.py` asserts: correct output shape, values in [0,1] range, reproducibility

**Todo List:**
1. Implement `LightCurveGenerator` class with `generate(params, seed) -> np.ndarray`:
   - Base brightness: `B(t) = albedo * A_proj(t) * (lambertian_term + specular_highlight)`
   - `A_proj(t)` varies with rotation phase per shape:
     - sphere: constant projected area
     - flat plate: `|cos(2π * rotation_rate * t)|`
     - cylinder: mixed cosine + constant baseline
     - tumbling: sum of two sinusoids at different frequencies (two-axis tumble)
   - Gaussian noise: `B_noisy = B + N(0, sigma)`, `sigma = max(B) / SNR`
   - Normalize output to [0, 1]
2. Implement `generate_dataset(n_samples, output_path, seed)`:
   - Sample parameters across all size/shape combinations
   - Save to CSV with header comment: `# SYNTHETIC DATA — see README`
3. Write `tests/test_generator.py`

**Reference:** Kaasalainen & Torppa (2001). Sinusoidal approximation with noise is honest
and sufficient for this scope. Cited in README.

---

### Sub-Task 3 — Light-Curve Inversion (Primary Characterization Math)

**Status:** `[ ] pending`

**Intent:**
Implement the deterministic light-curve inversion module that forms the primary characterization
method. This is where size, shape, and rotation state are actually *estimated from physics* — ML
in Sub-Task 4 only refines on top of what this module produces. This separation is architecturally
and scientifically important: every output traces to a computable formula, not a model weight.

**Expected Outcomes:**
- `pipeline/characterize/inversion.py` exposes `invert(light_curve: np.ndarray, cadence_s: float)
  -> InversionResult` (dataclass)
- `InversionResult` contains:
  - `rotation_rate_hz: float` — dominant period from FFT peak
  - `rotation_rate_uncertainty: float` — half-width of FFT peak bin
  - `amplitude: float` — peak-to-trough ratio of the normalized curve
  - `size_estimate_m: float` — amplitude-based size estimate (documented formula)
  - `shape_hint: str` — "sphere" | "flat_plate" | "cylinder" | "tumbling" | "unknown"
  - `fourier_coefficients: np.ndarray` — first N Fourier coefficients (configurable, default N=8)
  - `snr_estimate: float` — estimated SNR of the input curve
- All estimates include a documented derivation comment in code
- `tests/test_inversion.py` asserts: known synthetic inputs return correct rotation rate and
  amplitude within tolerance; sphere input returns shape_hint "sphere"

**Todo List:**
1. Implement Fourier decomposition in `inversion.py`:
   - Compute FFT of the light curve
   - Extract dominant frequency: `rotation_rate_hz = argmax(|FFT|[1:N/2]) / (N * cadence_s)`
   - Estimate rotation_rate_uncertainty from FFT bin width
   - Extract first N Fourier coefficients (real part of FFT / N, normalized)
2. Implement amplitude-based size estimation:
   - `amplitude = (max(lc) - min(lc)) / mean(lc)` (peak-to-trough ratio)
   - Size heuristic (documented as heuristic, not ground truth):
     - amplitude < 0.1 → "small" (≈ 1–3cm), likely near-spherical
     - 0.1 ≤ amplitude < 0.4 → "medium" (≈ 3–7cm)
     - amplitude ≥ 0.4 → "large" (≈ 7–10cm), likely tumbling/flat
   - `size_estimate_m`: map size class to midpoint estimate
3. Implement shape hinting from Fourier coefficient structure:
   - Constant (low variance): sphere
   - Single dominant frequency + low harmonics: flat plate
   - Two comparable frequencies: tumbling
   - High harmonic content: cylinder or complex tumbling
4. Estimate SNR from ratio of signal power to high-frequency noise floor
5. Write `tests/test_inversion.py`

---

### Sub-Task 4 — Feature Extraction + ML Classifier (Secondary Refinement)

**Status:** `[ ] pending`

**Intent:**
Extract a feature vector that combines the inversion output (physical estimates) with statistical
curve descriptors, then train a RandomForest classifier to refine the shape and size labels.
The key architectural point: the ML model's input is the *inversion output + derived features*,
not the raw light curve. This keeps the ML interpretable and grounded in physics.

**Expected Outcomes:**
- `pipeline/characterize/features.py` extracts ~15–18 features from an `InversionResult` plus
  the raw light curve (for statistical features)
- `pipeline/characterize/model.py` wraps two RandomForestClassifiers (size, shape) with
  train/save/load/predict methods
- `train.py` at root: calls generator → inversion → feature extraction → train/test split →
  trains → saves → prints `classification_report`
- Classifier achieves >80% accuracy on held-out synthetic test split
- `tests/test_features.py` verifies feature extraction produces correct shape, no NaN values

**Todo List:**
1. Implement `extract_features(lc: np.ndarray, inv: InversionResult) -> np.ndarray`
   in `features.py`:
   - From InversionResult: rotation_rate_hz, rotation_rate_uncertainty, amplitude,
     size_estimate_m, snr_estimate, fourier_coefficients[0:4] (4 features), shape_hint encoded
   - Statistical from raw curve: mean, std, skewness, kurtosis, peak count,
     coefficient of variation, 10th/90th percentile range
   - Total: ~18 features
2. Implement `CharacterizeModel` in `model.py`:
   - `train(X, y_size, y_shape)` — fits two RandomForestClassifiers (n_estimators=100)
   - `save(path)` / `load(path)` — joblib.dump / joblib.load
   - `predict(lc, inv) -> dict` — returns
     `{size_class, shape, size_confidence, shape_confidence, inversion_result}`
   - The returned dict includes the inversion_result so every prediction is inspectable
3. Write `train.py`:
   - `generate_dataset(n_samples=2000)` → invert each curve → extract features →
     train/test split 80/20 → train model → print `classification_report` → save
4. Write `tests/test_features.py`

---

### Sub-Task 5 — Mini-MegaTORTORA Validation

**Status:** `[ ] pending`

**Intent:**
Validate the characterization pipeline against real published light-curve data from the
Mini-MegaTORTORA (MMT) database. This is the scientific honesty checkpoint. The validation must
report accuracy metrics including where they degrade on noisy real-world data — the README and UI
must state this plainly. Real measured light curves are noisier than synthetic ones.

**Expected Outcomes:**
- `data/validation/` contains at least 10–20 real MMT light-curve samples (CSV format)
- `validate_mmtortora.py` runs the full inversion + ML prediction on each sample and produces
  a validation report: accuracy on rotation rate estimation, shape hint accuracy, size class
  accuracy, and notes on noise-induced degradation
- Validation report printed to stdout and saved as `data/validation/validation_report.txt`
- README validation section cites results honestly, including failure modes

**Todo List:**
1. Source Mini-MegaTORTORA light curves:
   - MMT database: http://mmt9.ru/index.html (Karpov et al. public catalog)
   - Download 10–20 light curves from the public catalog covering known object types
   - Save as CSVs in `data/validation/` with source metadata (object ID, observation date)
2. Implement `validate_mmtortora.py`:
   - Load each validation curve
   - Run `invert()` → compute rotation rate estimate
   - Run `CharacterizeModel.predict()` on extracted features
   - Compare inversion rotation rate vs. published period (where available)
   - Report: mean absolute error on rotation rate, shape hint match rate, size class accuracy
   - Report noise analysis: compute SNR of each real curve vs. synthetic training data
   - Flag where SNR is below training distribution (honest degradation report)
3. Write a plain-English summary paragraph for the README:
   "On synthetic data: X% shape accuracy, Y% size accuracy. On MMT real curves (n=Z):
   rotation rate MAE = W Hz. Shape accuracy drops to X% on real data due to higher noise
   floor (median SNR = V vs. synthetic SNR > U). Results are physics-grounded estimates,
   not precise state vectors."
4. Add `data/validation/` to `.gitignore` exception so validation curves are committed

**Note:** If MMT download proves inaccessible at time of build (network, format change), fall back
to Schildknecht et al. published light-curve tables (freely available in ESA proceedings). Document
whichever source was actually used.

---

### Sub-Task 6 — TLE Ingestion + Debris Density Map

**Status:** `[ ] pending`

**Intent:**
Fetch real Two-Line Element orbital data, propagate each object to current epoch using SGP4,
bin by altitude into LEO bands, and compute object count density. Foundation of the risk map.

**Expected Outcomes:**
- `pipeline/map/tle_fetcher.py` fetches TLE data from CelesTrak GP endpoint (or Space-Track),
  with 24h cache and graceful fallback to committed snapshot
- `pipeline/map/density.py` produces a DataFrame:
  `[altitude_band_km, band_label, object_count, band_width_km, density_per_km]`
- Altitude bands (200–2000km, 200km steps):
  200–400, 400–600, 600–800, 800–1000, 1000–1200, 1200–1400, 1400–1600, 1600–2000
- `data/tle_snapshot_fallback.csv` committed to repo for offline/demo use
- `tests/test_density.py` loads the fallback snapshot and asserts expected output structure

**Todo List:**
1. Implement `TLEFetcher` in `tle_fetcher.py`:
   - `fetch(force_refresh=False) -> list[str]`
   - CelesTrak GP endpoint: `https://celestrak.org/SOCRATES/query.php` or catalog GP:
     `https://celestrak.org/pub/TLE/catalog.txt`
   - Cache: write raw TLE text to `data/cache/tle_cache.txt` with timestamp sidecar JSON
   - Fallback: cache → committed fallback CSV
2. Implement `compute_density(tle_lines: list[str]) -> pd.DataFrame` in `density.py`:
   - Parse TLE pairs with `sgp4.api.Satrec.twoline2rv`
   - Propagate each object to current UTC: position vector → altitude = `|r| - 6371`
   - Filter to 200–2000km; bin with `pd.cut`
3. Commit a real TLE snapshot as `data/tle_snapshot_fallback.csv`
4. Write `tests/test_density.py`

---

### Sub-Task 7 — Grid-Based Risk-Density Map

**Status:** `[ ] pending`

**Intent:**
Combine TLE density data with characterized detections to produce a recency- and
confidence-weighted risk-density map per altitude band. This is explicitly NOT a Bayesian filter
or particle filter — it is a grid-based accumulator. Documentation says so plainly.

**Expected Outcomes:**
- `pipeline/map/risk_map.py` produces a risk-scored DataFrame:
  `[altitude_band_km, band_label, tracked_object_count, density_per_km,
   detection_count, detection_confidence_weighted, recency_weight,
   composite_risk_density, last_updated]`
- Risk map is additive: calling `update(new_detections)` adds to existing state with recency decay
- `tests/test_risk_map.py` verifies: higher-density bands score higher; older detections
  decay correctly; output contains no null values

**Todo List:**
1. Implement `RiskDensityMap` class in `risk_map.py`:
   - Internal state: DataFrame of detections with `[altitude_band, confidence, timestamp]`
   - `update(detections: list[dict])`: append new detections to state
   - `compute() -> pd.DataFrame`:
     - Start from TLE density DataFrame
     - For each band: compute `detection_weight = sum(confidence * recency_weight)`
       where `recency_weight = exp(-age_days / decay_half_life)`, `decay_half_life=7`
     - `composite_risk_density = w1 * density_norm + w2 * detection_weight_norm`
     - Default weights: w1=0.5, w2=0.5 (tunable)
   - Include `last_updated` timestamp on output
2. Add a docstring: "This is a grid-based accumulator, not a Bayesian filter or particle
   filter. It does not track individual objects. It provides a live, evolving regional
   risk picture updated with each new detection batch."
3. Write `tests/test_risk_map.py`

---

### Sub-Task 8 — Prioritization: Hohmann Delta-V + Risk Tiers

**Status:** `[ ] pending`

**Intent:**
For each high-risk altitude band, compute the delta-v a remediation or avoidance action targeting
that region would require, then weigh it against the risk removed. Use standard orbital mechanics
(Hohmann transfer + plane-change). Output tiers: HIGH-PRIORITY / MONITOR / LOW. Export as
CSV/JSON with delta-v cost on every entry.

**This is region-level costing only.** Stage 1 gives size/shape/rotation — not a state vector.
Stage 2 works at region level. Object-level delta-v is not supportable.

**Expected Outcomes:**
- `pipeline/prioritize/scorer.py` exposes `PriorityScorer.score(risk_df) -> pd.DataFrame`
- Output DataFrame columns:
  `[rank, band_label, altitude_km_mid, composite_risk_density, severity_index,
   delta_v_hohmann_ms, delta_v_plane_change_ms, delta_v_total_ms,
   priority_score, tier, explanation_text]`
- `tier` values: "HIGH-PRIORITY" | "MONITOR" | "LOW"
- `priority_score = composite_risk_density × severity_index / delta_v_total_ms`
- Export function: `export_csv(df, path)` and `export_json(df, path)`
- `tests/test_scorer.py` validates: delta-v increases with altitude, priority ordering is
  correct, tiers are assigned to all rows, no null values

**Todo List:**
1. Implement Hohmann transfer delta-v in `scorer.py`:
   - Reference orbit: 400km circular (notional chase vehicle starting orbit)
   - Target: midpoint of each altitude band
   - Semi-major axes: `a1 = 6371 + 400`, `a2 = 6371 + altitude_mid` (all in km)
   - Transfer orbit: `a_t = (a1 + a2) / 2`
   - `dv1 = sqrt(GM/a1) * (sqrt(2*a2/(a1+a2)) - 1)` [first burn, km/s]
   - `dv2 = sqrt(GM/a2) * (1 - sqrt(2*a1/(a1+a2)))` [second burn, km/s]
   - `dv_hohmann = |dv1| + |dv2|` → convert to m/s
   - `GM = 398600.4418 km³/s²`
2. Implement plane-change estimate:
   - Assume worst-case 5° plane change at apoapsis (representative for LEO debris fields)
   - `dv_plane = 2 * v_apoapsis * sin(5° / 2)`
   - `v_apoapsis = sqrt(GM * (2/a2 - 1/a_t))`
   - `dv_total = dv_hohmann + dv_plane` (m/s)
3. Implement severity index:
   - `severity_index = size_class_weight * kinetic_energy_proxy`
   - `size_class_weight`: small=1.0, medium=2.5, large=5.0
   - `kinetic_energy_proxy`: proportional to (size_estimate_m)² × relative_velocity²
   - Relative velocity: 7.5 km/s mean LEO relative velocity (constant, documented)
4. Implement tier assignment:
   - Compute `priority_score` for all bands
   - Normalize to [0, 1]; tier thresholds: HIGH-PRIORITY ≥ 0.6, MONITOR 0.3–0.6, LOW < 0.3
5. Implement `export_csv(df, path)` and `export_json(df, path)`
6. Write `tests/test_scorer.py`

---

## Week 2 — IBM Granite, UI, Integration, Deployment

### Sub-Task 9 — IBM Granite Integration

**Status:** `[ ] pending`

**Intent:**
Wire in IBM Granite (granite-3-3-8b-instruct) via watsonx.ai to generate natural language
explanations from structured pipeline data. Granite translates numbers already computed into
plain language — it never produces or guesses numbers itself. This is the IBM technology showcase.

**Granite is used in three places:**
- **Characterize:** "This debris fragment shows a rotation rate of X Hz (±Y Hz), estimated size
  class [medium], shape consistent with [tumbling]. Key indicators: amplitude = Z, Fourier
  coefficients show two competing frequencies. Confidence: A%."
- **Map:** Situation report paragraph on current risk landscape across LEO bands
- **Prioritize:** Mission-brief recommendation for top-tier bands, with delta-v context

**Expected Outcomes:**
- `ai/granite.py` exposes three functions: `explain_characterization()`, `write_situation_report()`,
  `write_mission_brief()`
- Each builds a structured prompt from computed numbers, calls watsonx.ai, returns a string
- If `WATSONX_API_KEY` is not set, each returns a meaningful pre-written fallback template
- Responses cached in `st.session_state` to avoid re-calling on every Streamlit rerender
- Token usage minimal: structured data summaries, responses capped at 300 tokens
- Prompts explicitly instruct Granite not to invent numbers — only explain the provided data

**Todo List:**
1. Set up IBM Cloud Lite account (cloud.ibm.com, free, no credit card):
   - Create watsonx.ai instance → create project → generate API key
   - Note WATSONX_API_KEY and WATSONX_PROJECT_ID
2. Implement `GraniteClient` in `ai/granite.py`:
   - `__init__`: initialize `ibm_watsonx_ai.foundation_models.ModelInference`
     with model_id=`ibm/granite-3-3-8b-instruct`, credentials, project_id
   - `generate(prompt, max_tokens=300) -> str`
   - try/except → log error → return fallback on any failure
3. Implement `explain_characterization(prediction: dict, inv_result: InversionResult) -> str`:
   - Prompt includes: rotation_rate_hz, amplitude, size_estimate_m, shape, confidence,
     top Fourier coefficients
   - Instruction: "Explain in 2–3 sentences as if briefing a mission analyst. Use only the
     numbers provided. Do not invent or estimate additional values."
4. Implement `write_situation_report(risk_df: pd.DataFrame) -> str`:
   - Top 3 riskiest bands, their composite_risk_density, detection counts, recency
   - 3–4 sentence situation report
5. Implement `write_mission_brief(priority_df: pd.DataFrame) -> str`:
   - Top 3 HIGH-PRIORITY bands, priority_score, delta_v_total_ms, tier
   - Concise mission-brief paragraph; include delta-v context ("requiring ~X m/s total delta-v")
6. Fallback strings must be substantive, not placeholder error messages

---

### Sub-Task 10 — Streamlit Pages (All Three Stages)

**Status:** `[ ] pending`

**Intent:**
Build all three Streamlit stage pages, each visualizing one pipeline stage with full inspectability
(numbers behind every output), the Granite AI explanation, and clean navigation between stages.

**Expected Outcomes:**
- `pages/01_characterize.py`: light curve → inversion results → ML refinement → Granite explanation
- `pages/02_map.py`: TLE refresh → risk-density map → Granite situation report
- `pages/03_prioritize.py`: priority tiers → delta-v table → Granite brief → CSV/JSON export
- Each output card shows its underlying computed values (inspectable)
- Shared sidebar: data source toggle, Space-Track credentials (masked), Granite status indicator

**Todo List:**

**Page 1 — Characterize (`pages/01_characterize.py`):**
1. Two tabs: "Generate Demo Curve" (sliders: shape, rotation rate, SNR) and "Upload CSV"
2. Plot light curve with Plotly line chart
3. "Inversion Results" expander: rotation_rate_hz, amplitude, shape_hint, size_estimate_m,
   fourier_coefficients — all shown as computed values (inspectable)
4. "ML Refinement" section: `st.metric` cards for size_class, shape, confidence
5. Granite explanation in `st.info` box with IBM Granite badge
6. Cache model with `@st.cache_resource`

**Page 2 — Map (`pages/02_map.py`):**
1. Sidebar: data source radio + credential fields + Refresh button
2. "Last updated" timestamp, object count, detection count
3. Plotly horizontal bar chart: altitude bands on Y, composite_risk_density on X, color by risk
4. "Map Methodology" expander: plain English — grid-based accumulator, NOT Bayesian filter,
   recency decay formula, data sources, synthetic data disclaimer
5. Expandable raw data table (all columns)
6. Granite situation report in `st.info`

**Page 3 — Prioritize (`pages/03_prioritize.py`):**
1. Priority tier cards: HIGH-PRIORITY bands highlighted in red, MONITOR in amber, LOW in green
2. Plotly bar chart of priority_score per band
3. Full ranked table with columns: band, tier, priority_score, delta_v_total_ms,
   composite_risk_density, severity_index — all visible
4. "Delta-V Methodology" expander: Hohmann transfer formula, plane-change assumption (5°),
   reference orbit (400km), GM value used — fully inspectable
5. Granite mission brief in `st.success`
6. Two download buttons: CSV export and JSON export

---

### Sub-Task 11 — Home Dashboard + Full Integration

**Status:** `[ ] pending`

**Intent:**
Build the `app.py` home page as a compelling entry point and wire all three stages into an
end-to-end summary view. First thing judges see.

**Expected Outcomes:**
- Home page: Delta-V title + tagline + pipeline flow diagram + live summary dashboard
- Summary dashboard: top-tier band, its delta-v cost, top risk band, Granite status —
  all from `st.session_state` if pipeline has been run, showing "--" otherwise
- "Run Full Pipeline Demo" button runs all three stages in sequence
- All session state keys documented at top of `app.py`

**Todo List:**
1. Build home page:
   - Header: Delta-V title, tagline ("Physics-grounded debris characterization, risk mapping,
     and delta-v-costed prioritization for LEO.")
   - Pipeline flow: three columns — Characterize / Map / Prioritize — each with stage name,
     one-sentence description, and → arrow
   - 4 `st.metric` cards: Top Risk Band, Top Priority Tier, Delta-V Cost (ms), Granite Status
2. Implement "Run Full Pipeline Demo":
   - Generate demo light curve → invert → extract features → predict → store in session_state
   - Fetch TLE (use cache) → compute density → update risk map → store
   - Score priorities → store
   - Rerun to populate summary metrics
3. Initialize all session state keys on first load

---

### Sub-Task 12 — README, Deployment, and Submission Polish

**Status:** `[ ] pending`

**Intent:**
Write the final README, deploy to Streamlit Cloud, and prepare submission artifacts.
The README is a primary judging artifact.

**Expected Outcomes:**
- `README.md` complete, submission-ready, honest about data provenance
- App deployed at a live public Streamlit Cloud URL
- `data/tle_snapshot_fallback.csv` committed
- `.env.example` accurate
- 2–3 minute demo video covering all three stages + Granite explanations

**Todo List:**
1. Final `README.md` — must include:
   - Title (Delta-V), tagline, problem statement
   - Pipeline architecture (ASCII diagram)
   - **Scope section**: grid-based map (not Bayesian), region-level delta-v (not object-level),
     why these constraints are scientifically correct given the underlying data
   - **Data transparency**: synthetic light curves (physics-based, Kaasalainen & Torppa 2001),
     real TLE from CelesTrak, validation against Mini-MegaTORTORA (cite results + degradation)
   - **Validation results**: accuracy on synthetic vs. real data (actual numbers from Sub-Task 5)
   - IBM Technology: what Granite does (explain numbers, not produce them)
   - Setup: `pip install -r requirements.txt`, `.env` setup, `python train.py`,
     `python validate_mmtortora.py`, `streamlit run app.py`
   - Live demo URL
   - References: CelesTrak, Kaasalainen & Torppa (2001), Karpov et al. (MMT), NASA ORDEM
2. Commit fresh CelesTrak TLE snapshot
3. Deploy on Streamlit Cloud: connect GitHub, set secrets
4. Verify all three pages at live URL
5. Record demo video: home → Characterize (inversion results + ML + Granite) →
   Map (risk chart + methodology expander) → Prioritize (tiers + delta-v table + export)

---

## Build Schedule (12-Day Sprint)

| Day | Focus | Sub-Task | Checkpoint |
|---|---|---|---|
| **Day 1** | Scaffold | 1 | `streamlit run app.py` shows placeholder |
| **Day 2** | Generator | 2 | `tests/test_generator.py` passes |
| **Day 3** | Inversion math | 3 | `tests/test_inversion.py` passes; known input → correct rotation rate |
| **Day 4** | Features + ML | 4 | `python train.py` → >80% accuracy report |
| **Day 5** | TLE ingestion + density | 6 | `compute_density()` returns correct DataFrame |
| **Day 6** | Risk map | 7 | `RiskDensityMap.compute()` returns weighted output |
| **Day 7** | Prioritize + delta-v | 8 | `PriorityScorer.score()` returns tiers; delta-v increases with altitude |
| **Day 7 (parallel)** | MMT validation | 5 | `validate_mmtortora.py` produces report (can run alongside scorer) |
| **Day 8** | Granite | 9 | Granite functions return explanation text or fallback |
| **Day 9** | Streamlit pages | 10 | All three pages render locally with correct data |
| **Day 10** | Home + integration | 11 | Full pipeline demo button works end-to-end |
| **Day 11** | README + deploy | 12 | App live on Streamlit Cloud |
| **Day 12** | Video + submission | 12 | Demo video recorded, submission submitted |

---

## Realism Watchlist — Adjust Now if Needed

These are the items with non-trivial execution risk. Flag during build if any need scope change.

| Item | Risk Level | Mitigation |
|---|---|---|
| Mini-MegaTORTORA data access | **Medium** — site is active but format may vary | Fall back to Schildknecht et al. ESA proceedings tables; document source used |
| Fourier inversion accuracy on real noisy curves | **Medium** — expected degradation | Report it honestly; the point is the degradation is measured, not hidden |
| Hohmann delta-v formula correctness | **Low** — standard textbook | Unit-test that dv increases monotonically with altitude from 400km reference |
| ibm-watsonx-ai SDK version compatibility | **Low** | Pin version in requirements.txt; test fallback path thoroughly |
| Streamlit Cloud secrets for Granite | **Low** | Graceful fallback ensures app works without credentials |
| 12-day schedule tightness | **Medium** | Days 7 and 12 have buffer; stretch goals deferred until all sub-tasks green |

---

## Stretch Goals (Only After All Sub-Tasks Green)

- 3D globe: Plotly `scatter_geo` with altitude color bands
- "What-If Breakup" scenario: add a simulated breakup at chosen altitude, watch risk map update
- Time-slider: debris density evolution over a simulated month
- PDF mission report: `reportlab`
- Expand validation corpus beyond 20 curves

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATA_SOURCE` | No | `celestrak` | TLE data source: `celestrak` or `spacetrack` |
| `SPACETRACK_USER` | Only if spacetrack | — | Space-Track.org username |
| `SPACETRACK_PASS` | Only if spacetrack | — | Space-Track.org password |
| `WATSONX_API_KEY` | No | — | IBM Cloud API key for watsonx.ai |
| `WATSONX_PROJECT_ID` | No | — | watsonx.ai project ID |
| `WATSONX_URL` | No | `https://us-south.ml.cloud.ibm.com` | watsonx.ai regional endpoint |

---

## Key Invariants (Never Violate These)

1. **Granite never produces numbers.** It only receives numbers computed by the pipeline and
   translates them into plain language. If a Granite response appears to contain a delta-v
   figure or risk score not present in the prompt, the prompt is wrong — fix the prompt.
2. **Delta-v is region-level only.** No code path should compute a per-object delta-v.
   The scorer operates on altitude-band DataFrames, not on individual detection records.
3. **Every output is inspectable.** Any risk score, priority tier, or delta-v value shown in
   the UI must have an "expand" path that shows the raw numbers and formula behind it.
4. **Synthetic data is labeled.** Any CSV or UI element displaying synthetic data carries
   a comment or disclaimer. No synthetic result is presented as a real observation.
5. **The risk map is not Bayesian.** No code comment, docstring, README line, or UI tooltip
   should describe the map as Bayesian, probabilistic inference, or a particle filter.
   It is a recency- and confidence-weighted grid accumulator.
