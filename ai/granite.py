"""
ai/granite.py — IBM Granite (watsonx.ai) Integration

Granite is used in exactly three places in Delta-V:
  1. explain_characterization() — translates inversion + ML output into analyst brief
  2. write_situation_report()   — summarises the current LEO risk landscape
  3. write_mission_brief()      — mission-brief recommendation for top-priority bands

DESIGN CONSTRAINT: Granite never produces or guesses numbers.
Every prompt contains the computed numbers from the pipeline. Granite's only
job is to translate those numbers into plain English. If a Granite response
appears to contain a delta-v figure or risk score not present in the prompt,
the prompt is wrong — fix the prompt, not the guard.

Prompts instruct Granite to:
  - Use ONLY the numbers provided in the data block
  - Not invent, estimate, or add context beyond what is given
  - Write for a mission analyst audience (precise, concise, no hedging)

Fallback behaviour:
  If WATSONX_API_KEY is not set, or any API call fails, each function returns
  a pre-written template string built from the same structured data.
  Fallbacks are substantive — they are real analyst text, not error messages.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config import get_config, granite_available

logger = logging.getLogger(__name__)

# Model identifier — pinned to avoid silent version changes
GRANITE_MODEL_ID  = "ibm/granite-3-3-8b-instruct"
GRANITE_MAX_TOKENS = 300

# ── Lazy singleton client ─────────────────────────────────────────────────────
_client: Optional["GraniteClient"] = None


def get_client() -> Optional["GraniteClient"]:
    """Return a cached GraniteClient, or None if credentials are unavailable."""
    global _client
    if _client is not None:
        return _client
    cfg = get_config()
    if not granite_available(cfg):
        return None
    try:
        _client = GraniteClient(
            api_key=cfg["watsonx_api_key"],
            project_id=cfg["watsonx_project_id"],
            url=cfg["watsonx_url"],
        )
        return _client
    except Exception as exc:
        logger.warning("GraniteClient init failed: %s", exc)
        return None


# ── Client ────────────────────────────────────────────────────────────────────

class GraniteClient:
    """Wraps ibm_watsonx_ai ModelInference for Granite text generation."""

    def __init__(self, api_key: str, project_id: str, url: str):
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params

        credentials = Credentials(url=url, api_key=api_key)
        self._model = ModelInference(
            model_id=GRANITE_MODEL_ID,
            credentials=credentials,
            project_id=project_id,
            params={
                Params.MAX_NEW_TOKENS: GRANITE_MAX_TOKENS,
                Params.TEMPERATURE: 0.3,
                Params.REPETITION_PENALTY: 1.1,
            },
        )

    def generate(self, prompt: str) -> str:
        """Call Granite and return the generated text string."""
        response = self._model.generate_text(prompt=prompt)
        return response.strip() if response else ""


# ── Public API ────────────────────────────────────────────────────────────────

def explain_characterization(prediction: dict, session_state_key: str = "granite_explain") -> str:
    """Generate a 2-3 sentence analyst brief from a characterization prediction.

    Parameters
    ----------
    prediction : dict
        Output of CharacterizeModel.predict(). Must contain:
            size_class, shape, size_confidence, shape_confidence,
            inversion_result (InversionResult dataclass)
    session_state_key : str
        Streamlit session_state key for caching. Ignored outside Streamlit.

    Returns
    -------
    str — analyst brief text (from Granite or fallback)
    """
    inv = prediction.get("inversion_result")
    size_class     = prediction.get("size_class", "medium")
    shape          = prediction.get("shape", "unknown")
    size_conf      = prediction.get("size_confidence", 0.0)
    shape_conf     = prediction.get("shape_confidence", 0.0)
    rot_hz         = getattr(inv, "rotation_rate_hz", 0.0) if inv else 0.0
    rot_unc        = getattr(inv, "rotation_rate_uncertainty_hz", 0.0) if inv else 0.0
    amplitude      = getattr(inv, "amplitude", 0.0) if inv else 0.0
    size_m         = getattr(inv, "size_estimate_m", 0.05) if inv else 0.05
    snr            = getattr(inv, "snr_estimate", 0.0) if inv else 0.0
    shape_hint     = getattr(inv, "shape_hint", "unknown") if inv else "unknown"
    coeffs         = getattr(inv, "fourier_coefficients", []) if inv else []
    top_coeffs     = list(coeffs[:3]) if len(coeffs) >= 3 else list(coeffs)

    prompt = f"""<|system|>
You are a space situational awareness analyst writing a technical assessment.
You must use ONLY the numbers provided in the DATA BLOCK below.
Do not invent, estimate, or add values not present in the data.
Write exactly 2-3 sentences in plain technical English.
<|user|>
DATA BLOCK:
- Debris size class: {size_class} (ML confidence: {size_conf*100:.0f}%)
- Estimated size: {size_m*100:.1f} cm
- Shape: {shape} (ML confidence: {shape_conf*100:.0f}%)
- Inversion shape hint: {shape_hint}
- Rotation rate (apparent, from FFT): {rot_hz:.4f} Hz (±{rot_unc:.4f} Hz)
- Light curve amplitude (peak-to-trough/mean): {amplitude:.3f}
- Signal-to-noise estimate: {snr:.1f}
- Top Fourier coefficients: {[f"{c:.4f}" for c in top_coeffs]}

Write a 2-3 sentence analyst brief describing this debris detection.
Use only the numbers above. Do not add orbital data or position estimates.
<|assistant|>"""

    client = get_client()
    if client is None:
        return _fallback_characterization(
            size_class, shape, size_m, rot_hz, rot_unc, amplitude,
            size_conf, shape_conf, snr, shape_hint
        )
    try:
        return client.generate(prompt)
    except Exception as exc:
        logger.warning("Granite explain_characterization failed: %s", exc)
        return _fallback_characterization(
            size_class, shape, size_m, rot_hz, rot_unc, amplitude,
            size_conf, shape_conf, snr, shape_hint
        )


def write_situation_report(risk_df: pd.DataFrame, session_state_key: str = "granite_sitrep") -> str:
    """Generate a 3-4 sentence situation report from the risk-density map.

    Parameters
    ----------
    risk_df : pd.DataFrame
        Output of RiskDensityMap.compute(). Must contain:
            band_label, composite_risk_density, tracked_object_count,
            detection_count, last_updated

    Returns
    -------
    str — situation report text (from Granite or fallback)
    """
    if risk_df is None or risk_df.empty:
        return "No risk map data available. Run the Map stage first."

    top3 = risk_df.sort_values("composite_risk_density", ascending=False).head(3)
    total_objects = int(risk_df["tracked_object_count"].sum())
    total_detections = int(risk_df["detection_count"].sum())
    last_updated = risk_df["last_updated"].iloc[0] if "last_updated" in risk_df.columns else "unknown"

    band_lines = "\n".join(
        f"  - {row['band_label']}: risk density {row['composite_risk_density']:.3f}, "
        f"{row['tracked_object_count']} tracked objects, {row['detection_count']} characterised detections"
        for _, row in top3.iterrows()
    )

    prompt = f"""<|system|>
You are a space situational awareness analyst writing a situation report.
You must use ONLY the numbers provided in the DATA BLOCK below.
Do not invent risk scores, object counts, or altitude values not in the data.
Write exactly 3-4 sentences in plain technical English.
<|user|>
DATA BLOCK:
- Total tracked objects in LEO bands: {total_objects}
- Total characterised debris detections: {total_detections}
- Last map update: {last_updated}
- Top 3 highest-risk altitude bands:
{band_lines}

Note: This risk map is a grid-based accumulator, not a Bayesian filter.
It does not track individual objects — only regional density.

Write a 3-4 sentence situation report on the current LEO debris risk landscape.
Use only the numbers above.
<|assistant|>"""

    client = get_client()
    if client is None:
        return _fallback_situation_report(top3, total_objects, total_detections)
    try:
        return client.generate(prompt)
    except Exception as exc:
        logger.warning("Granite write_situation_report failed: %s", exc)
        return _fallback_situation_report(top3, total_objects, total_detections)


def write_mission_brief(priority_df: pd.DataFrame, session_state_key: str = "granite_brief") -> str:
    """Generate a mission-brief recommendation from the priority scoring output.

    Parameters
    ----------
    priority_df : pd.DataFrame
        Output of PriorityScorer.score(). Must contain:
            band_label, tier, priority_score, dv_total_ms,
            composite_risk_density, explanation_text

    Returns
    -------
    str — mission brief text (from Granite or fallback)
    """
    if priority_df is None or priority_df.empty:
        return "No priority data available. Run the Prioritize stage first."

    high_priority = priority_df[priority_df["tier"] == "HIGH-PRIORITY"].head(3)
    monitor = priority_df[priority_df["tier"] == "MONITOR"].head(2)

    if high_priority.empty:
        top = priority_df.head(3)
    else:
        top = high_priority

    band_lines = "\n".join(
        f"  - {row['band_label']}: tier={row['tier']}, priority={row['priority_score']:.3f}, "
        f"delta-v required={row['dv_total_ms']:.0f} m/s, "
        f"risk density={row['composite_risk_density']:.3f}"
        for _, row in top.iterrows()
    )

    monitor_lines = ""
    if not monitor.empty:
        monitor_lines = "\nMONITOR bands:\n" + "\n".join(
            f"  - {row['band_label']}: delta-v={row['dv_total_ms']:.0f} m/s"
            for _, row in monitor.iterrows()
        )

    prompt = f"""<|system|>
You are a space mission planner writing a mission brief recommendation.
You must use ONLY the numbers provided in the DATA BLOCK below.
Do not invent delta-v values, risk scores, or band labels not in the data.
Write exactly 3-4 sentences in plain technical English suitable for a mission analyst.
<|user|>
DATA BLOCK:
HIGH-PRIORITY bands (delta-v cost is Hohmann transfer + 5-degree plane change from 400 km reference orbit):
{band_lines}
{monitor_lines}

Note: delta-v costs are region-level estimates only, not per individual object.
The reference orbit is 400 km circular. Plane change assumes 5 degrees worst case.

Write a concise mission brief recommendation for the prioritised debris regions.
Reference the specific delta-v costs from the data. Use only the numbers above.
<|assistant|>"""

    client = get_client()
    if client is None:
        return _fallback_mission_brief(top, monitor)
    try:
        return client.generate(prompt)
    except Exception as exc:
        logger.warning("Granite write_mission_brief failed: %s", exc)
        return _fallback_mission_brief(top, monitor)


# ── Fallback text generators ──────────────────────────────────────────────────
# These are substantive analyst-quality text, not error messages.

def _fallback_characterization(
    size_class, shape, size_m, rot_hz, rot_unc, amplitude,
    size_conf, shape_conf, snr, shape_hint
) -> str:
    size_cm = size_m * 100
    rot_period = 1.0 / rot_hz if rot_hz > 0 else 0.0
    return (
        f"Light curve inversion indicates a {size_class}-class debris fragment "
        f"(estimated {size_cm:.1f} cm) with a {shape} morphology "
        f"(ML confidence: size {size_conf*100:.0f}%, shape {shape_conf*100:.0f}%). "
        f"The dominant FFT frequency is {rot_hz:.4f} Hz (±{rot_unc:.4f} Hz), "
        f"corresponding to an apparent rotation period of {rot_period:.2f} s; "
        f"Fourier coefficient structure is consistent with a {shape_hint} body. "
        f"Light curve amplitude of {amplitude:.3f} and SNR estimate of {snr:.1f} "
        f"suggest {'reliable' if snr > 10 else 'marginal'} characterization confidence "
        f"— results should be treated as order-of-magnitude estimates."
    )


def _fallback_situation_report(top3: pd.DataFrame, total_objects: int, total_detections: int) -> str:
    if top3.empty:
        return "Insufficient data to generate a situation report."
    top = top3.iloc[0]
    lines = []
    for _, row in top3.iterrows():
        lines.append(f"{row['band_label']} (risk density {row['composite_risk_density']:.3f})")
    top_bands_str = ", ".join(lines)
    return (
        f"Current LEO risk map covers {total_objects} tracked objects across 8 altitude bands "
        f"(200–2000 km), supplemented by {total_detections} characterised debris detections. "
        f"The highest composite risk densities are observed in: {top_bands_str}. "
        f"The {top['band_label']} band shows the highest risk density at "
        f"{top['composite_risk_density']:.3f}, driven by tracked object concentration "
        f"and recency-weighted detection accumulation. "
        f"Note: this map is a grid-based regional accumulator — it does not track individual objects."
    )


def _fallback_mission_brief(top: pd.DataFrame, monitor: pd.DataFrame) -> str:
    if top.empty:
        return "Insufficient data to generate a mission brief."

    top_row = top.iloc[0]
    high_bands = ", ".join(
        f"{row['band_label']} (~{row['dv_total_ms']:.0f} m/s)"
        for _, row in top.iterrows()
    )
    monitor_str = ""
    if not monitor.empty:
        monitor_bands = ", ".join(
            f"{row['band_label']} (~{row['dv_total_ms']:.0f} m/s)"
            for _, row in monitor.iterrows()
        )
        monitor_str = f" Bands requiring continued monitoring: {monitor_bands}."

    return (
        f"Priority assessment identifies the following HIGH-PRIORITY debris regions: "
        f"{high_bands} (delta-v from 400 km reference, Hohmann + 5° plane change). "
        f"The top-priority band, {top_row['band_label']}, has a priority score of "
        f"{top_row['priority_score']:.3f} and requires approximately "
        f"{top_row['dv_total_ms']:.0f} m/s total delta-v for a remediation approach manoeuvre. "
        f"Risk density for this region is {top_row['composite_risk_density']:.3f}."
        f"{monitor_str} "
        f"All delta-v figures are region-level estimates — not per-object trajectories."
    )


def granite_status() -> dict:
    """Return a dict describing Granite availability for UI display."""
    cfg = get_config()
    available = granite_available(cfg)
    client = get_client() if available else None
    return {
        "available": available and client is not None,
        "model_id": GRANITE_MODEL_ID,
        "status_text": "Connected" if (available and client is not None) else "Fallback mode",
        "status_colour": "green" if (available and client is not None) else "grey",
    }
