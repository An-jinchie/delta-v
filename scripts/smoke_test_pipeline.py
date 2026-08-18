"""
scripts/smoke_test_pipeline.py — End-to-end smoke test of the full pipeline + Granite fallback.
Run from project root: python scripts/smoke_test_pipeline.py
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Stage 1 ───────────────────────────────────────────────────────────────────
from pipeline.characterize.generator import LightCurveGenerator, LightCurveParams
from pipeline.characterize.inversion import invert
from pipeline.characterize.model import CharacterizeModel

gen = LightCurveGenerator()
params = LightCurveParams("medium", "tumbling", 0.6, 0.0, 0.15, 20.0, seed=42)
lc = gen.generate(params)
inv = invert(lc, cadence_s=0.1)
model = CharacterizeModel.load("data/models/characterize_model.pkl")
pred = model.predict(lc, inv)
print(f"Stage 1 OK: size={pred['size_class']} ({pred['size_confidence']*100:.0f}%), "
      f"shape={pred['shape']} ({pred['shape_confidence']*100:.0f}%)")
print(f"  Inversion: rot={inv.rotation_rate_hz:.4f} Hz, amplitude={inv.amplitude:.3f}, "
      f"shape_hint={inv.shape_hint}, SNR={inv.snr_estimate:.1f}")

# ── Stage 2 ───────────────────────────────────────────────────────────────────
from pipeline.map.tle_fetcher import TLEFetcher
from pipeline.map.density import compute_density
from pipeline.map.risk_map import RiskDensityMap

fetcher = TLEFetcher(fallback_path="data/tle_snapshot_fallback.csv")
lines = fetcher._load_fallback()
density_df = compute_density(lines)
rmap = RiskDensityMap()
rmap.update([{
    "altitude_band_km": "400-600",
    "confidence": pred["size_confidence"],
    "timestamp": time.time(),
    "size_class": pred["size_class"],
}])
risk_df = rmap.compute(density_df)
top_band = risk_df.sort_values("composite_risk_density", ascending=False).iloc[0]
print(f"Stage 2 OK: top risk = {top_band['band_label']} "
      f"(composite_risk={top_band['composite_risk_density']:.3f}, "
      f"objects={top_band['tracked_object_count']})")

# ── Stage 3 ───────────────────────────────────────────────────────────────────
from pipeline.prioritize.scorer import PriorityScorer

scorer = PriorityScorer()
priority_df = scorer.score(risk_df)
top = priority_df.iloc[0]
print(f"Stage 3 OK: top priority = {top['band_label']} | tier={top['tier']} | "
      f"dv={top['dv_total_ms']:.0f} m/s | priority={top['priority_score']:.3f}")

# ── Granite fallback ──────────────────────────────────────────────────────────
from ai.granite import (
    explain_characterization, write_situation_report, write_mission_brief, granite_status
)

status = granite_status()
print(f"\nGranite: {status['status_text']} (model={status['model_id']})")

brief_char    = explain_characterization(pred)
brief_sitrep  = write_situation_report(risk_df)
brief_mission = write_mission_brief(priority_df)

assert len(brief_char) > 50,    "Characterization fallback too short"
assert len(brief_sitrep) > 50,  "Situation report fallback too short"
assert len(brief_mission) > 50, "Mission brief fallback too short"

# Verify fallback text contains key computed values
assert str(int(top['dv_total_ms'])) in brief_mission or \
       f"{top['dv_total_ms']:.0f}" in brief_mission, \
       f"Mission brief missing delta-v value: {top['dv_total_ms']:.0f}"

print("\n--- Characterization Brief ---")
print(brief_char)
print("\n--- Situation Report ---")
print(brief_sitrep)
print("\n--- Mission Brief ---")
print(brief_mission)

print("\nAll pipeline stages and Granite fallback: OK")
