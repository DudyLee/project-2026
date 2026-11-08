"""Independent metric recomputation and frozen-model reload verification; no fits."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
import pvsd_trial as api
import evaluate_holdout_trial_v0_1 as ev


def main():
    cfg = ev.read(ev.CONFIG)
    out = ROOT/cfg["output"]
    frozen = ev.read(out/"pretest_freeze.json")
    for relative, digest in frozen["hashes"].items():
        assert api.sha256(ROOT/relative) == digest, relative
    pf = ev.read(out/"prediction_freeze.json")
    assert pf["TEST_numeric_targets_loaded"] is False
    assert pf["predictions_sha256"] == api.sha256(out/"predictions_before_targets.parquet")
    before = pd.read_parquet(out/"predictions_before_targets.parquet")
    scored = pd.read_parquet(out/"scored_predictions.parquet")
    assert len(scored) == 52560 and not scored.duplicated(ev.KEYS).any()
    pd.testing.assert_frame_equal(before, scored[before.columns])
    inputs = ev.load_test_columns(ROOT/cfg["source"], list(dict.fromkeys(ev.META+api.RAW_FEATURES)), cfg)
    labels = ev.load_test_columns(ROOT/cfg["source"], list(dict.fromkeys(ev.KEYS+api.TARGETS)), cfg)
    pd.testing.assert_frame_equal(scored[ev.KEYS], labels[ev.KEYS])
    for t in api.TARGETS:
        np.testing.assert_array_equal(scored["true__"+t], labels[t])
    model = api.TrialPredictor(ROOT/cfg["model_dir"])
    x = api.encode_inputs(inputs[api.RAW_FEATURES])
    differences = {}
    with threadpool_limits(limits=cfg["thread_limit"]):
        for t in api.TARGETS:
            est = model.bundle["models"][t]["estimators"]
            if t == "E_P10_lux":
                p = np.maximum(est["ratio"].predict(x), 0)*(np.maximum(np.expm1(est["mean_log"].predict(x)), 0)+1)
            else:
                p = est["value"].predict(x)
                if t == "E_mean_lux":
                    p = np.expm1(p)
                p = np.clip(p, 0, 1) if t in api.FRACTIONS else np.maximum(p, 0)
            difference = float(np.max(np.abs(p-scored["pred__"+t])))
            assert difference < 1e-9
            differences[t] = difference
    checked = 0
    for name, grouping in [("room_metrics.csv", ["room"]), ("room_angle_metrics.csv", ["room", "PVSD_Angle_deg"]), ("room_hour_metrics.csv", ["room", "Hour"]), ("pooled_metrics.csv", [])]:
        for row in pd.read_csv(out/name).to_dict("records"):
            group = scored
            for c in grouping:
                group = group[group["Room_ID" if c == "room" else c] == row[c]]
            y = group["true__"+row["target"]].to_numpy()
            p = group[("pred__" if row["method"] == "Frozen_trial_v0_1" else "baseline__")+row["target"]].to_numpy()
            assert len(y) == row["n"]
            expected = {"mae": mean_absolute_error(y, p), "rmse": np.sqrt(mean_squared_error(y, p)),
                        "r2": r2_score(y, p, force_finite=False) if np.var(y) > 0 else float("nan"),
                        "bias": float(np.mean(p-y)), "within_tolerance_fraction": float(np.mean(np.abs(p-y) <= cfg["tolerances_descriptive"][row["target"]]))}
            for key, value in expected.items():
                np.testing.assert_allclose(row[key], value, atol=1e-10, rtol=1e-10, equal_nan=True)
            checked += 1
    rm = pd.read_csv(out/"room_metrics.csv")
    for row in pd.read_csv(out/"summary_metrics.csv").to_dict("records"):
        r = rm[(rm.method == row["method"])&(rm.target == row["target"])]
        assert len(r) == 3
        np.testing.assert_allclose(row["macro_MAE"], r.mae.mean(), atol=1e-12)
        np.testing.assert_allclose(row["worst_MAE"], r.mae.max(), atol=1e-12)
        assert row["worst_MAE_room"] == r.sort_values(["mae", "room"], ascending=[False, True]).iloc[0].room
    b = pd.read_csv(out/"train_angle_mean_baseline.csv").set_index("PVSD_Angle_deg")
    for t in api.TARGETS:
        np.testing.assert_allclose(scored["baseline__"+t], scored.PVSD_Angle_deg.map(b[t]), atol=1e-14)
    assert datetime.fromisoformat(frozen["frozen_utc"]) < datetime.fromisoformat(pf["frozen_utc"])
    assert api.sha256(ROOT/cfg["model_dir"]/"model.joblib") == cfg["model_sha256"]
    assert api.sha256(ROOT/cfg["source"]) == cfg["source_sha256"]
    verification = {"status": "passed", "verification_status": "VERIFIED", "completed_utc": ev.utc(),
        "model_and_input_hashes_unchanged": True, "predictions_frozen_before_target_load": True,
        "rows": len(scored), "model_values": len(scored)*5, "independently_recomputed_metric_rows": checked,
        "max_prediction_reload_difference": differences, "new_estimator_fits": 0,
        "note": "Reload reproduction of the same frozen predictions and independent score-formula checks; not another model selection or new evaluation claim."}
    ev.dump(out/"verification.json", verification)
    run = ev.read(out/"run_manifest.json")
    run.update({"status": "completed_verified", "verification": "reports/holdout_trial_v0_1/verification.json"})
    ev.dump(out/"run_manifest.json", run)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
