"""Authorized one-time holdout evaluation of frozen trial 0.1; no estimator.fit."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import pvsd_trial as api
import train_logo_v1 as base

CONFIG = ROOT/"configs/holdout_trial_v0_1.json"
KEYS = ["Room_ID", "HOY_0based", "PVSD_Angle_deg"]
META = ["Room_ID", "Month", "Day", "DOY_1based", "Hour", "HOY_0based", "PVSD_Angle_deg"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def utc():
    return datetime.now(timezone.utc).isoformat()


def score(y, p, tolerance):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    error = p-y
    return {"n": len(y), **base.metrics(y, p), "true_mean": float(y.mean()), "pred_mean": float(p.mean()),
            "true_sd": float(y.std()), "true_zero_fraction": float((y == 0).mean()),
            "absolute_error_p90": float(np.quantile(np.abs(error), .9)), "absolute_error_p95": float(np.quantile(np.abs(error), .95)),
            "absolute_tolerance": tolerance, "within_tolerance_fraction": float((np.abs(error) <= tolerance).mean())}


def freeze(cfg, out):
    if (out/"pretest_freeze.json").exists():
        raise RuntimeError("Evaluation already frozen; refusing overwrite")
    assert api.sha256(ROOT/cfg["source"]) == cfg["source_sha256"]
    predictor = api.TrialPredictor(ROOT/cfg["model_dir"])
    assert predictor.model_hash == cfg["model_sha256"]
    assert predictor.manifest["training_rooms"] == cfg["development_rooms"]
    assert not set(cfg["development_rooms"]) & set(cfg["test_rooms"])
    train, skipped = base.load_train(ROOT/cfg["source"], read(ROOT/"configs/model_plan.json"))
    assert skipped == cfg["test_rows"]
    offsets = (train.HOY_0based-((train.DOY_1based-1)*24+train.Hour)).unique()
    assert len(offsets) == 1
    baseline = train.groupby("PVSD_Angle_deg", sort=True)[api.TARGETS].mean().reset_index()
    baseline.to_csv(out/"train_angle_mean_baseline.csv", index=False)
    # Match fixed hyperparameters; these are still previously used development rooms.
    candidate_scores = pd.read_csv(ROOT/"reports/trial_v0_1/parameter_selection_room_metrics.csv")
    old_hgb = pd.read_csv(ROOT/"reports/experiment_v1/fold_metrics.csv")
    reference = []
    for target in api.TARGETS:
        spec = predictor.bundle["models"][target]["spec"]
        if target in ["E_mean_lux", "E_P10_lux"]:
            rows = candidate_scores[(candidate_scores.target == target)&(candidate_scores.candidate == spec["id"])][["room", "mae", "rmse", "r2", "bias"]].copy()
        else:
            rows = old_hgb[(old_hgb.feature_set == "B_add_solar_weather")&(old_hgb.model == "HistGB")&(old_hgb.target == target)][["validation_room", "mae", "rmse", "r2", "bias"]].rename(columns={"validation_room": "room"})
        assert len(rows) == 7 and set(rows.room) == set(cfg["development_rooms"])
        rows.insert(0, "target", target)
        reference.append(rows)
    pd.concat(reference, ignore_index=True).to_csv(out/"development_reference_room_metrics.csv", index=False)
    files = [CONFIG, ROOT/cfg["source"], ROOT/"pvsd_trial.py", Path(__file__),
             ROOT/"scripts/verify_holdout_trial_v0_1.py", ROOT/"scripts/train_logo_v1.py", ROOT/"configs/model_plan.json",
             ROOT/"docs/holdout_trial_v0_1_protocol.md", ROOT/"docs/approvals/holdout_trial_v0_1_approval.json",
             ROOT/"reports/trial_v0_1/parameter_freeze.json", ROOT/"reports/trial_v0_1/parameter_selection_room_metrics.csv",
             ROOT/"reports/experiment_v1/fold_metrics.csv", out/"train_angle_mean_baseline.csv", out/"development_reference_room_metrics.csv"]
    files += [ROOT/cfg["model_dir"]/name for name in ["manifest.json", "model.joblib", "metadata.json"]]
    frozen = {"evaluation_id": cfg["evaluation_id"], "frozen_utc": utc(), "stage": "before_TEST_feature_prediction_and_numeric_targets",
              "model_version": api.VERSION, "training_rooms": cfg["development_rooms"], "test_rooms": cfg["test_rooms"],
              "git_head_before_freeze": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "config": cfg, "fixed_models": {t: predictor.bundle["models"][t]["spec"] for t in api.TARGETS},
              "hashes": {p.relative_to(ROOT).as_posix(): api.sha256(p) for p in files},
              "observed_TRAIN_HOY_minus_DOY_hour_offset": int(offsets[0]),
              "TEST_numeric_targets_loaded": False, "estimator_fits": 0}
    dump(out/"pretest_freeze.json", frozen)
    print("Frozen model, rules, TRAIN-only baseline and development references. TEST numeric targets not loaded.", flush=True)


def load_test_columns(source, columns, cfg):
    """Only requested columns retained/converted; numeric targets requested after prediction freeze."""
    records, skipped = [], 0
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert set(columns) <= set(reader.fieldnames)
        for row in reader:
            if row["Sample_Role"] == "TRAIN":
                assert row["Room_ID"] in cfg["development_rooms"]
                skipped += 1
                continue
            assert row["Sample_Role"] == "TEST" and row["Room_ID"] in cfg["test_rooms"]
            records.append({c: row[c] for c in columns})
    frame = pd.DataFrame.from_records(records)
    for name in columns:
        if name != "Room_ID":
            frame[name] = pd.to_numeric(frame[name], errors="raise")
    assert len(frame) == cfg["test_rows"] and skipped == 122640
    assert set(frame.Room_ID) == set(cfg["test_rooms"])
    assert not frame.duplicated(KEYS).any()
    assert np.isfinite(frame.select_dtypes("number").to_numpy()).all()
    assert frame.groupby(["Room_ID", "PVSD_Angle_deg"]).size().eq(4380).all()
    assert set(frame.PVSD_Angle_deg) == set(cfg["angles"])
    return frame


def evaluate(cfg, out):
    frozen = read(out/"pretest_freeze.json")
    for relative, digest in frozen["hashes"].items():
        assert api.sha256(ROOT/relative) == digest, relative
    if (out/"run_manifest.json").exists():
        raise RuntimeError("Evaluation already started; preserve its first results")
    run = {"evaluation_id": cfg["evaluation_id"], "status": "running", "started_utc": utc(), "pid": os.getpid(),
           "pretest_freeze_sha256": api.sha256(out/"pretest_freeze.json"), "test_rooms": cfg["test_rooms"],
           "model_sha256": cfg["model_sha256"], "estimator_fits": 0, "TEST_used_for_training_or_tuning": False,
           "new_simulations": 0, "new_visualizations": 0}
    dump(out/"run_manifest.json", run)
    started = time.perf_counter()
    inputs = load_test_columns(ROOT/cfg["source"], list(dict.fromkeys(META+api.RAW_FEATURES)), cfg)
    assert inputs.Hour.between(7, 18).all() and inputs.DOY_1based.between(1, 365).all()
    assert inputs.HOY_0based.eq((inputs.DOY_1based-1)*24+inputs.Hour+frozen["observed_TRAIN_HOY_minus_DOY_hour_offset"]).all(), "Source HOY mapping differs from TRAIN"
    assert not inputs.duplicated(["Room_ID", "DOY_1based", "Hour", "PVSD_Angle_deg"]).any()
    assert set(inputs.DOY_1based) == set(range(1, 366)) and set(inputs.Hour) == set(range(7, 19))
    for name in ["DOY_1based", "Hour"]:
        assert inputs[name].eq(np.floor(inputs[name])).all()
    assert not set(api.TARGETS) & set(inputs.columns)
    predictor = api.TrialPredictor(ROOT/cfg["model_dir"])
    x = api.encode_inputs(inputs[api.RAW_FEATURES])
    assert list(x.columns) == predictor.bundle["encoded_features"]
    pd.testing.assert_frame_equal(x, base.encode_inputs(inputs, api.RAW_FEATURES))
    result = inputs[META].copy()
    baseline = pd.read_csv(out/"train_angle_mean_baseline.csv").set_index("PVSD_Angle_deg")
    with threadpool_limits(limits=cfg["thread_limit"]):
        for target in api.TARGETS:
            result["pred__"+target] = api.predict_target(predictor.bundle["models"][target], x)
            result["baseline__"+target] = inputs.PVSD_Angle_deg.map(baseline[target])
    # Freeze predictions before requesting TEST numeric targets. No model fitting here.
    result.to_parquet(out/"predictions_before_targets.parquet", index=False)
    prediction_freeze = {"frozen_utc": utc(), "rows": len(result), "model_target_values": len(result)*5,
                         "predictions_sha256": api.sha256(out/"predictions_before_targets.parquet"),
                         "pretest_freeze_sha256": api.sha256(out/"pretest_freeze.json"), "model_sha256": predictor.model_hash,
                         "TEST_numeric_targets_loaded": False, "columns": list(result.columns)}
    dump(out/"prediction_freeze.json", prediction_freeze)
    print(f"Saved {len(result)} TEST rows / {len(result)*5} model values before numeric labels. Now scoring frozen predictions.", flush=True)
    labels = load_test_columns(ROOT/cfg["source"], list(dict.fromkeys(KEYS+api.TARGETS)), cfg)
    for target in api.TARGETS:
        assert labels[target].ge(0).all()
        if target in api.FRACTIONS:
            assert labels[target].le(1).all()
    labels = labels.rename(columns={t: "true__"+t for t in api.TARGETS})
    scored = result.merge(labels, on=KEYS, how="left", validate="one_to_one")
    assert len(scored) == cfg["test_rows"] and np.isfinite(scored.select_dtypes("number")).all().all()
    scored.to_parquet(out/"scored_predictions.parquet", index=False)
    room_metrics, pooled, stratified = [], [], {"angle": [], "hour": []}
    for method, prefix in [("Frozen_trial_v0_1", "pred__"), ("TRAIN_angle_mean", "baseline__")]:
        for target in api.TARGETS:
            tol = cfg["tolerances_descriptive"][target]
            for room, group in scored.groupby("Room_ID", sort=True):
                room_metrics.append({"method": method, "target": target, "room": room, **score(group["true__"+target], group[prefix+target], tol)})
            pooled.append({"method": method, "target": target, **score(scored["true__"+target], scored[prefix+target], tol)})
            if method == "Frozen_trial_v0_1":
                for name, column in [("angle", "PVSD_Angle_deg"), ("hour", "Hour")]:
                    for (room, value), group in scored.groupby(["Room_ID", column], sort=True):
                        stratified[name].append({"method": method, "target": target, "room": room, column: int(value), **score(group["true__"+target], group[prefix+target], tol)})
    fm = pd.DataFrame(room_metrics)
    fm.to_csv(out/"room_metrics.csv", index=False)
    pd.DataFrame(pooled).to_csv(out/"pooled_metrics.csv", index=False)
    for name, rows in stratified.items():
        pd.DataFrame(rows).to_csv(out/f"room_{name}_metrics.csv", index=False)
    summary = fm.groupby(["method", "target"], sort=False).agg(room_count=("room", "nunique"), macro_MAE=("mae", "mean"), worst_MAE=("mae", "max"),
             macro_RMSE=("rmse", "mean"), median_R2=("r2", "median"), min_R2=("r2", "min"), macro_bias=("bias", "mean"),
             macro_within_tolerance=("within_tolerance_fraction", "mean"), min_within_tolerance=("within_tolerance_fraction", "min")).reset_index()
    summary["worst_MAE_room"] = [fm[(fm.method == r.method)&(fm.target == r.target)].sort_values(["mae", "room"], ascending=[False, True]).iloc[0].room for r in summary.itertuples()]
    summary.to_csv(out/"summary_metrics.csv", index=False)
    profiles = inputs[["Room_ID"]+api.ROOM_FEATURES].drop_duplicates()
    assert len(profiles) == 3
    representatives = inputs.groupby("Room_ID", sort=True).head(1).copy()
    profiles = representatives[["Room_ID"]+api.ROOM_FEATURES].copy()
    profiles["Coverage_Flags"] = predictor.coverage_flags(representatives[["Room_ID"]+api.INPUT_FEATURES])
    profiles.to_csv(out/"test_room_feature_coverage.csv", index=False)
    coverage = []
    for room, group in inputs.groupby("Room_ID", sort=True):
        for name in api.INPUT_FEATURES:
            limits = predictor.metadata["training_feature_ranges"][name]
            outside = (group[name] < limits["min"]-1e-9)|(group[name] > limits["max"]+1e-9)
            coverage.append({"room": room, "feature": name, "test_min": float(group[name].min()), "test_max": float(group[name].max()),
                             "train_min": limits["min"], "train_max": limits["max"], "outside_rows": int(outside.sum()), "rows": len(group)})
    pd.DataFrame(coverage).to_csv(out/"feature_range_coverage.csv", index=False)
    dev = pd.read_csv(out/"development_reference_room_metrics.csv")
    comparison = []
    for target in api.TARGETS:
        d = dev[dev.target == target]
        test = summary[(summary.method == "Frozen_trial_v0_1")&(summary.target == target)].iloc[0]
        comparison.append({"target": target, "development_macro_MAE": float(d.mae.mean()), "development_worst_MAE": float(d.mae.max()),
                           "test_macro_MAE": float(test.macro_MAE), "test_worst_MAE": float(test.worst_MAE),
                           "note": "Different room sets and training sizes; descriptive comparison, not improvement attribution"})
    pd.DataFrame(comparison).to_csv(out/"development_vs_holdout.csv", index=False)
    for relative, digest in frozen["hashes"].items():
        assert api.sha256(ROOT/relative) == digest, relative
    run.update({"status": "scored_pending_verification", "completed_utc": utc(), "elapsed_seconds": time.perf_counter()-started,
                "TEST_prediction_rows": len(result), "TEST_scored_rows": len(scored), "TEST_model_target_values": len(result)*5,
                "TEST_baseline_target_values": len(result)*5, "prediction_freeze_sha256": api.sha256(out/"prediction_freeze.json"),
                "holdout_state": "observed_for_frozen_trial_0.1.0; no longer untouched for future result-informed changes"})
    dump(out/"run_manifest.json", run)
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "score"])
    args = parser.parse_args()
    cfg = read(CONFIG)
    out = ROOT/cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    if Path(sys.prefix).resolve() != (ROOT/".conda/envs/pvsd-ml-v1").resolve():
        raise RuntimeError("Use the project conda environment")
    if args.phase == "freeze":
        freeze(cfg, out)
    else:
        evaluate(cfg, out)


if __name__ == "__main__":
    main()
