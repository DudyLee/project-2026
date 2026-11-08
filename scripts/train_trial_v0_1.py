"""Freeze retained families, select two bounded parameter pairs, refit TRAIN only."""
from __future__ import annotations
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits
import pvsd_trial as api
import train_logo_v1 as base
import train_nested_v3 as v3
import train_target_specific_v7 as v7


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    started = time.perf_counter()
    cfg_path = ROOT / "configs/trial_v0_1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out, model_dir = ROOT/cfg["report_output"], ROOT/cfg["model_output"]
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if (out/"run_manifest.json").exists() or (model_dir/"manifest.json").exists():
        raise RuntimeError("Version already started; do not overwrite a trial release")
    if Path(sys.prefix).resolve() != (ROOT/".conda/envs/pvsd-ml-v1").resolve():
        raise RuntimeError("Use the project's dedicated conda environment")
    source = ROOT/cfg["source"]
    assert api.sha256(source) == cfg["source_sha256"]
    plan = json.loads((ROOT/cfg["feature_plan"]).read_text(encoding="utf-8"))
    old = json.loads((ROOT/cfg["parameter_source"]).read_text(encoding="utf-8"))
    hgb = json.loads((ROOT/cfg["histgb_parameter_source"]).read_text(encoding="utf-8"))["hist_gradient_boosting"]
    frame, skipped = base.load_train(source, plan)
    assert sorted(frame.Room_ID.unique()) == cfg["development_rooms"]
    assert not set(frame.Room_ID) & set(cfg["protected_test_rooms"])
    x = base.encode_inputs(frame, api.RAW_FEATURES)
    pd.testing.assert_frame_equal(x, api.encode_inputs(frame))
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "threadpoolctl", "pyarrow"]}
    tracked = [cfg_path, source, ROOT/cfg["parameter_source"], ROOT/cfg["histgb_parameter_source"], ROOT/cfg["feature_plan"], Path(__file__), ROOT/"pvsd_trial.py", ROOT/"docs/trial_v0_1_protocol.md", ROOT/"scripts/train_logo_v1.py", ROOT/"scripts/train_nested_v3.py", ROOT/"scripts/train_target_specific_v7.py", ROOT/"scripts/train_cross_room_v5.py"]
    run = {"version": cfg["version"], "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "train_rows": len(frame), "training_rooms": cfg["development_rooms"], "test_rows_skipped": skipped,
           "TEST_predictions": 0, "TEST_scores": 0, "TEST_fits": 0, "new_simulations": 0,
           "versions": versions, "python": sys.version, "input_hashes": {p.relative_to(ROOT).as_posix(): api.sha256(p) for p in tracked},
           "scope": cfg["scope"]}
    dump(out/"run_manifest.json", run)
    dump(out/"config_snapshot.json", cfg)
    print(f"TRAIN {len(frame)} rows, 7 rooms; skipped TEST {skipped}. Selecting parameters.", flush=True)
    scores, fit_log, selected, folds = [], [], {}, []
    splits = list(LeaveOneGroupOut().split(x, groups=frame.Room_ID))
    for fold, (ti, vi) in enumerate(splits, 1):
        folds.append({"fold": fold, "fit_rooms": sorted(frame.iloc[ti].Room_ID.unique()), "validation_room": frame.iloc[vi].Room_ID.iloc[0], "fit_rows": len(ti), "validation_rows": len(vi)})
    dump(out/"fold_splits.json", folds)
    with threadpool_limits(limits=cfg["thread_limit"]):
        for target, design in cfg["models"].items():
            if "fixed" in design:
                selected[target] = {"target": target, "family": design["family"], "transform": design["transform"], "id": design["fixed"], "params": hgb}
                continue
            candidates = []
            for order, cid in enumerate(design["candidates"]):
                spec = {"target": target, "family": design["family"], "transform": design["transform"], "id": cid, "params": v3.params_for(old, design["family"], cid)}
                oof = frame[["Room_ID", "DOY_1based", "Hour", "PVSD_Angle_deg", target]].copy()
                oof["prediction"] = np.nan
                room_errors = []
                for split, (ti, vi) in zip(folds, splits):
                    t0 = time.perf_counter()
                    fitted, labels = v7.fit_target(spec, x.iloc[ti], frame.iloc[ti])
                    pred = api.predict_target(fitted, x.iloc[vi])
                    score = base.metrics(frame[target].iloc[vi].to_numpy(), pred)
                    room_errors.append(score["mae"])
                    scores.append({"target": target, "candidate": cid, "fold": split["fold"], "room": split["validation_room"], **score})
                    oof.loc[vi, "prediction"] = pred
                    fit_log.append({"stage": "development_parameter_selection", "target": target, "candidate": cid, **split,
                                    "estimators": len(fitted["estimators"]), "training_label_hashes": labels,
                                    "training_inputs_sha256": v7.T.array_sha(x.iloc[ti].to_numpy()), "seconds": time.perf_counter()-t0})
                    print(f"{target} {cid} held {split['validation_room']} MAE={score['mae']:.6f}", flush=True)
                    dump(out/"progress.json", {"stage": "parameter_selection", "completed_estimator_fits": sum(r["estimators"] for r in fit_log), "total_estimator_fits": 48})
                oof.to_parquet(out/f"selection_oof_{target}_{cid}.parquet", index=False)
                candidates.append({"spec": spec, "order": order, "macro_MAE": float(np.mean(room_errors)), "worst_MAE": float(max(room_errors))})
            winner = min(candidates, key=lambda c: (c["macro_MAE"], c["order"]))
            selected[target] = winner["spec"]
            dump(out/f"selection_{target}.json", {"scope": "Development parameter selection; not independent test", "candidates": candidates, "selected": winner["spec"]["id"]})
        pd.DataFrame(scores).to_csv(out/"parameter_selection_room_metrics.csv", index=False)
        frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "selection_rule": cfg["parameter_selection"], "models": selected, "TEST_used": False}
        dump(out/"parameter_freeze.json", frozen)
        print("Parameters frozen. Fitting all seven development rooms.", flush=True)
        fitted_models = {}
        for target, spec in selected.items():
            t0 = time.perf_counter()
            fitted, labels = v7.fit_target(spec, x, frame)
            fitted_models[target] = fitted
            fit_log.append({"stage": "full_seven_refit", "target": target, "candidate": spec["id"], "fit_rooms": cfg["development_rooms"],
                            "fit_rows": len(frame), "estimators": len(fitted["estimators"]), "training_label_hashes": labels,
                            "training_inputs_sha256": v7.T.array_sha(x.to_numpy()), "seconds": time.perf_counter()-t0})
            print(f"Refit {target} {spec['family']} {spec['id']}", flush=True)
    bundle = {"version": cfg["version"], "raw_features": api.RAW_FEATURES, "encoded_features": list(x.columns), "models": fitted_models}
    joblib.dump(bundle, model_dir/"model.joblib", compress=3)
    profiles = frame[["Room_ID"]+api.ROOM_FEATURES].drop_duplicates()
    assert len(profiles) == 7
    metadata = {"version": cfg["version"], "angles": api.ANGLES, "input_features": api.INPUT_FEATURES,
                "raw_features": api.RAW_FEATURES, "encoded_features": list(x.columns), "targets": api.TARGETS,
                "units": {t: "fraction" if t in api.FRACTIONS else "kWh" if t == "PV_AC_kWh" else "lux" for t in api.TARGETS},
                "training_rows": len(frame), "training_room_features": profiles.to_dict("records"),
                "training_feature_ranges": {c: {"min": float(frame[c].min()), "max": float(frame[c].max())} for c in api.INPUT_FEATURES},
                "models": selected, "protected_test_rooms": cfg["protected_test_rooms"],
                "coverage_warning": "Marginal ranges and exact geometry matches only; not a confidence interval or reliability certificate",
                "provenance": "EPW identity, hour alignment, model north and louver definitions unconfirmed",
                "scope": cfg["scope"]}
    dump(model_dir/"metadata.json", metadata)
    dump(model_dir/"manifest.json", {"version": cfg["version"], "created_utc": datetime.now(timezone.utc).isoformat(),
          "versions": versions, "python": sys.version, "files": {name: api.sha256(model_dir/name) for name in ["model.joblib", "metadata.json"]},
          "parameter_freeze_sha256": api.sha256(out/"parameter_freeze.json"), "training_source_sha256": cfg["source_sha256"],
          "training_rows": len(frame), "training_rooms": cfg["development_rooms"], "TEST_predictions": 0, "TEST_scores": 0})
    dump(out/"fit_log.json", fit_log)
    examples = ROOT/"examples/trial_v0_1"
    examples.mkdir(parents=True, exist_ok=True)
    # One known room/hour per development room; no target labels exported to inputs.
    sample = frame[(frame.PVSD_Angle_deg == 0)&(frame.DOY_1based == 172)&(frame.Hour == 12)][["Room_ID"]+api.INPUT_FEATURES].copy()
    assert len(sample) == 7
    sample.insert(1, "Request_ID", [f"demo_{i+1}" for i in range(len(sample))])
    sample.to_csv(examples/"input.csv", index=False, encoding="utf-8-sig")
    dump(examples/"input.json", sample.to_dict("records"))
    pd.DataFrame(columns=["Room_ID", "Request_ID"]+api.INPUT_FEATURES).to_csv(examples/"input_template.csv", index=False, encoding="utf-8-sig")
    assert sum(r["estimators"] for r in fit_log) == 48
    assert api.sha256(source) == cfg["source_sha256"]
    run.update({"status": "trained_pending_interface_verification", "completed_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter()-started, "parameter_selection_estimator_fits": 42, "full_seven_estimator_fits": 6,
                "parameter_freeze_sha256": api.sha256(out/"parameter_freeze.json"), "model_manifest_sha256": api.sha256(model_dir/"manifest.json")})
    dump(out/"run_manifest.json", run)
    dump(out/"progress.json", {"stage": "trained_pending_interface_verification", "completed_estimator_fits": 48, "total_estimator_fits": 48})
    print(f"Training finished in {time.perf_counter()-started:.1f}s; TEST untouched.", flush=True)


if __name__ == "__main__":
    main()
