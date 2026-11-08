"""Contract, reload, source-integrity and TRAIN-only integration verification."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import pvsd_trial as api
import train_logo_v1 as base
import train_target_specific_v7 as v7


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    started = time.perf_counter()
    out = ROOT/"reports/trial_v0_1"
    examples = ROOT/"examples/trial_v0_1"
    model_dir = ROOT/"models/trial_v0_1"
    plan = json.loads((ROOT/"configs/model_plan.json").read_text(encoding="utf-8"))
    run = json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in run["input_hashes"].items():
        assert api.sha256(ROOT/relative) == expected, relative
    frame, skipped = base.load_train(ROOT/"data/raw/SEND_ML_BIM_EPW_07-18.csv", plan)
    assert len(frame) == 122640 and skipped == 52560
    assert not set(frame.Room_ID) & api.PROTECTED_ROOMS
    t0 = time.perf_counter()
    predictor = api.TrialPredictor()
    load_seconds = time.perf_counter()-t0
    conditions = frame[frame.PVSD_Angle_deg == 0][["Room_ID"]+api.INPUT_FEATURES].reset_index(drop=True)
    assert len(conditions) == 30660
    t0 = time.perf_counter()
    result = predictor.predict(conditions)
    batch_seconds = time.perf_counter()-t0
    assert len(result) == len(frame)
    assert result.groupby("Input_Row").PVSD_Angle_deg.apply(list).map(lambda x: x == api.ANGLES).all()
    keys = ["Room_ID", "DOY_1based", "Hour", "PVSD_Angle_deg"]
    ordered = frame.sort_values(keys).reset_index(drop=True)
    predicted = result.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered[keys], predicted[keys], check_dtype=False)
    legacy_x = base.encode_inputs(ordered, api.RAW_FEATURES)
    pd.testing.assert_frame_equal(legacy_x, api.encode_inputs(ordered))
    max_diff = {}
    with threadpool_limits(limits=4):
        for target in api.TARGETS:
            direct = v7.predict_target(predictor.bundle["models"][target], legacy_x)
            diff = float(np.max(np.abs(direct-predicted[target].to_numpy())))
            assert diff < 1e-9, (target, diff)
            max_diff[target] = diff
            assert predicted[target].ge(0).all()
            if target in api.FRACTIONS:
                assert predicted[target].le(1).all()
    # Confirm the fit log binds full-seven TRAIN inputs/labels, with six estimators.
    log = json.loads((out/"fit_log.json").read_text(encoding="utf-8"))
    assert sum(r["estimators"] for r in log if r["stage"] == "full_seven_refit") == 6
    full_x = base.encode_inputs(frame, api.RAW_FEATURES)
    for item in log:
        assert not set(item["fit_rooms"]) & api.PROTECTED_ROOMS
        if item["stage"] == "full_seven_refit":
            assert item["fit_rows"] == 122640 and len(item["fit_rooms"]) == 7
            assert item["training_inputs_sha256"] == v7.T.array_sha(full_x.to_numpy())
            y = frame[item["target"]].to_numpy()
            labels = {"mean_log": np.log1p(frame.E_mean_lux.to_numpy()), "ratio": y/(frame.E_mean_lux.to_numpy()+1)} if item["target"] == "E_P10_lux" else {"value": np.log1p(y) if item["target"] == "E_mean_lux" else y}
            assert {k: v7.T.array_sha(v) for k, v in labels.items()} == item["training_label_hashes"]
    sample = api.read_conditions(examples/"input.json")
    expected = predictor.predict(sample)
    pd.testing.assert_frame_equal(expected, predictor.predict(sample, batch_size=1))
    pd.testing.assert_frame_equal(expected, api.TrialPredictor().predict(sample))
    t0 = time.perf_counter()
    for _ in range(5):
        predictor.predict(sample[:1])
    one_record_seconds = (time.perf_counter()-t0)/5
    # Bad calls must fail before the model prediction function is entered.
    invalid = []
    def bad(name, records):
        with patch.object(api, "predict_target", side_effect=AssertionError("Invalid request reached model")) as spy:
            try:
                predictor.predict(records)
            except ValueError:
                invalid.append(name)
            else:
                raise AssertionError("Invalid input accepted: " + name)
            assert spy.call_count == 0
    for room in sorted(api.PROTECTED_ROOMS):
        bad("sealed_"+room, [{**sample[0], "Room_ID": " "+room.lower()+" "}])
    bad("empty", [])
    bad("missing_weather", [{k: v for k,v in sample[0].items() if k != "EPW_DNI_Wh_m2"}])
    for target in api.TARGETS:
        bad("target_leakage_"+target, [{**sample[0], target: 1}])
    bad("caller_angle", [{**sample[0], "PVSD_Angle_deg": 30}])
    for name, column, value in [
        ("nan", "Hour", float("nan")), ("infinity", "Room_Area_m2", float("inf")),
        ("bool", "DOY_1based", True), ("text", "Hour", "12"), ("negative_radiation", "EPW_GHI_Wh_m2", -1),
        ("zero_area", "Room_Area_m2", 0), ("shade_range", "Solar_Obstruction_Ratio", 2),
        ("hour_outside", "Hour", 19), ("fractional_hour", "Hour", 8.5), ("leap_day", "DOY_1based", 366),
        ("negative_glass", "Glass_Area_m2", -1), ("azimuth_range", "Solar_Azimuth_mid_deg", 361),
        ("altitude_range", "Solar_Altitude_mid_deg", 91), ("null_identifier", "Room_ID", None),
        ("bad_orientation", "Orientation_X_model", 2),
    ]:
        bad(name, [{**sample[0], column: value}])
    bad("zero_orientation", [{**sample[0], "Orientation_X_model": 0, "Orientation_Y_model": 0}])
    duplicate = pd.DataFrame(sample)
    duplicate = pd.concat([duplicate, duplicate[["Hour"]]], axis=1)
    bad("duplicate_column", duplicate)
    unknown = [{**sample[0], "Room_ID": "NEW_DEMO_ROOM", "Room_Area_m2": 10000}]
    flags = predictor.predict(unknown).Coverage_Flags.iloc[0]
    assert "UNSEEN_ROOM_FEATURE_COMBINATION" in flags and "OUTSIDE_TRAIN_RANGE:Room_Area_m2" in flags
    assert "PROVENANCE_UNCONFIRMED" in flags and "NEW_OR_UNSPECIFIED_ROOM" in flags
    mismatch = predictor.predict([{**sample[0], "Room_Area_m2": 10000}]).Coverage_Flags.iloc[0]
    assert "ROOM_ID_GEOMETRY_MISMATCH" in mismatch
    with tempfile.TemporaryDirectory(prefix="trial_verify_", dir=out) as td:
        temp = Path(td)
        # Isolated bundle has no data/, configs/ or old experiment models.
        isolated = temp/"portable_model"
        shutil.copytree(model_dir, isolated)
        def cli(input_path, output_path, expected_code=0):
            r = subprocess.run([sys.executable, str(ROOT/"pvsd_trial.py"), "--model-dir", str(isolated), "--input", str(input_path), "--output", str(output_path)], cwd=temp, capture_output=True, text=True, timeout=60)
            assert r.returncode == expected_code, (r.returncode, r.stdout, r.stderr)
            return r
        csv_path, json_path = temp/"out.csv", temp/"out.json"
        cli(examples/"input.csv", csv_path)
        cli(examples/"input.json", json_path)
        cli(examples/"input.csv", csv_path, 2)  # Explicit overwrite protection.
        got_csv = pd.read_csv(csv_path, keep_default_na=False)
        got_json = pd.DataFrame(json.loads(json_path.read_text(encoding="utf-8")))
        pd.testing.assert_frame_equal(expected, got_csv, check_dtype=False, atol=1e-10, rtol=1e-12)
        pd.testing.assert_frame_equal(expected, got_json, check_dtype=False, atol=1e-10, rtol=1e-12)
        for room in sorted(api.PROTECTED_ROOMS):
            blocked = temp/(room+".json")
            dump(blocked, [{**sample[0], "Room_ID": room}])
            rejected_out = temp/(room+"_output.csv")
            cli(blocked, rejected_out, 2)
            assert not rejected_out.exists()
        # Reject duplicate CSV headers/JSON keys rather than silently changing input.
        duplicate_csv = temp/"duplicate.csv"
        duplicate_csv.write_text("Hour,Hour\n7,8\n", encoding="utf-8")
        duplicate_json = temp/"duplicate.json"
        duplicate_json.write_text('[{"Hour":7,"Hour":8}]', encoding="utf-8")
        for path in [duplicate_csv, duplicate_json]:
            try:
                api.read_conditions(path)
            except ValueError:
                invalid.append(path.name)
            else:
                raise AssertionError("Duplicate fields accepted")
        (isolated/"metadata.json").write_text("{}", encoding="utf-8")
        try:
            api.TrialPredictor(isolated)
        except ValueError as exc:
            assert "integrity" in str(exc)
        else:
            raise AssertionError("Damaged model metadata accepted")
    expected.to_csv(examples/"expected_output.csv", index=False, encoding="utf-8-sig")
    dump(examples/"expected_output.json", expected.to_dict("records"))
    assert api.sha256(ROOT/"data/raw/SEND_ML_BIM_EPW_07-18.csv") == run["input_hashes"]["data/raw/SEND_ML_BIM_EPW_07-18.csv"]
    check = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter()-started,
             "conditions_checked": len(conditions), "four_angle_rows_checked": len(result), "target_values_checked": len(result)*5,
             "maximum_difference_from_legacy_direct_inference": max_diff,
             "full7_fit_input_label_hashes_verified": True, "encoded_columns_and_values_match_existing_baseline": True,
             "CSV_JSON_Python_equivalence": True, "single_and_batch_equivalence": True, "portable_bundle_without_source_data": True,
             "reload_and_integrity_checks": True, "invalid_cases_rejected_before_prediction": invalid,
             "protected_room_API_and_CLI_rejection": sorted(api.PROTECTED_ROOMS),
             "coverage_flags_checked": True, "TEST_predictions": 0, "TEST_scores": 0, "TEST_fits": 0, "additional_training_fits": 0,
             "timing_descriptive_this_machine": {"load_seconds": load_seconds, "four_angles_one_record_seconds_warm_average_5": one_record_seconds,
                     "all_30660_conditions_seconds": batch_seconds},
             "meaning": "Software/fit-lineage verification using TRAIN inputs only; these checks do not establish predictive accuracy or out-of-room reliability"}
    dump(out/"verification.json", check)
    run.update({"status": "trial_ready_verified", "verification": "reports/trial_v0_1/verification.json", "verification_sha256": api.sha256(out/"verification.json")})
    dump(out/"run_manifest.json", run)
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
