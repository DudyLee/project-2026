"""Existing-data, frozen-model BM702 diagnosis; no fitting or visualization."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import pvsd_trial as api
import train_logo_v1 as base

CONFIG = ROOT/"configs/diagnosis_BM702_v1.json"
KEYS = ["Room_ID", "PVSD_Angle_deg", "HOY_0based"]
TEXT = {"Room_ID", "Sample_Role", "Level", "Revit_Window_Element_IDs"}


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_subset(path, allowed):
    records, skipped = [], 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["Room_ID"] not in allowed:
                skipped += 1
                continue
            records.append(row)
    frame = pd.DataFrame.from_records(records)
    for name in frame.columns:
        if name not in TEXT:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame, skipped


def stats(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    e = p-y
    return {"n": len(y), **base.metrics(y, p), "true_mean": float(y.mean()), "pred_mean": float(p.mean()),
            "true_p90": float(np.quantile(y, .9)), "over_fraction": float((e > 1e-9).mean()),
            "under_fraction": float((e < -1e-9).mean()), "sum_absolute_error": float(np.abs(e).sum())}


def main():
    started = time.perf_counter()
    cfg = read(CONFIG)
    out = ROOT/cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    if (out/"run_manifest.json").exists():
        raise RuntimeError("Diagnosis already started; preserve existing evidence")
    assert api.sha256(ROOT/cfg["source"]) == cfg["source_sha256"]
    assert api.sha256(ROOT/cfg["model_dir"]/"model.joblib") == cfg["model_sha256"]
    allowed = set(cfg["development_rooms"]+[cfg["focal_room"]])
    d, skipped = load_subset(ROOT/cfg["source"], allowed)
    bim, _ = load_subset(ROOT/cfg["bim_source"], allowed)
    assert len(d) == 140160 and skipped == 35040 and len(bim) == 8
    assert not set(d.Room_ID) & set(cfg["excluded_from_this_diagnosis"])
    assert d.loc[d.Room_ID == cfg["focal_room"], "Sample_Role"].eq("TEST").all()
    assert d.loc[d.Room_ID != cfg["focal_room"], "Sample_Role"].eq("TRAIN").all()
    inputs = [CONFIG, ROOT/cfg["source"], ROOT/cfg["bim_source"], ROOT/cfg["model_dir"]/"model.joblib",
              ROOT/cfg["model_dir"]/"metadata.json", ROOT/cfg["model_dir"]/"manifest.json", ROOT/"pvsd_trial.py", Path(__file__),
              ROOT/"reports/holdout_trial_v0_1/scored_predictions.parquet"]
    hashes = {p.relative_to(ROOT).as_posix(): api.sha256(p) for p in inputs}
    run = {"analysis_id": cfg["analysis_id"], "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "input_hashes": hashes, "numeric_rows": len(d), "other_TEST_rows_skipped": skipped,
           "new_estimator_fits": 0, "new_simulations": 0, "new_visualizations": 0, "scope": cfg["scope"]}
    dump(out/"run_manifest.json", run)
    checks = []
    def check(name, violations, detail):
        checks.append({"check": name, "violations": int(violations), "detail": detail})
    check("missing", d.isna().sum().sum(), "Retained eight rooms only")
    check("nonfinite_numeric", (~np.isfinite(d.select_dtypes("number"))).sum().sum(), "All numeric fields")
    check("duplicate_case_hour", d.duplicated(KEYS).sum(), "Room/angle/HOY")
    check("case_count_4380", (~d.groupby(["Room_ID", "PVSD_Angle_deg"]).size().eq(4380)).sum(), "32 cases, each 4380 retained hours")
    check("calendar_DOY", (pd.to_datetime({"year": np.full(len(d),2001), "month":d.Month,"day":d.Day}).dt.dayofyear != d.DOY_1based).sum(), "Generic non-leap calendar; not confirmation of simulation year")
    offsets = d.HOY_0based-((d.DOY_1based-1)*24+d.Hour)
    check("HOY_offset_consistency", int(offsets.nunique() != 1), "Observed indexing consistency only; hour interval remains unconfirmed")
    static = [c for c in bim.columns if c != "Room_ID"]+["Sensor_Count", "Floor_0based"]
    check("within_room_static_features", (d.groupby("Room_ID")[static].nunique(dropna=False) != 1).sum().sum(), "Must be fixed across all four angles and hours")
    profiles = d[["Room_ID"]+static].drop_duplicates().sort_values("Room_ID").reset_index(drop=True)
    check("unique_room_profiles", abs(len(profiles)-8), "One profile per room")
    bm = bim.set_index("Room_ID")
    for c in [c for c in bim.columns if c != "Room_ID"]:
        observed = profiles.set_index("Room_ID")[c].loc[bm.index]
        if c in TEXT:
            mismatch = observed.astype(str).to_numpy() != bm[c].astype(str).to_numpy()
        else:
            mismatch = ~np.isclose(observed.to_numpy(), bm[c].to_numpy(), rtol=1e-9, atol=1e-9)
        check("BIM_join__"+c, mismatch.sum(), "Cross-file consistency; not independent original-geometry verification")
    check("WFR_identity", (~np.isclose(profiles.WFR_window_to_floor, profiles.Window_Area_Nominal_m2/profiles.Room_Area_m2, atol=1e-9)).sum(), "Nominal window area / room area")
    check("obstruction_identity", (~np.isclose(profiles.Solar_Obstruction_Ratio, 1-profiles.Annual_Incident_Shaded_kWh_m2/profiles.Annual_Incident_Unshaded_kWh_m2, atol=1e-9)).sum(), "1 - annual shaded / unshaded")
    weather = [c for c in d if c.startswith("EPW_") or c.startswith("Solar_Altitude") or c.startswith("Solar_Azimuth")]
    check("weather_equal_at_HOY", (d.groupby("HOY_0based")[weather].nunique() != 1).sum().sum(), "Weather/solar values identical across retained rooms and angles at each timestamp")
    consistency = []
    for room, g in d.groupby("Room_ID", sort=True):
        n = g.Sensor_Count.to_numpy(dtype=float)
        q = g.E_P10_lux.to_numpy(); m = g.E_mean_lux.to_numpy()
        f3 = g.F_300_fraction.to_numpy(); fu = g.F_UDI_fraction.to_numpy()
        weight = (n-np.ceil(.1*(n-1)))/n
        violations = {
            "negative_targets": int((g[api.TARGETS] < -1e-10).sum().sum()),
            "fraction_outside_unit": int(((g[list(api.FRACTIONS)] < -1e-10)|(g[list(api.FRACTIONS)] > 1+1e-10)).sum().sum()),
            "fraction_sensor_count_noninteger": int((np.abs(np.r_[f3*n,fu*n]-np.rint(np.r_[f3*n,fu*n])) > 1e-6).sum()),
            "mean_below_300_times_F300": int((m+1e-7 < 300*f3).sum()),
            "mean_below_100_times_FUDI": int((m+1e-7 < 100*fu).sum()),
            "mean_below_P10_min_mass": int((m+1e-7 < q*weight).sum()),
            "FUDI_below_F300_minus_mean_over3000": int((fu+1e-7 < f3-m/3000).sum()),
        }
        consistency.append({"room": room, "Sensor_Count": int(n[0]), **violations})
    pd.DataFrame(checks).to_csv(out/"data_checks.csv", index=False)
    pd.DataFrame(consistency).to_csv(out/"daylight_consistency_checks.csv", index=False)
    profiles.to_csv(out/"room_profiles.csv", index=False)
    orientation = profiles.groupby(["Orientation_X_model", "Orientation_Y_model"]).agg(rooms=("Room_ID", lambda s: ";".join(s)), room_count=("Room_ID", "size")).reset_index()
    orientation.to_csv(out/"orientation_coverage.csv", index=False)
    print("Internal CSV/BIM/weather and daylight consistency checks complete.", flush=True)

    model = api.TrialPredictor(ROOT/cfg["model_dir"])
    x = api.encode_inputs(d[api.RAW_FEATURES])
    with threadpool_limits(limits=4):
        for t in cfg["targets"]:
            d["pred__"+t] = api.predict_target(model.bundle["models"][t], x)
        pair = model.bundle["models"]["E_P10_lux"]["estimators"]
        d["paired_mean"] = np.maximum(np.expm1(pair["mean_log"].predict(x)), 0)
        d["ratio_pred"] = np.maximum(pair["ratio"].predict(x), 0)
    d["ratio_true"] = d.E_P10_lux/(d.E_mean_lux+1)
    d["DNI_band"] = pd.cut(d.EPW_DNI_Wh_m2, cfg["DNI_edges"], labels=["zero", "0_to_200", "200_to_500", "above_500"], include_lowest=True).astype(str)
    d["GHI_band"] = pd.cut(d.EPW_GHI_Wh_m2, cfg["GHI_edges"], labels=["0_to_50", "50_to_200", "200_to_500", "above_500"], include_lowest=True).astype(str)
    d["Altitude_band"] = pd.cut(d.Solar_Altitude_mid_deg, cfg["altitude_edges"], labels=["below_horizon", "0_to_20", "20_to_50", "above_50"], include_lowest=True).astype(str)
    d["Daypart"] = np.select([d.Hour <= 12, d.Hour <= 17], ["07_to_12", "13_to_17"], default="18")
    focal = d[d.Room_ID == cfg["focal_room"]].copy()
    ix = focal.index
    prior = pd.read_parquet(ROOT/"reports/holdout_trial_v0_1/scored_predictions.parquet")
    prior = prior[prior.Room_ID == cfg["focal_room"]].set_index(KEYS).loc[pd.MultiIndex.from_frame(focal[KEYS])]
    reload_diff = {t: float(np.max(np.abs(prior["pred__"+t].to_numpy()-focal["pred__"+t].to_numpy()))) for t in cfg["targets"]}
    assert max(reload_diff.values()) < 1e-9
    room_scores, strata, target_profiles = [], [], []
    for room, g in d.groupby("Room_ID", sort=True):
        for t in cfg["targets"]:
            room_scores.append({"room": room, "target": t, "scope": "observed_holdout" if room == cfg["focal_room"] else "TRAIN_in_sample_not_validation", **stats(g[t],g["pred__"+t])})
        target_profiles.append({"room": room, "n": len(g), "mean_lux": float(g.E_mean_lux.mean()), "P10_mean_lux": float(g.E_P10_lux.mean()),
                                "FUDI_mean": float(g.F_UDI_fraction.mean()), "ratio_mean": float(g.ratio_true.mean()),
                                "ratio_median": float(g.ratio_true.median()), "ratio_p90": float(g.ratio_true.quantile(.9)),
                                "P10_zero_fraction": float(g.E_P10_lux.eq(0).mean())})
    for t in cfg["targets"]:
        total = np.abs(focal["pred__"+t]-focal[t]).sum()
        for axis in cfg["strata"]:
            for label, g in focal.groupby(axis, sort=True):
                s = stats(g[t],g["pred__"+t])
                strata.append({"target": t, "axis": axis, "group": str(label), **s, "fraction_of_target_absolute_error": s["sum_absolute_error"]/total})
    pd.DataFrame(room_scores).to_csv(out/"native_room_metrics.csv",index=False)
    pd.DataFrame(target_profiles).to_csv(out/"target_profiles.csv",index=False)
    pd.DataFrame(strata).to_csv(out/"BM702_stratified_errors.csv",index=False)
    # Simple room distances based only on three continuous geometry/shading inputs.
    train_profiles = profiles[profiles.Room_ID.isin(cfg["development_rooms"])].set_index("Room_ID")
    focus_profile = profiles[profiles.Room_ID == cfg["focal_room"]].iloc[0]
    geo = ["Room_Area_m2", "Glass_Area_m2", "Solar_Obstruction_Ratio"]
    scales = (train_profiles[geo].max()-train_profiles[geo].min()).to_numpy()
    distances = []
    for room, r in train_profiles.iterrows():
        distances.append({"TRAIN_room": room, "continuous_feature_scaled_distance": float(np.sqrt(np.mean(((r[geo].to_numpy(dtype=float)-focus_profile[geo].to_numpy(dtype=float))/scales)**2))),
                          "orientation_X": r.Orientation_X_model, "orientation_Y": r.Orientation_Y_model})
    pd.DataFrame(distances).sort_values("continuous_feature_scaled_distance").to_csv(out/"geometry_neighbours.csv",index=False)

    # Nonphysical diagnostic only: demonstrate exact branch saturation for negative X.
    probe_x = x.loc[ix].copy()
    probe_x["Orientation_X_model"] = 0.0
    probes = []
    with threadpool_limits(limits=4):
        for t in cfg["targets"]:
            p = api.predict_target(model.bundle["models"][t], probe_x)
            delta = p-focal["pred__"+t].to_numpy()
            probes.append({"target": t, "probe": "X=-1_to_0_with_Y=0_NONPHYSICAL_not_a_candidate", "rows": len(p),
                           "maximum_absolute_prediction_change": float(np.abs(delta).max()), "mean_absolute_prediction_change": float(np.abs(delta).mean()),
                           "exact_equal_fraction": float((delta == 0).mean())})
    pd.DataFrame(probes).to_csv(out/"orientation_branch_probe.csv",index=False)
    # P10 error = mean term + ratio term + interaction; ORACLEs never performance candidates.
    m = focal.E_mean_lux.to_numpy(); mh = focal.paired_mean.to_numpy()
    r = focal.ratio_true.to_numpy(); rh = focal.ratio_pred.to_numpy(); y = focal.E_P10_lux.to_numpy()
    error = focal["pred__E_P10_lux"].to_numpy()-y
    parts = {"mean_term": r*(mh-m), "ratio_term": (m+1)*(rh-r), "interaction_term": (mh-m)*(rh-r)}
    identity_error = float(np.abs(sum(parts.values())-error).max())
    assert identity_error < 1e-9
    oracle_rows = []
    for name, p in [("native_pair", focal["pred__E_P10_lux"].to_numpy()), ("ORACLE_true_mean_only", rh*(m+1)), ("ORACLE_true_ratio_only", r*(mh+1)), ("ORACLE_both", r*(m+1))]:
        oracle_rows.append({"diagnostic": name, "deployable": name == "native_pair", **stats(y,p)})
    pd.DataFrame(oracle_rows).to_csv(out/"P10_oracle_diagnostics.csv",index=False)
    components = focal[KEYS+["DOY_1based","Hour","PVSD_Angle_deg"]].loc[:,lambda z:~z.columns.duplicated()].copy()
    components["true_mean"] = m; components["paired_mean"] = mh
    components["true_P10"] = y; components["pred_P10"] = focal["pred__E_P10_lux"].to_numpy()
    components["ratio_true"] = r; components["ratio_pred"] = rh
    for name, a in parts.items(): components[name] = a
    components.to_parquet(out/"P10_components.parquet",index=False)
    decomposition = {"identity_max_abs_error": identity_error, "true_ratio_mean": float(r.mean()), "pred_ratio_mean": float(rh.mean()),
                     "true_mean_mean": float(m.mean()), "paired_mean_mean": float(mh.mean()), "paired_mean_metrics": stats(m,mh),
                     "terms": {name:{"mean":float(a.mean()),"mean_absolute":float(np.abs(a).mean())} for name,a in parts.items()},
                     "warning": "Terms interact/cancel; absolute magnitudes do not add to MAE. Oracle inputs unavailable at deployment."}
    dump(out/"P10_decomposition.json",decomposition)
    print("Strata, orientation branch probe and P10 component diagnostics complete.",flush=True)

    # Exact training-room contribution to RF ratio leaves using the original bootstrap samples.
    train = d[d.Room_ID.isin(cfg["development_rooms"])].copy()
    xt = x.loc[train.index].to_numpy(dtype=np.float32)
    xb = x.loc[ix].to_numpy(dtype=np.float32)
    room_order = cfg["development_rooms"]
    room_code = pd.Categorical(train.Room_ID, categories=room_order).codes
    ratio_labels = train.ratio_true.to_numpy()
    rf = pair["ratio"]
    samples = rf.estimators_samples_
    weights = np.zeros((len(focal),len(room_order)))
    contributions = np.zeros_like(weights)
    for tree, sample in zip(rf.estimators_,samples):
        leaf_train = tree.apply(xt)
        leaf_b = tree.apply(xb)
        counts = np.zeros((tree.tree_.node_count,len(room_order)))
        sums = np.zeros_like(counts)
        np.add.at(counts,(leaf_train[sample],room_code[sample]),1)
        np.add.at(sums,(leaf_train[sample],room_code[sample]),ratio_labels[sample])
        denominator = counts.sum(axis=1)[leaf_b]
        assert (denominator > 0).all()
        weights += counts[leaf_b]/denominator[:,None]/len(rf.estimators_)
        contributions += sums[leaf_b]/denominator[:,None]/len(rf.estimators_)
    attribution_error = float(np.abs(contributions.sum(axis=1)-focal.ratio_pred.to_numpy()).max())
    assert attribution_error < 1e-10 and np.allclose(weights.sum(axis=1),1)
    attrib = []
    for i, room in enumerate(room_order):
        attrib.append({"TRAIN_room": room,"mean_leaf_weight":float(weights[:,i].mean()),"mean_ratio_contribution":float(contributions[:,i].mean()),
                       "weight_07_to_12":float(weights[focal.Hour.to_numpy()<=12,i].mean()),
                       "weight_13_to_17":float(weights[focal.Hour.between(13,17).to_numpy(),i].mean())})
    pd.DataFrame(attrib).sort_values("mean_leaf_weight",ascending=False).to_csv(out/"RF_ratio_room_attribution.csv",index=False)
    dump(out/"mechanical_verification.json",{"status":"passed","native_reload_max_differences":reload_diff,
         "P10_decomposition_max_error":identity_error,"RF_ratio_leaf_reconstruction_max_error":attribution_error,
         "model_hash_unchanged":api.sha256(ROOT/cfg["model_dir"]/"model.joblib")==cfg["model_sha256"],
         "scope":"Numerical consistency and exact model mechanism reproduction; no causal identification"})
    for relative,digest in hashes.items(): assert api.sha256(ROOT/relative)==digest,relative
    run.update({"status":"completed","completed_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.perf_counter()-started,
                "internal_check_violations":int(pd.DataFrame(checks).violations.sum()),
                "daylight_necessary_inequality_violations":int(pd.DataFrame(consistency).drop(columns=["room","Sensor_Count"]).to_numpy().sum())})
    dump(out/"run_manifest.json",run)
    print(json.dumps({"status":"completed","seconds":run["elapsed_seconds"],"checks":run["internal_check_violations"],"daylight_checks":run["daylight_necessary_inequality_violations"],"decomposition":decomposition,"oracle_metrics":oracle_rows,"branch_probe":probes,"rf_attribution":attrib},ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
