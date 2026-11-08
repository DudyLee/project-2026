"""Offline Tower A four-angle inference. Runtime needs no source data or labels.

Python: TrialPredictor().predict(list_of_records) -> pandas.DataFrame
CLI: python -m pvsd_trial --input conditions.csv --output predictions.csv
Only load the trusted, versioned project joblib package (pickle-based).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import numbers
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models/trial_v0_1"
ANGLES = [0, 30, 60, 90]
PROTECTED_ROOMS = {"BM532", "BM702", "BMG26"}
RAW_FEATURES = [
    "Room_Area_m2", "Glass_Area_m2", "Orientation_X_model", "Orientation_Y_model",
    "Solar_Obstruction_Ratio", "DOY_1based", "Hour", "PVSD_Angle_deg",
    "EPW_GHI_Wh_m2", "EPW_DNI_Wh_m2", "EPW_DHI_Wh_m2",
    "Solar_Altitude_mid_deg", "Solar_Azimuth_mid_deg",
]
INPUT_FEATURES = [c for c in RAW_FEATURES if c != "PVSD_Angle_deg"]
ROOM_FEATURES = RAW_FEATURES[:5]
TARGETS = ["E_mean_lux", "E_P10_lux", "F_300_fraction", "F_UDI_fraction", "PV_AC_kWh"]
FRACTIONS = {"F_300_fraction", "F_UDI_fraction"}
IDENTIFIERS = ["Room_ID", "Request_ID"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode_inputs(frame):
    """Same deterministic encoding/order as the frozen experimental baseline."""
    result = pd.DataFrame(index=frame.index)
    for name in RAW_FEATURES:
        if name == "DOY_1based":
            phase = 2 * np.pi * (frame[name] - 1) / 365
            result["DOY_sin"], result["DOY_cos"] = np.sin(phase), np.cos(phase)
        elif name == "Solar_Azimuth_mid_deg":
            phase = np.deg2rad(frame[name])
            result["Solar_Azimuth_sin"], result["Solar_Azimuth_cos"] = np.sin(phase), np.cos(phase)
        else:
            result[name] = frame[name].astype(float)
    return result


def predict_target(fitted, encoded):
    est, spec = fitted["estimators"], fitted["spec"]
    if spec["transform"] == "relative":
        # P10 has its own companion RF mean model; never use true/output XGB mean.
        mean = np.maximum(np.expm1(est["mean_log"].predict(encoded)), 0)
        prediction = np.maximum(est["ratio"].predict(encoded), 0) * (mean + 1)
    else:
        prediction = est["value"].predict(encoded)
        if spec["transform"] == "log1p":
            prediction = np.expm1(prediction)
        elif spec["transform"] != "identity":
            raise ValueError("Unknown frozen target transform")
    prediction = np.clip(prediction, 0, 1) if spec["target"] in FRACTIONS else np.maximum(prediction, 0)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite model prediction; no output written")
    return np.asarray(prediction, dtype=np.float64)


class TrialPredictor:
    """Load once and reuse. All predictions carry scope/coverage flags."""

    def __init__(self, model_dir=DEFAULT_MODEL):
        self.model_dir = Path(model_dir)
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["version"] != VERSION:
            raise ValueError("Model/interface version mismatch")
        for package in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]:
            if importlib.metadata.version(package) != self.manifest["versions"][package]:
                raise RuntimeError(f"Dependency version mismatch: {package}; use the recorded trial environment")
        for name in ["model.joblib", "metadata.json"]:
            if sha256(self.model_dir / name) != self.manifest["files"][name]:
                raise ValueError(f"Model integrity check failed: {name}")
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.bundle = joblib.load(self.model_dir / "model.joblib")
        if self.bundle["version"] != VERSION or self.bundle["raw_features"] != RAW_FEATURES:
            raise ValueError("Unexpected model feature schema")
        if list(self.bundle["models"]) != TARGETS or self.metadata["angles"] != ANGLES:
            raise ValueError("Unexpected output schema")
        self.model_hash = self.manifest["files"]["model.joblib"]

    def validate(self, records):
        if isinstance(records, pd.DataFrame):
            if records.columns.duplicated().any():
                raise ValueError("Duplicate input column")
            frame = records.copy()
        elif isinstance(records, list) and all(isinstance(r, dict) for r in records):
            frame = pd.DataFrame.from_records(records)
        else:
            raise ValueError("Input must be a nonempty list of records or a DataFrame")
        if frame.empty:
            raise ValueError("Input is empty")
        missing = set(INPUT_FEATURES) - set(frame.columns)
        extra = set(frame.columns) - set(INPUT_FEATURES + IDENTIFIERS)
        if missing or extra:
            raise ValueError(f"Missing fields: {sorted(missing)}; unexpected fields: {sorted(extra)}. Supply 12 non-angle features only, plus optional identifiers.")
        frame = frame.reset_index(drop=True)
        for name in IDENTIFIERS:
            if name not in frame:
                frame[name] = ""
            if not frame[name].map(lambda v: isinstance(v, str)).all():
                raise ValueError(f"{name} must be a string, or omit the field")
            frame[name] = frame[name].str.strip()
        if frame.Room_ID.str.upper().isin(PROTECTED_ROOMS).any():
            raise ValueError("Protected TEST Room_ID: BM532/BM702/BMG26 remain sealed; prediction refused")
        for name in INPUT_FEATURES:
            # Reject booleans, complex numbers, missing and text values in Python/JSON.
            if not frame[name].map(lambda v: isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))).all():
                raise ValueError(f"{name} must contain finite numeric values")
            frame[name] = frame[name].astype(float)
            if not np.isfinite(frame[name]).all():
                raise ValueError(f"{name} must contain finite numeric values")
        rules = {
            "Room_Area_m2": (0, None, True), "Glass_Area_m2": (0, None, False),
            "Orientation_X_model": (-1, 1, False), "Orientation_Y_model": (-1, 1, False),
            "Solar_Obstruction_Ratio": (0, 1, False), "DOY_1based": (1, 365, False),
            "Hour": (7, 18, False), "EPW_GHI_Wh_m2": (0, None, False),
            "EPW_DNI_Wh_m2": (0, None, False), "EPW_DHI_Wh_m2": (0, None, False),
            "Solar_Altitude_mid_deg": (-90, 90, False), "Solar_Azimuth_mid_deg": (0, 360, False),
        }
        for name, (lo, hi, strict) in rules.items():
            v = frame[name]
            invalid = (v <= lo) if strict else (v < lo)
            if hi is not None:
                invalid |= v > hi
            if invalid.any():
                raise ValueError(f"{name} outside physical/interface bounds at input row {int(np.flatnonzero(invalid)[0])+1}")
        for name in ["DOY_1based", "Hour"]:
            if (frame[name] != np.floor(frame[name])).any():
                raise ValueError(f"{name} must be integer")
        norm = np.hypot(frame.Orientation_X_model, frame.Orientation_Y_model)
        if not np.isclose(norm, 1, atol=1e-4, rtol=0).all():
            raise ValueError("Orientation_X_model/Y_model must form a unit vector in the original model coordinates")
        return frame

    def coverage_flags(self, frame):
        profiles = self.metadata["training_room_features"]
        profile_values = np.array([[p[c] for c in ROOM_FEATURES] for p in profiles])
        known_ids = {p["Room_ID"] for p in profiles}
        flags = []
        for row in frame.to_dict("records"):
            current = ["PROVENANCE_UNCONFIRMED"]
            current.append("DEVELOPMENT_ROOM" if row["Room_ID"] in known_ids else "NEW_OR_UNSPECIFIED_ROOM")
            matches = np.isclose(profile_values, [row[c] for c in ROOM_FEATURES], atol=1e-6, rtol=1e-6).all(axis=1)
            if not matches.any():
                current.append("UNSEEN_ROOM_FEATURE_COMBINATION")
            if row["Room_ID"] in known_ids and not any(matches[i] and p["Room_ID"] == row["Room_ID"] for i, p in enumerate(profiles)):
                current.append("ROOM_ID_GEOMETRY_MISMATCH")
            orientation = [row["Orientation_X_model"], row["Orientation_Y_model"]]
            if not np.isclose(profile_values[:, 2:4], orientation, atol=1e-6, rtol=0).all(axis=1).any():
                current.append("UNSEEN_MODEL_ORIENTATION")
            for name in INPUT_FEATURES:
                limits = self.metadata["training_feature_ranges"][name]
                if row[name] < limits["min"] - 1e-9 or row[name] > limits["max"] + 1e-9:
                    current.append("OUTSIDE_TRAIN_RANGE:" + name)
            flags.append(";".join(current))
        return flags

    def predict(self, records, batch_size=5000):
        """Return four rows per input record in input order; no labels required."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        frame = self.validate(records)
        flags = self.coverage_flags(frame)
        outputs = []
        with threadpool_limits(limits=4):
            for start in range(0, len(frame), batch_size):
                stop = min(start + batch_size, len(frame))
                block = frame.iloc[start:stop]
                expanded = block.loc[block.index.repeat(4)].reset_index(drop=True)
                expanded["PVSD_Angle_deg"] = np.tile(ANGLES, len(block))
                x = encode_inputs(expanded)
                if list(x.columns) != self.bundle["encoded_features"]:
                    raise ValueError("Encoded feature order mismatch")
                out = expanded[["Room_ID", "Request_ID", "DOY_1based", "Hour", "PVSD_Angle_deg"]].copy()
                out.insert(0, "Input_Row", np.repeat(np.arange(start+1, stop+1), 4))
                out[["DOY_1based", "Hour"]] = out[["DOY_1based", "Hour"]].astype(int)
                for target in TARGETS:
                    out[target] = predict_target(self.bundle["models"][target], x)
                out["Model_Version"] = VERSION
                out["Model_SHA256"] = self.model_hash
                out["Usage_Status"] = "RESEARCH_TRIAL_NOT_INDEPENDENTLY_TESTED"
                out["Coverage_Flags"] = np.repeat(flags[start:stop], 4)
                outputs.append(out)
        return pd.concat(outputs, ignore_index=True)


def _unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_conditions(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_json_keys)
    if path.suffix.lower() != ".csv":
        raise ValueError("Input must be .csv or .json")
    import csv
    with path.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column")
    frame = pd.read_csv(path, dtype={c: str for c in IDENTIFIERS}, keep_default_na=False)
    for name in INPUT_FEATURES:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tower A trial: 12 non-angle features -> 4 angles x 5 predictions")
    parser.add_argument("--input", required=True, type=Path, help="CSV or JSON list of records")
    parser.add_argument("--output", required=True, type=Path, help="CSV or JSON output; existing files are protected")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output must be different files")
        if args.output.suffix.lower() not in [".csv", ".json"]:
            raise ValueError("Output must be .csv or .json")
        if args.output.exists() and not args.overwrite:
            raise ValueError("Output exists; choose a new path or explicitly use --overwrite")
        predictor = TrialPredictor(args.model_dir)
        result = predictor.predict(read_conditions(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv":
            result.to_csv(args.output, index=False, encoding="utf-8-sig")
        else:
            args.output.write_text(json.dumps(result.to_dict("records"), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Trial {VERSION}: {len(result)//4} input records -> {len(result)} angle rows. Saved: {args.output}")
        print("Research trial only; inspect Coverage_Flags. TEST rooms remain sealed.")
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Input/model error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
