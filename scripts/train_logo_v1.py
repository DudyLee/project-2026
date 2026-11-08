"""Fixed-parameter TRAIN-only leave-one-room-out regression experiment."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / 'configs' / 'experiment_v1.json'
FRACTIONS = {'F_300_fraction', 'F_UDI_fraction'}


def dump_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train(source: Path, plan: dict) -> tuple[pd.DataFrame, int]:
    """TEST rows are discarded before target conversion or numeric operations."""
    metadata = ['Room_ID', 'Sample_Role', 'Month', 'Day', 'Hour', 'DOY_1based', 'HOY_0based', 'PVSD_Angle_deg']
    columns = list(dict.fromkeys(metadata + sum(plan['feature_sets'].values(), []) + plan['targets']))
    records, skipped = [], 0
    with source.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        assert set(columns).issubset(reader.fieldnames), 'Required columns missing'
        for row in reader:
            if row['Sample_Role'] != 'TRAIN':
                assert row['Sample_Role'] == 'TEST', 'Unexpected sample role'
                skipped += 1
                continue
            assert row['Room_ID'] in plan['validation']['development_rooms']
            records.append({c: row[c] for c in columns})
    frame = pd.DataFrame.from_records(records)
    for c in columns:
        if c not in ['Room_ID', 'Sample_Role']:
            frame[c] = pd.to_numeric(frame[c], errors='raise')
    assert len(frame) == plan['validation']['development_rows']
    assert skipped == plan['validation']['final_test_rows']
    assert set(frame.Room_ID) == set(plan['validation']['development_rooms'])
    assert not set(frame.Room_ID).intersection(plan['validation']['final_test_rooms'])
    assert not frame.duplicated(['Room_ID', 'PVSD_Angle_deg', 'HOY_0based']).any()
    assert frame.groupby(['Room_ID', 'PVSD_Angle_deg']).size().eq(4380).all()
    assert np.isfinite(frame.select_dtypes('number')).all().all()
    return frame, skipped


def encode_inputs(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Deterministic encodings only. No statistics fitted using validation rows."""
    encoded = pd.DataFrame(index=frame.index)
    for c in columns:
        if c == 'DOY_1based':
            phase = 2 * np.pi * (frame[c] - 1) / 365
            encoded['DOY_sin'] = np.sin(phase)
            encoded['DOY_cos'] = np.cos(phase)
        elif c == 'Solar_Azimuth_mid_deg':
            phase = np.deg2rad(frame[c])
            encoded['Solar_Azimuth_sin'] = np.sin(phase)
            encoded['Solar_Azimuth_cos'] = np.cos(phase)
        else:
            encoded[c] = frame[c].astype(float)
    return encoded


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - y
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        'mae': float(np.mean(np.abs(error))),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'r2': float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else None,
        'bias': float(np.mean(error)),
    }


def bound_prediction(pred: np.ndarray, target: str) -> np.ndarray:
    return np.clip(pred, 0, 1) if target in FRACTIONS else np.maximum(pred, 0)


def main() -> None:
    started = time.perf_counter()
    cfg = json.loads(CFG_PATH.read_text(encoding='utf-8'))
    plan_path = ROOT / cfg['feature_plan']
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    source = ROOT / cfg['source']
    out = ROOT / cfg['output']
    model_root = ROOT / cfg['model_output']
    for folder in [out, model_root, out/'predictions', out/'figures']:
        folder.mkdir(parents=True, exist_ok=True)
    if (out/'fold_metrics.csv').exists():
        raise RuntimeError('Results already exist. Use a new experiment configuration/output directory.')
    expected_env = ROOT/'.conda'/'envs'/'pvsd-ml-v1'
    assert Path(sys.prefix).resolve() == expected_env.resolve(), 'Run this in the dedicated project conda environment'
    hashes = {'source_sha256': sha256(source), 'config_sha256': sha256(CFG_PATH), 'feature_plan_sha256': sha256(plan_path), 'script_sha256': sha256(Path(__file__))}
    frame, skipped = load_train(source, plan)
    versions = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib','pyarrow','threadpoolctl']}
    run = {
        'experiment_id': cfg['experiment_id'], 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'python': sys.version, 'python_executable': sys.executable, 'conda_prefix': sys.prefix,
        'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'thread_limit': cfg['thread_limit'],
        'versions': versions, 'hashes': hashes, 'config': cfg,
        'train_rows': len(frame), 'train_rooms': sorted(frame.Room_ID.unique().tolist()),
        'test_rows_skipped': skipped, 'test_predictions': 0, 'test_scores': 0,
        'raw_input_columns': plan['feature_sets'], 'targets': plan['targets'],
        'configuration_selected_using_results': False,
        'scope': 'Fixed-parameter development cross-validation only; no final test or full-seven-room deployment refit',
    }
    dump_json(out/'run_manifest.json', run)
    dump_json(out/'experiment_config_snapshot.json', cfg)
    print(f"Loaded TRAIN={len(frame)} rows / {frame.Room_ID.nunique()} rooms; skipped TEST={skipped} rows.", flush=True)
    folds = list(LeaveOneGroupOut().split(frame, groups=frame.Room_ID))
    assert len(folds) == 7
    split_records = []
    for fold, (ti, vi) in enumerate(folds, 1):
        fit_rooms = sorted(frame.iloc[ti].Room_ID.unique().tolist())
        val_rooms = frame.iloc[vi].Room_ID.unique().tolist()
        assert len(fit_rooms) == 6 and len(val_rooms) == 1 and not set(fit_rooms).intersection(val_rooms)
        split_records.append({'fold': fold, 'training_rooms': fit_rooms, 'validation_room': val_rooms[0], 'training_rows':len(ti), 'validation_rows':len(vi)})
    dump_json(out/'fold_splits.json', split_records)
    fold_metrics, stratified, encoded_names = [], [], {}
    completed = 0
    oof_paths = []
    with threadpool_limits(limits=cfg['thread_limit']):
        for feature_set, raw_columns in plan['feature_sets'].items():
            assert not set(raw_columns).intersection(plan['targets'])
            x = encode_inputs(frame, raw_columns)
            encoded_names[feature_set] = list(x.columns)
            assert x.shape[1] == cfg['feature_encoding']['encoded_counts'][feature_set]
            for model_name in ['Ridge', 'HistGB']:
                oof = frame[['Room_ID','Month','Day','Hour','DOY_1based','HOY_0based','PVSD_Angle_deg']].copy()
                oof['fold'] = -1
                for target in plan['targets']:
                    oof['true__'+target] = frame[target]
                    oof['pred__'+target] = np.nan
                    oof['raw__'+target] = np.nan
                for split, (ti, vi) in zip(split_records, folds):
                    assert oof.loc[vi,'fold'].eq(-1).all()
                    oof.loc[vi,'fold'] = split['fold']
                    for target in plan['targets']:
                        t0 = time.perf_counter()
                        estimator = make_pipeline(StandardScaler(), Ridge(**cfg['ridge'])) if model_name == 'Ridge' else HistGradientBoostingRegressor(**cfg['hist_gradient_boosting'])
                        estimator.fit(x.iloc[ti], frame[target].iloc[ti].to_numpy())
                        fit_seconds = time.perf_counter()-t0
                        t0=time.perf_counter()
                        raw = estimator.predict(x.iloc[vi])
                        prediction_seconds = time.perf_counter()-t0
                        assert np.isfinite(raw).all()
                        pred = bound_prediction(raw, target)
                        y = frame[target].iloc[vi].to_numpy()
                        violation = (raw < 0) | ((raw > 1) if target in FRACTIONS else False)
                        row = {
                            'feature_set':feature_set, 'raw_feature_count':len(raw_columns), 'encoded_feature_count':x.shape[1],
                            'model':model_name, 'target':target, 'unit':'fraction' if target in FRACTIONS else 'kWh' if target=='PV_AC_kWh' else 'lux',
                            'fold':split['fold'], 'validation_room':split['validation_room'], 'training_rooms':';'.join(split['training_rooms']),
                            'n_train':len(ti), 'n_validation':len(vi), **metrics(y,pred),
                            **{'raw_'+k:v for k,v in metrics(y,raw).items()},
                            'clipped_count':int(violation.sum()), 'clipped_fraction':float(violation.mean()),
                            'positive_target_mae':float(np.mean(np.abs(pred[y>0]-y[y>0]))) if (y>0).any() else None,
                            'fit_seconds':fit_seconds,'predict_seconds':prediction_seconds,
                        }
                        fold_metrics.append(row)
                        oof.loc[vi,'pred__'+target]=pred
                        oof.loc[vi,'raw__'+target]=raw
                        model_dir=model_root/feature_set/model_name
                        model_dir.mkdir(parents=True,exist_ok=True)
                        bundle={'estimator':estimator,'raw_features':raw_columns,'encoded_features':list(x.columns),'target':target,'split':split,'prediction_bounds':[0,1] if target in FRACTIONS else [0,None],'encoding':cfg['feature_encoding'],'sklearn_version':versions['scikit-learn']}
                        joblib.dump(bundle,model_dir/f"fold_{split['fold']:02d}_{split['validation_room']}__{target}.joblib",compress=3)
                        completed+=1
                        pd.DataFrame(fold_metrics).to_csv(out/'fold_metrics.partial.csv',index=False)
                        dump_json(out/'progress.json',{'status':'running','completed_fits':completed,'total_fits':140,'feature_set':feature_set,'model':model_name,'target':target,'validation_room':split['validation_room'],'elapsed_seconds':time.perf_counter()-started})
                        print(f"[{completed:03d}/140] {feature_set} {model_name} {split['validation_room']} {target}: MAE={row['mae']:.6g}, R2={row['r2']}, fit={fit_seconds:.2f}s",flush=True)
                assert oof['fold'].gt(0).all()
                assert not oof.filter(regex='^(true|pred|raw)__').isna().any().any()
                oof_path=out/'predictions'/f'{feature_set}__{model_name}.parquet'
                oof.to_parquet(oof_path,index=False,compression='zstd')
                oof_paths.append(oof_path.relative_to(ROOT).as_posix())
                oof.head(100).to_csv(out/'predictions'/f'{feature_set}__{model_name}__sample.csv',index=False)
                for target in plan['targets']:
                    for by in ['Room_ID','Hour','Month','PVSD_Angle_deg']:
                        for value,group in oof.groupby(by):
                            stratified.append({'feature_set':feature_set,'model':model_name,'target':target,'group_by':by,'group_value':str(value),'n':len(group),**metrics(group['true__'+target].to_numpy(),group['pred__'+target].to_numpy())})
    fm=pd.DataFrame(fold_metrics)
    assert len(fm)==140 and completed==140
    fm.to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(stratified).to_csv(out/'stratified_metrics.csv',index=False)
    summary=fm.groupby(['feature_set','model','target','unit'],sort=False).agg(
        room_count=('validation_room','nunique'),mae_mean=('mae','mean'),mae_std=('mae','std'),mae_min=('mae','min'),mae_max=('mae','max'),
        rmse_mean=('rmse','mean'),r2_mean=('r2','mean'),r2_min=('r2','min'),r2_max=('r2','max'),
        raw_mae_mean=('raw_mae','mean'),clipped_fraction_mean=('clipped_fraction','mean'),fit_seconds_sum=('fit_seconds','sum')).reset_index()
    summary.to_csv(out/'summary_metrics.csv',index=False)
    pairs=fm.pivot(index=['model','target','unit','validation_room'],columns='feature_set',values='mae').reset_index()
    pairs['mae_reduction']=pairs.A_bim_time_angle-pairs.B_add_solar_weather
    pairs['mae_reduction_pct']=100*pairs.mae_reduction/pairs.A_bim_time_angle
    pairs['weather_better']=pairs.mae_reduction>0
    pairs.to_csv(out/'paired_room_comparison.csv',index=False)
    comparison=[]
    for (model,target,unit),g in pairs.groupby(['model','target','unit'],sort=False):
        a,b=g.A_bim_time_angle.mean(),g.B_add_solar_weather.mean()
        comparison.append({'model':model,'target':target,'unit':unit,'base_mae':float(a),'weather_mae':float(b),'mae_reduction_pct':float(100*(a-b)/a) if a>0 else None,'rooms_improved':int(g.weather_better.sum()),'rooms_total':len(g)})
    pd.DataFrame(comparison).to_csv(out/'weather_feature_comparison.csv',index=False)
    assert sha256(source)==hashes['source_sha256'], 'Raw source changed during experiment'
    dump_json(out/'encoded_feature_names.json',encoded_names)
    run.update({'status':'completed','completed_fits':completed,'folds':split_records,'encoded_input_columns':encoded_names,'prediction_files':oof_paths,'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,'raw_source_unchanged':True,'model_file_count':len(list(model_root.rglob('*.joblib')))})
    dump_json(out/'run_manifest.json',run)
    dump_json(out/'progress.json',{'status':'completed','completed_fits':completed,'total_fits':140,'elapsed_seconds':run['elapsed_seconds']})
    print('COMPLETED: 140 fits; four TRAIN-only out-of-fold prediction files; no TEST prediction or score.',flush=True)
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import traceback
        output=ROOT/'reports'/'experiment_v1'
        if output.exists():
            dump_json(output/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'time_utc':datetime.now(timezone.utc).isoformat()})
        raise
