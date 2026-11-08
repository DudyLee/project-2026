"""Approved P10 target-handling comparison on TRAIN-only room folds."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import shutil
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/experiment_v2.json'


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def training_scale_weights(frame):
    """Call with fitting rows only; no statistics from the held-out room."""
    scales=frame.groupby('Room_ID').E_P10_lux.mean()
    raw=1/np.sqrt(1+scales)
    weights=frame.Room_ID.map(raw).to_numpy(dtype=float)
    normalizer=float(weights.mean())
    return weights/normalizer,scales,raw,normalizer


def main():
    started=time.perf_counter()
    cfg=json.loads(CONFIG.read_text())
    plan=json.loads((ROOT/'configs/model_plan.json').read_text())
    base=runpy.run_path(str(ROOT/'scripts/train_logo_v1.py'),run_name='v2_import')
    out=ROOT/cfg['output']
    models=ROOT/cfg['model_output']
    for path in [out,out/'predictions',models,models/'mean_stage',out/'figures']:
        path.mkdir(parents=True,exist_ok=True)
    if (out/'fold_metrics.csv').exists():raise RuntimeError('V2 results already exist; do not overwrite.')
    assert Path(os.sys.prefix).resolve()==(ROOT/'.conda/envs/pvsd-ml-v1').resolve()
    source=ROOT/cfg['source']
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    original_run=json.loads((ROOT/'reports/experiment_v1/run_manifest.json').read_text())
    assert source_hash==original_run['hashes']['source_sha256']
    d,skipped=base['load_train'](source,plan)
    assert set(d.Room_ID).isdisjoint(cfg['protected_test_rooms'])
    cols=plan['feature_sets'][cfg['feature_set']]
    assert len(cols)==13 and not set(cols).intersection(plan['targets'])
    x=base['encode_inputs'](d,cols)
    assert x.shape[1]==15
    folds=list(LeaveOneGroupOut().split(d,groups=d.Room_ID))
    assert len(folds)==7
    keys=['Room_ID','PVSD_Angle_deg','HOY_0based']
    old_p10=pd.read_parquet(ROOT/'reports/experiment_v1/predictions/B_add_solar_weather__HistGB.parquet')
    old_log_p10=pd.read_parquet(ROOT/'reports/diagnosis_v1/predictions/HistGB_log1p__E_P10_lux.parquet')
    old_log_mean=pd.read_parquet(ROOT/'reports/diagnosis_v1/predictions/HistGB_log1p__E_mean_lux.parquet')
    for reference in [old_p10,old_log_p10,old_log_mean]:assert reference[keys].equals(d[keys])
    assert np.array_equal(old_p10.true__E_P10_lux,d.E_P10_lux)
    assert np.array_equal(old_log_p10.truth,d.E_P10_lux)
    assert np.array_equal(old_log_mean.truth,d.E_mean_lux)
    weights_log,metrics,mean_metrics,splits,mean_models,source_model_hashes=[],[],[],[],{},[]
    save_cols=['Room_ID','Month','Day','Hour','DOY_1based','HOY_0based','PVSD_Angle_deg']
    outputs={name:d[save_cols].copy() for name in cfg['p10_methods']}
    for p in outputs.values():
        p['fold']=-1
        p['truth']=d.E_P10_lux
        p['true_mean_for_evaluation_only']=d.E_mean_lux
        for c in ['predicted_mean','raw_prediction','prediction']:p[c]=np.nan
    outputs['Relative_P10']['ratio_prediction']=np.nan
    outputs['Relative_P10']['raw_ratio_prediction']=np.nan
    mean_oof=d[save_cols].copy()
    mean_oof['truth']=d.E_mean_lux
    mean_oof['prediction']=np.nan
    run={'experiment_id':cfg['experiment_id'],'status':'running','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source_hash,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'train_rows':len(d),'train_rooms':sorted(d.Room_ID.unique().tolist()),'test_rows_skipped':skipped,'test_predictions':0,'test_scores':0,'config':cfg,'conda_prefix':os.sys.prefix,'python_executable':os.sys.executable,'library_versions':original_run['versions']}
    write_json(out/'run_manifest.json',run)
    shutil.copyfile(CONFIG,out/'config_snapshot.json')
    write_json(out/'raw_and_encoded_inputs.json',{'raw':cols,'encoded':list(x.columns)})
    completed=0
    with threadpool_limits(limits=cfg['thread_limit']):
        for fold,(ti,vi) in enumerate(folds,1):
            room=d.iloc[vi].Room_ID.unique().item()
            fit_rooms=sorted(d.iloc[ti].Room_ID.unique().tolist())
            assert room not in fit_rooms and len(fit_rooms)==6
            split={'fold':fold,'training_rooms':fit_rooms,'validation_room':room,'n_train':len(ti),'n_validation':len(vi)}
            splits.append(split)
            mean_path=ROOT/cfg['mean_stage']['reuse_frozen_source']/f'fold_{fold:02d}_{room}.joblib'
            bundle=joblib.load(mean_path)
            assert bundle['train_rooms']==fit_rooms and bundle['held_room']==room
            assert bundle['target']=='E_mean_lux' and bundle['target_log1p'] is True
            assert bundle['raw_features']==cols and bundle['encoded_features']==list(x.columns)
            archived=models/'mean_stage'/mean_path.name
            shutil.copyfile(mean_path,archived)
            digest=hashlib.sha256(mean_path.read_bytes()).hexdigest()
            assert hashlib.sha256(archived.read_bytes()).hexdigest()==digest
            source_model_hashes.append({'source':mean_path.relative_to(ROOT).as_posix(),'archived':archived.relative_to(ROOT).as_posix(),'sha256':digest,'fold':fold,'held_room':room})
            mean_model=bundle['estimator']
            mean_models[fold]=mean_model
            predicted_mean=np.maximum(np.expm1(mean_model.predict(x.iloc[vi])),0)
            assert np.allclose(predicted_mean,old_log_mean.prediction.iloc[vi],rtol=1e-10,atol=1e-10)
            mean_oof.loc[vi,'prediction']=predicted_mean
            mean_metrics.append({'method':'Mean_log1p','room':room,'fold':fold,**base['metrics'](d.E_mean_lux.iloc[vi].to_numpy(),predicted_mean)})
            for method,reference in [('Original_HistGB_B',old_p10.pred__E_P10_lux),('Unweighted_log1p_P10',old_log_p10.prediction)]:
                metrics.append({'method':method,'room':room,'fold':fold,**base['metrics'](d.E_P10_lux.iloc[vi].to_numpy(),reference.iloc[vi].to_numpy()),'fit_seconds':0.0,'newly_fitted':False})
            w,scales,raw_weights,normalizer=training_scale_weights(d.iloc[ti])
            assert np.isclose(w.mean(),1) and np.isfinite(w).all() and (w>0).all()
            assert set(scales.index)==set(fit_rooms) and room not in scales.index
            for fit_room in fit_rooms:
                weights_log.append({'fold':fold,'held_room':room,'fitting_room':fit_room,'fitting_room_mean_P10':float(scales[fit_room]),'raw_weight':float(raw_weights[fit_room]),'normalized_weight':float(raw_weights[fit_room]/normalizer),'training_rows':int(d.iloc[ti].Room_ID.eq(fit_room).sum())})
            for method in cfg['p10_methods']:
                estimator=HistGradientBoostingRegressor(**cfg['hist_gradient_boosting'])
                y=d.E_P10_lux.iloc[ti].to_numpy()
                if method=='Weighted_log1p_P10':
                    fitted_target=np.log1p(y)
                    kwargs={'sample_weight':w}
                else:
                    fitted_target=y/(d.E_mean_lux.iloc[ti].to_numpy()+cfg['epsilon_lux'])
                    kwargs={}
                t0=time.perf_counter()
                estimator.fit(x.iloc[ti],fitted_target,**kwargs)
                seconds=time.perf_counter()-t0
                raw_estimate=estimator.predict(x.iloc[vi])
                if method=='Weighted_log1p_P10':
                    raw=np.expm1(raw_estimate)
                    pred=np.maximum(raw,0)
                else:
                    # Only X-derived mean predictions enter the primary reconstruction.
                    raw=raw_estimate*(predicted_mean+cfg['epsilon_lux'])
                    ratio=np.maximum(raw_estimate,0)
                    pred=ratio*(predicted_mean+cfg['epsilon_lux'])
                    outputs[method].loc[vi,'ratio_prediction']=ratio
                    outputs[method].loc[vi,'raw_ratio_prediction']=raw_estimate
                assert np.isfinite(pred).all()
                p=outputs[method]
                p.loc[vi,'fold']=fold
                p.loc[vi,'predicted_mean']=predicted_mean
                p.loc[vi,'raw_prediction']=raw
                p.loc[vi,'prediction']=pred
                row={'method':method,'room':room,'fold':fold,**base['metrics'](d.E_P10_lux.iloc[vi].to_numpy(),pred),'raw_mae':float(np.abs(raw-d.E_P10_lux.iloc[vi].to_numpy()).mean()),'clipped_fraction':float((raw<0).mean()),'fit_seconds':seconds,'newly_fitted':True}
                metrics.append(row)
                md=models/method
                md.mkdir(exist_ok=True)
                model_bundle={'estimator':estimator,'method':method,'target':'E_P10_lux','raw_features':cols,'encoded_features':list(x.columns),'training_rooms':fit_rooms,'held_room':room,'fold':fold,'epsilon_lux':cfg['epsilon_lux'],'mean_model':archived.relative_to(ROOT).as_posix(),'weight_scales':scales.to_dict() if method=='Weighted_log1p_P10' else None,'weight_normalizer':normalizer if method=='Weighted_log1p_P10' else None}
                joblib.dump(model_bundle,md/f'fold_{fold:02d}_{room}.joblib',compress=3)
                completed+=1
                pd.DataFrame(metrics).to_csv(out/'fold_metrics.partial.csv',index=False)
                write_json(out/'progress.json',{'status':'running','completed_new_fits':completed,'total_new_fits':14,'reused_mean_models':fold,'elapsed_seconds':time.perf_counter()-started})
                print(f'[{completed}/14] {method} {room}: MAE={row["mae"]:.6f} lux, RMSE={row["rmse"]:.6f}',flush=True)
    assert completed==14
    fm=pd.DataFrame(metrics)
    assert len(fm)==28 and fm.groupby('method').size().eq(7).all()
    fm.to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(weights_log).to_csv(out/'training_room_weights.csv',index=False)
    pd.DataFrame(mean_metrics).to_csv(out/'mean_stage_fold_metrics.csv',index=False)
    mean_oof.to_parquet(out/'predictions/Mean_log1p.parquet',index=False,compression='zstd')
    segment_metrics=[]
    for method,p in outputs.items():
        assert not p.isna().any().any() and p.fold.ge(1).all()
        assert set(p.Room_ID).isdisjoint(cfg['protected_test_rooms'])
        p.to_parquet(out/'predictions'/f'{method}.parquet',index=False,compression='zstd')
        p.head(100).to_csv(out/'predictions'/f'{method}__sample.csv',index=False)
        for by in ['Room_ID','Hour','Month','PVSD_Angle_deg']:
            for value,g in p.groupby(by):
                segment_metrics.append({'method':method,'group_by':by,'group_value':str(value),'n':len(g),**base['metrics'](g.truth.to_numpy(),g.prediction.to_numpy())})
    pd.DataFrame(segment_metrics).to_csv(out/'stratified_metrics.csv',index=False)
    summary=fm.groupby('method',sort=False).agg(room_count=('room','nunique'),mae_mean=('mae','mean'),mae_median=('mae','median'),mae_min=('mae','min'),mae_max=('mae','max'),rmse_mean=('rmse','mean'),r2_median=('r2','median'),r2_min=('r2','min')).reset_index()
    pivot=fm.pivot(index='room',columns='method',values='mae')
    compare=[]
    for method in cfg['p10_methods']:
        for room,r in pivot.iterrows():
            compare.append({'method':method,'room':room,'mae':float(r[method]),'original_mae':float(r.Original_HistGB_B),'unweighted_log_mae':float(r.Unweighted_log1p_P10),'mae_reduction_vs_original_pct':float(100*(r.Original_HistGB_B-r[method])/r.Original_HistGB_B),'mae_reduction_vs_unweighted_log_pct':float(100*(r.Unweighted_log1p_P10-r[method])/r.Unweighted_log1p_P10)})
    pd.DataFrame(compare).to_csv(out/'paired_room_comparison.csv',index=False)
    for ref in ['Original_HistGB_B','Unweighted_log1p_P10']:
        ref_mean=float(summary.loc[summary.method.eq(ref),'mae_mean'].iloc[0])
        summary['mae_reduction_vs_'+ref+'_pct']=100*(ref_mean-summary.mae_mean)/ref_mean
        summary['rooms_improved_vs_'+ref]=summary.method.map(lambda method:int((pivot[method]<pivot[ref]).sum()))
    summary.to_csv(out/'summary_metrics.csv',index=False)
    # Retrospective error attribution only; these labels never feed inference above.
    p=outputs['Relative_P10']
    true_ratio=p.truth/(p.true_mean_for_evaluation_only+cfg['epsilon_lux'])
    ratio_term=(p.ratio_prediction-true_ratio)*(p.true_mean_for_evaluation_only+cfg['epsilon_lux'])
    mean_term=p.ratio_prediction*(p.predicted_mean-p.true_mean_for_evaluation_only)
    assert np.allclose((ratio_term+mean_term).to_numpy(),(p.prediction-p.truth).to_numpy(),rtol=1e-8,atol=1e-9)
    decomposition=[]
    for room,g in p.groupby('Room_ID'):
        ids=g.index
        decomposition.append({'room':room,'ratio_estimation_term_mean_abs_lux':float(ratio_term.loc[ids].abs().mean()),'mean_propagation_term_mean_abs_lux':float(mean_term.loc[ids].abs().mean()),'ratio_term_bias_lux':float(ratio_term.loc[ids].mean()),'mean_term_bias_lux':float(mean_term.loc[ids].mean()),'actual_mae_lux':float((g.prediction-g.truth).abs().mean()),'note':'Retrospective label-based attribution; terms can cancel, so their absolute magnitudes do not add to MAE.'})
    pd.DataFrame(decomposition).to_csv(out/'ratio_error_decomposition.csv',index=False)
    write_json(out/'fold_splits.json',splits)
    write_json(out/'reused_mean_model_provenance.json',source_model_hashes)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    run.update({'status':'completed','finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,'new_P10_fits':14,'reused_mean_model_count':7,'saved_model_count':len(list(models.rglob('*.joblib'))),'source_unchanged':True})
    write_json(out/'run_manifest.json',run)
    write_json(out/'progress.json',{'status':'completed','completed_new_fits':14,'reused_mean_models':7,'elapsed_seconds':run['elapsed_seconds']})
    print('Completed: 14 new P10 fits; 7 frozen mean models reused; TEST predictions/scores = 0.',flush=True)
    print(summary.to_string(index=False),flush=True)
    print(pivot.to_string(),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        output=ROOT/'reports/experiment_v2'
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/'failure.json',{'error':repr(error),'traceback':traceback.format_exc()})
        raise
