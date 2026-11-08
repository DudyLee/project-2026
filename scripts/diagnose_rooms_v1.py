"""TRAIN-only data, input-support, frozen-model and controlled-ablation diagnosis."""
from pathlib import Path
import csv
import hashlib
import json
import runpy
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs'/'diagnosis_v1.json'
STATIC=['Room_Area_m2','Glass_Area_m2','Orientation_X_model','Orientation_Y_model','Solar_Obstruction_Ratio']


def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    clock=time.perf_counter()
    cfg=json.loads(CFG.read_text())
    plan=json.loads((ROOT/'configs/model_plan.json').read_text())
    baseline_cfg=json.loads((ROOT/'configs/experiment_v1.json').read_text())
    base=runpy.run_path(str(ROOT/'scripts/train_logo_v1.py'),run_name='diagnosis_import')
    out=ROOT/cfg['output']
    for folder in [out,out/'predictions',out/'figures',ROOT/cfg['model_output']]:folder.mkdir(parents=True,exist_ok=True)
    if (out/'ablation_fold_metrics.csv').exists():raise RuntimeError('Diagnostic results already exist; use a new version.')
    d,skipped=base['load_train'](ROOT/cfg['source'],plan)
    assert set(d.Room_ID)==set(cfg['development_rooms']) and skipped==52560
    extra_cols=['Floor_0based','WFR_window_to_floor','Sensor_Count','Window_Count','Window_Area_Nominal_m2']
    extras=[]
    with (ROOT/cfg['source']).open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            if row['Sample_Role']!='TRAIN':continue
            extras.append({k:row[k] for k in ['Room_ID','PVSD_Angle_deg','HOY_0based']+extra_cols})
    e=pd.DataFrame(extras)
    for c in e:
        if c!='Room_ID':e[c]=pd.to_numeric(e[c],errors='raise')
    d=d.merge(e,on=['Room_ID','PVSD_Angle_deg','HOY_0based'],validate='one_to_one',sort=False)
    assert len(d)==122640 and not set(d.Room_ID).intersection(cfg['protected_test_rooms'])
    source_hash=hashlib.sha256((ROOT/cfg['source']).read_bytes()).hexdigest()
    old=json.loads((ROOT/'reports/experiment_v1/run_manifest.json').read_text())
    assert source_hash==old['hashes']['source_sha256']
    save_json(out/'config_snapshot.json',cfg)
    manifest={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source_hash,'config_sha256':hashlib.sha256(CFG.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'train_rows':len(d),'train_rooms':sorted(d.Room_ID.unique().tolist()),'test_rows_skipped':skipped,'test_predictions':0,'test_scores':0,'variants':cfg['variants'],'scope':'Exploratory diagnostics on development folds observed to fail; not independent final evaluation'}
    save_json(out/'run_manifest.json',manifest)
    for target in cfg['targets']:
        assert d[target].ge(0).all()
    count=d.Sensor_Count.to_numpy()
    checks={
        'duplicate_keys':int(d.duplicated(['Room_ID','PVSD_Angle_deg','HOY_0based']).sum()),
        'missing_cells':int(d.isna().sum().sum()),
        'fraction_count_integrality_failures':{c:int((np.abs(d[c]*d.Sensor_Count-np.round(d[c]*d.Sensor_Count))>1e-7).sum()) for c in ['F_300_fraction','F_UDI_fraction']},
        'mean_below_300_times_F300':int((d.E_mean_lux+1e-7<300*d.F_300_fraction).sum()),
        'mean_below_100_times_FUDI':int((d.E_mean_lux+1e-7<100*d.F_UDI_fraction).sum()),
        'mean_below_nonnegative_grid_P10_bound':int((d.E_mean_lux+1e-7<d.E_P10_lux*(count-np.ceil(.1*(count-1)))/count).sum()),
        'P10_ge300_but_F300_too_small':int(((d.E_P10_lux>=300)&(d.F_300_fraction<.9-1/d.Sensor_Count-1e-7)).sum()),
        'within_room_static_variation':d.groupby('Room_ID')[STATIC+extra_cols].nunique().max().astype(int).to_dict(),
        'raw_sensor_grid_available':False,
        'interpretation':'These check internal consistency only; sensor mapping, percentile computation and geometry cannot be independently certified without original simulation grids and export code.'
    }
    save_json(out/'train_data_checks.json',checks)
    rooms=[]
    for room,g in d.groupby('Room_ID'):
        r={'Room_ID':room,**g[STATIC+extra_cols].iloc[0].to_dict(),'sensor_density_per_m2':float(g.Sensor_Count.iloc[0]/g.Room_Area_m2.iloc[0])}
        for target in cfg['targets']:
            r.update({target+'__mean':float(g[target].mean()),target+'__std':float(g[target].std(ddof=0)),target+'__max':float(g[target].max()),target+'__zero_fraction':float(g[target].eq(0).mean())})
        lit=g.E_mean_lux>1
        r['P10_to_mean_median_when_mean_gt1']=float((g.loc[lit,'E_P10_lux']/g.loc[lit,'E_mean_lux']).median())
        rooms.append(r)
    room_summary=pd.DataFrame(rooms)
    room_summary.to_csv(out/'train_room_profiles.csv',index=False)
    coverage=[]
    for room in cfg['development_rooms']:
        ti=d.Room_ID.ne(room)
        vi=~ti
        for c in STATIC+extra_cols:
            lo,hi=d.loc[ti,c].min(),d.loc[ti,c].max()
            value=d.loc[vi,c].iloc[0]
            coverage.append({'validation_room':room,'feature':c,'held_value':float(value),'training_min':float(lo),'training_max':float(hi),'training_unique':int(d.loc[ti,c].nunique()),'outside_training_range':bool(value<lo or value>hi)})
    pd.DataFrame(coverage).to_csv(out/'fold_feature_support.csv',index=False)
    baseline_oof=pd.read_parquet(ROOT/'reports/experiment_v1/predictions/B_add_solar_weather__HistGB.parquet')
    assert set(baseline_oof.Room_ID)==set(cfg['development_rooms'])
    keys=['Room_ID','PVSD_Angle_deg','HOY_0based']
    assert d[keys].equals(baseline_oof[keys])
    folds=list(LeaveOneGroupOut().split(d,groups=d.Room_ID))
    base_cols=plan['feature_sets']['B_add_solar_weather']
    xbase=base['encode_inputs'](d,base_cols)
    original_metrics=pd.read_csv(ROOT/'reports/experiment_v1/fold_metrics.csv')
    original_metrics=original_metrics[(original_metrics.model=='HistGB')&(original_metrics.feature_set=='B_add_solar_weather')&original_metrics.target.isin(cfg['targets'])]
    models=ROOT/'models/experiment_v1/B_add_solar_weather/HistGB'
    sensitivity,templates,training_error,segments,peaks=[],[],[],[],[]
    room_stat=d.groupby('Room_ID')[STATIC].first()
    with threadpool_limits(limits=cfg['thread_limit']):
        for room,target in cfg['focus'].items():
            fold=cfg['development_rooms'].index(room)+1
            fit=d.Room_ID.ne(room)
            val=~fit
            g=d.loc[val]
            x=xbase.loc[val]
            model=joblib.load(models/f'fold_{fold:02d}_{room}__{target}.joblib')['estimator']
            raw=model.predict(x)
            assert np.allclose(raw,baseline_oof.loc[val,'raw__'+target],atol=1e-10,rtol=1e-10)
            pred=np.maximum(raw,0)
            truth=g[target].to_numpy()
            trpred=np.maximum(model.predict(xbase.loc[fit]),0)
            training_error.append({'room':room,'target':target,'training_mae':float(np.abs(trpred-d.loc[fit,target].to_numpy()).mean()),'validation_mae':float(np.abs(pred-truth).mean()),'true_mean':float(truth.mean()),'predicted_mean':float(pred.mean()),'true_std_population':float(truth.std()),'bias':float((pred-truth).mean()),'overestimate_fraction':float((pred>truth).mean()),'zero_prediction_mae':float(np.abs(truth).mean()),'r2':base['metrics'](truth,pred)['r2']})
            focus=g[['Month','Day','Hour','PVSD_Angle_deg','HOY_0based',target]].copy()
            focus['prediction']=pred
            focus['abs_error']=np.abs(pred-truth)
            focus.to_csv(out/f'{room}__{target}__baseline_predictions.csv',index=False)
            for by in ['Hour','Month','PVSD_Angle_deg']:
                for value,gg in focus.groupby(by):
                    segments.append({'room':room,'target':target,'group_by':by,'group_value':int(value),'n':len(gg),'true_mean':float(gg[target].mean()),'pred_mean':float(gg.prediction.mean()),**base['metrics'](gg[target].to_numpy(),gg.prediction.to_numpy())})
            peaks.extend(focus.nlargest(25,'abs_error').assign(room=room,target_name=target).rename(columns={target:'truth'}).to_dict('records'))
            for feature in STATIC:
                values=set(d.loc[fit,feature].unique().tolist())|{float(g[feature].iloc[0])}
                for value in sorted(values):
                    altered=g.copy()
                    altered[feature]=value
                    alt=np.maximum(model.predict(base['encode_inputs'](altered,base_cols)),0)
                    sensitivity.append({'room':room,'target':target,'changed_feature':feature,'substituted_value':float(value),'mean_abs_prediction_change':float(np.abs(alt-pred).mean()),'fraction_predictions_unchanged':float(np.isclose(alt,pred,atol=1e-9,rtol=0).mean()),'diagnostic_mae_against_held_room':float(np.abs(alt-truth).mean())})
            for other in sorted(d.loc[fit,'Room_ID'].unique()):
                altered=g.copy()
                for c in STATIC:altered[c]=room_stat.loc[other,c]
                alt=np.maximum(model.predict(base['encode_inputs'](altered,base_cols)),0)
                reference=d[d.Room_ID.eq(other)].set_index(['PVSD_Angle_deg','HOY_0based']).loc[pd.MultiIndex.from_frame(g[['PVSD_Angle_deg','HOY_0based']]),target].to_numpy()
                templates.append({'room':room,'target':target,'training_room_template':other,'mean_abs_prediction_change':float(np.abs(alt-pred).mean()),'fraction_predictions_unchanged':float(np.isclose(alt,pred,atol=1e-9,rtol=0).mean()),'baseline_prediction_to_other_room_true_mae':float(np.abs(pred-reference).mean()),'baseline_prediction_other_room_true_correlation':float(np.corrcoef(pred,reference)[0,1]),'other_room_target_mean':float(reference.mean()),'held_room_target_mean':float(truth.mean())})
    for name,rows in [('frozen_model_train_validation',training_error),('focus_error_segments',segments),('worst_25_records_per_focus',peaks),('static_feature_sensitivity',sensitivity),('training_room_template_diagnostic',templates)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    all_metrics=[]
    for _,r in original_metrics.iterrows():
        all_metrics.append({'variant':'Original_HistGB_B','room':r.validation_room,'fold':int(r.fold),'target':r.target,'mae':float(r.mae),'rmse':float(r.rmse),'r2':float(r.r2),'bias':float(r.bias),'fit_seconds':0.0,'raw_feature_count':13})
    completed=0
    with threadpool_limits(limits=cfg['thread_limit']):
        for variant in cfg['variants']:
            columns=base_cols+['Floor_0based','WFR_window_to_floor'] if variant=='HistGB_add_floor_WFR' else [c for c in base_cols if c!='Solar_Obstruction_Ratio'] if variant=='HistGB_without_obstruction' else base_cols
            x=base['encode_inputs'](d,columns)
            for target in cfg['targets']:
                oof=d[['Room_ID','Month','Hour','PVSD_Angle_deg','HOY_0based']].copy()
                oof['truth']=d[target]
                oof['prediction']=np.nan
                oof['raw_prediction']=np.nan
                for fold,(ti,vi) in enumerate(folds,1):
                    held=d.iloc[vi].Room_ID.unique().tolist()
                    assert len(held)==1 and held[0] not in set(d.iloc[ti].Room_ID)
                    room=held[0]
                    if variant=='ExtraTrees_B':
                        model=ExtraTreesRegressor(n_estimators=100,max_depth=12,min_samples_leaf=100,max_features=1.0,n_jobs=4,random_state=2026)
                    else:
                        params=dict(baseline_cfg['hist_gradient_boosting'])
                        if variant=='HistGB_absolute':params['loss']='absolute_error'
                        model=HistGradientBoostingRegressor(**params)
                    y=d[target].iloc[ti].to_numpy()
                    if variant=='HistGB_log1p':y=np.log1p(y)
                    t0=time.perf_counter()
                    model.fit(x.iloc[ti],y)
                    seconds=time.perf_counter()-t0
                    raw=model.predict(x.iloc[vi])
                    if variant=='HistGB_log1p':raw=np.expm1(raw)
                    assert np.isfinite(raw).all()
                    pred=np.maximum(raw,0)
                    truth=d[target].iloc[vi].to_numpy()
                    r={'variant':variant,'room':room,'fold':fold,'target':target,**base['metrics'](truth,pred),'fit_seconds':seconds,'raw_feature_count':len(columns)}
                    all_metrics.append(r)
                    oof.loc[vi,'prediction']=pred
                    oof.loc[vi,'raw_prediction']=raw
                    model_dir=ROOT/cfg['model_output']/variant/target
                    model_dir.mkdir(parents=True,exist_ok=True)
                    joblib.dump({'estimator':model,'raw_features':columns,'encoded_features':list(x.columns),'target':target,'target_log1p':variant=='HistGB_log1p','train_rooms':sorted(d.iloc[ti].Room_ID.unique().tolist()),'held_room':room},model_dir/f'fold_{fold:02d}_{room}.joblib',compress=3)
                    completed+=1
                    pd.DataFrame(all_metrics).to_csv(out/'ablation_fold_metrics.partial.csv',index=False)
                    save_json(out/'progress.json',{'completed_cv_fits':completed,'total_cv_fits':70,'status':'running','elapsed_seconds':time.perf_counter()-clock})
                    print(f'[{completed}/70] {variant} {room} {target}: MAE={r["mae"]:.6g}',flush=True)
                assert not oof.isna().any().any()
                assert not set(oof.Room_ID).intersection(cfg['protected_test_rooms'])
                oof.to_parquet(out/'predictions'/f'{variant}__{target}.parquet',index=False,compression='zstd')
    am=pd.DataFrame(all_metrics)
    assert len(am)==84 and completed==70
    am.to_csv(out/'ablation_fold_metrics.csv',index=False)
    summary=am.groupby(['variant','target']).agg(mae_mean=('mae','mean'),rmse_mean=('rmse','mean'),r2_median=('r2','median'),r2_min=('r2','min'),rooms=('room','nunique')).reset_index()
    reference=summary[summary.variant.eq('Original_HistGB_B')].set_index('target')
    summary['mae_reduction_pct']=summary.apply(lambda r:100*(reference.loc[r.target,'mae_mean']-r.mae_mean)/reference.loc[r.target,'mae_mean'],axis=1)
    summary.to_csv(out/'ablation_summary.csv',index=False)
    known=[]
    with threadpool_limits(limits=cfg['thread_limit']):
        for room,target in cfg['focus'].items():
            g=d[d.Room_ID.eq(room)]
            val=g.Month.isin([3,6,9,12])
            x=base['encode_inputs'](g,base_cols)
            model=HistGradientBoostingRegressor(**baseline_cfg['hist_gradient_boosting'])
            model.fit(x.loc[~val],g.loc[~val,target])
            pred=np.maximum(model.predict(x.loc[val]),0)
            truth=g.loc[val,target].to_numpy()
            previous=baseline_oof.loc[g.index[val],'pred__'+target].to_numpy()
            known.append({'room':room,'target':target,'training_months':[1,2,4,5,7,8,10,11],'validation_months':[3,6,9,12],'n_train':int((~val).sum()),'n_validation':int(val.sum()),'known_room_mae':float(np.abs(pred-truth).mean()),'unseen_room_baseline_mae_same_times':float(np.abs(previous-truth).mean()),'known_room_r2':base['metrics'](truth,pred)['r2'],'scope':'Temporal prediction for a room with calibration data; not a new-room validation score'})
            pd.DataFrame({'Room_ID':room,'HOY_0based':g.loc[val,'HOY_0based'],'PVSD_Angle_deg':g.loc[val,'PVSD_Angle_deg'],'truth':truth,'known_room_prediction':pred,'unseen_room_baseline_prediction':previous}).to_parquet(out/'predictions'/f'known_room_month_holdout__{room}__{target}.parquet',index=False)
            joblib.dump({'estimator':model,'target':target,'room':room,'validation_months':[3,6,9,12],'raw_features':base_cols},ROOT/cfg['model_output']/f'known_room__{room}__{target}.joblib',compress=3)
    save_json(out/'known_room_time_diagnostic.json',known)
    assert hashlib.sha256((ROOT/cfg['source']).read_bytes()).hexdigest()==source_hash
    manifest.update({'status':'completed','finished_utc':datetime.now(timezone.utc).isoformat(),'cv_fits':70,'known_room_fits':2,'model_files':len(list((ROOT/cfg['model_output']).rglob('*.joblib'))),'elapsed_seconds':time.perf_counter()-clock,'source_unchanged':True})
    save_json(out/'run_manifest.json',manifest)
    save_json(out/'progress.json',{'status':'completed','completed_cv_fits':70,'known_room_fits':2,'elapsed_seconds':manifest['elapsed_seconds']})
    print(summary.to_string(index=False),flush=True)
    print(json.dumps({'training_error':training_error,'known_room':known},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        out=ROOT/'reports/diagnosis_v1'
        out.mkdir(parents=True,exist_ok=True)
        save_json(out/'failure.json',{'error':repr(error),'traceback':traceback.format_exc()})
        raise
