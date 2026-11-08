"""Approved TRAIN-only, grouped nested model comparison. No TEST predictions."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/experiment_v3.json'
BASE=runpy.run_path(str(ROOT/'scripts/train_logo_v1.py'),run_name='v3_shared')
TARGETS=['E_mean_lux','E_P10_lux','F_300_fraction','F_UDI_fraction','PV_AC_kWh']
PRIMITIVES=['mean_log','ratio','F_300_fraction','F_UDI_fraction','PV_AC_kWh']
META=['Room_ID','Month','Day','Hour','DOY_1based','HOY_0based','PVSD_Angle_deg']
KEYS=['Room_ID','PVSD_Angle_deg','HOY_0based']

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def make_estimator(family,params):
    return {'HistGB':HistGradientBoostingRegressor,'RandomForest':RandomForestRegressor,'XGBoost':XGBRegressor}[family](**params)

def params_for(cfg,family,cid):
    item=next(c for c in cfg['families'][family]['candidates'] if c['id']==cid)
    return cfg['families'][family]['common']|{k:v for k,v in item.items() if k!='id'}

def training_targets(frame):
    return {'mean_log':np.log1p(frame.E_mean_lux.to_numpy()),
            'ratio':frame.E_P10_lux.to_numpy()/(frame.E_mean_lux.to_numpy()+1),
            **{t:frame[t].to_numpy() for t in PRIMITIVES[2:]}}

def reconstruct(raw):
    mean=np.maximum(np.expm1(raw['mean_log']),0)
    return {'E_mean_lux':mean,'E_P10_lux':np.maximum(raw['ratio'],0)*(mean+1),
            **{t:BASE['bound_prediction'](raw[t],t) for t in PRIMITIVES[2:]}}

def empty_oof(d):
    p=d[META].copy();p['fold']=-1
    for t in TARGETS:p['true__'+t]=d[t];p['pred__'+t]=np.nan
    return p

def predict_bundle(bundle,x):
    """Uses X only. Every selected relative pair carries its own mean component."""
    raw={k:est.predict(x) for k,est in bundle['estimators'].items()}
    sel=bundle['selected_candidates']
    prediction={'E_mean_lux':np.maximum(np.expm1(raw[sel['E_mean_lux']+'__mean_log']),0)}
    cid=sel['E_P10_lux']
    prediction['E_P10_lux']=np.maximum(raw[cid+'__ratio'],0)*(np.maximum(np.expm1(raw[cid+'__mean_log']),0)+1)
    for t in PRIMITIVES[2:]:prediction[t]=BASE['bound_prediction'](raw[sel[t]+'__'+t],t)
    return prediction

def main():
    started=time.perf_counter();cfg=json.loads(CONFIG.read_text());plan=json.loads((ROOT/'configs/model_plan.json').read_text())
    out=ROOT/cfg['output'];models=ROOT/cfg['model_output']
    for p in [out,out/'predictions',out/'figures',models]:p.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():raise RuntimeError('V3 has already started. Use a new version rather than overwrite experiment evidence.')
    assert Path(sys.prefix).resolve()==(ROOT/'.conda/envs/pvsd-ml-v1').resolve()
    source=ROOT/cfg['source'];source_hash=digest(source)
    assert source_hash==json.loads((ROOT/'reports/experiment_v1/run_manifest.json').read_text())['hashes']['source_sha256']
    d,skipped=BASE['load_train'](source,plan)
    assert set(d.Room_ID).isdisjoint(cfg['protected_test_rooms'])
    cols=plan['feature_sets'][cfg['feature_set']];x=BASE['encode_inputs'](d,cols)
    assert len(cols)==13 and x.shape[1]==15
    families=list(cfg['families']);methods=['Frozen_HistGB']+families+['Nested_select']
    predictions={m:empty_oof(d) for m in methods}
    old=pd.read_parquet(ROOT/'reports/experiment_v1/predictions/B_add_solar_weather__HistGB.parquet')
    mean=pd.read_parquet(ROOT/'reports/experiment_v2/predictions/Mean_log1p.parquet')
    ratio=pd.read_parquet(ROOT/'reports/experiment_v2/predictions/Relative_P10.parquet')
    reference_files=[ROOT/'reports/experiment_v1/predictions/B_add_solar_weather__HistGB.parquet',ROOT/'reports/experiment_v2/predictions/Mean_log1p.parquet',ROOT/'reports/experiment_v2/predictions/Relative_P10.parquet']
    reference_hashes={str(p.relative_to(ROOT)):digest(p) for p in reference_files}
    for ref in [old,mean,ratio]:assert ref[KEYS].equals(d[KEYS])
    for t in TARGETS:
        assert np.array_equal(old['true__'+t],d[t])
        predictions['Frozen_HistGB']['pred__'+t]=mean.prediction if t=='E_mean_lux' else ratio.prediction if t=='E_P10_lux' else old['pred__'+t]
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','pyarrow','threadpoolctl']}
    run={'experiment_id':cfg['experiment_id'],'status':'running','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
         'source_sha256':source_hash,'config_sha256':digest(CONFIG),'script_sha256':digest(Path(__file__)),
         'reference_hashes':reference_hashes,'versions':versions,'python':sys.version,'conda_prefix':sys.prefix,
         'train_rows':len(d),'train_rooms':sorted(d.Room_ID.unique().tolist()),'test_rows_skipped':skipped,'test_predictions':0,'test_scores':0,
         'config':cfg,'inner_primitive_fits_planned':630,'outer_primitive_fits_maximum':126,'scope':cfg['validation']['scope']}
    dump(out/'run_manifest.json',run);dump(out/'config_snapshot.json',cfg)
    dump(out/'raw_and_encoded_inputs.json',{'raw':cols,'encoded':list(x.columns)})
    inner_rows=[];scores=[];selections=[];splits=[];fold_metrics=[];fit_counts={'inner':0,'outer':0};raw_checks=[]
    with threadpool_limits(limits=cfg['thread_limit']):
        for fold,(ti,vi) in enumerate(LeaveOneGroupOut().split(d,groups=d.Room_ID),1):
            room=d.iloc[vi].Room_ID.unique().item();fit_rooms=sorted(d.iloc[ti].Room_ID.unique().tolist())
            print(f'OUTER {fold}/7: score {room}; fit rooms {fit_rooms}',flush=True)
            train=d.iloc[ti].reset_index(drop=True);xt=x.iloc[ti].reset_index(drop=True);xv=x.iloc[vi]
            inner_splits=list(GroupKFold(n_splits=3).split(train,groups=train.Room_ID))
            split={'outer_fold':fold,'held_room':room,'training_rooms':fit_rooms,'training_rows':len(ti),'scoring_rows':len(vi),'inner':[]}
            for inner,(a,b) in enumerate(inner_splits,1):
                fit_set=sorted(train.iloc[a].Room_ID.unique().tolist());val_set=sorted(train.iloc[b].Room_ID.unique().tolist())
                assert len(fit_set)==4 and len(val_set)==2 and not set(fit_set)&set(val_set) and room not in fit_set+val_set
                split['inner'].append({'inner_fold':inner,'fit_rooms':fit_set,'validation_rooms':val_set,'n_fit':len(a),'n_validation':len(b)})
            splits.append(split);dump(out/'fold_splits.json',splits)
            all_fold_scores=[]
            for family in families:
                family_scores=[]
                for candidate in cfg['families'][family]['candidates']:
                    cid=candidate['id'];params=params_for(cfg,family,cid);candidate_scores={t:[] for t in TARGETS}
                    for inner,(a,b) in enumerate(inner_splits,1):
                        yf=training_targets(train.iloc[a]);raw={};t0=time.perf_counter()
                        for primitive in PRIMITIVES:
                            est=make_estimator(family,params);est.fit(xt.iloc[a],yf[primitive]);raw[primitive]=est.predict(xt.iloc[b]);fit_counts['inner']+=1
                        pred=reconstruct(raw)
                        for held in sorted(train.iloc[b].Room_ID.unique()):
                            mask=train.iloc[b].Room_ID.to_numpy()==held
                            for t in TARGETS:
                                met=BASE['metrics'](train.iloc[b][t].to_numpy()[mask],pred[t][mask]);candidate_scores[t].append(met['mae'])
                                inner_rows.append({'outer_fold':fold,'outer_held_room':room,'family':family,'candidate':cid,'inner_fold':inner,'inner_validation_room':held,'target':t,**met})
                        print(f'  {family}/{cid} inner {inner}/3: {time.perf_counter()-t0:.1f}s; total fits {sum(fit_counts.values())}',flush=True)
                        dump(out/'progress.json',{'status':'running','outer_fold':fold,'outer_held_room':room,'family':family,'candidate':cid,'inner_fold':inner,'fit_counts':fit_counts,'elapsed_seconds':time.perf_counter()-started})
                    for t in TARGETS:
                        assert len(candidate_scores[t])==6
                        row={'outer_fold':fold,'outer_held_room':room,'family':family,'candidate':cid,'target':t,'inner_mean_room_mae':float(np.mean(candidate_scores[t]))}
                        scores.append(row);family_scores.append(row);all_fold_scores.append(row)
                # Selection uses inner scores only; outer scoring labels are not read here.
                selected={t:min((r for r in family_scores if r['target']==t),key=lambda r:r['inner_mean_room_mae'])['candidate'] for t in TARGETS}
                needed={(selected['E_mean_lux'],'mean_log'),(selected['E_P10_lux'],'mean_log'),(selected['E_P10_lux'],'ratio')}
                needed.update((selected[t],t) for t in PRIMITIVES[2:])
                fitted={};yf=training_targets(train)
                for cid,primitive in sorted(needed):
                    est=make_estimator(family,params_for(cfg,family,cid));est.fit(xt,yf[primitive]);fit_counts['outer']+=1;fitted[cid+'__'+primitive]=est
                bundle={'family':family,'outer_fold':fold,'held_room':room,'train_rooms':fit_rooms,'selected_candidates':selected,'estimators':fitted,'raw_features':cols,'encoded_features':list(x.columns),'target_handling':cfg['target_handling'],'versions':versions}
                md=models/family;md.mkdir(exist_ok=True);joblib.dump(bundle,md/f'fold_{fold:02d}_{room}.joblib',compress=3)
                pred=predict_bundle(bundle,xv)
                for t in TARGETS:
                    assert np.isfinite(pred[t]).all();predictions[family].loc[vi,'pred__'+t]=pred[t]
                    selections.append({'outer_fold':fold,'held_room':room,'method':family,'target':t,'selected_family':family,'selected_candidate':selected[t],
                                       'inner_mean_room_mae':next(r['inner_mean_room_mae'] for r in family_scores if r['target']==t and r['candidate']==selected[t])})
                # Retain pre-clipping violations of every fitted primitive on outer X.
                for key,est in fitted.items():
                    raw=est.predict(xv);primitive=key.split('__',1)[1]
                    physical=np.expm1(raw) if primitive=='mean_log' else raw
                    violation=(physical<0)|((physical>1) if primitive in ['F_300_fraction','F_UDI_fraction'] else False)
                    raw_checks.append({'outer_fold':fold,'room':room,'family':family,'component':key,'clipped_or_negative_fraction':float(violation.mean())})
                pd.DataFrame(inner_rows).to_csv(out/'inner_room_metrics.partial.csv',index=False)
                pd.DataFrame(scores).to_csv(out/'inner_candidate_scores.partial.csv',index=False)
            for t in TARGETS:
                winner=min((r for r in all_fold_scores if r['target']==t),key=lambda r:r['inner_mean_room_mae'])
                family=winner['family'];predictions['Nested_select'].loc[vi,'pred__'+t]=predictions[family].loc[vi,'pred__'+t]
                selections.append({'outer_fold':fold,'held_room':room,'method':'Nested_select','target':t,'selected_family':family,'selected_candidate':winner['candidate'],'inner_mean_room_mae':winner['inner_mean_room_mae']})
            # All choices are frozen before calculating any outer metric.
            for method,p in predictions.items():
                p.loc[vi,'fold']=fold
                for t in TARGETS:
                    met=BASE['metrics'](d.iloc[vi][t].to_numpy(),p.loc[vi,'pred__'+t].to_numpy())
                    fold_metrics.append({'method':method,'target':t,'outer_fold':fold,'room':room,**met})
            pd.DataFrame(fold_metrics).to_csv(out/'fold_metrics.partial.csv',index=False)
            pd.DataFrame(selections).to_csv(out/'selections.partial.csv',index=False)
            for m,p in predictions.items():p.to_parquet(out/'predictions'/f'{m}.partial.parquet',index=False,compression='zstd')
            print(f'OUTER {fold}/7 COMPLETE; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    assert fit_counts['inner']==630 and len(fold_metrics)==175 and len(inner_rows)==1260 and len(scores)==210
    for method,p in predictions.items():
        assert p.fold.gt(0).all() and not p.filter(regex='^(true|pred)__').isna().any().any()
        p.to_parquet(out/'predictions'/f'{method}.parquet',index=False,compression='zstd')
    for name,rows in [('fold_metrics',fold_metrics),('inner_room_metrics',inner_rows),('inner_candidate_scores',scores),('selections',selections),('raw_prediction_checks',raw_checks)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    fm=pd.DataFrame(fold_metrics)
    fm.groupby(['method','target'],sort=False).agg(room_count=('room','nunique'),mae_mean=('mae','mean'),mae_median=('mae','median'),mae_max=('mae','max'),rmse_mean=('rmse','mean'),r2_mean=('r2','mean'),r2_median=('r2','median'),r2_min=('r2','min')).reset_index().to_csv(out/'summary_metrics.csv',index=False)
    assert digest(source)==source_hash
    assert all(digest(ROOT/p)==h for p,h in reference_hashes.items())
    run.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,fit_counts=fit_counts,outer_bundle_count=21,source_unchanged=True,reference_files_unchanged=True)
    dump(out/'run_manifest.json',run);dump(out/'progress.json',{'status':'completed','fit_counts':fit_counts,'elapsed_seconds':run['elapsed_seconds']})
    print(f'COMPLETED: {fit_counts}; 21 outer bundles; 5 OOF files; TEST untouched.',flush=True)

if __name__=='__main__':main()
