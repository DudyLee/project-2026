"""Approved TRAIN-only 2x2 feature/target ablation with grouped nested tuning."""
from __future__ import annotations
import copy
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
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/experiment_v5.json'
V3 = runpy.run_path(str(ROOT/'scripts/train_nested_v3.py'), run_name='v5_v3_helpers')
BASE = V3['BASE']
TARGETS = ['E_mean_lux', 'E_P10_lux']
PRIMITIVES = ['mean_log', 'ratio', 'asinh']
META = V3['META']
KEYS = V3['KEYS']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()
def dump(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def load_development(cfg):
    plan = copy.deepcopy(read(ROOT/'configs/model_plan.json'))
    plan['targets'] = TARGETS
    plan['feature_sets'] = {'base': plan['feature_sets'][cfg['base_feature_set']],
                            'required_for_derivation': ['Window_Count', 'Window_Area_Nominal_m2']}
    return BASE['load_train'](ROOT/cfg['source'], plan)

def features(frame, cfg):
    cols = read(ROOT/'configs/model_plan.json')['feature_sets'][cfg['base_feature_set']]
    x = BASE['encode_inputs'](frame, cols)
    assert (frame.Room_Area_m2 > 0).all() and (frame.Window_Count > 0).all()
    enhanced = x.copy()
    enhanced['Glass_to_floor_fraction'] = frame.Glass_Area_m2/frame.Room_Area_m2
    enhanced['Window_Count'] = frame.Window_Count.astype(float)
    enhanced['Mean_nominal_window_area_m2'] = frame.Window_Area_Nominal_m2/frame.Window_Count
    assert x.shape[1] == 15 and enhanced.shape[1] == 18
    assert np.isfinite(enhanced).all().all()
    assert not set(enhanced.columns).intersection(TARGETS + ['Room_ID', 'Sensor_Count'])
    return {'base': x, 'enhanced': enhanced}

def transformed_targets(frame):
    return {'mean_log': np.log1p(frame.E_mean_lux.to_numpy()),
            'ratio': frame.E_P10_lux.to_numpy()/(frame.E_mean_lux.to_numpy()+1),
            'asinh': np.arcsinh(frame.E_P10_lux.to_numpy())}

def reconstruct(raw):
    mean = np.maximum(np.expm1(raw['mean_log']), 0)
    return {'mean': mean, 'relative': np.maximum(raw['ratio'], 0)*(mean+1),
            'asinh': np.maximum(np.sinh(raw['asinh']), 0)}

def extra_metrics(y, pred):
    return BASE['metrics'](y, pred) | {'asinh_mae': float(np.abs(np.arcsinh(pred)-np.arcsinh(y)).mean())}

def predict_bundle(bundle, x, formulation):
    """Inference consumes only encoded X; no true mean or target input."""
    sel = bundle['selected_candidates']
    def raw(cid, component): return bundle['estimators'][cid+'__'+component].predict(x)
    mean = np.maximum(np.expm1(raw(sel['mean'], 'mean_log')), 0)
    cid = sel[formulation]
    if formulation == 'relative':
        p10 = np.maximum(raw(cid, 'ratio'), 0)*(np.maximum(np.expm1(raw(cid, 'mean_log')), 0)+1)
    else:
        p10 = np.maximum(np.sinh(raw(cid, 'asinh')), 0)
    return {'E_mean_lux': mean, 'E_P10_lux': p10}

def main():
    start = time.perf_counter(); cfg = read(CONFIG); params_cfg = read(ROOT/cfg['parameter_source'])
    out = ROOT/cfg['output']; modeldir = ROOT/cfg['model_output']
    for p in [out, out/'predictions', out/'inner_predictions', out/'figures', modeldir]: p.mkdir(parents=True, exist_ok=True)
    if (out/'run_manifest.json').exists(): raise RuntimeError('V5 already started; do not overwrite experiment evidence.')
    assert Path(sys.prefix).resolve() == (ROOT/'.conda/envs/pvsd-ml-v1').resolve()
    inputs = [CONFIG, ROOT/cfg['parameter_source'], ROOT/'configs/model_plan.json', ROOT/cfg['source'],
              ROOT/'scripts/train_logo_v1.py', ROOT/'scripts/train_nested_v3.py']
    for family in cfg['families'] + ['Frozen_HistGB']:
        inputs.append(ROOT/'reports/experiment_v3/predictions'/f'{family}.parquet')
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    assert hashes[cfg['source']] == read(ROOT/'reports/experiment_v1/run_manifest.json')['hashes']['source_sha256']
    d, skipped = load_development(cfg); xs = features(d, cfg)
    assert set(d.Room_ID).isdisjoint(cfg['protected_test_rooms'])
    versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib', 'pyarrow']}
    run = {'status': 'running', 'experiment_id': cfg['experiment_id'], 'started_utc': datetime.now(timezone.utc).isoformat(),
           'pid': os.getpid(), 'input_hashes': hashes, 'script_sha256': sha(__file__), 'versions': versions,
           'conda_prefix': sys.prefix, 'python': sys.version, 'train_rows': len(d), 'train_rooms': sorted(d.Room_ID.unique().tolist()),
           'test_rows_skipped': skipped, 'test_predictions': 0, 'test_scores': 0, 'scope': cfg['scope']}
    dump(out/'run_manifest.json', run); dump(out/'config_snapshot.json', cfg)
    dump(out/'feature_names.json', {k: list(v.columns) for k, v in xs.items()})
    methods = [p+'_'+family for family in cfg['families'] for p in cfg['profiles']]
    oofs = {}
    for method in methods:
        p = d[META].copy(); p['fold'] = -1
        for t in TARGETS: p['true__'+t] = d[t]; p['pred__'+t] = np.nan
        oofs[method] = p
    inner_records=[]; score_records=[]; selections=[]; splits=[]; fit_log=[]; index=[]; support=[]
    fit_counts={'inner':0, 'outer':0}; comparisons=[]
    with threadpool_limits(limits=cfg['thread_limit']):
        for fold,(ti,vi) in enumerate(LeaveOneGroupOut().split(d,groups=d.Room_ID),1):
            train=d.iloc[ti].reset_index(drop=True); held=d.iloc[vi].Room_ID.unique().item()
            trainrooms=sorted(train.Room_ID.unique().tolist()); groupsplits=list(GroupKFold(3).split(train,groups=train.Room_ID))
            split={'fold':fold,'held_room':held,'training_rooms':trainrooms,'training_rows':len(ti),'scoring_rows':len(vi),'inner':[]}
            for inner,(a,b) in enumerate(groupsplits,1):
                fr=sorted(train.iloc[a].Room_ID.unique().tolist()); vr=sorted(train.iloc[b].Room_ID.unique().tolist())
                assert len(fr)==4 and len(vr)==2 and not set(fr)&set(vr) and held not in fr+vr
                split['inner'].append({'inner':inner,'fitting_rooms':fr,'validation_rooms':vr,'fitting_rows':len(a),'validation_rows':len(b)})
            splits.append(split);dump(out/'fold_splits.json',splits)
            print(f'OUTER {fold}/7: held {held}, fitting {trainrooms}',flush=True)
            for fs,x in xs.items():
                xt=x.iloc[ti].reset_index(drop=True);xv=x.iloc[vi]
                static=['Room_Area_m2','Glass_Area_m2','Orientation_X_model','Orientation_Y_model','Solar_Obstruction_Ratio']
                if fs=='enhanced':static+=list(cfg['added_features'])
                for col in static:
                    lo=float(xt[col].min());hi=float(xt[col].max());v=float(xv[col].iloc[0])
                    assert xv[col].nunique()==1
                    support.append({'fold':fold,'room':held,'feature_set':fs,'feature':col,'fitting_min':lo,'fitting_max':hi,'held_value':v,
                                    'outside_fitting_range':v<lo or v>hi,'fitting_unique_values':int(xt[col].nunique())})
                for family in cfg['families']:
                    local_scores=[];ip=train[META].copy();ip['inner_fold']=-1
                    for t in TARGETS:ip['true__'+t]=train[t]
                    for candidate in params_cfg['families'][family]['candidates']:
                        cid=candidate['id'];params=V3['params_for'](params_cfg,family,cid)
                        for output in ['mean','relative','asinh']:ip[cid+'__'+output]=np.nan
                        candidate_room_scores={k:[] for k in ['mean','relative','asinh']}
                        for inner,(a,b) in enumerate(groupsplits,1):
                            t0=time.perf_counter();ys=transformed_targets(train.iloc[a]);raw={};ip.loc[b,'inner_fold']=inner
                            for primitive in PRIMITIVES:
                                est=V3['make_estimator'](family,params);est.fit(xt.iloc[a],ys[primitive]);raw[primitive]=est.predict(xt.iloc[b]);fit_counts['inner']+=1
                                fit_log.append({'stage':'inner','outer_fold':fold,'held_room':held,'feature_set':fs,'family':family,'candidate':cid,'primitive':primitive,
                                                'inner_fold':inner,'fitting_rooms':split['inner'][inner-1]['fitting_rooms'],'n_fit':len(a),
                                                'training_target_sha256':array_sha(ys[primitive]),'encoded_feature_count':xt.shape[1]})
                            preds=reconstruct(raw)
                            for output,pred in preds.items():
                                assert np.isfinite(pred).all();ip.loc[b,cid+'__'+output]=pred
                                target='E_mean_lux' if output=='mean' else 'E_P10_lux'
                                for room in sorted(train.iloc[b].Room_ID.unique()):
                                    mask=train.iloc[b].Room_ID.to_numpy()==room
                                    met=extra_metrics(train.iloc[b][target].to_numpy()[mask],pred[mask])
                                    inner_records.append({'outer_fold':fold,'outer_held_room':held,'feature_set':fs,'family':family,'candidate':cid,
                                                          'output':output,'target':target,'inner_fold':inner,'validation_room':room,**met})
                                    candidate_room_scores[output].append(met['mae'])
                            print(f'  {fs}/{family}/{cid}, inner {inner}/3: {time.perf_counter()-t0:.1f}s; fits {fit_counts}',flush=True)
                            dump(out/'progress.json',{'status':'running','outer_fold':fold,'held_room':held,'feature_set':fs,'family':family,'candidate':cid,
                                                      'inner_fold':inner,'fit_counts':fit_counts,'elapsed_seconds':time.perf_counter()-start})
                        for output,values in candidate_room_scores.items():
                            assert len(values)==6
                            row={'outer_fold':fold,'held_room':held,'feature_set':fs,'family':family,'candidate':cid,'output':output,'inner_mean_room_mae':float(np.mean(values))}
                            local_scores.append(row);score_records.append(row)
                    assert ip.inner_fold.ge(1).all() and not ip.isna().any().any()
                    ip.to_parquet(out/'inner_predictions'/f'fold_{fold:02d}_{held}__{fs}__{family}.parquet',index=False,compression='zstd')
                    selected={output:min((r for r in local_scores if r['output']==output),key=lambda r:r['inner_mean_room_mae'])['candidate'] for output in ['mean','relative','asinh']}
                    # All choices above use inner held-room scores only, before outer scoring.
                    needed={(selected['mean'],'mean_log'),(selected['relative'],'mean_log'),(selected['relative'],'ratio'),(selected['asinh'],'asinh')}
                    fitted={};ys=transformed_targets(train);model_evidence=[]
                    for cid,primitive in sorted(needed):
                        est=V3['make_estimator'](family,V3['params_for'](params_cfg,family,cid));est.fit(xt,ys[primitive]);fit_counts['outer']+=1
                        fitted[cid+'__'+primitive]=est
                        evidence={'stage':'outer','outer_fold':fold,'held_room':held,'feature_set':fs,'family':family,'candidate':cid,'primitive':primitive,
                                  'fitting_rooms':trainrooms,'n_fit':len(train),'training_target_sha256':array_sha(ys[primitive]),'encoded_feature_count':xt.shape[1]}
                        fit_log.append(evidence);model_evidence.append(evidence)
                    bundle={'family':family,'feature_set':fs,'fold':fold,'held_room':held,'train_rooms':trainrooms,'selected_candidates':selected,
                            'estimators':fitted,'encoded_features':list(xt.columns),'target_handling':cfg['transforms'],'versions':versions,'fit_evidence':model_evidence}
                    path=modeldir/f'{fs}__{family}__fold_{fold:02d}_{held}.joblib';joblib.dump(bundle,path,compress=3)
                    index.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'fold':fold,'room':held,'family':family,'feature_set':fs,'estimators':len(fitted)})
                    for output,cid in selected.items():
                        selections.append({'outer_fold':fold,'held_room':held,'feature_set':fs,'family':family,'output':output,'candidate':cid,
                                           'inner_mean_room_mae':next(r['inner_mean_room_mae'] for r in local_scores if r['output']==output and r['candidate']==cid)})
                    for profile,spec in cfg['profiles'].items():
                        if spec['features']!=fs:continue
                        method=profile+'_'+family;preds=predict_bundle(bundle,xv,spec['P10'])
                        for t,pred in preds.items():
                            assert np.isfinite(pred).all() and (pred>=0).all();oofs[method].loc[vi,'pred__'+t]=pred
                        oofs[method].loc[vi,'fold']=fold
            for name,records in [('inner_room_metrics',inner_records),('inner_candidate_scores',score_records),('selections',selections)]:
                pd.DataFrame(records).to_csv(out/f'{name}.partial.csv',index=False)
            dump(out/'fit_log.partial.json',fit_log);dump(out/'model_index.json',index)
            print(f'OUTER {fold} COMPLETE; elapsed {time.perf_counter()-start:.1f}s',flush=True)
    assert fit_counts['inner']==504 and fit_counts['outer']<=112 and len(index)==28
    # External scoring is deliberately postponed until every model/parameter is frozen.
    frozen=pd.read_parquet(ROOT/'reports/experiment_v3/predictions/Frozen_HistGB.parquet')
    assert frozen[KEYS].equals(d[KEYS])
    oofs['Frozen_HistGB']=frozen[META+['fold']+sum(([f'true__{t}',f'pred__{t}'] for t in TARGETS),[])].copy()
    for family in cfg['families']:
        old=pd.read_parquet(ROOT/'reports/experiment_v3/predictions'/f'{family}.parquet')
        assert old[KEYS].equals(oofs['A_'+family][KEYS])
        for target in TARGETS:
            delta=np.max(np.abs(old['pred__'+target]-oofs['A_'+family]['pred__'+target]))
            assert delta<1e-8,(family,target,delta)
            comparisons.append({'family':family,'target':target,'max_absolute_prediction_difference_from_v3':float(delta)})
    metrics=[]
    for method,p in oofs.items():
        assert not p.isna().any().any() and p.fold.ge(1).all()
        assert set(p.Room_ID).isdisjoint(cfg['protected_test_rooms'])
        for room,g in p.groupby('Room_ID',sort=True):
            for t in TARGETS:metrics.append({'method':method,'room':room,'target':t,**extra_metrics(g['true__'+t].to_numpy(),g['pred__'+t].to_numpy())})
        p.to_parquet(out/'predictions'/f'{method}.parquet',index=False,compression='zstd')
    for name,records in [('fold_metrics',metrics),('inner_room_metrics',inner_records),('inner_candidate_scores',score_records),('selections',selections),('feature_support',support)]:
        pd.DataFrame(records).to_csv(out/f'{name}.csv',index=False)
    dump(out/'fit_log.json',fit_log);dump(out/'baseline_reproduction.json',comparisons)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    run.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
               fit_counts=fit_counts,outer_bundle_count=len(index),input_files_unchanged=True)
    dump(out/'run_manifest.json',run);dump(out/'progress.json',{'status':'completed','fit_counts':fit_counts,'elapsed_seconds':run['elapsed_seconds']})
    print(f'COMPLETED: {fit_counts}; 28 outer bundles, 9 OOF files; TEST predictions/scores = 0.',flush=True)

if __name__=='__main__': main()
