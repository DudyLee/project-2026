"""Execute frozen v8 cells with TRAIN-only, grouped nested validation."""
from __future__ import annotations
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import directional_v8 as D
import train_target_specific_v7 as V7

ROOT = D.ROOT


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return D.base.sha256(Path(p))
def idx(d, rooms): return d.index[d.Room_ID.isin(rooms)].to_numpy()
def entries(cell):
    return [(cell, t, False) for t in D.TARGETS] + ([('S1_direct', 'E_P10_lux', True)] if cell == 'S1' else [])
def col(method, target, lam=None):
    return method+'__'+target+('::'+str(lam) if lam is not None else '')


def evaluate(d, cfg, out, protocols):
    records, strata, components = [], [], []
    lookup = d.set_index(D.KEYS)
    for protocol in protocols:
        pred = pd.read_parquet(out/'outer_predictions'/(protocol['job_id']+'.parquet'))
        weather = lookup.loc[pd.MultiIndex.from_frame(pred[D.KEYS]), 'EPW_DNI_Wh_m2'].to_numpy()
        pred['DNI_bin'] = pd.cut(weather, cfg['DNI_edges'], labels=['0','(0,200]','(200,500]','>500']).astype(str)
        for room, g in pred.groupby('Room_ID', sort=True):
            for method, spec in cfg['methods'].items():
                for target in spec['targets']:
                    suffix = col(method, target)
                    cs = {k:g[k+'__'+suffix].to_numpy() for k in ['paired_mean','ratio','component_clipped'] if k+'__'+suffix in g}
                    meta = {'protocol':protocol['protocol'], 'fold':protocol['fold'], 'job_id':protocol['job_id'], 'room':room, 'method':method, 'target':target}
                    records.append(meta | D.metrics(g['true__'+target],g['pred__'+suffix],g['raw__'+suffix],cfg['tolerances_descriptive'][target],target,cs))
                    if cs:
                        components.append(meta | {'paired_mean_MAE':float(np.abs(cs['paired_mean']-g.true__E_mean_lux).mean()),
                            'ratio_MAE':float(np.abs(cs['ratio']-g.true__E_P10_lux/(g.true__E_mean_lux+1)).mean()),
                            'ratio_true_mean':float((g.true__E_P10_lux/(g.true__E_mean_lux+1)).mean()),
                            'ratio_pred_mean':float(cs['ratio'].mean())})
                    for dimension in ['Hour','PVSD_Angle_deg','DNI_bin']:
                        for value, h in g.groupby(dimension, sort=True, observed=True):
                            strata.append(meta | {'dimension':dimension,'value':str(value)} | D.base.metrics(h['true__'+target],h['pred__'+suffix]) | {'n':len(h)})
    fm = pd.DataFrame(records)
    assert len(fm) == 182
    fm.to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(out/'stratified_metrics.csv',index=False)
    pd.DataFrame(components).to_csv(out/'P10_components.csv',index=False)
    summary, gates, paired, decisions = D.summarize(fm,cfg)
    summary.to_csv(out/'summary_metrics.csv',index=False)
    gates.to_csv(out/'development_gate.csv',index=False)
    paired.to_csv(out/'paired_room_comparison.csv',index=False)
    D.dump(out/'candidate_decision.json',{'targets':decisions,'full_seven_refit':False,'deployed_model_changed':False,
        'TEST_predictions':0,'scope':'Repeated-use development evidence; the previously observed holdout motivated the plan but supplies no v8 fitting, scoring or selection rows.'})


def main():
    started = time.perf_counter()
    cfg = D.read(D.CONFIG)
    out, modeldir = ROOT/cfg['output'], ROOT/cfg['model_output']
    for folder in [out, modeldir, out/'inner_predictions', out/'outer_predictions', out/'selection_freezes']:
        folder.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists(): raise RuntimeError('v8 already started: refusing overwrite')
    assert Path(sys.prefix).resolve() == (ROOT/'.conda/envs/pvsd-ml-v1').resolve()
    protected = [p for p in (ROOT/'models/trial_v0_1').rglob('*') if p.is_file()]
    frozen = [D.CONFIG, ROOT/cfg['source'], ROOT/cfg['parameter_source'], ROOT/'configs/model_plan.json',
        ROOT/'docs/experiment_v8_protocol.md',ROOT/'docs/approvals/directional_v8_approval.json',
        Path(__file__), ROOT/'scripts/directional_v8.py', ROOT/'scripts/train_logo_v1.py',
        ROOT/'scripts/train_target_specific_v7.py', ROOT/'scripts/train_cross_room_v5.py',ROOT/'scripts/train_nested_v3.py',ROOT/'pvsd_trial.py'] + protected
    hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in frozen}
    assert hashes[cfg['source']] == cfg['source_sha256']
    d, skipped = D.base.load_train(ROOT/cfg['source'],D.read(ROOT/'configs/model_plan.json'))
    assert sorted(d.Room_ID.unique()) == sorted(cfg['development_rooms']) and skipped == 52560
    directions = {'posX':(1,0),'negY':(0,-1),'posY':(0,1)}
    for direction, rooms in cfg['orientation_groups'].items():
        assert set(map(tuple,d.loc[d.Room_ID.isin(rooms),['Orientation_X_model','Orientation_Y_model']].drop_duplicates().to_numpy())) == {directions[direction]}
    xx, yy = D.features(d), D.labels(d)
    jobs, protocols = D.splits(d,cfg)
    tree_specs = D.read(ROOT/cfg['parameter_source'])['models']
    D.dump(out/'config_snapshot.json',cfg)
    D.dump(out/'tree_specs.json',{t:tree_specs[t] for t in D.TARGETS})
    D.dump(out/'fold_splits.json',{'jobs':jobs,'protocols':protocols})
    D.dump(out/'feature_names.json',{k:list(x.columns) for k,x in xx.items()})
    run = {'experiment_id':cfg['experiment_id'],'status':'running','started_utc':now(),'pid':os.getpid(),
        'python':sys.version,'conda_prefix':sys.prefix,'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','pyarrow']},
        'input_hashes':hashes,'train_rows':len(d),'train_rooms':sorted(d.Room_ID.unique()),
        'test_rows_skipped_before_numeric_conversion':skipped,'TEST_fit_rows':0,'TEST_predictions':0,'TEST_scores':0,
        'new_simulations':0,'visualizations':0,'full_seven_refit':False,'unique_outer_jobs':len(jobs),'logical_outer_folds':len(protocols),'scope':cfg['scope']}
    D.dump(out/'run_manifest.json',run)
    fitlog, preplog, inner_rooms, inner_scores, selections, model_index = [], [], [], [], [], []
    counts = {'inner_ridge':0,'outer_ridge':0,'outer_tree':0}
    print(f'Loaded {len(d)} TRAIN rows; skipped {skipped} TEST rows. 9 outer jobs, 378 inner Ridge fits.',flush=True)
    with threadpool_limits(limits=cfg['thread_limit']):
        for number, job in enumerate(jobs,1):
            jid=job['id']; ti,vi=idx(d,job['fit_rooms']),idx(d,job['held_rooms'])
            print(f"OUTER {number}/9 {jid}: fit {job['fit_rooms']}, hold {job['held_rooms']}",flush=True)
            ip=d.loc[ti,D.META].copy();ip['inner_fold']=-1
            for t in D.TARGETS:ip['true__'+t]=d.loc[ti,t]
            for cell in ['S0','S1']:
                x=xx[cfg['methods'][cell]['features']]
                for inner in job['inner']:
                    a,b=idx(d,inner['fit_rooms']),idx(d,inner['held_rooms'])
                    ip.loc[b,'inner_fold']=inner['fold']
                    prep=D.preprocessing(cfg);za=prep.fit_transform(x.loc[a]);zb=prep.transform(x.loc[b])
                    assert za.shape[1] == (39 if cell=='S0' else 45)
                    prepmeta={'job_id':jid,'phase':'inner','inner_fold':inner['fold'],'cell':cell,'fit_rooms':inner['fit_rooms']}
                    preplog.append(prepmeta | D.preprocess_diagnostics(prep,x.loc[a]))
                    zhash=D.array_sha(za)
                    primitive_names=['mean_log','ratio','udi']+(['p10_asinh'] if cell=='S1' else [])
                    for lam in cfg['ridge']['lambdas']:
                        values={}
                        for primitive in primitive_names:
                            tick=time.perf_counter()
                            est=Ridge(alpha=lam*len(a),solver=cfg['ridge']['solver'],fit_intercept=cfg['ridge']['fit_intercept'])
                            est.fit(za,yy[primitive][a]);values[primitive]=est.predict(zb)
                            counts['inner_ridge']+=1
                            fitlog.append(prepmeta | {'primitive':primitive,'lambda':lam,'alpha':lam*len(a),'fit_rows':len(a),
                                'input_sha256':zhash,'label_sha256':D.array_sha(yy[primitive][a]),'seconds':time.perf_counter()-tick})
                        for method,target,direct in entries(cell):
                            p,_,_=D.reconstruct(values,target,direct)
                            ip.loc[b,'pred__'+col(method,target,lam)]=p
                for method,target,direct in entries(cell):
                    for lam in cfg['ridge']['lambdas']:
                        local=[]
                        for room,g in ip.groupby('Room_ID',sort=True):
                            m=D.base.metrics(g['true__'+target],g['pred__'+col(method,target,lam)])
                            record={'job_id':jid,'cell':cell,'method':method,'target':target,'lambda':lam,'room':room,'inner_fold':int(g.inner_fold.iloc[0]),'n':len(g)} | m
                            inner_rooms.append(record);local.append(m['mae'])
                        inner_scores.append({'job_id':jid,'cell':cell,'method':method,'target':target,'lambda':lam,'mean_MAE':float(np.mean(local)),'worst_MAE':float(np.max(local)),'rooms':len(local)})
            assert ip.inner_fold.gt(0).all() and not ip.isna().any().any()
            ip.to_parquet(out/'inner_predictions'/(jid+'.parquet'),index=False)
            chosen={}
            for cell in ['S0','S1']:
                for method,target,direct in entries(cell):
                    options=[s for s in inner_scores if s['job_id']==jid and s['method']==method and s['target']==target]
                    best=min(options,key=lambda s:(s['mean_MAE'],-s['lambda']))
                    chosen[method+'::'+target]=best['lambda']
                    selections.append(best | {'selected_utc':now(),'outer_held_rooms':job['held_rooms']})
            D.dump(out/'selection_freezes'/(jid+'.json'),{'job_id':jid,'selected':chosen,'selected_utc':now(),'outer_predictions_computed':False,'outer_labels_used_for_selection':False})
            bundle={'job':job,'raw_inputs':D.api.RAW_FEATURES,'trees':{},'splines':{}}
            for method in ['T0','T1']:
                x=xx[cfg['methods'][method]['features']];bundle['trees'][method]={}
                for target in D.TARGETS:
                    tick=time.perf_counter()
                    fitted,labelhashes=V7.fit_target(tree_specs[target],x.loc[ti],d.loc[ti])
                    bundle['trees'][method][target]=fitted
                    for primitive,labelhash in labelhashes.items():
                        counts['outer_tree']+=1
                        fitlog.append({'job_id':jid,'phase':'outer_tree','cell':method,'target':target,'primitive':primitive,'fit_rooms':job['fit_rooms'],'fit_rows':len(ti),
                            'input_sha256':D.array_sha(x.loc[ti]),'label_sha256':labelhash,'seconds_for_target':time.perf_counter()-tick})
            for cell in ['S0','S1']:
                x=xx[cfg['methods'][cell]['features']]
                prep=D.preprocessing(cfg);z=prep.fit_transform(x.loc[ti]);zhash=D.array_sha(z)
                pm={'job_id':jid,'phase':'outer','inner_fold':None,'cell':cell,'fit_rooms':job['fit_rooms']}
                preplog.append(pm | D.preprocess_diagnostics(prep,x.loc[ti]))
                item={'preprocessor':prep,'selected':{},'estimators':{}}
                for method,target,direct in entries(cell):
                    lam=chosen[method+'::'+target];item['selected'][method+'::'+target]=lam
                    for primitive in D.primitives(target,direct):
                        key=str(lam)+'::'+primitive
                        if key in item['estimators']: continue
                        tick=time.perf_counter()
                        est=Ridge(alpha=lam*len(ti),solver=cfg['ridge']['solver'],fit_intercept=cfg['ridge']['fit_intercept'])
                        est.fit(z,yy[primitive][ti]);item['estimators'][key]=est;counts['outer_ridge']+=1
                        fitlog.append(pm | {'primitive':primitive,'lambda':lam,'alpha':lam*len(ti),'fit_rows':len(ti),
                            'input_sha256':zhash,'label_sha256':D.array_sha(yy[primitive][ti]),'seconds':time.perf_counter()-tick})
                bundle['splines'][cell]=item
            modelpath=modeldir/(jid+'.joblib');joblib.dump(bundle,modelpath,compress=3)
            # All model fits and lambda selection are frozen before outer outcomes are accessed for scoring.
            predictions=D.predict_bundle(bundle,d.loc[vi,D.api.RAW_FEATURES],cfg)
            op=d.loc[vi,D.META].copy()
            for target in D.TARGETS:op['true__'+target]=d.loc[vi,target]
            for (method,target),(p,raw,cs) in predictions.items():
                suffix=col(method,target);op['pred__'+suffix]=p;op['raw__'+suffix]=raw
                for key,value in cs.items():op[key+'__'+suffix]=value
            assert not op.isna().any().any()
            op.to_parquet(out/'outer_predictions'/(jid+'.parquet'),index=False)
            model_index.append({'job_id':jid,'path':modelpath.relative_to(ROOT).as_posix(),'sha256':sha(modelpath),'fit_rooms':job['fit_rooms'],'held_rooms':job['held_rooms']})
            for name,value in [('fit_log.json',fitlog),('preprocessing_audit.json',preplog),('selections.json',selections),('model_index.json',model_index)]:D.dump(out/name,value)
            pd.DataFrame(inner_rooms).to_csv(out/'inner_room_metrics.csv',index=False)
            pd.DataFrame(inner_scores).to_csv(out/'inner_candidate_scores.csv',index=False)
            run.update({'completed_outer_jobs':number,'fit_counts':counts.copy(),'elapsed_seconds':time.perf_counter()-started})
            D.dump(out/'run_manifest.json',run)
            print(f"FINISHED {jid}; fits {counts}; elapsed {time.perf_counter()-started:.1f}s",flush=True)
    assert counts['inner_ridge']==378 and counts['outer_tree']==72 and sum(counts.values())<=cfg['maximum_new_estimator_fits']
    assert len(preplog)==72 and len(inner_rooms)==700 and len(inner_scores)==126 and len(selections)==63
    evaluate(d,cfg,out,protocols)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    run.update({'status':'completed_pending_verification','completed_utc':now(),'elapsed_seconds':time.perf_counter()-started,
        'fit_counts':counts,'total_estimator_fits':sum(counts.values()),'preprocessor_fits':len(preplog),'protected_inputs_unchanged':True})
    D.dump(out/'run_manifest.json',run)
    print(f'COMPLETE {sum(counts.values())} estimator fits in {time.perf_counter()-started:.1f}s; verification pending.',flush=True)


if __name__=='__main__':main()
