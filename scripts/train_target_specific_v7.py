"""Bounded target-specific models with nested mean versus worst-room selection."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits
import train_cross_room_v5 as T

ROOT = T.ROOT
CONFIG = ROOT/'configs/experiment_v7.json'
TARGETS = T.V3['TARGETS']
META, KEYS, BASE = T.META, T.KEYS, T.BASE


def catalog(cfg):
    old = T.read(ROOT/cfg['parameter_source'])
    result = {}
    for target, designs in cfg['candidate_design'].items():
        result[target] = []
        for design in designs:
            for size in design['parameters']:
                params = T.V3['params_for'](old, design['family'], size)
                if design['loss'] == 'absolute':
                    if design['family'] == 'XGBoost': params['objective'] = 'reg:absoluteerror'
                    elif design['family'] == 'HistGB': params['loss'] = 'absolute_error'
                    else: raise ValueError('Unsupported declared loss')
                result[target].append({**design, 'id': design['prefix']+'__'+size,
                                       'params': params, 'target': target, 'order': len(result[target])})
    return result


def fit_target(spec, x, frame):
    y = frame[spec['target']].to_numpy()
    transform = spec['transform']
    if transform == 'relative':
        ys = {'mean_log': np.log1p(frame.E_mean_lux.to_numpy()),
              'ratio': y/(frame.E_mean_lux.to_numpy()+1)}
    else:
        ys = {'value': np.log1p(y) if transform == 'log1p' else np.arcsinh(y) if transform == 'asinh' else y}
    estimators = {}
    for component, labels in ys.items():
        estimator = T.V3['make_estimator'](spec['family'], spec['params'])
        estimator.fit(x, labels)
        estimators[component] = estimator
    return {'spec': spec, 'estimators': estimators}, {k:T.array_sha(v) for k,v in ys.items()}


def predict_target(fitted, x):
    transform = fitted['spec']['transform']
    est = fitted['estimators']
    if transform == 'relative':
        prediction = np.maximum(est['ratio'].predict(x), 0)*(np.maximum(np.expm1(est['mean_log'].predict(x)), 0)+1)
    else:
        raw = est['value'].predict(x)
        prediction = np.expm1(raw) if transform == 'log1p' else np.sinh(raw) if transform == 'asinh' else raw
    prediction = BASE['bound_prediction'](prediction, fitted['spec']['target'])
    assert np.isfinite(prediction).all()
    return np.asarray(prediction, dtype=np.float64)


def predict_bundle(bundle, inputs, policy):
    """Public inference accepts the 13 raw predictors only; no target or Room_ID required."""
    assert set(bundle['raw_inputs']).issubset(inputs.columns)
    x = BASE['encode_inputs'](inputs, bundle['raw_inputs'])
    assert list(x.columns) == bundle['encoded_inputs']
    return {target:predict_target(bundle['fitted'][target+'::'+cid], x)
            for target,cid in bundle['selected'][policy].items()}


def choose(scores, cfg):
    baseline = min((r for r in scores if r['baseline']), key=lambda r:(r['mean_mae'], r['order']))
    mean = min(scores, key=lambda r:(r['mean_mae'], r['order']))
    limit = baseline['mean_mae']*cfg['selection']['guard_mean_ratio']+cfg['selection']['selection_tolerance']
    eligible = [r for r in scores if r['mean_mae'] <= limit]
    worst = min(eligible, key=lambda r:(r['worst_mae'], r['mean_mae'], r['order']))
    return {'Target_baseline':baseline, 'Mean_select':mean, 'Worst_guarded':worst}


def metrics(y, p, tolerance):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    return BASE['metrics'](y,p) | {'within_tolerance':float((np.abs(y-p)<=tolerance).mean()),
        'absolute_tolerance':tolerance, 'true_mean':float(y.mean()), 'pred_mean':float(p.mean()), 'n':len(y)}


def evaluate(predictions, cfg, out):
    rows = []
    for method, frame in predictions.items():
        for room,g in frame.groupby('Room_ID',sort=True):
            for target in TARGETS:
                rows.append({'method':method,'room':room,'target':target,
                             **metrics(g['true__'+target],g['pred__'+target],cfg['tolerance_descriptive'][target])})
    fm = pd.DataFrame(rows)
    fm.to_csv(out/'fold_metrics.csv',index=False)
    summary = fm.groupby(['method','target'],sort=False).agg(
        room_count=('room','nunique'),mae_mean=('mae','mean'),mae_median=('mae','median'),mae_max=('mae','max'),
        rmse_mean=('rmse','mean'),bias_mean=('bias','mean'),r2_median=('r2','median'),r2_min=('r2','min'),
        within_tolerance_mean=('within_tolerance','mean'),within_tolerance_min=('within_tolerance','min')).reset_index()
    summary['worst_room'] = [fm[(fm.method==r.method)&(fm.target==r.target)].sort_values(['mae','room'],ascending=[False,True]).iloc[0].room for r in summary.itertuples()]
    summary.to_csv(out/'summary_metrics.csv',index=False)
    comparisons, gates = [], []
    rule = cfg['development_gate']; tol=rule['numerical_tolerance']
    for method in ['Mean_select','Worst_guarded']:
        for reference in rule['against']:
            for target in TARGETS:
                a=fm[(fm.method==reference)&(fm.target==target)].set_index('room')
                b=fm[(fm.method==method)&(fm.target==target)].set_index('room').loc[a.index]
                for room in a.index:
                    comparisons.append({'method':method,'reference':reference,'target':target,'room':room,
                        'reference_mae':a.loc[room,'mae'],'candidate_mae':b.loc[room,'mae'],
                        'mae_change':b.loc[room,'mae']-a.loc[room,'mae'],
                        'reference_bias':a.loc[room,'bias'],'candidate_bias':b.loc[room,'bias']})
                worst_reduction=1-b.mae.max()/a.mae.max();mean_change=b.mae.mean()/a.mae.mean()-1
                nonworse=int((b.mae<=a.mae+tol).sum())
                passed=bool(worst_reduction>=rule['minimum_worst_room_MAE_reduction_fraction']-tol and
                            b.mae.mean()<=a.mae.mean()+tol and nonworse>=rule['minimum_rooms_nonworse'])
                gates.append({'method':method,'reference':reference,'target':target,
                    'worst_MAE_reduction_fraction':float(worst_reduction),'macro_MAE_change_fraction':float(mean_change),
                    'rooms_improved':int((b.mae<a.mae-tol).sum()),'rooms_nonworse':nonworse,
                    'rooms_tied':int((np.abs(b.mae-a.mae)<=tol).sum()),'passes_gate':passed})
    pd.DataFrame(comparisons).to_csv(out/'paired_room_comparison.csv',index=False)
    pd.DataFrame(gates).to_csv(out/'development_gate.csv',index=False)
    decision={'primary_candidate':'Worst_guarded','rule':rule,'targets':{},'TEST_unsealed':False,'final_refit':False,
        'scope':'Development screening after repeated use of seven rooms. Passing does not establish independent generalization.'}
    for target in TARGETS:
        selected=[r for r in gates if r['method']=='Worst_guarded' and r['target']==target]
        passed=all(r['passes_gate'] for r in selected)
        decision['targets'][target]={'passes_both_references':passed,'comparisons':selected,
            'status':'eligible_for_further_validation_only' if passed else 'retain_existing_reference'}
    T.dump(out/'candidate_decision.json',decision)
    return summary,decision


def main():
    started=time.perf_counter();cfg=T.read(CONFIG);plan=T.read(ROOT/'configs/model_plan.json')
    out=ROOT/cfg['output'];models=ROOT/cfg['model_output']
    for p in [out,out/'predictions',out/'inner_predictions',out/'figures',models]:p.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():raise RuntimeError('V7 already started; refusing overwrite')
    assert Path(sys.prefix).resolve()==(ROOT/'.conda/envs/pvsd-ml-v1').resolve()
    references=sorted(set(cfg['retained_reference'].values())|set(cfg['target_baseline_reference'].values()))
    files=[CONFIG,ROOT/cfg['source'],ROOT/cfg['parameter_source'],ROOT/'configs/model_plan.json',
           ROOT/'docs/experiment_v7_protocol.md',ROOT/'scripts/train_logo_v1.py',ROOT/'scripts/train_nested_v3.py',ROOT/'scripts/train_cross_room_v5.py']
    files += [ROOT/f'reports/experiment_v3/predictions/{f}.parquet' for f in references]
    hashes={p.relative_to(ROOT).as_posix():T.sha(p) for p in files}
    assert hashes[cfg['source']]==T.read(ROOT/'reports/experiment_v1/run_manifest.json')['hashes']['source_sha256']
    d,skipped=BASE['load_train'](ROOT/cfg['source'],plan)
    cols=plan['feature_sets'][cfg['feature_set']];x=BASE['encode_inputs'](d,cols)
    assert len(cols)==13 and x.shape[1]==15 and not set(d.Room_ID)&set(cfg['protected_test_rooms'])
    specs=catalog(cfg);T.dump(out/'candidate_catalog.json',specs);T.dump(out/'config_snapshot.json',cfg)
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','xgboost','joblib','pyarrow']}
    run={'experiment_id':cfg['experiment_id'],'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
         'pid':os.getpid(),'input_hashes':hashes,'script_sha256':T.sha(__file__),'versions':versions,
         'python':sys.version,'conda_prefix':sys.prefix,'train_rows':len(d),'train_rooms':sorted(d.Room_ID.unique()),
         'test_rows_skipped_before_numeric_conversion':skipped,'TEST_predictions':0,'TEST_scores':0,
         'scope':cfg['scope'],'planned_inner_estimator_fits':cfg['planned_inner_estimator_fits']}
    T.dump(out/'run_manifest.json',run);T.dump(out/'feature_names.json',{'raw':cols,'encoded':list(x.columns)})
    oofs={policy:T.V3['empty_oof'](d) for policy in cfg['policies']}
    splits=[];inner_records=[];score_records=[];selections=[];fit_log=[];index=[]
    fit_counts={'inner':0,'outer':0}
    with threadpool_limits(limits=cfg['thread_limit']):
        for fold,(ti,vi) in enumerate(LeaveOneGroupOut().split(d,groups=d.Room_ID),1):
            held=d.iloc[vi].Room_ID.unique().item();train=d.iloc[ti].reset_index(drop=True)
            xt=x.iloc[ti].reset_index(drop=True);xv=x.iloc[vi]
            groups=list(GroupKFold(3).split(train,groups=train.Room_ID))
            split={'fold':fold,'held_room':held,'fitting_rooms':sorted(train.Room_ID.unique()),'n_fit':len(ti),'n_score':len(vi),'inner':[]}
            for inner,(a,b) in enumerate(groups,1):
                fr=sorted(train.iloc[a].Room_ID.unique());vr=sorted(train.iloc[b].Room_ID.unique())
                assert len(fr)==4 and len(vr)==2 and not set(fr)&set(vr) and held not in fr+vr
                split['inner'].append({'inner_fold':inner,'fitting_rooms':fr,'validation_rooms':vr,'n_fit':len(a),'n_validation':len(b)})
            splits.append(split);T.dump(out/'fold_splits.json',splits)
            ip=train[META].copy();ip['inner_fold']=-1
            for target in TARGETS:ip['true__'+target]=train[target]
            selected={policy:{} for policy in cfg['policies']};fitted={};evidence=[]
            print(f'OUTER {fold}/7 held {held}',flush=True)
            for target in TARGETS:
                scores=[]
                for spec in specs[target]:
                    key=target+'::'+spec['id'];ip[key]=np.nan;room_errors=[]
                    for inner,(a,b) in enumerate(groups,1):
                        t0=time.perf_counter();model,label_hashes=fit_target(spec,xt.iloc[a],train.iloc[a]);p=predict_target(model,xt.iloc[b])
                        fit_counts['inner']+=len(label_hashes)
                        ip.loc[b,key]=p;ip.loc[b,'inner_fold']=inner
                        fit_log.append({'stage':'inner','outer_fold':fold,'held_room':held,'target':target,'candidate':spec['id'],
                            'inner_fold':inner,'fitting_rooms':split['inner'][inner-1]['fitting_rooms'],'n_fit':len(a),
                            'encoded_input_sha256':T.array_sha(xt.iloc[a].to_numpy()),'target_hashes':label_hashes,'estimator_count':len(label_hashes)})
                        for room in split['inner'][inner-1]['validation_rooms']:
                            mask=train.iloc[b].Room_ID.to_numpy()==room
                            met=metrics(train.iloc[b][target].to_numpy()[mask],p[mask],cfg['tolerance_descriptive'][target])
                            room_errors.append(met['mae'])
                            inner_records.append({'outer_fold':fold,'held_room':held,'inner_fold':inner,'room':room,'target':target,'candidate':spec['id'],**met})
                        del model
                        T.dump(out/'progress.json',{'status':'running','outer_fold':fold,'held_room':held,'target':target,'candidate':spec['id'],
                            'inner_fold':inner,'fit_counts':fit_counts,'elapsed_seconds':time.perf_counter()-started})
                    assert len(room_errors)==6
                    score={'outer_fold':fold,'held_room':held,'target':target,'candidate':spec['id'],'order':spec['order'],
                           'baseline':spec['baseline'],'mean_mae':float(np.mean(room_errors)),'worst_mae':float(np.max(room_errors))}
                    scores.append(score);score_records.append(score)
                    print(f'  {target}/{spec["id"]}: 3 inner folds completed; fits {fit_counts}',flush=True)
                chosen=choose(scores,cfg)
                for policy,row in chosen.items():
                    selected[policy][target]=row['candidate']
                    selections.append({'outer_fold':fold,'held_room':held,'target':target,'policy':policy,'candidate':row['candidate'],
                        'inner_mean_mae':row['mean_mae'],'inner_worst_mae':row['worst_mae'],
                        'guard_limit':chosen['Target_baseline']['mean_mae']*cfg['selection']['guard_mean_ratio']})
                for cid in sorted({r['candidate'] for r in chosen.values()}):
                    spec=next(s for s in specs[target] if s['id']==cid)
                    model,label_hashes=fit_target(spec,xt,train);key=target+'::'+cid
                    fitted[key]=model;fit_counts['outer']+=len(label_hashes)
                    ev={'stage':'outer','outer_fold':fold,'held_room':held,'target':target,'candidate':cid,
                        'fitting_rooms':split['fitting_rooms'],'n_fit':len(train),'encoded_input_sha256':T.array_sha(xt.to_numpy()),
                        'target_hashes':label_hashes,'estimator_count':len(label_hashes)}
                    fit_log.append(ev);evidence.append(ev)
                for policy,row in chosen.items():
                    oofs[policy].loc[vi,'pred__'+target]=predict_target(fitted[target+'::'+row['candidate']],xv)
                    oofs[policy].loc[vi,'fold']=fold
            assert not ip.isna().any().any() and ip.inner_fold.ge(1).all()
            ip.to_parquet(out/'inner_predictions'/f'fold_{fold:02d}_{held}.parquet',index=False,compression='zstd')
            bundle={'fold':fold,'held_room':held,'fitting_rooms':split['fitting_rooms'],'raw_inputs':cols,'encoded_inputs':list(x.columns),
                    'selected':selected,'fitted':fitted,'fit_evidence':evidence,'versions':versions}
            path=models/f'fold_{fold:02d}_{held}.joblib';joblib.dump(bundle,path,compress=3)
            index.append({'fold':fold,'room':held,'path':path.relative_to(ROOT).as_posix(),'sha256':T.sha(path),'candidate_models':len(fitted),
                          'estimators':sum(len(m['estimators']) for m in fitted.values())})
            for name,rows in [('inner_room_metrics',inner_records),('inner_candidate_scores',score_records),('selections',selections)]:
                pd.DataFrame(rows).to_csv(out/f'{name}.partial.csv',index=False)
            T.dump(out/'model_index.json',index);T.dump(out/'fit_log.partial.json',fit_log)
            print(f'OUTER {fold} complete; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    assert fit_counts['inner']==cfg['planned_inner_estimator_fits'] and fit_counts['outer']<=cfg['maximum_outer_estimator_fits']
    T.dump(out/'selection_freeze.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'selections':selections,
        'note':'All model/loss/parameter choices complete before outer scoring below.'})
    # Only now use saved outer scoring evidence to compare procedures and references.
    old={name:pd.read_parquet(ROOT/f'reports/experiment_v3/predictions/{name}.parquet') for name in references}
    retained=T.V3['empty_oof'](d);retained['fold']=oofs['Target_baseline']['fold']
    reproduction=[]
    for target in TARGETS:
        prior=old[cfg['target_baseline_reference'][target]]
        assert prior[KEYS].equals(d[KEYS]) and np.array_equal(prior['true__'+target],d[target])
        difference=float(np.max(np.abs(prior['pred__'+target]-oofs['Target_baseline']['pred__'+target])))
        reproduction.append({'target':target,'v3_method':cfg['target_baseline_reference'][target],'max_abs_difference':difference})
        assert difference<1e-8,(target,difference)
        retained['pred__'+target]=old[cfg['retained_reference'][target]]['pred__'+target]
    oofs['Retained_reference']=retained
    for policy,frame in oofs.items():
        assert not frame.isna().any().any() and frame.fold.ge(1).all()
        assert set(frame.Room_ID).isdisjoint(cfg['protected_test_rooms'])
        frame.to_parquet(out/'predictions'/f'{policy}.parquet',index=False,compression='zstd')
    summary,decision=evaluate(oofs,cfg,out)
    for name,rows in [('inner_room_metrics',inner_records),('inner_candidate_scores',score_records),('selections',selections)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    T.dump(out/'fit_log.json',fit_log);T.dump(out/'baseline_reproduction.json',reproduction)
    assert all(T.sha(ROOT/p)==h for p,h in hashes.items())
    run.update(status='completed',fit_counts=fit_counts,outer_bundle_count=len(index),finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,input_files_unchanged=True,inner_candidate_scores=len(score_records),
        inner_room_scores=len(inner_records),selection_tasks=len(selections),outer_method_room_target_scores=140)
    T.dump(out/'run_manifest.json',run);T.dump(out/'progress.json',{'status':'completed','fit_counts':fit_counts,'elapsed_seconds':run['elapsed_seconds']})
    print(f'COMPLETED {fit_counts}; 7 bundles; TEST predictions/scores 0.',flush=True)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
