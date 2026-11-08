"""Independently score saved OOF predictions and audit grouped selections/model reloads."""
from pathlib import Path
from datetime import datetime, timezone
import json
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_cross_room_v5 as T

ROOT=T.ROOT
OUT=ROOT/'reports/experiment_v5'

def calculate(y,p,prediction_transform_dtype=None):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);e=p-y
    total=np.square(y-y.mean()).sum()
    return {'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.square(e).mean())),
            'r2':float(1-np.square(e).sum()/total) if total>0 else None,'bias':float(e.mean()),
            'asinh_mae':float(np.abs(np.arcsinh(p.astype(prediction_transform_dtype) if prediction_transform_dtype is not None else p)-np.arcsinh(y)).mean())}

def check_metrics(actual,expected):
    for k,v in expected.items():
        if v is None: assert actual[k] is None or pd.isna(actual[k])
        else: assert np.isclose(float(actual[k]),v,rtol=1e-10,atol=1e-10),(k,actual[k],v)

def training_labels(frame):
    # Explicit independent reconstruction for every saved fit record.
    return {'mean_log':np.log1p(np.asarray(frame['E_mean_lux'],dtype=float)),
            'ratio':np.asarray(frame['E_P10_lux'],dtype=float)/(np.asarray(frame['E_mean_lux'],dtype=float)+1.0),
            'asinh':np.arcsinh(np.asarray(frame['E_P10_lux'],dtype=float)/1.0)}

def main():
    started=time.perf_counter();cfg=T.read(T.CONFIG);run=T.read(OUT/'run_manifest.json')
    assert run['status']=='completed' and run['fit_counts']['inner']==504
    assert all(T.sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    assert T.sha(ROOT/'scripts/train_cross_room_v5.py')==run['script_sha256']
    d,skipped=T.load_development(cfg);xs=T.features(d,cfg)
    protected=set(cfg['protected_test_rooms']);assert set(d.Room_ID).isdisjoint(protected)
    # Derived features are input-only and contain no fitted outer-room statistics.
    assert np.array_equal(xs['enhanced'].Glass_to_floor_fraction,d.Glass_Area_m2/d.Room_Area_m2)
    assert np.array_equal(xs['enhanced'].Mean_nominal_window_area_m2,d.Window_Area_Nominal_m2/d.Window_Count)
    splits=T.read(OUT/'fold_splits.json');fitlog=T.read(OUT/'fit_log.json');index=T.read(OUT/'model_index.json')
    assert len(fitlog)==sum(run['fit_counts'].values()) and len(index)==28
    label_hashes={}
    for split in splits:
        trainrooms=set(split['training_rooms']);held=split['held_room']
        assert len(trainrooms)==6 and held not in trainrooms and not trainrooms&protected
        train=d[d.Room_ID.isin(trainrooms)].reset_index(drop=True)
        for primitive,arr in training_labels(train).items():label_hashes[(split['fold'],0,primitive)]=T.array_sha(arr)
        for inner in split['inner']:
            fr=set(inner['fitting_rooms']);vr=set(inner['validation_rooms'])
            assert len(fr)==4 and len(vr)==2 and fr|vr==trainrooms and not fr&vr
            for primitive,arr in training_labels(train[train.Room_ID.isin(fr)]).items():
                label_hashes[(split['fold'],inner['inner'],primitive)]=T.array_sha(arr)
    for r in fitlog:
        assert r['training_target_sha256']==label_hashes[(r['outer_fold'],r.get('inner_fold',0),r['primitive'])]
        expected=next(s for s in splits if s['fold']==r['outer_fold'])
        fr=expected['training_rooms'] if r['stage']=='outer' else expected['inner'][r['inner_fold']-1]['fitting_rooms']
        assert r['fitting_rooms']==fr and r['held_room'] not in fr and not set(fr)&protected
        assert r['n_fit']==len(fr)*17520
    inner_metrics=pd.read_csv(OUT/'inner_room_metrics.csv');scores=pd.read_csv(OUT/'inner_candidate_scores.csv');selections=pd.read_csv(OUT/'selections.csv')
    assert len(inner_metrics)==1008 and len(scores)==168 and len(selections)==84
    inner_checked=0
    params_cfg=T.read(ROOT/cfg['parameter_source'])
    for item in index:
        fold=item['fold'];room=item['room'];fs=item['feature_set'];fam=item['family']
        ip=pd.read_parquet(OUT/'inner_predictions'/f'fold_{fold:02d}_{room}__{fs}__{fam}.parquet')
        fitting=next(s for s in splits if s['fold']==fold)
        assert len(ip)==105120 and set(ip.Room_ID)==set(fitting['training_rooms']) and room not in set(ip.Room_ID)
        truth=d[d.Room_ID.isin(fitting['training_rooms'])].reset_index(drop=True)
        assert ip[T.KEYS].equals(truth[T.KEYS])
        for t in T.TARGETS:assert np.array_equal(ip['true__'+t],truth[t])
        for inn in fitting['inner']:
            assert set(ip.loc[ip.inner_fold==inn['inner'],'Room_ID'])==set(inn['validation_rooms'])
        subset=inner_metrics[(inner_metrics.outer_fold==fold)&(inner_metrics.feature_set==fs)&(inner_metrics.family==fam)]
        for r in subset.to_dict('records'):
            g=ip[ip.Room_ID==r['validation_room']]
            values=g[r['candidate']+'__'+r['output']]
            # Inner XGBoost transforms were evaluated on native float32 arrays,
            # then predictions were stored in float64 columns. Reproduce that
            # arithmetic instead of loosening tolerances or changing scores.
            if fam=='XGBoost': assert np.array_equal(values,values.to_numpy().astype(np.float32).astype(np.float64))
            check_metrics(r,calculate(g['true__'+r['target']],values,np.float32 if fam=='XGBoost' else None))
            inner_checked+=1
    for r in scores.to_dict('records'):
        subset=inner_metrics[(inner_metrics.outer_fold==r['outer_fold'])&(inner_metrics.feature_set==r['feature_set'])&
                             (inner_metrics.family==r['family'])&(inner_metrics.candidate==r['candidate'])&(inner_metrics.output==r['output'])]
        assert len(subset)==6 and np.isclose(r['inner_mean_room_mae'],subset.mae.mean(),rtol=1e-12,atol=1e-12)
    for r in selections.to_dict('records'):
        subset=scores[(scores.outer_fold==r['outer_fold'])&(scores.feature_set==r['feature_set'])&(scores.family==r['family'])&(scores.output==r['output'])]
        # idxmin is stable and preserves the declared candidate order.
        selected=subset.loc[subset.inner_mean_room_mae.idxmin()]
        assert selected.candidate==r['candidate']
    preds={p.stem:pd.read_parquet(p) for p in (OUT/'predictions').glob('*.parquet')}
    assert len(preds)==9
    scored=[];details=[]
    reported=pd.read_csv(OUT/'fold_metrics.csv')
    for method,p in preds.items():
        assert p[T.KEYS].equals(d[T.KEYS]) and len(p)==122640 and not set(p.Room_ID)&protected
        assert not p.isna().any().any()
        for t in T.TARGETS:
            assert np.array_equal(p['true__'+t],d[t]) and (p['pred__'+t]>=0).all()
        for room,g in p.groupby('Room_ID'):
            for t in T.TARGETS:
                met=calculate(g['true__'+t],g['pred__'+t]);scored.append({'method':method,'room':room,'target':t,**met})
                stored=reported[(reported.method==method)&(reported.room==room)&(reported.target==t)].iloc[0]
                check_metrics(stored,met)
                err=g['pred__'+t]-g['true__'+t]
                details.append({'method':method,'room':room,'target':t,'actual_mean_lux':float(g['true__'+t].mean()),
                                'predicted_mean_lux':float(g['pred__'+t].mean()),'overprediction_fraction':float((err>0).mean()),
                                'absolute_error_p95_lux':float(np.quantile(np.abs(err),.95))})
    shared_mean_checks=[]
    for family in cfg['families']:
        for a,b in [('A','C'),('B','D')]:
            error=float(np.max(np.abs(preds[a+'_'+family].pred__E_mean_lux-preds[b+'_'+family].pred__E_mean_lux)))
            # RF parallel tree summation can vary by a few float64 ulps.
            assert error<=1e-9
            shared_mean_checks.append({'family':family,'pair':a+'/'+b,'max_difference_lux':error})
    # All stored outer estimators reloaded; both output paths verified on held-room X only.
    reloaded=0;estimator_count=0
    with threadpool_limits(limits=cfg['thread_limit']):
        for item in index:
            path=ROOT/item['path'];assert T.sha(path)==item['sha256'];bundle=joblib.load(path)
            assert bundle['held_room']==item['room'] and bundle['held_room'] not in bundle['train_rooms']
            assert not set(bundle['train_rooms'])&protected
            selected_rows=selections[(selections.outer_fold==item['fold'])&(selections.feature_set==item['feature_set'])&(selections.family==item['family'])]
            assert bundle['selected_candidates']==dict(zip(selected_rows.output,selected_rows.candidate))
            indices=d.index[d.Room_ID==item['room']].to_numpy()[np.linspace(0,17519,31,dtype=int)]
            x=xs[item['feature_set']].loc[indices]
            assert list(x.columns)==bundle['encoded_features']
            for key,est in bundle['estimators'].items():
                cid=key.split('__')[0];expected=T.V3['params_for'](params_cfg,item['family'],cid)
                for k,v in expected.items(): assert est.get_params()[k]==v,(item['path'],key,k)
                estimator_count+=1
            for profile,spec in cfg['profiles'].items():
                if spec['features']!=item['feature_set']:continue
                got=T.predict_bundle(bundle,x,spec['P10'])
                chosen=bundle['selected_candidates'];ests=bundle['estimators']
                independent_mean=np.clip(np.expm1(ests[chosen['mean']+'__mean_log'].predict(x)),0,None)
                pcid=chosen[spec['P10']]
                if spec['P10']=='asinh':
                    independent_p10=np.clip(np.sinh(ests[pcid+'__asinh'].predict(x)),0,None)
                else:
                    independent_p10=np.clip(ests[pcid+'__ratio'].predict(x),0,None)*(np.clip(np.expm1(ests[pcid+'__mean_log'].predict(x)),0,None)+1.0)
                assert np.allclose(got['E_mean_lux'],independent_mean,rtol=1e-12,atol=1e-9)
                assert np.allclose(got['E_P10_lux'],independent_p10,rtol=1e-12,atol=1e-9)
                for target,values in got.items():
                    stored=preds[profile+'_'+item['family']].loc[indices,'pred__'+target]
                    assert np.allclose(values,stored,rtol=1e-9,atol=1e-8)
            reloaded+=1
    assert estimator_count==run['fit_counts']['outer']
    for family in cfg['families']:
        reference=pd.read_parquet(ROOT/'reports/experiment_v3/predictions'/f'{family}.parquet')
        assert reference[T.KEYS].equals(preds['A_'+family][T.KEYS])
        for target in T.TARGETS:
            assert np.allclose(reference['pred__'+target],preds['A_'+family]['pred__'+target],rtol=0,atol=1e-8)
    fm=pd.DataFrame(scored)
    summary=fm.groupby(['method','target']).agg(room_count=('room','nunique'),mae_mean=('mae','mean'),mae_median=('mae','median'),
             mae_max=('mae','max'),rmse_mean=('rmse','mean'),r2_median=('r2','median'),r2_min=('r2','min'),
             asinh_mae_mean=('asinh_mae','mean'),bias_mean=('bias','mean')).reset_index()
    summary.to_csv(OUT/'summary_metrics.csv',index=False);pd.DataFrame(details).to_csv(OUT/'room_error_details.csv',index=False)
    criteria=[];pairs=[];tol=cfg['promotion_rule']['numerical_tolerance_lux']
    for fam in cfg['families']:
        for target in T.TARGETS:
            baseline=fm[(fm.method=='A_'+fam)&(fm.target==target)].set_index('room').sort_index()
            for profile in ['B','C','D']:
                method=profile+'_'+fam;new=fm[(fm.method==method)&(fm.target==target)].set_index('room').sort_index()
                delta=new.mae-baseline.mae
                improvement=1-new.mae.mean()/baseline.mae.mean();nonworse=int((delta<=tol).sum())
                worst_change=float(new.mae.max()-baseline.mae.max())
                passed=improvement>=.10 and nonworse>=5 and worst_change<=tol
                criteria.append({'method':method,'baseline':'A_'+fam,'target':target,'macro_mae_reduction_fraction':float(improvement),
                                 'rooms_improved':int((delta < -tol).sum()),'rooms_tied':int((delta.abs()<=tol).sum()),
                                 'rooms_worse':int((delta>tol).sum()),'rooms_nonworse':nonworse,'worst_room_mae_change_lux':worst_change,
                                 'passes_development_rule':passed,'mean_prediction_unchanged':target=='E_mean_lux' and profile=='C'})
                for room in new.index:
                    pairs.append({'method':method,'baseline':'A_'+fam,'target':target,'room':room,'baseline_mae_lux':float(baseline.loc[room,'mae']),
                                  'new_mae_lux':float(new.loc[room,'mae']),'mae_change_lux':float(delta.loc[room]),
                                  'baseline_bias_lux':float(baseline.loc[room,'bias']),'new_bias_lux':float(new.loc[room,'bias']),
                                  'asinh_mae_change':float(new.loc[room,'asinh_mae']-baseline.loc[room,'asinh_mae'])})
    pd.DataFrame(criteria).to_csv(OUT/'promotion_criteria.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_room_changes.csv',index=False)
    assert all(T.sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    verification={'status':'passed','outer_room_target_metrics_recomputed':len(scored),'inner_room_metrics_recomputed':inner_checked,
                  'candidate_mean_scores_recomputed':len(scores),'inner_selections_recomputed':len(selections),
                  'training_target_hashes_reconstructed':len(fitlog),'outer_bundles_reloaded':reloaded,'outer_estimators_reloaded':estimator_count,
                  'reload_samples_per_held_room':31,'paired_comparisons':len(pairs),'development_criteria_evaluated':len(criteria),
                  'shared_mean_predictions_match_within_1e_9_lux':shared_mean_checks,'no_true_target_required_at_inference':True,'grouped_disjointness_checked':True,
                  'both_P10_inverse_formulas_independently_checked':True,'v3_baseline_predictions_reproduced':True,
                  'inner_XGBoost_asinh_metric_native_float32_arithmetic_reproduced':True,
                  'input_hashes_unchanged':True,'TEST_rows_skipped_before_numeric_conversion':skipped,'TEST_predictions':0,'TEST_scores':0,
                  'evaluation_script_sha256':T.sha(__file__),'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,
                  'scope':'Saved prediction scoring, selection, label-hash and model-reload verification. Not independent TEST performance or a second full training run.'}
    T.dump(OUT/'verification.json',verification)
    T.dump(OUT/'candidate_decision.json',{'status':'development_only','rule':cfg['promotion_rule'],'passing_candidates':[r for r in criteria if r['passes_development_rule']],
                                        'deployed':False,'final_refit':False,'test_unsealed':False,'automatic_replacement_of_previous_models':False})
    print(summary[['method','target','mae_mean','mae_max','asinh_mae_mean']].to_string(index=False))
    print(pd.DataFrame(criteria)[['method','target','macro_mae_reduction_fraction','rooms_nonworse','worst_room_mae_change_lux','passes_development_rule']].to_string(index=False))
    print(json.dumps(verification,ensure_ascii=False))

if __name__=='__main__':main()
