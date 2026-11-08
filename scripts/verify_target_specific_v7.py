"""Independent score/selection audit, model reload, and bounded reproducibility reruns."""
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_target_specific_v7 as V

T=V.T;ROOT=V.ROOT;OUT=ROOT/'reports/experiment_v7'


def stat(y,p,tol):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64);e=p-y
    denominator=float(((y-y.mean())**2).sum())
    return {'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt((e*e).mean())),
        'bias':float(e.mean()),'r2':float(1-(e*e).sum()/denominator) if denominator>0 else None,
        'within_tolerance':float((np.abs(e)<=tol).mean()),'absolute_tolerance':tol,
        'true_mean':float(y.mean()),'pred_mean':float(p.mean()),'n':len(y)}


def compare(saved,computed):
    for k,v in computed.items():
        if v is None:assert pd.isna(saved[k]) or saved[k] is None
        else:assert np.isclose(saved[k],v,rtol=1e-10,atol=1e-10),(k,saved[k],v)


def labels(frame,spec):
    y=frame[spec['target']].to_numpy();tr=spec['transform']
    if tr=='relative':return {'mean_log':np.log1p(frame.E_mean_lux.to_numpy()),'ratio':y/(frame.E_mean_lux.to_numpy()+1)}
    return {'value':np.log1p(y) if tr=='log1p' else np.arcsinh(y) if tr=='asinh' else y}


def main():
    cfg=T.read(V.CONFIG);run=T.read(OUT/'run_manifest.json');specs=T.read(OUT/'candidate_catalog.json')
    assert run['status']=='completed' and run['fit_counts']['inner']==546
    assert T.sha(ROOT/'scripts/train_target_specific_v7.py')==run['script_sha256']
    assert all(T.sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    assert specs==V.catalog(cfg)
    d,skipped=V.BASE['load_train'](ROOT/cfg['source'],T.read(ROOT/'configs/model_plan.json'))
    columns=T.read(OUT/'feature_names.json')['raw'];x=V.BASE['encode_inputs'](d,columns)
    assert len(columns)==13 and x.shape[1]==15 and skipped==52560
    ir=pd.read_csv(OUT/'inner_room_metrics.csv');cs=pd.read_csv(OUT/'inner_candidate_scores.csv');sel=pd.read_csv(OUT/'selections.csv')
    assert len(ir)==1008 and len(cs)==168 and len(sel)==105
    folds=T.read(OUT/'fold_splits.json');index=T.read(OUT/'model_index.json');log=T.read(OUT/'fit_log.json')
    assert len(folds)==len(index)==7
    oofs={policy:pd.read_parquet(OUT/'predictions'/f'{policy}.parquet') for policy in cfg['policies']+['Retained_reference']}
    for frame in oofs.values():
        assert frame[V.KEYS].equals(d[V.KEYS]) and len(frame)==122640 and not frame.isna().any().any()
        assert set(frame.Room_ID).isdisjoint(cfg['protected_test_rooms'])
        for target in V.TARGETS:assert np.array_equal(frame['true__'+target],d[target])
    reload_delta=0.;refit_delta=0.;refit_count=0;reruns=[];audited_fit_entries=0
    with threadpool_limits(limits=cfg['thread_limit']):
        for fold in folds:
            k=fold['fold'];held=fold['held_room'];fit_rooms=fold['fitting_rooms']
            assert len(fit_rooms)==6 and held not in fit_rooms and not set(fit_rooms)&set(cfg['protected_test_rooms'])
            train=d[d.Room_ID.isin(fit_rooms)].reset_index(drop=True);xt=V.BASE['encode_inputs'](train,columns)
            ip=pd.read_parquet(OUT/'inner_predictions'/f'fold_{k:02d}_{held}.parquet')
            assert ip[V.KEYS].equals(train[V.KEYS]) and set(ip.Room_ID)==set(fit_rooms)
            for group in fold['inner']:
                assert len(group['fitting_rooms'])==4 and len(group['validation_rooms'])==2
                assert not set(group['fitting_rooms'])&set(group['validation_rooms'])
                assert set(ip.loc[ip.inner_fold==group['inner_fold'],'Room_ID'])==set(group['validation_rooms'])
            expected_choices={p:{} for p in cfg['policies']}
            for target in V.TARGETS:
                recomputed=[]
                for spec in specs[target]:
                    key=target+'::'+spec['id'];values=[]
                    assert np.array_equal(ip['true__'+target],train[target])
                    for room,g in ip.groupby('Room_ID',sort=True):
                        met=stat(g['true__'+target],g[key],cfg['tolerance_descriptive'][target]);values.append(met['mae'])
                        saved=ir[(ir.outer_fold==k)&(ir.target==target)&(ir.candidate==spec['id'])&(ir.room==room)].iloc[0]
                        compare(saved,met)
                    s={'candidate':spec['id'],'mean':float(np.mean(values)),'worst':float(np.max(values)),
                       'order':spec['order'],'baseline':spec['baseline']}
                    saved=cs[(cs.outer_fold==k)&(cs.target==target)&(cs.candidate==spec['id'])].iloc[0]
                    compare(saved,{'mean_mae':s['mean'],'worst_mae':s['worst']});recomputed.append(s)
                baseline=sorted([s for s in recomputed if s['baseline']],key=lambda s:(s['mean'],s['order']))[0]
                avg=sorted(recomputed,key=lambda s:(s['mean'],s['order']))[0]
                eligible=[s for s in recomputed if s['mean']<=baseline['mean']*1.05+cfg['selection']['selection_tolerance']]
                worst=sorted(eligible,key=lambda s:(s['worst'],s['mean'],s['order']))[0]
                for policy,s in [('Target_baseline',baseline),('Mean_select',avg),('Worst_guarded',worst)]:
                    saved=sel[(sel.outer_fold==k)&(sel.target==target)&(sel.policy==policy)].iloc[0]
                    assert saved.candidate==s['candidate']
                    compare(saved,{'inner_mean_mae':s['mean'],'inner_worst_mae':s['worst'],'guard_limit':baseline['mean']*1.05})
                    expected_choices[policy][target]=s['candidate']
            entry=next(i for i in index if i['fold']==k);path=ROOT/entry['path'];assert T.sha(path)==entry['sha256']
            bundle=joblib.load(path)
            assert bundle['selected']==expected_choices and bundle['fitting_rooms']==fit_rooms and bundle['held_room']==held
            held_frame=d[d.Room_ID==held]
            for policy in cfg['policies']:
                pred=V.predict_bundle(bundle,held_frame[columns],policy)
                saved=oofs[policy][oofs[policy].Room_ID==held]
                for target,p in pred.items():
                    difference=float(np.max(np.abs(saved['pred__'+target].to_numpy()-p)))
                    assert difference<1e-8,(policy,held,target,difference)
                    reload_delta=max(reload_delta,difference)
            # Reconstruct every training subset and label hash without using the held room.
            for evidence in (e for e in log if e['outer_fold']==k):
                if evidence['stage']=='outer':
                    subset=train;subx=xt;expected_rooms=fit_rooms
                else:
                    group=fold['inner'][evidence['inner_fold']-1];expected_rooms=group['fitting_rooms']
                    mask=train.Room_ID.isin(expected_rooms);subset=train.loc[mask];subx=xt.loc[mask]
                assert evidence['fitting_rooms']==expected_rooms and held not in expected_rooms
                assert evidence['n_fit']==len(subset) and T.array_sha(subx.to_numpy())==evidence['encoded_input_sha256']
                spec=next(s for s in specs[evidence['target']] if s['id']==evidence['candidate'])
                yh={c:T.array_sha(y) for c,y in labels(subset,spec).items()}
                assert yh==evidence['target_hashes'];audited_fit_entries+=1
            # Bounded rerun on first outer/inner split: five new absolute-loss fits plus one relative P10 pair.
            if k==1:
                group=fold['inner'][0];a=train.Room_ID.isin(group['fitting_rooms']);b=train.Room_ID.isin(group['validation_rooms'])
                repeat_specs=[next(s for s in specs[target] if s['loss']=='absolute') for target in V.TARGETS]
                repeat_specs.append(next(s for s in specs['E_P10_lux'] if s['transform']=='relative'))
                for spec in repeat_specs:
                    estimators={}
                    for component,y in labels(train.loc[a],spec).items():
                        estimator=T.V3['make_estimator'](spec['family'],spec['params']);estimator.fit(xt.loc[a],y)
                        estimators[component]=estimator;refit_count+=1
                    repeated=V.predict_target({'spec':spec,'estimators':estimators},xt.loc[b])
                    expected=ip.loc[b,spec['target']+'::'+spec['id']].to_numpy()
                    difference=float(np.max(np.abs(repeated-expected)));assert difference<1e-8
                    refit_delta=max(refit_delta,difference)
                    reruns.append({'target':spec['target'],'candidate':spec['id'],'outer_fold':k,'inner_fold':1,'max_abs_prediction_difference':difference})
            print(f'Verified fold {k}/7 {held}: inner choices, all outer predictions, fit-subset hashes',flush=True)
    assert sum(e['estimator_count'] for e in log if e['stage']=='inner')==run['fit_counts']['inner']
    assert sum(e['estimator_count'] for e in log if e['stage']=='outer')==run['fit_counts']['outer']
    fm=pd.read_csv(OUT/'fold_metrics.csv');assert len(fm)==140
    for method,frame in oofs.items():
        for room,g in frame.groupby('Room_ID'):
            for target in V.TARGETS:
                saved=fm[(fm.method==method)&(fm.room==room)&(fm.target==target)].iloc[0]
                compare(saved,stat(g['true__'+target],g['pred__'+target],cfg['tolerance_descriptive'][target]))
    summary=pd.read_csv(OUT/'summary_metrics.csv')
    for row in summary.itertuples():
        g=fm[(fm.method==row.method)&(fm.target==row.target)]
        compare(row._asdict(),{'mae_mean':g.mae.mean(),'mae_median':g.mae.median(),'mae_max':g.mae.max(),
            'rmse_mean':g.rmse.mean(),'bias_mean':g.bias.mean(),'r2_median':g.r2.median(),'r2_min':g.r2.min(),
            'within_tolerance_mean':g.within_tolerance.mean(),'within_tolerance_min':g.within_tolerance.min()})
        assert row.worst_room==g.sort_values(['mae','room'],ascending=[False,True]).iloc[0].room
    gates=pd.read_csv(OUT/'development_gate.csv');dec=T.read(OUT/'candidate_decision.json');tol=cfg['development_gate']['numerical_tolerance']
    for row in gates.itertuples():
        a=fm[(fm.method==row.reference)&(fm.target==row.target)].set_index('room').sort_index()
        b=fm[(fm.method==row.method)&(fm.target==row.target)].set_index('room').sort_index()
        reduction=1-b.mae.max()/a.mae.max();nonworse=int((b.mae<=a.mae+tol).sum())
        compare(row._asdict(),{'worst_MAE_reduction_fraction':reduction,'macro_MAE_change_fraction':b.mae.mean()/a.mae.mean()-1,
            'rooms_nonworse':nonworse,'rooms_improved':int((b.mae<a.mae-tol).sum()),'rooms_tied':int((np.abs(b.mae-a.mae)<=tol).sum())})
        passed=bool(reduction>=.10-tol and b.mae.mean()<=a.mae.mean()+tol and nonworse>=5)
        assert passed==row.passes_gate
    for target in V.TARGETS:
        passed=bool(gates[(gates.method=='Worst_guarded')&(gates.target==target)].passes_gate.all())
        assert passed==dec['targets'][target]['passes_both_references']
    for target in V.TARGETS:
        for policy,mapping in [('Retained_reference',cfg['retained_reference']),('Target_baseline',cfg['target_baseline_reference'])]:
            ref=pd.read_parquet(ROOT/f'reports/experiment_v3/predictions/{mapping[target]}.parquet')
            assert ref[V.KEYS].equals(d[V.KEYS])
            assert np.allclose(ref['pred__'+target],oofs[policy]['pred__'+target],rtol=0,atol=1e-8)
    assert all(T.sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    result={'status':'passed','inner_room_scores_recomputed':len(ir),'inner_candidate_scores_recomputed':len(cs),
        'selection_tasks_recomputed':len(sel),'outer_model_bundles_reloaded':len(index),
        'outer_prediction_values_recomputed':122640*5*3,'model_reload_max_abs_difference':reload_delta,
        'fit_subset_and_label_hash_entries_checked':audited_fit_entries,'outer_metrics_recomputed':len(fm),'summary_metrics_checked':len(summary),
        'gate_comparisons_recomputed':len(gates),'baseline_reproduction':'passed for all five outputs',
        'bounded_reproducibility_estimator_refits':refit_count,'bounded_reproducibility_max_abs_difference':refit_delta,'reruns':reruns,
        'inference_received_only_13_features_no_room_ID_or_targets':True,'TEST_rows_discarded_before_numeric_conversion':skipped,
        'TEST_predictions':0,'TEST_scores':0,'frozen_inputs_unchanged':True,'script_sha256':T.sha(__file__),
        'finished_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'All numerical scores and selection/inference evidence checked; bounded 7-estimator repeat, not an independent repetition of the entire experiment.'}
    T.dump(OUT/'verification.json',result)
    print('VERIFICATION PASSED; TEST predictions/scores 0; bounded reproducibility fits '+str(refit_count),flush=True)


if __name__=='__main__':main()
