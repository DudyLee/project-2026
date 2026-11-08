"""Recompute selections/scores and verify saved v8 models without estimator fitting."""
from __future__ import annotations
from datetime import datetime, timezone
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
import directional_v8 as D
from train_directional_v8 import idx, col, entries, sha

ROOT=D.ROOT


def near(a,b,atol=1e-8):
    np.testing.assert_allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=1e-10,atol=atol,equal_nan=True)


def main():
    started=time.perf_counter();cfg=D.read(D.CONFIG);out=ROOT/cfg['output'];run=D.read(out/'run_manifest.json')
    assert run['status'] in ['completed_pending_verification','completed_verified']
    assert all(sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    assert run['TEST_fit_rows']==run['TEST_predictions']==run['TEST_scores']==0
    d,skipped=D.base.load_train(ROOT/cfg['source'],D.read(ROOT/'configs/model_plan.json'))
    assert skipped==52560 and len(d)==122640 and not set(d.Room_ID)&set(cfg['excluded_test_rooms'])
    xx,yy=D.features(d),D.labels(d)
    # Independently spell out the six interactions, including degree-to-radian conversion.
    b=d.EPW_DNI_Wh_m2.to_numpy()*np.cos(d.Solar_Altitude_mid_deg.to_numpy()*np.pi/180)
    u=b*np.sin(d.Solar_Azimuth_mid_deg.to_numpy()*np.pi/180);v=b*np.cos(d.Solar_Azimuth_mid_deg.to_numpy()*np.pi/180)
    expected=np.column_stack([u,v,u*d.Orientation_X_model,v*d.Orientation_X_model,u*d.Orientation_Y_model,v*d.Orientation_Y_model])
    near(xx['F1'][D.EXTRA],expected)
    jobs,protocols=D.splits(d,cfg)
    assert D.read(out/'fold_splits.json')=={'jobs':jobs,'protocols':protocols}
    selection=D.read(out/'selections.json');fitlog=D.read(out/'fit_log.json');preplog=D.read(out/'preprocessing_audit.json')
    modelindex={r['job_id']:r for r in D.read(out/'model_index.json')}
    recordedinner=pd.read_csv(out/'inner_room_metrics.csv');scoretable=pd.read_csv(out/'inner_candidate_scores.csv')
    assert len(selection)==63 and len(recordedinner)==700 and len(scoretable)==126 and len(preplog)==72
    selection_checks=0;inner_score_checks=0;model_values=0;bound_checks=0;preprocess_checks=0;label_checks=0
    predictions={};baseline_differences={}
    with threadpool_limits(limits=cfg['thread_limit']):
        for job in jobs:
            jid=job['id'];ti,vi=idx(d,job['fit_rooms']),idx(d,job['held_rooms'])
            assert not set(job['fit_rooms'])&set(job['held_rooms'])
            ip=pd.read_parquet(out/'inner_predictions'/(jid+'.parquet'))
            pd.testing.assert_frame_equal(ip[D.META],d.loc[ti,D.META].reset_index(drop=True))
            assert set(ip.Room_ID)==set(job['fit_rooms']) and not ip.isna().any().any()
            chosen={}
            for cell in ['S0','S1']:
                for method,target,direct in entries(cell):
                    candidates=[]
                    for lam in cfg['ridge']['lambdas']:
                        maes=[]
                        for room,g in ip.groupby('Room_ID',sort=True):
                            orig=d[d.Room_ID.eq(room)].set_index(D.KEYS).loc[pd.MultiIndex.from_frame(g[D.KEYS])]
                            near(g['true__'+target],orig[target])
                            score=mean_absolute_error(g['true__'+target],g['pred__'+col(method,target,lam)])
                            r=recordedinner[(recordedinner.job_id==jid)&(recordedinner.method==method)&(recordedinner.target==target)&(recordedinner['lambda']==lam)&(recordedinner.room==room)]
                            assert len(r)==1;near(score,r.mae.iloc[0]);maes.append(score);inner_score_checks+=1
                        r=scoretable[(scoretable.job_id==jid)&(scoretable.method==method)&(scoretable.target==target)&(scoretable['lambda']==lam)]
                        assert len(r)==1;near([np.mean(maes),np.max(maes)],r[['mean_MAE','worst_MAE']].iloc[0])
                        candidates.append((float(np.mean(maes)),lam))
                    best=min(candidates,key=lambda q:(q[0],-q[1]))
                    s=[r for r in selection if r['job_id']==jid and r['method']==method and r['target']==target]
                    assert len(s)==1 and s[0]['lambda']==best[1]
                    chosen[method+'::'+target]=best[1];selection_checks+=1
            freeze=D.read(out/'selection_freezes'/(jid+'.json'))
            assert freeze['selected']==chosen and not freeze['outer_predictions_computed'] and not freeze['outer_labels_used_for_selection']
            item=modelindex[jid];path=ROOT/item['path'];assert sha(path)==item['sha256']
            bundle=joblib.load(path)
            assert bundle['job']==job and bundle['raw_inputs']==D.api.RAW_FEATURES
            # Refit only unsupervised preprocessing to verify every stored train-fold statistic and input/label hash.
            for pm in [r for r in preplog if r['job_id']==jid]:
                a=idx(d,pm['fit_rooms']);x=xx[cfg['methods'][pm['cell']]['features']]
                if pm['phase']=='inner':
                    inner=next(r for r in job['inner'] if r['fold']==pm['inner_fold'])
                    assert pm['fit_rooms']==inner['fit_rooms'] and not set(pm['fit_rooms'])&set(inner['held_rooms'])
                else:assert pm['fit_rooms']==job['fit_rooms']
                assert not set(pm['fit_rooms'])&set(job['held_rooms'])
                prep=D.preprocessing(cfg);z=prep.fit_transform(x.loc[a]);diag=D.preprocess_diagnostics(prep,x.loc[a])
                for key in ['fit_rows','raw_columns','expanded_columns','input_sha256','constant_input_columns','scaler_n_samples_seen']:assert pm[key]==diag[key]
                for key in ['scaler_mean','scaler_var','spline_knots']:near(pm[key],diag[key])
                if pm['phase']=='outer':
                    saved=bundle['splines'][pm['cell']]
                    near(saved['preprocessor'].transform(x.loc[a]),z)
                    for key,est in saved['estimators'].items():
                        lam,primitive=key.split('::');assert est.alpha==float(lam)*len(a) and est.solver=='svd' and est.fit_intercept
                    for key,value in saved['selected'].items():assert chosen[key]==value
                phase='inner' if pm['phase']=='inner' else 'outer'
                logs=[r for r in fitlog if r['job_id']==jid and r['cell']==pm['cell'] and r['phase']==phase and r.get('inner_fold')==pm['inner_fold']]
                assert len(logs)>0
                for record in logs:
                    assert record['input_sha256']==D.array_sha(z) and record['label_sha256']==D.array_sha(yy[record['primitive']][a])
                    assert record['fit_rows']==len(a) and record['alpha']==record['lambda']*len(a)
                    label_checks+=1
                preprocess_checks+=1
            for method in ['T0','T1']:
                for target,fitted in bundle['trees'][method].items():
                    assert fitted['spec']==D.read(out/'tree_specs.json')[target]
                    x=xx[cfg['methods'][method]['features']]
                    for primitive,est in fitted['estimators'].items():
                        for key,value in fitted['spec']['params'].items():assert est.get_params()[key]==value
                        p=primitive if primitive!='value' else ('mean_log' if target=='E_mean_lux' else 'udi')
                        logs=[r for r in fitlog if r['job_id']==jid and r['phase']=='outer_tree' and r['cell']==method and r['target']==target and r['primitive']==primitive]
                        assert len(logs)==1
                        assert logs[0]['input_sha256']==D.array_sha(x.loc[ti]) and logs[0]['label_sha256']==D.array_sha(yy[p][ti])
                        assert logs[0]['fit_rooms']==job['fit_rooms'];label_checks+=1
            # Input-only reloading and inference; no Room_ID or output columns supplied.
            p=D.predict_bundle(bundle,d.loc[vi,D.api.RAW_FEATURES],cfg)
            op=pd.read_parquet(out/'outer_predictions'/(jid+'.parquet'))
            pd.testing.assert_frame_equal(op[D.META],d.loc[vi,D.META].reset_index(drop=True))
            for target in D.TARGETS:near(op['true__'+target],d.loc[vi,target])
            for (method,target),(pred,raw,cs) in p.items():
                suffix=col(method,target);near(pred,op['pred__'+suffix]);near(raw,op['raw__'+suffix])
                for key,value in cs.items():near(value,op[key+'__'+suffix])
                assert (pred>=0).all() and np.isfinite(raw).all()
                if target=='F_UDI_fraction':assert (pred<=1).all()
                model_values+=len(pred);bound_checks+=1
            predictions[jid]=op
            print(f'VERIFIED {jid}: nested selections, preprocessing, model reload',flush=True)
    assert selection_checks==63 and inner_score_checks==700 and preprocess_checks==72 and label_checks==len(fitlog)
    fm=pd.read_csv(out/'fold_metrics.csv');verifiedmetrics=0
    for protocol in protocols:
        op=predictions[protocol['job_id']]
        for room,g in op.groupby('Room_ID',sort=True):
            for method,spec in cfg['methods'].items():
                for target in spec['targets']:
                    suffix=col(method,target);y=g['true__'+target].to_numpy();p=g['pred__'+suffix].to_numpy();raw=g['raw__'+suffix].to_numpy()
                    r=fm[(fm.protocol==protocol['protocol'])&(fm.room==room)&(fm.method==method)&(fm.target==target)]
                    assert len(r)==1 and r.n.iloc[0]==len(g)
                    near([mean_absolute_error(y,p),np.sqrt(mean_squared_error(y,p)),r2_score(y,p),np.mean(p-y)],r[['mae','rmse','r2','bias']].iloc[0])
                    near(np.mean(np.abs(p-y)<=cfg['tolerances_descriptive'][target]),r.within_tolerance.iloc[0])
                    near(np.mean((raw<0)|((raw>1) if target=='F_UDI_fraction' else False)),r.raw_out_of_bounds_fraction.iloc[0])
                    verifiedmetrics+=1
    summary,gates,paired,decisions=D.summarize(fm,cfg)
    for name,frame in [('summary_metrics.csv',summary),('development_gate.csv',gates),('paired_room_comparison.csv',paired)]:
        pd.testing.assert_frame_equal(pd.read_csv(out/name),frame,check_dtype=False,atol=1e-8,rtol=1e-10)
    assert D.read(out/'candidate_decision.json')['targets']==decisions
    rule=cfg['gate'];eps=rule['numerical_tolerance']
    for g in gates.itertuples():
        def room_errors(protocol,method):
            return fm[(fm.protocol==protocol)&(fm.method==method)&(fm.target==g.target)].set_index('room').mae.sort_index()
        a,b=room_errors('LeaveOneRoomOut','T0'),room_errors('LeaveOneRoomOut',g.method)
        oa,ob=room_errors('LeaveOneOrientationOut','T0'),room_errors('LeaveOneOrientationOut',g.method)
        checks=[b.max()<=a.max()*(1-rule['primary_worst_MAE_reduction'])+eps,
                b.mean()<=a.mean()*(1+rule['primary_macro_MAE_increase_max'])+eps,
                (b<=a+eps).sum()>=rule['primary_rooms_nonworse_min'],
                ob.mean()<=oa.mean()*(1+rule['orientation_macro_MAE_increase_max'])+eps,
                ob.max()<=oa.max()*(1+rule['orientation_worst_MAE_increase_max'])+eps]
        assert bool(g.passes_all)==bool(all(checks))
    assert verifiedmetrics==182
    # Reproduce the previously archived exact-parameter baseline on all seven rooms.
    current=pd.concat([predictions['room_'+r] for r in cfg['development_rooms']],ignore_index=True).set_index(D.KEYS).sort_index()
    refs={'E_mean_lux':('reports/trial_v0_1/selection_oof_E_mean_lux_depth3.parquet','prediction'),
          'E_P10_lux':('reports/trial_v0_1/selection_oof_E_P10_lux_depth12.parquet','prediction'),
          'F_UDI_fraction':('reports/experiment_v1/predictions/B_add_solar_weather__HistGB.parquet','pred__F_UDI_fraction')}
    for target,(path,column) in refs.items():
        old=pd.read_parquet(ROOT/path).set_index(D.KEYS).sort_index()
        assert set(old.index.get_level_values('Room_ID'))==set(cfg['development_rooms']) and old.index.equals(current.index)
        delta=float(np.max(np.abs(current['pred__T0__'+target]-old[column])))
        near(current['pred__T0__'+target],old[column],atol=1e-6)
        baseline_differences[target]={'max_absolute_difference':delta,'source':path,'source_sha256':sha(ROOT/path),'rows':len(old)}
    assert run['total_estimator_fits']==len(fitlog)<=cfg['maximum_new_estimator_fits']
    assert all(sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    result={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-started,
        'source_and_frozen_code_hashes_unchanged':True,'protected_trial_model_unchanged':True,'TEST_fit_rows':0,'TEST_predictions':0,'TEST_scores':0,
        'train_rows':len(d),'test_rows_skipped':skipped,'independent_inner_room_metrics':inner_score_checks,'independent_lambda_selections':selection_checks,
        'preprocessing_refits_for_verification_only':preprocess_checks,'new_estimator_fits_in_verifier':0,'fit_label_and_input_hash_checks':label_checks,
        'saved_bundles_reloaded':len(jobs),'prediction_values_reproduced':model_values,'bounded_output_arrays':bound_checks,
        'independent_outer_room_metric_records':verifiedmetrics,'summary_gate_and_decision_checks':'passed',
        'baseline_reproduction':baseline_differences,'posX_room_BM223_reuse_verified':True,
        'limitations':'Code/data lineage checks verify the implemented procedure, not source EPW/north/grid provenance or independent generalization.'}
    D.dump(out/'verification.json',result)
    run['status']='completed_verified';run['verification']='reports/experiment_v8/verification.json';D.dump(out/'run_manifest.json',run)
    print(result,flush=True)


if __name__=='__main__':main()
