"""Verify v10 selections, fold preprocessing, saved inference and score gates."""
from datetime import datetime,timezone
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from threadpoolctl import threadpool_limits
import pair_balance_v10 as V
from train_pair_balance_v10 import idx,sha

D=V.D;ROOT=V.ROOT
def close(a,b):np.testing.assert_allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=1e-10,atol=1e-8,equal_nan=True)


def main():
    tick=time.perf_counter();cfg=D.read(V.CONFIG);out=ROOT/cfg['output'];run=D.read(out/'run_manifest.json')
    assert run['status'] in ['completed_pending_verification','completed_verified']
    assert all(sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    d,audit,profiles=V.V9.load(cfg);assert audit==D.read(out/'input_source_audit.json')
    xx,q=V.V9.features(d,cfg);x=xx['F2']
    close(q,d.Glass_Area_m2.to_numpy()*d[V.V9.ANNUAL].to_numpy()/(100*d.Room_Area_m2.to_numpy()))
    assert not set(x.columns)&{'Room_ID',V.TARGET,'Sample_Role'}
    jobs,protocols=D.splits(d,cfg);assert D.read(out/'fold_splits.json')=={'jobs':jobs,'protocols':protocols}
    assert D.read(ROOT/cfg['reference_experiment']/'fold_splits.json')=={'jobs':jobs,'protocols':protocols}
    logs=D.read(out/'fit_log.json');preps=D.read(out/'preprocessing_audit.json');selected=D.read(out/'selections.json');index={r['job_id']:r for r in D.read(out/'model_index.json')}
    params=D.read(out/'parameters.json');assert params['objective']=='reg:squarederror'
    inner=pd.read_csv(out/'inner_room_metrics.csv');scores=pd.read_csv(out/'inner_candidate_scores.csv')
    assert len(logs)==252 and len(preps)==36 and len(selected)==36 and len(inner)==400 and len(scores)==72
    counters={'inner_scores':0,'selections':0,'preprocessing':0,'fit_hashes':0,'prediction_values':0,'outer_scores':0,'strata':0};ops={}
    with threadpool_limits(limits=cfg['thread_limit']):
        for job in jobs:
            jid=job['id'];ti,vi=idx(d,job['fit_rooms']),idx(d,job['held_rooms'])
            ip=pd.read_parquet(out/'inner_predictions'/(jid+'.parquet'));pd.testing.assert_frame_equal(ip[D.META],d.loc[ti,D.META].reset_index(drop=True))
            close(ip.true__E_mean_lux,d.loc[ti,V.TARGET]);assert not ip.isna().any().any()
            choices={}
            for method,spec in cfg['methods'].items():
                options=[]
                for parameter in spec['parameters']:
                    maes=[]
                    for room,g in ip.groupby('Room_ID',sort=True):
                        score=mean_absolute_error(g.true__E_mean_lux,g[f'pred__{method}__{parameter}'])
                        r=inner[(inner.job_id==jid)&(inner.method==method)&(inner.parameter==parameter)&(inner.room==room)]
                        assert len(r)==1;close(score,r.mae.iloc[0]);maes.append(score);counters['inner_scores']+=1
                    r=scores[(scores.job_id==jid)&(scores.method==method)&(scores.parameter==parameter)]
                    assert len(r)==1;close([np.mean(maes),np.max(maes)],r[['macro_MAE','worst_MAE']].iloc[0]);options.append((float(np.mean(maes)),parameter))
                best=min(options,key=lambda z:(z[0],z[1] if spec['family']=='XGBoost' else -z[1]))
                choices[method]=best[1];r=[r for r in selected if r['job_id']==jid and r['method']==method]
                assert len(r)==1 and r[0]['parameter']==best[1];counters['selections']+=1
            freeze=D.read(out/'selection_freezes'/(jid+'.json'));assert freeze['selected']==choices and not freeze['outer_predictions_computed'] and not freeze['outer_labels_used_for_selection']
            item=index[jid];path=ROOT/item['path'];assert sha(path)==item['sha256'];bundle=joblib.load(path)
            assert bundle['job']==job and bundle['selected']==choices and bundle['raw_inputs']==V.RAW
            for pm in [r for r in preps if r['job_id']==jid]:
                split=job if pm['phase']=='outer' else next(r for r in job['inner'] if r['fold']==pm['inner_fold'])
                assert pm['fit_rooms']==split['fit_rooms']
                a=idx(d,pm['fit_rooms']);prep=D.preprocessing(cfg);z=prep.fit_transform(x.loc[a]);assert z.shape[1]==48
                diag=D.preprocess_diagnostics(prep,x.loc[a])
                for key in ['fit_rows','raw_columns','expanded_columns','input_sha256','constant_input_columns','scaler_n_samples_seen']:assert diag[key]==pm[key]
                for key in ['scaler_mean','scaler_var','spline_knots']:close(diag[key],pm[key])
                if pm['phase']=='outer':close(bundle['preprocessor'].transform(x.loc[a]),z)
                fitrecords=[r for r in logs if r['job_id']==jid and r['phase']==pm['phase'] and r['inner_fold']==pm['inner_fold']]
                assert len(fitrecords)==(4 if pm['phase']=='outer' else 8)
                for r in fitrecords:
                    assert r['fit_rooms']==split['fit_rooms'] and r['held_rooms']==split['held_rooms'] and r['sample_weight'] is None
                    assert not set(r['fit_rooms'])&(set(job['held_rooms'])|set(r['held_rooms'])|set(cfg['excluded_test_rooms']))
                    spec=cfg['methods'][r['method']];y=d.loc[a,V.TARGET].to_numpy()
                    if spec['normalized']:y=y/q[a]
                    y=np.log1p(y);input_values=x.loc[a] if spec['family']=='XGBoost' else z
                    assert r['fit_rows']==len(a) and r['label_sha256']==D.array_sha(y) and r['input_sha256']==D.array_sha(input_values)
                    counters['fit_hashes']+=1
                counters['preprocessing']+=1
            for method,est in bundle['estimators'].items():
                if cfg['methods'][method]['family']=='XGBoost':
                    for key,value in (params|{'max_depth':choices[method]}).items():assert est.get_params()[key]==value
                else:assert est.alpha==choices[method]*len(ti) and est.solver=='svd' and est.fit_intercept
            op=pd.read_parquet(out/'outer_predictions'/(jid+'.parquet'));pd.testing.assert_frame_equal(op[D.META],d.loc[vi,D.META].reset_index(drop=True))
            close(op.q,q[vi]);close(op.true__E_mean_lux,d.loc[vi,V.TARGET])
            pred=V.predict_bundle(bundle,d.loc[vi,V.RAW],cfg)
            z=bundle['preprocessor'].transform(x.loc[vi])
            for method,(p,raw) in pred.items():
                close(p,op['pred__'+method]);close(raw,op['raw__'+method]);assert np.isfinite(raw).all() and (p>=0).all()
                # Independent inverse-transform reconstruction, checking both q branches.
                est=bundle['estimators'][method];logp=np.asarray(est.predict(x.loc[vi] if cfg['methods'][method]['family']=='XGBoost' else z),dtype=float)
                native=np.expm1(logp)*(q[vi] if cfg['methods'][method]['normalized'] else 1)
                close(native,raw);close(np.maximum(native,0),p);counters['prediction_values']+=len(p)
            old=pd.read_parquet(ROOT/cfg['reference_experiment']/'outer_predictions'/(jid+'.parquet'));pd.testing.assert_frame_equal(old[D.META],op[D.META])
            for method,oldname in cfg['references'].items():
                for kind in ['pred','raw']:np.testing.assert_array_equal(op[kind+'__'+method],old[kind+'__'+oldname])
            ops[jid]=op;print(f'VERIFIED {jid}: nested selections, preprocessing, unweighted labels, inference and references',flush=True)
    fm=pd.read_csv(out/'fold_metrics.csv')
    for fold in protocols:
        for room,g in ops[fold['job_id']].groupby('Room_ID',sort=True):
            for method in list(cfg['references'])+V.METHODS:
                r=fm[(fm.protocol==fold['protocol'])&(fm.room==room)&(fm.method==method)];assert len(r)==1
                y=g.true__E_mean_lux.to_numpy();p=g['pred__'+method].to_numpy();raw=g['raw__'+method].to_numpy();ae=np.abs(p-y)
                close([mean_absolute_error(y,p),np.sqrt(mean_squared_error(y,p)),r2_score(y,p),np.mean(p-y),np.mean(ae<=50),np.quantile(ae,.9),np.mean(raw<0),mean_absolute_error(y,raw)],
                    r[['mae','rmse','r2','bias','within_50_lux','absolute_error_P90','raw_negative_fraction','raw_MAE']].iloc[0]);counters['outer_scores']+=1
    summary,gates,paired,decision=V.summarize(fm,cfg)
    for name,data in [('summary_metrics',summary),('development_gate',gates),('paired_room_comparison',paired)]:pd.testing.assert_frame_equal(pd.read_csv(out/(name+'.csv')),data,check_dtype=False,atol=1e-8,rtol=1e-10)
    assert D.read(out/'candidate_decision.json')==decision
    def errs(protocol,method):return fm[(fm.protocol==protocol)&(fm.method==method)].set_index('room').mae.sort_index()
    eps=cfg['gate']['numerical_tolerance']
    for r in gates.itertuples():
        a=errs('LeaveOneRoomOut',r.method);s=errs('LeaveOneRoomOut','v8_S1');c=errs('LeaveOneRoomOut','v9_C')
        oa=errs('LeaveOneOrientationOut',r.method);ot=errs('LeaveOneOrientationOut','v8_T1');os=errs('LeaveOneOrientationOut','v8_S1')
        flags=[a.mean()<50,a.BM432<=120+eps,a.LM608<=40+eps,a.max()<=c.max()+eps,(a<=s+eps).sum()>=5,oa.mean()<=ot.mean()+eps,oa.max()<=ot.max()+eps,oa.LM608<=os.LM608+eps]
        assert r.reaches_primary_targets==all(flags[:3]) and r.passes_all==all(flags)
        assert r.partial_progress==bool(a.mean()<s.mean() and a.BM432<=c.BM432+eps and a.LM608<=s.LM608+eps)
    strata=pd.read_csv(out/'stratified_metrics.csv')
    for (protocol,room),group in strata.groupby(['protocol','room'],sort=False):
        fold=next(f for f in protocols if f['protocol']==protocol and room in next(j['held_rooms'] for j in jobs if j['id']==f['job_id']))
        g=ops[fold['job_id']].query('Room_ID == @room').copy()
        dni=d[d.Room_ID.eq(room)].set_index(D.KEYS).loc[pd.MultiIndex.from_frame(g[D.KEYS]),'EPW_DNI_Wh_m2'].to_numpy()
        g['DNI_bin']=pd.cut(dni,[-1,0,200,500,np.inf],labels=['0','(0,200]','(200,500]','>500']).astype(str)
        g['true_lux_bin']=pd.cut(g.true__E_mean_lux,[-1,100,300,1000,np.inf],labels=['<=100','(100,300]','(300,1000]','>1000']).astype(str)
        g['midday']=np.where(g.Hour.between(10,14),'10-14','other_hours')
        for r in group.itertuples():
            h=g[g[r.dimension].astype(str)==str(r.value)];assert len(h)==r.n;close(mean_absolute_error(h.true__E_mean_lux,h['pred__'+r.method]),r.mae);counters['strata']+=1
    assert counters['inner_scores']==400 and counters['selections']==36 and counters['preprocessing']==36 and counters['fit_hashes']==252 and counters['outer_scores']==98
    assert run['total_estimator_fits']==252 and all(sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    assert run['TEST_numeric_rows']==run['TEST_fit_rows']==run['TEST_predictions']==run['TEST_scores']==0
    result={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-tick,'checks':counters,'saved_model_bundles_reloaded':9,
        'preprocessing_refits_for_verification_only':36,'new_estimator_fits':0,'TEST_numeric_rows':0,'TEST_predictions':0,'TEST_scores':0,
        'references_exactly_reproduced':True,'sample_weights_all_none_verified':True,'input_source_gate_rechecked':True,'frozen_inputs_and_trial_model_unchanged':True,
        'scope':'Model reload inference and numerical procedure verified; not a second training run, source-physics verification or independent test.'}
    D.dump(out/'verification.json',result);run['status']='completed_verified';run['verification']='reports/experiment_v10/verification.json';D.dump(out/'run_manifest.json',run);print(result,flush=True)


if __name__=='__main__':main()
