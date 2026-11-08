"""Verify v9 lineage, inner selection, saved inference, metrics and gates; no fits."""
from datetime import datetime,timezone
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from threadpoolctl import threadpool_limits
import mean_exposure_v9 as V
from train_mean_exposure_v9 import idx,sha

D=V.D;ROOT=V.ROOT


def close(a,b):np.testing.assert_allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=1e-10,atol=1e-8,equal_nan=True)


def main():
    started=time.perf_counter();cfg=D.read(V.CONFIG);out=ROOT/cfg['output'];run=D.read(out/'run_manifest.json')
    assert run['status'] in ['completed_pending_verification','completed_verified']
    assert all(sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    d,audit,profiles=V.load(cfg);assert D.read(out/'input_source_audit.json')==audit
    assert run['TEST_numeric_rows']==run['TEST_fit_rows']==run['TEST_predictions']==run['TEST_scores']==0
    x,q=V.features(d,cfg)
    independent_q=d.Glass_Area_m2.to_numpy()*d[V.ANNUAL].to_numpy()/(100*d.Room_Area_m2.to_numpy())
    close(q,independent_q)
    close(x['F2']['q_DHI'],independent_q*d.EPW_DHI_Wh_m2)
    close(x['F2']['q_horizontal_DNI'],independent_q*d.EPW_DNI_Wh_m2*np.cos(d.Solar_Altitude_mid_deg*np.pi/180))
    assert not set(x['F2'].columns)&{'Room_ID',V.TARGET,'Sample_Role'}
    jobs,protocols=D.splits(d,cfg)
    assert D.read(out/'fold_splits.json')=={'jobs':jobs,'protocols':protocols}
    assert D.read(ROOT/cfg['reference_experiment']/'fold_splits.json')=={'jobs':jobs,'protocols':protocols}
    inner=pd.read_csv(out/'inner_room_metrics.csv');scores=pd.read_csv(out/'inner_candidate_scores.csv')
    selections=D.read(out/'selections.json');fitlog=D.read(out/'fit_log.json');index={r['job_id']:r for r in D.read(out/'model_index.json')}
    params=D.read(out/'parameters.json');assert params['objective']=='reg:absoluteerror'
    assert len(fitlog)==189 and len(inner)==300 and len(scores)==54 and len(selections)==27
    checked_inner=0;checked_selections=0;checked_fits=0;checked_values=0;ops={}
    with threadpool_limits(limits=cfg['thread_limit']):
        for job in jobs:
            jid=job['id'];ti,vi=idx(d,job['fit_rooms']),idx(d,job['held_rooms'])
            ip=pd.read_parquet(out/'inner_predictions'/(jid+'.parquet'))
            pd.testing.assert_frame_equal(ip[D.META],d.loc[ti,D.META].reset_index(drop=True))
            close(ip.true__E_mean_lux,d.loc[ti,V.TARGET]);assert not ip.isna().any().any()
            chosen={}
            for method in V.METHODS:
                options=[]
                for depth in cfg['depths']:
                    maes=[]
                    for room,g in ip.groupby('Room_ID',sort=True):
                        score=mean_absolute_error(g.true__E_mean_lux,g[f'pred__{method}__depth{depth}'])
                        stored=inner[(inner.job_id==jid)&(inner.method==method)&(inner.depth==depth)&(inner.room==room)]
                        assert len(stored)==1;close(score,stored.mae.iloc[0]);maes.append(score);checked_inner+=1
                    record=scores[(scores.job_id==jid)&(scores.method==method)&(scores.depth==depth)]
                    assert len(record)==1;close([np.mean(maes),np.max(maes)],record[['macro_MAE','worst_MAE']].iloc[0])
                    options.append((np.mean(maes),depth))
                best=min(options);chosen[method]=best[1]
                selected=[r for r in selections if r['job_id']==jid and r['method']==method]
                assert len(selected)==1 and selected[0]['depth']==best[1];checked_selections+=1
            freeze=D.read(out/'selection_freezes'/(jid+'.json'))
            assert freeze['selected']==chosen and not freeze['outer_predictions_computed'] and not freeze['outer_labels_used_for_selection']
            for r in [r for r in fitlog if r['job_id']==jid]:
                if r['phase']=='inner':split=next(g for g in job['inner'] if g['fold']==r['inner_fold'])
                else:split=job
                assert r['fit_rooms']==split['fit_rooms'] and r['held_rooms']==split['held_rooms']
                assert not set(r['fit_rooms'])&(set(r['held_rooms'])|set(job['held_rooms'])|set(cfg['excluded_test_rooms']))
                a=idx(d,r['fit_rooms']);method=r['method'];yy=d.loc[a,V.TARGET].to_numpy();ww=None
                if method=='C':yy=yy/q[a];ww=q[a]/np.mean(q[a]);close(r['q_fit_mean'],np.mean(q[a]))
                assert r['fit_rows']==len(a) and r['label_sha256']==D.array_sha(yy)
                assert r['input_sha256']==D.array_sha(x[cfg['methods'][method]['features']].loc[a])
                assert r['weight_sha256']==(D.array_sha(ww) if ww is not None else None)
                if ww is not None:
                    close(ww.mean(),1)
                    # Algebraic loss identity, unrelated to observed outer errors.
                    arbitrary=np.full(len(a),17.0)
                    close(ww*np.abs(yy-arbitrary),np.abs(d.loc[a,V.TARGET].to_numpy()-q[a]*arbitrary)/np.mean(q[a]))
                checked_fits+=1
            record=index[jid];path=ROOT/record['path'];assert sha(path)==record['sha256'];bundle=joblib.load(path)
            assert bundle['job']==job and bundle['selected']==chosen and bundle['raw_inputs']==V.RAW
            for method,est in bundle['estimators'].items():
                for key,value in (params|{'max_depth':chosen[method]}).items():assert est.get_params()[key]==value
            op=pd.read_parquet(out/'outer_predictions'/(jid+'.parquet'))
            pd.testing.assert_frame_equal(op[D.META],d.loc[vi,D.META].reset_index(drop=True));close(op.q,q[vi]);close(op.true__E_mean_lux,d.loc[vi,V.TARGET])
            predictions=V.predict_bundle(bundle,d.loc[vi,V.RAW],cfg)
            for method,(pred,raw) in predictions.items():
                close(pred,op['pred__'+method]);close(raw,op['raw__'+method]);assert np.isfinite(raw).all() and (pred>=0).all();checked_values+=len(pred)
            old=pd.read_parquet(ROOT/cfg['reference_experiment']/'outer_predictions'/(jid+'.parquet'))
            pd.testing.assert_frame_equal(old[D.META],op[D.META])
            for method in cfg['references']:
                for kind in ['pred','raw']:np.testing.assert_array_equal(op[kind+'__'+method],old[kind+'__'+method+'__E_mean_lux'])
            ops[jid]=op
            print(f'VERIFIED {jid}: weights/labels, depth selection, reloaded predictions, references',flush=True)
    assert checked_inner==300 and checked_selections==27 and checked_fits==189
    fm=pd.read_csv(out/'fold_metrics.csv');checked_outer=0
    for fold in protocols:
        for room,g in ops[fold['job_id']].groupby('Room_ID',sort=True):
            for method in cfg['references']+V.METHODS:
                r=fm[(fm.protocol==fold['protocol'])&(fm.room==room)&(fm.method==method)]
                assert len(r)==1;y=g.true__E_mean_lux.to_numpy();p=g['pred__'+method].to_numpy();ae=np.abs(y-p)
                close([mean_absolute_error(y,p),np.sqrt(mean_squared_error(y,p)),r2_score(y,p),np.mean(p-y),np.mean(ae<=50),np.quantile(ae,.9)],
                      r[['mae','rmse','r2','bias','within_50_lux','absolute_error_P90']].iloc[0])
                checked_outer+=1
    summary,gates,paired,decision=V.summarize(fm,cfg)
    for name,data in [('summary_metrics',summary),('development_gate',gates),('paired_room_comparison',paired)]:
        pd.testing.assert_frame_equal(pd.read_csv(out/(name+'.csv')),data,check_dtype=False,atol=1e-8,rtol=1e-10)
    assert D.read(out/'candidate_decision.json')==decision
    eps=cfg['gate']['numerical_tolerance']
    for r in gates.itertuples():
        def errors(protocol,method):return fm[(fm.protocol==protocol)&(fm.method==method)].set_index('room').mae.sort_index()
        a=errors('LeaveOneRoomOut',r.method);t=errors('LeaveOneRoomOut','T1');s=errors('LeaveOneRoomOut','S1')
        oa=errors('LeaveOneOrientationOut',r.method);ot=errors('LeaveOneOrientationOut','T1')
        conditions=[a.mean()<50,a.BM432<=120+eps,a.max()<=s.max()+eps,(a<=t+eps).sum()>=5,oa.mean()<=ot.mean()+eps,oa.max()<=ot.max()+eps]
        assert r.passes_all==all(conditions) and r.reaches_primary_targets==all(conditions[:2])
    assert checked_outer==84 and run['total_estimator_fits']==189
    # Check all saved stratum summaries against the recorded outer predictions.
    strata=pd.read_csv(out/'stratified_metrics.csv');stratum_checks=0
    for (protocol,room),group in strata.groupby(['protocol','room'],sort=False):
        fold=next(f for f in protocols if f['protocol']==protocol and room in next(j['held_rooms'] for j in jobs if j['id']==f['job_id']))
        g=ops[fold['job_id']].query('Room_ID == @room').copy()
        weather=d[d.Room_ID.eq(room)].set_index(D.KEYS).loc[pd.MultiIndex.from_frame(g[D.KEYS]),'EPW_DNI_Wh_m2'].to_numpy()
        g['DNI_bin']=pd.cut(weather,[-1,0,200,500,np.inf],labels=['0','(0,200]','(200,500]','>500']).astype(str)
        g['true_lux_bin']=pd.cut(g.true__E_mean_lux,[-1,100,300,1000,np.inf],labels=['<=100','(100,300]','(300,1000]','>1000']).astype(str)
        g['midday']=np.where(g.Hour.between(10,14),'10-14','other_hours')
        for r in group.itertuples():
            h=g[g[r.dimension].astype(str)==str(r.value)];assert len(h)==r.n
            close(mean_absolute_error(h.true__E_mean_lux,h['pred__'+r.method]),r.mae);stratum_checks+=1
    assert all(sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    result={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-started,
        'independent_inner_room_MAE_checks':checked_inner,'independent_depth_selection_checks':checked_selections,'fit_input_label_weight_hash_checks':checked_fits,
        'saved_model_bundles_reloaded':9,'saved_prediction_values_reproduced':checked_values,'independent_outer_metric_records':checked_outer,
        'stratified_MAE_checks':stratum_checks,'reference_predictions_exactly_reproduced':True,'normalization_loss_identity_verified':True,
        'source_gate_rechecked':True,'frozen_inputs_and_trial_model_unchanged':True,'new_estimator_fits':0,'TEST_numeric_rows':0,'TEST_fit_rows':0,'TEST_predictions':0,'TEST_scores':0,
        'scope':'Saved inference, selections, numerical metrics and lineage verified; not a second training run or upstream physical-source verification.'}
    D.dump(out/'verification.json',result);run['status']='completed_verified';run['verification']='reports/experiment_v9/verification.json';D.dump(out/'run_manifest.json',run)
    print(result,flush=True)


if __name__=='__main__':main()
