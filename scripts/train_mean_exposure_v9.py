"""Approved A/B/C XGBoost mean-illumination experiment, maximum 189 fits."""
from datetime import datetime,timezone
import importlib.metadata
import os
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
import mean_exposure_v9 as V
from train_directional_v8 import idx

D=V.D;ROOT=V.ROOT
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return D.base.sha256(Path(p))


def evaluate(d,cfg,out,protocols):
    records=[];strata=[]
    source=d.set_index(D.KEYS)
    for fold in protocols:
        op=pd.read_parquet(out/'outer_predictions'/(fold['job_id']+'.parquet'))
        dni=source.loc[pd.MultiIndex.from_frame(op[D.KEYS]),'EPW_DNI_Wh_m2'].to_numpy()
        op['DNI_bin']=pd.cut(dni,[-1,0,200,500,np.inf],labels=['0','(0,200]','(200,500]','>500']).astype(str)
        op['true_lux_bin']=pd.cut(op.true__E_mean_lux,[-1,100,300,1000,np.inf],labels=['<=100','(100,300]','(300,1000]','>1000']).astype(str)
        op['midday']=np.where(op.Hour.between(10,14),'10-14','other_hours')
        for room,g in op.groupby('Room_ID',sort=True):
            for method in cfg['references']+V.METHODS:
                meta={'protocol':fold['protocol'],'fold':fold['fold'],'job_id':fold['job_id'],'room':room,'method':method}
                records.append(meta|V.metrics(g.true__E_mean_lux,g['pred__'+method],g['raw__'+method]))
                for dimension in ['DNI_bin','true_lux_bin','Hour','PVSD_Angle_deg','midday']:
                    for value,h in g.groupby(dimension,sort=True,observed=True):
                        strata.append(meta|{'dimension':dimension,'value':str(value)}|V.metrics(h.true__E_mean_lux,h['pred__'+method],h['raw__'+method]))
    fm=pd.DataFrame(records);assert len(fm)==84
    fm.to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(strata).to_csv(out/'stratified_metrics.csv',index=False)
    summary,gates,paired,decision=V.summarize(fm,cfg)
    for name,data in [('summary_metrics',summary),('development_gate',gates),('paired_room_comparison',paired)]:data.to_csv(out/(name+'.csv'),index=False)
    D.dump(out/'candidate_decision.json',decision)


def main():
    started=time.perf_counter();cfg=D.read(V.CONFIG);out=ROOT/cfg['output'];models=ROOT/cfg['model_output']
    for p in [out,models,out/'inner_predictions',out/'outer_predictions',out/'selection_freezes']:p.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():raise RuntimeError('Existing v9 run; refusing overwrite')
    assert Path(sys.prefix).resolve()==(ROOT/'.conda/envs/pvsd-ml-v1').resolve()
    assert sha(ROOT/cfg['source'])==cfg['source_sha256'] and sha(ROOT/cfg['bim_source'])==cfg['bim_sha256']
    d,audit,profiles=V.load(cfg);D.dump(out/'input_source_audit.json',audit);profiles.to_csv(out/'development_profiles.csv')
    x,q=V.features(d,cfg);jobs,protocols=D.splits(d,cfg)
    oldsplits=D.read(ROOT/cfg['reference_experiment']/'fold_splits.json')
    assert oldsplits=={'jobs':jobs,'protocols':protocols}
    oldrun=D.read(ROOT/cfg['reference_experiment']/'run_manifest.json');assert oldrun['status']=='completed_verified'
    assert oldrun['input_hashes'][cfg['source']]==cfg['source_sha256']
    params=D.read(ROOT/cfg['parameter_source'])['models'][V.TARGET]['params'].copy();params['objective']=cfg['objective']
    inputs=[V.CONFIG,Path(__file__),ROOT/'scripts/mean_exposure_v9.py',ROOT/'scripts/directional_v8.py',ROOT/'scripts/train_logo_v1.py',ROOT/'scripts/train_directional_v8.py',ROOT/'pvsd_trial.py',
        ROOT/cfg['source'],ROOT/cfg['bim_source'],ROOT/'data/raw/SEND_README.txt',ROOT/cfg['parameter_source'],ROOT/'configs/model_plan.json',
        ROOT/'docs/approvals/mean_exposure_v9_approval.json',ROOT/'docs/next_experiment_BM432_mean.md',ROOT/cfg['reference_experiment']/'fold_splits.json',
        ROOT/cfg['reference_experiment']/'run_manifest.json',ROOT/cfg['reference_experiment']/'verification.json']
    inputs += [ROOT/cfg['reference_experiment']/'outer_predictions'/(j['id']+'.parquet') for j in jobs]
    inputs += [p for p in (ROOT/'models/trial_v0_1').rglob('*') if p.is_file()]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}
    run={'experiment_id':cfg['experiment_id'],'status':'running','started_utc':now(),'pid':os.getpid(),'python':sys.version,'conda_prefix':sys.prefix,
        'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','xgboost','joblib','pyarrow']},'input_hashes':hashes,
        'TRAIN_rows':len(d),'TRAIN_rooms':sorted(d.Room_ID.unique()),'TEST_numeric_rows':0,'TEST_rows_skipped':52560,'TEST_fit_rows':0,'TEST_predictions':0,'TEST_scores':0,
        'new_simulations':0,'visualizations':0,'full_seven_refit':False,'scope':cfg['scope']}
    for name,value in [('run_manifest.json',run),('config_snapshot.json',cfg),('parameters.json',params),('fold_splits.json',{'jobs':jobs,'protocols':protocols}),('feature_names.json',{k:list(v.columns) for k,v in x.items()})]:D.dump(out/name,value)
    fitlog=[];innerrooms=[];innerscores=[];selections=[];modelindex=[];counts={'inner':0,'outer':0}
    print('Input source gate passed: TRAIN=122640, TEST skipped=52560. Frozen 3 methods x 2 depths, 189 fits.',flush=True)
    with threadpool_limits(limits=cfg['thread_limit']):
        for number,job in enumerate(jobs,1):
            jid=job['id'];ti,vi=idx(d,job['fit_rooms']),idx(d,job['held_rooms'])
            print(f'OUTER {number}/9 {jid}',flush=True)
            ip=d.loc[ti,D.META].copy();ip['true__E_mean_lux']=d.loc[ti,V.TARGET];ip['inner_fold']=-1
            for inner in job['inner']:
                a,b=idx(d,inner['fit_rooms']),idx(d,inner['held_rooms']);ip.loc[b,'inner_fold']=inner['fold']
                for method in V.METHODS:
                    y,w=V.training_arrays(d,q,a,method);xx=x[cfg['methods'][method]['features']]
                    for depth in cfg['depths']:
                        t=time.perf_counter();est=XGBRegressor(**(params|{'max_depth':depth}))
                        est.fit(xx.loc[a],y,sample_weight=w)
                        p,_=V.predict(est,xx.loc[b],q[b],method);ip.loc[b,f'pred__{method}__depth{depth}']=p
                        counts['inner']+=1
                        fitlog.append({'job_id':jid,'phase':'inner','inner_fold':inner['fold'],'method':method,'depth':depth,'fit_rooms':inner['fit_rooms'],'held_rooms':inner['held_rooms'],
                            'fit_rows':len(a),'input_sha256':D.array_sha(xx.loc[a]),'label_sha256':D.array_sha(y),'weight_sha256':D.array_sha(w) if w is not None else None,
                            'q_fit_mean':float(q[a].mean()) if method=='C' else None,'seconds':time.perf_counter()-t})
            assert not ip.isna().any().any() and ip.inner_fold.gt(0).all()
            ip.to_parquet(out/'inner_predictions'/(jid+'.parquet'),index=False)
            chosen={}
            for method in V.METHODS:
                choices=[]
                for depth in cfg['depths']:
                    errors=[]
                    for room,g in ip.groupby('Room_ID',sort=True):
                        m=V.metrics(g.true__E_mean_lux,g[f'pred__{method}__depth{depth}']);errors.append(m['mae'])
                        innerrooms.append({'job_id':jid,'method':method,'depth':depth,'room':room,'inner_fold':int(g.inner_fold.iloc[0])}|m)
                    score={'job_id':jid,'method':method,'depth':depth,'macro_MAE':float(np.mean(errors)),'worst_MAE':float(np.max(errors))}
                    innerscores.append(score);choices.append(score)
                best=min(choices,key=lambda r:(r['macro_MAE'],r['depth']));chosen[method]=best['depth'];selections.append(best|{'selected_utc':now()})
            D.dump(out/'selection_freezes'/(jid+'.json'),{'job_id':jid,'selected':chosen,'selected_utc':now(),'outer_predictions_computed':False,'outer_labels_used_for_selection':False})
            bundle={'job':job,'raw_inputs':V.RAW,'selected':chosen,'estimators':{}}
            for method in V.METHODS:
                a=ti;y,w=V.training_arrays(d,q,a,method);xx=x[cfg['methods'][method]['features']];t=time.perf_counter()
                est=XGBRegressor(**(params|{'max_depth':chosen[method]}));est.fit(xx.loc[a],y,sample_weight=w);bundle['estimators'][method]=est;counts['outer']+=1
                fitlog.append({'job_id':jid,'phase':'outer','inner_fold':None,'method':method,'depth':chosen[method],'fit_rooms':job['fit_rooms'],'held_rooms':job['held_rooms'],
                    'fit_rows':len(a),'input_sha256':D.array_sha(xx.loc[a]),'label_sha256':D.array_sha(y),'weight_sha256':D.array_sha(w) if w is not None else None,
                    'q_fit_mean':float(q[a].mean()) if method=='C' else None,'seconds':time.perf_counter()-t})
            path=models/(jid+'.joblib');joblib.dump(bundle,path,compress=3)
            predictions=V.predict_bundle(bundle,d.loc[vi,V.RAW],cfg)
            op=d.loc[vi,D.META].copy();op['true__E_mean_lux']=d.loc[vi,V.TARGET];op['q']=q[vi]
            for method,(p,raw) in predictions.items():op['pred__'+method]=p;op['raw__'+method]=raw
            old=pd.read_parquet(ROOT/cfg['reference_experiment']/'outer_predictions'/(jid+'.parquet'))
            pd.testing.assert_frame_equal(old[D.META],op[D.META].reset_index(drop=True))
            np.testing.assert_array_equal(old.true__E_mean_lux,op.true__E_mean_lux)
            for reference in cfg['references']:
                for kind in ['pred','raw']:op[kind+'__'+reference]=old[kind+'__'+reference+'__E_mean_lux'].to_numpy()
            op.to_parquet(out/'outer_predictions'/(jid+'.parquet'),index=False)
            modelindex.append({'job_id':jid,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'fit_rooms':job['fit_rooms'],'held_rooms':job['held_rooms']})
            for name,value in [('fit_log.json',fitlog),('selections.json',selections),('model_index.json',modelindex)]:D.dump(out/name,value)
            pd.DataFrame(innerrooms).to_csv(out/'inner_room_metrics.csv',index=False);pd.DataFrame(innerscores).to_csv(out/'inner_candidate_scores.csv',index=False)
            run.update({'completed_jobs':number,'fit_counts':counts.copy(),'elapsed_seconds':time.perf_counter()-started});D.dump(out/'run_manifest.json',run)
            print(f'FINISHED {jid}: depths {chosen}, fits {counts}, elapsed {time.perf_counter()-started:.1f}s',flush=True)
    assert counts=={'inner':162,'outer':27} and len(innerrooms)==300 and len(innerscores)==54 and len(selections)==27
    evaluate(d,cfg,out,protocols)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    run.update({'status':'completed_pending_verification','completed_utc':now(),'elapsed_seconds':time.perf_counter()-started,'total_estimator_fits':189,'protected_inputs_unchanged':True})
    D.dump(out/'run_manifest.json',run)
    print('COMPLETE: 189 fits; no TEST operations. Verification pending.',flush=True)


if __name__=='__main__':main()
