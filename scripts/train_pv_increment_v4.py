"""Approved 49-fit PV increment and grouped threshold pilot. TRAIN only."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'scripts/train_logo_v1.py'),run_name='v4_shared')
CONFIG=ROOT/'configs/experiment_v4.json'
PV='PV_AC_kWh'
KEYS=['Room_ID','HOY_0based','PVSD_Angle_deg']
META=['Room_ID','Month','Day','Hour','DOY_1based','HOY_0based','PVSD_Angle_deg']
TARGETS=['E_mean_lux','E_P10_lux','F_300_fraction','F_UDI_fraction',PV]

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float64).tobytes()).hexdigest()

def baseline_values(frame,values):
    """Pair supplied values strictly by Room_ID + HOY, preserving row order."""
    z=frame.PVSD_Angle_deg.to_numpy()==0
    ref=pd.Series(np.asarray(values)[z],index=pd.MultiIndex.from_frame(frame.loc[z,['Room_ID','HOY_0based']]))
    assert ref.index.is_unique and len(ref)*4==len(frame)
    result=ref.reindex(pd.MultiIndex.from_frame(frame[['Room_ID','HOY_0based']])).to_numpy()
    assert np.isfinite(result).all()
    return result

def gains_from_absolute(frame,values):return np.asarray(values)-baseline_values(frame,values)

def predict_delta(estimator,x,angles):
    """Inference receives predictors and angles only, with a known zero anchor."""
    nonzero=np.asarray(angles)!=0
    result=np.zeros(len(x));result[nonzero]=estimator.predict(x.loc[nonzero])
    assert np.isfinite(result).all()
    return result

def choices(gain_matrix,threshold):
    assert gain_matrix.ndim==2 and gain_matrix.shape[1]==4
    assert np.all(gain_matrix[:,0]==0)
    index=np.argmax(gain_matrix,axis=1)
    value=gain_matrix[np.arange(len(index)),index]
    margin=np.inf if threshold is None else threshold
    return np.where(value>margin,index,0)

def sorted_room(frame,room,gain_column):
    g=frame[frame.Room_ID.eq(room)].sort_values(['HOY_0based','PVSD_Angle_deg'])
    assert len(g)==17520
    angles=g.PVSD_Angle_deg.to_numpy().reshape(4380,4)
    assert np.array_equal(angles,np.tile([0,30,60,90],(4380,1)))
    return g,g[gain_column].to_numpy().reshape(4380,4),g['true_PV'].to_numpy().reshape(4380,4)

def threshold_scores(frame,variant,grid,outer_fold,held_room):
    records=[]
    for t in grid:
        for room in sorted(frame.Room_ID.unique()):
            g,gm,y=sorted_room(frame,room,'pred_'+variant+'_gain')
            k=choices(gm,t['kWh']);actual=y[np.arange(4380),k];gain=actual-y[:,0]
            records.append({'outer_fold':outer_fold,'outer_held_room':held_room,'variant':variant,'threshold_label':t['label'],'threshold_kWh':t['kWh'],
                'validation_room':room,'inner_fold':int(g.inner_fold.unique().item()),'net_gain_kWh':float(gain.sum()),'PV_kWh':float(actual.sum()),'switch_fraction':float(np.mean(k!=0))})
    f=pd.DataFrame(records)
    summaries=[]
    for t in grid:
        v=f.loc[f.threshold_label.eq(t['label']),'net_gain_kWh']
        assert len(v)==6
        summaries.append({'outer_fold':outer_fold,'outer_held_room':held_room,'variant':variant,'threshold_label':t['label'],'threshold_kWh':t['kWh'],
                          'mean_room_gain_kWh':float(v.mean()),'median_room_gain_kWh':float(v.median()),'min_room_gain_kWh':float(v.min())})
    best=max(r['mean_room_gain_kWh'] for r in summaries)
    chosen=[r for r in summaries if abs(r['mean_room_gain_kWh']-best)<=1e-12][-1]
    assert chosen['mean_room_gain_kWh']>=-1e-12
    return records,summaries,chosen

def parameters(cfg3,candidate):
    c=next(r for r in cfg3['families']['XGBoost']['candidates'] if r['id']==candidate)
    return cfg3['families']['XGBoost']['common']|{k:v for k,v in c.items() if k!='id'}

def main():
    started=time.perf_counter();cfg=json.loads(CONFIG.read_text());plan=json.loads((ROOT/'configs/model_plan.json').read_text());cfg3=json.loads((ROOT/'configs/experiment_v3.json').read_text())
    out=ROOT/cfg['output'];model_dir=ROOT/cfg['model_output']
    for p in [out,out/'predictions',out/'figures',model_dir]:p.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():raise RuntimeError('V4 has already started; preserve experiment evidence and use a new version.')
    assert Path(sys.prefix).resolve()==(ROOT/'.conda/envs/pvsd-ml-v1').resolve()
    paths=[ROOT/cfg['source'],CONFIG,ROOT/cfg['source_proposal'],ROOT/'configs/experiment_v3.json',ROOT/'reports/experiment_v3/selections.csv',ROOT/'reports/experiment_v3/fold_splits.json',
           ROOT/'reports/experiment_v3/predictions/Frozen_HistGB.parquet',ROOT/'reports/experiment_v3/predictions/XGBoost.parquet']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert sha(ROOT/cfg['source'])==json.loads((ROOT/'reports/experiment_v3/run_manifest.json').read_text())['source_sha256']
    d,skipped=BASE['load_train'](ROOT/cfg['source'],plan);raw=plan['feature_sets'][cfg['feature_set']];x=BASE['encode_inputs'](d,raw)
    assert len(raw)==13 and x.shape[1]==15 and not set(raw)&set(TARGETS)
    frozen=pd.read_parquet(ROOT/'reports/experiment_v3/predictions/Frozen_HistGB.parquet');absolute=pd.read_parquet(ROOT/'reports/experiment_v3/predictions/XGBoost.parquet')
    for ref in [frozen,absolute]:
        assert ref[KEYS].equals(d[KEYS])
        for t in TARGETS:assert np.array_equal(ref['true__'+t],d[t])
    selection3=pd.read_csv(ROOT/'reports/experiment_v3/selections.csv');splits3=json.loads((ROOT/'reports/experiment_v3/fold_splits.json').read_text())
    run={'experiment_id':cfg['experiment_id'],'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'config':cfg,
        'input_hashes':hashes,'script_sha256':sha(Path(__file__)),'python_executable':sys.executable,'conda_prefix':sys.prefix,
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','joblib','pyarrow','threadpoolctl']},
        'train_rows':len(d),'train_rooms':sorted(d.Room_ID.unique().tolist()),'TEST_rows_discarded_before_numeric_conversion':skipped,'TEST_predictions':0,'TEST_scores':0,
        'scope':cfg['scope'],'parameter_history':'Per-outer v3 parameters were selected inside the same six fitting rooms; thresholds reuse that development set. No outer held-room labels enter either selection. This is not an independent calibration sample.'}
    dump(out/'run_manifest.json',run);dump(out/'config_snapshot.json',cfg);dump(out/'feature_names.json',{'raw':raw,'encoded':list(x.columns)})
    outer=d[META].copy();outer['fold']=-1
    for t in TARGETS:outer['true__'+t]=d[t]
    outer['true_delta_PV']=np.nan;outer['pred_frozen_HistGB_PV']=frozen['pred__'+PV];outer['pred_absolute_XGBoost_PV']=absolute['pred__'+PV]
    outer['pred_absolute_gain']=gains_from_absolute(d,absolute['pred__'+PV].to_numpy());outer['pred_frozen_HistGB_gain']=gains_from_absolute(d,frozen['pred__'+PV].to_numpy());outer['pred_delta_gain']=np.nan
    threshold_rows=[];threshold_summary=[];selections=[];model_index=[];splits=[];counts={'inner_absolute':0,'inner_delta':0,'outer_delta':0};provenance=[]
    with threadpool_limits(limits=cfg['thread_limit']):
        for fold,(ti,vi) in enumerate(LeaveOneGroupOut().split(d,groups=d.Room_ID),1):
            held=d.iloc[vi].Room_ID.unique().item();fit_rooms=sorted(d.iloc[ti].Room_ID.unique().tolist());candidate=cfg['outer_frozen_XGBoost_candidates'][held]
            chosen3=selection3[(selection3.held_room==held)&(selection3.method=='XGBoost')&(selection3.target==PV)].iloc[0]
            assert chosen3.selected_candidate==candidate and int(chosen3.outer_fold)==fold
            params=parameters(cfg3,candidate);source=ROOT/f'models/experiment_v3/XGBoost/fold_{fold:02d}_{held}.joblib';bundle=joblib.load(source)
            assert bundle['train_rooms']==fit_rooms and bundle['held_room']==held and bundle['selected_candidates'][PV]==candidate
            old_est=bundle['estimators'][candidate+'__'+PV]
            for k,v in params.items():assert old_est.get_params()[k]==v
            sample=vi[np.linspace(0,len(vi)-1,17,dtype=int)]
            assert np.allclose(np.maximum(old_est.predict(x.iloc[sample]),0),absolute.loc[sample,'pred__'+PV],rtol=1e-9,atol=1e-9)
            source_hash=sha(source);provenance.append({'outer_fold':fold,'held_room':held,'source_bundle':str(source.relative_to(ROOT)),'sha256':source_hash,'candidate':candidate,'parameters':params})
            run['input_hashes'][str(source.relative_to(ROOT))]=source_hash
            train=d.iloc[ti].reset_index(drop=True);xt=x.iloc[ti].reset_index(drop=True);inner=train[META].copy();inner['true_PV']=train[PV];inner['inner_fold']=-1
            for c in ['raw_absolute_PV','pred_absolute_PV','pred_absolute_gain','pred_delta_gain']:inner[c]=np.nan
            outer_split={'outer_fold':fold,'held_room':held,'training_rooms':fit_rooms,'candidate':candidate,'inner':[]}
            print(f'OUTER {fold}/7 {held}: frozen {candidate}; fitting and threshold calibration use {fit_rooms}',flush=True)
            for ins,(a,b) in enumerate(GroupKFold(n_splits=3).split(train,groups=train.Room_ID),1):
                fit_set=sorted(train.iloc[a].Room_ID.unique().tolist());valid_set=sorted(train.iloc[b].Room_ID.unique().tolist())
                assert len(fit_set)==4 and len(valid_set)==2 and not set(fit_set)&set(valid_set) and held not in fit_set+valid_set
                outer_split['inner'].append({'inner_fold':ins,'fit_rooms':fit_set,'validation_rooms':valid_set,'n_absolute_fit':len(a),'n_delta_fit':int(train.iloc[a].PVSD_Angle_deg.ne(0).sum()),'n_validation':len(b)})
                inner.loc[b,'inner_fold']=ins
                for variant in ['absolute','delta']:
                    fit_frame=train.iloc[a];yf=fit_frame[PV].to_numpy();xx=xt.iloc[a]
                    if variant=='delta':
                        yf=yf-baseline_values(fit_frame,yf);nonzero=fit_frame.PVSD_Angle_deg.ne(0).to_numpy();yf=yf[nonzero];xx=xx.loc[nonzero]
                        assert len(yf)==52560 and np.any(yf<0)
                    est=XGBRegressor(**params);est.fit(xx,yf);counts['inner_'+variant]+=1
                    if variant=='absolute':
                        pr=est.predict(xt.iloc[b]);bounded=np.maximum(pr,0)
                        inner.loc[b,'raw_absolute_PV']=pr;inner.loc[b,'pred_absolute_PV']=bounded
                        inner.loc[b,'pred_absolute_gain']=gains_from_absolute(train.iloc[b],bounded)
                    else:inner.loc[b,'pred_delta_gain']=predict_delta(est,xt.iloc[b],train.iloc[b].PVSD_Angle_deg.to_numpy())
                    directory=model_dir/('inner_'+variant);directory.mkdir(exist_ok=True);path=directory/f'outer_{fold:02d}_{held}__inner_{ins}.joblib'
                    bb={'estimator':est,'variant':variant,'outer_fold':fold,'inner_fold':ins,'held_room':held,'fit_rooms':fit_set,'validation_rooms':valid_set,'candidate':candidate,'parameters':params,
                        'raw_features':raw,'encoded_features':list(x.columns),'n_fit':len(yf),'fit_label_sha256':array_sha(yf)}
                    joblib.dump(bb,path,compress=3);model_index.append({k:v for k,v in bb.items() if k not in ['estimator','parameters','raw_features','encoded_features']}|{'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
                print(f'  inner {ins}/3 complete; fits={sum(counts.values())}',flush=True)
            assert inner.inner_fold.gt(0).all() and np.isfinite(inner.select_dtypes('number')).all().all()
            oldsplit=splits3[fold-1]
            assert oldsplit['held_room']==held and oldsplit['training_rooms']==fit_rooms
            for n,o in zip(outer_split['inner'],oldsplit['inner']):assert n['fit_rooms']==o['fit_rooms'] and n['validation_rooms']==o['validation_rooms']
            inner.to_parquet(out/'predictions'/f'inner_OOF_outer_{fold:02d}_{held}.parquet',index=False,compression='zstd')
            for variant in ['absolute','delta']:
                rr,ss,choice=threshold_scores(inner,variant,cfg['thresholds'],fold,held);threshold_rows+=rr;threshold_summary+=ss
                selections.append(choice|{'candidate':candidate,'selection_rooms':fit_rooms})
                print(f"  {variant} selected {choice['threshold_label']}; inner mean gain {choice['mean_room_gain_kWh']:.6f} kWh",flush=True)
            # Choices are locked above, before outer labels are used for any scoring.
            yf=train[PV].to_numpy();yf=yf-baseline_values(train,yf);nonzero=train.PVSD_Angle_deg.ne(0).to_numpy();yf=yf[nonzero]
            est=XGBRegressor(**params);est.fit(xt.loc[nonzero],yf);counts['outer_delta']+=1
            pr=predict_delta(est,x.iloc[vi],d.iloc[vi].PVSD_Angle_deg.to_numpy())
            outer.loc[vi,'pred_delta_gain']=pr;outer.loc[vi,'fold']=fold
            outer.loc[vi,'true_delta_PV']=d.iloc[vi][PV].to_numpy()-baseline_values(d.iloc[vi],d.iloc[vi][PV].to_numpy())
            directory=model_dir/'outer_delta';directory.mkdir(exist_ok=True);path=directory/f'fold_{fold:02d}_{held}.joblib'
            bb={'estimator':est,'variant':'delta','outer_fold':fold,'inner_fold':None,'held_room':held,'fit_rooms':fit_rooms,'validation_rooms':[held],'candidate':candidate,'parameters':params,
                'raw_features':raw,'encoded_features':list(x.columns),'n_fit':len(yf),'fit_label_sha256':array_sha(yf)}
            joblib.dump(bb,path,compress=3);model_index.append({k:v for k,v in bb.items() if k not in ['estimator','parameters','raw_features','encoded_features']}|{'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
            splits.append(outer_split);dump(out/'fold_splits.json',splits);dump(out/'threshold_selections.json',selections);dump(out/'model_index.json',model_index)
            pd.DataFrame(threshold_rows).to_csv(out/'inner_threshold_room_scores.partial.csv',index=False)
            pd.DataFrame(threshold_summary).to_csv(out/'inner_threshold_summary.partial.csv',index=False)
            dump(out/'progress.json',{'status':'running','completed_outer_folds':fold,'fits':counts,'elapsed_seconds':time.perf_counter()-started})
    assert counts=={'inner_absolute':21,'inner_delta':21,'outer_delta':7} and len(model_index)==49 and len(selections)==14
    assert outer.fold.gt(0).all() and np.isfinite(outer.select_dtypes('number')).all().all()
    assert outer.loc[outer.PVSD_Angle_deg.eq(0),'pred_delta_gain'].eq(0).all()
    outer.to_parquet(out/'predictions/outer_PV_candidates.parquet',index=False,compression='zstd')
    pd.DataFrame(threshold_rows).to_csv(out/'inner_threshold_room_scores.csv',index=False);pd.DataFrame(threshold_summary).to_csv(out/'inner_threshold_summary.csv',index=False)
    dump(out/'reused_absolute_provenance.json',provenance)
    assert all(sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    run.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),fits=counts,elapsed_seconds=time.perf_counter()-started,models_saved=49,raw_and_frozen_inputs_unchanged=True)
    dump(out/'run_manifest.json',run);dump(out/'progress.json',{'status':'completed','fits':counts,'elapsed_seconds':run['elapsed_seconds']})
    print(f'COMPLETED: 49 fits; 14 inner-selected thresholds; TEST predictions/scores=0; elapsed {run["elapsed_seconds"]:.1f}s.',flush=True)

if __name__=='__main__':main()
