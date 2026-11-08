"""Reproduce the v8-v11 methods with user-supplied data; outputs stay local."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import methods as M


def now():return datetime.now(timezone.utc).isoformat()


def summarize(rows):
    result=[]
    for (v,p,m,t),g in rows.groupby(['version','protocol','method','target'],sort=False):
        expected=M.STUDY['excluded_test_rooms'] if p=='Holdout3Rooms' else M.STUDY['development_rooms']
        assert len(g)==len(expected) and set(g.room)==set(expected)
        worst=g.sort_values(['mae','room'],ascending=[False,True]).iloc[0]
        result.append(dict(version=v,protocol=p,method=m,target=t,room_count=len(g),
            macro_MAE=float(g.mae.mean()),worst_MAE=float(worst.mae),worst_room=worst.room,
            macro_RMSE=float(g.rmse.mean()),median_R2=None if g.r2.isna().all() else float(g.r2.median())))
    return pd.DataFrame(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Locally supplied SEND_ML_BIM_EPW_07-18.csv')
    parser.add_argument('--output',type=Path,default=M.ROOT/'runs/reproduction')
    parser.add_argument('--versions',type=int,nargs='+',choices=[8,9,10,11],default=[8,9,10,11])
    parser.add_argument('--preflight',action='store_true',help='Check inputs without training or loading TEST labels')
    parser.add_argument('--evaluate-test',action='store_true',help='Score the observed holdout only after every model is frozen')
    parser.add_argument('--save-predictions',action='store_true',help='Optionally save hourly predictions locally; ignored by Git')
    args=parser.parse_args();versions=sorted(set(args.versions))
    if M.sha(args.source)!=M.STUDY['source_sha256']:
        raise ValueError('Source hash differs from the archived study; use the original supplied dataset.')
    columns=list(dict.fromkeys(M.META+M.RAW+M.TARGETS))
    d=M.load_data(args.source,'TRAIN',columns);xx,q=M.features(d)
    for t in M.TARGETS:
        assert (d[t]>=0).all()
        if t.startswith('F_'):assert (d[t]<=1).all()
    if args.preflight:
        print(f'Preflight passed: {len(d)} TRAIN rows; no TEST numeric targets loaded.');return
    output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a new output directory; published and prior results are never overwritten.')
    if output.is_relative_to((M.ROOT/'results').resolve()):raise ValueError('Use runs/ or another local output directory.')
    output.mkdir(parents=True);(output/'models').mkdir()
    if args.save_predictions:(output/'predictions').mkdir()
    M.START=time.perf_counter();M.SELECT.clear()
    manifest=dict(started_utc=now(),source_sha256=M.sha(args.source),versions=versions,
        code_sha256={p.name:M.sha(p) for p in [Path(__file__),M.ROOT/'scripts/methods.py']},
        configuration_sha256={p.name:M.sha(p) for p in (M.ROOT/'configs').glob('*.json')},
        TEST_fit_rows=0,TEST_used_for_selection=False,new_simulations=0,status='training')
    M.dump(output/'run.json',manifest)
    jobs,protocols=M.outer_splits();rows=[];frozen=[]
    with threadpool_limits(limits=M.STUDY['thread_limit']):
        for v in versions:
            full=dict(id='all_seven',fit_rooms=M.STUDY['development_rooms'],held_rooms=[])
            for job in jobs+[full]:
                print(f'v{v}: {job["id"]}',flush=True)
                bundles={}
                for method,spec in M.specs(v).items():
                    targets=['E_P10_lux'] if method=='S1_direct' else M.TARGETS
                    bundles[method]=M.train_cell(d,xx,q,v,method,spec,targets,job)
                if job['id']=='all_seven':
                    path=output/'models'/f'v{v}_all_seven.joblib';joblib.dump(bundles,path,compress=3)
                    frozen.append(dict(version=v,path=str(path.relative_to(output)),sha256=M.sha(path)))
                else:
                    vi=M.idx(d,job['held_rooms']);vx={k:x.loc[vi] for k,x in xx.items()}
                    pred={m:M.predict_cell(b,vx,q[vi]) for m,b in bundles.items()}
                    if v==11:M.add_fusion(pred)
                    frame=M.prediction_frame(d.loc[vi],v,job,pred,True)
                    for p in protocols:
                        if p['job_id']==job['id']:rows+=M.score_frame(frame,v,p['protocol'],p['fold'],'reproduction')
                    if args.save_predictions:frame.to_parquet(output/'predictions'/f'v{v}_{job["id"]}.parquet',index=False)
        M.dump(output/'selection.json',M.SELECT)
        M.dump(output/'models_frozen.json',dict(time_utc=now(),models=frozen,TEST_numeric_inputs_loaded=False,TEST_numeric_targets_loaded=False))
        if args.evaluate_test:
            inputs=M.load_data(args.source,'TEST',list(dict.fromkeys(M.META+M.RAW)))
            tx,tq=M.features(inputs);frames={};hashes={}
            for item in frozen:
                v=item['version'];path=output/item['path'];assert M.sha(path)==item['sha256']
                bundles=joblib.load(path);pred={m:M.predict_cell(b,tx,tq) for m,b in bundles.items()}
                if v==11:M.add_fusion(pred)
                frame=M.prediction_frame(inputs,v,None,pred,False);frames[v]=frame
                hashes[str(v)]=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()
            M.dump(output/'predictions_frozen.json',dict(time_utc=now(),prediction_hashes=hashes,TEST_numeric_targets_loaded=False))
            labels=M.load_data(args.source,'TEST',M.META+M.TARGETS)[M.KEYS+M.TARGETS].rename(columns={t:'true__'+t for t in M.TARGETS})
            for v,frame in frames.items():
                assert hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()==hashes[str(v)]
                frame=frame.merge(labels,on=M.KEYS,validate='one_to_one')
                rows+=M.score_frame(frame,v,'Holdout3Rooms','all_seven','reproduction')
                if args.save_predictions:frame.to_parquet(output/'predictions'/f'v{v}_test.parquet',index=False)
    table=pd.DataFrame(rows)
    table.to_csv(output/'room_metrics.csv',index=False)
    summary=summarize(table);summary.to_csv(output/'summary_metrics.csv',index=False)
    manifest.update(status='complete',completed_utc=now(),elapsed_seconds=time.perf_counter()-M.START,
        TEST_scored=args.evaluate_test,summary_cells=len(summary),models=frozen)
    M.dump(output/'run.json',manifest)
    print(f'Complete: {len(summary)} summary cells in {output}')


if __name__=='__main__':main()
