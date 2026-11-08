"""Choose five-grid convex P10 blend weights within outer training rooms only."""
from __future__ import annotations
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_cross_room_v5 as T

ROOT=T.ROOT;CONFIG=ROOT/'configs/experiment_v6.json';TARGET='E_P10_lux';KEYS=T.KEYS

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    return T.BASE['metrics'](y,p)|{'asinh_mae':float(np.abs(np.arcsinh(p)-np.arcsinh(y)).mean())}

def predict_recipe(recipe,inputs):
    """Fold recipe verification/inference from BIM/time/weather X only, no labels."""
    path=ROOT/recipe['source_bundle'];assert T.sha(path)==recipe['source_bundle_sha256']
    model=joblib.load(path);cols=T.read(ROOT/'configs/model_plan.json')['feature_sets'][recipe['feature_set']]
    x=T.BASE['encode_inputs'](inputs,cols);assert list(x.columns)==model['encoded_features']
    relative_cid=recipe['relative_candidate'];direct_cid=recipe['asinh_candidate'];ests=model['estimators']
    mean=np.maximum(np.expm1(ests[relative_cid+'__mean_log'].predict(x)),0)
    relative=np.maximum(ests[relative_cid+'__ratio'].predict(x),0)*(mean+1)
    direct=np.maximum(np.sinh(ests[direct_cid+'__asinh'].predict(x)),0)
    return recipe['weight_on_A']*relative+(1-recipe['weight_on_A'])*direct

def main():
    start=time.perf_counter();cfg=T.read(CONFIG);out=ROOT/cfg['output'];models=ROOT/cfg['model_output'];src=ROOT/cfg['frozen_source_run']
    for p in [out,out/'predictions',out/'figures',models]:p.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():raise RuntimeError('V6 already started; refusing to overwrite experiment evidence.')
    assert Path(sys.prefix).resolve()==(ROOT/'.conda/envs/pvsd-ml-v1').resolve()
    assert T.read(src/'verification.json')['status']=='passed'
    folds=T.read(src/'fold_splits.json');selections=pd.read_csv(src/'selections.csv');index=T.read(src/'model_index.json')
    inputs=[CONFIG,ROOT/cfg['source_proposal'],src/'run_manifest.json',src/'verification.json',src/'fold_splits.json',src/'selections.csv',src/'inner_candidate_scores.csv',src/'model_index.json',ROOT/'configs/model_plan.json',ROOT/'scripts/train_cross_room_v5.py']
    inputs += [src/'predictions'/f'{m}.parquet' for m in cfg['models']]
    inputs += [src/'inner_predictions'/f'fold_{s["fold"]:02d}_{s["held_room"]}__base__RandomForest.parquet' for s in folds]
    inputs += [ROOT/i['path'] for i in index if i['feature_set']=='base' and i['family']=='RandomForest']
    hashes={p.relative_to(ROOT).as_posix():T.sha(p) for p in inputs}
    run={'status':'running','experiment_id':cfg['experiment_id'],'started_utc':datetime.now(timezone.utc).isoformat(),
         'input_hashes':hashes,'script_sha256':T.sha(__file__),'new_regression_estimator_fits':0,'weight_selection_tasks':0,
         'TEST_predictions':0,'TEST_scores':0,'source_csv_parsed_during_selection':False,'python':sys.version,'conda_prefix':sys.prefix,'scope':cfg['scope']}
    T.dump(out/'run_manifest.json',run);T.dump(out/'config_snapshot.json',cfg)
    roomscores=[];candidates=[];chosen=[];recipes=[]
    # Resolve all seven choices before reading any outer labels or scores.
    for fold in folds:
        k=fold['fold'];held=fold['held_room'];fitrooms=fold['training_rooms']
        s=selections[(selections.outer_fold==k)&(selections.feature_set=='base')&(selections.family=='RandomForest')]
        ac=s.loc[s.output=='relative','candidate'].item();cc=s.loc[s.output=='asinh','candidate'].item()
        ip=pd.read_parquet(src/'inner_predictions'/f'fold_{k:02d}_{held}__base__RandomForest.parquet')
        assert len(ip)==105120 and set(ip.Room_ID)==set(fitrooms) and held not in set(ip.Room_ID)
        assert not set(ip.Room_ID)&set(cfg['protected_test_rooms'])
        for inner in fold['inner']:
            assert set(ip.loc[ip.inner_fold==inner['inner'],'Room_ID'])==set(inner['validation_rooms'])
            assert not set(inner['fitting_rooms'])&set(inner['validation_rooms'])
        pa=ip[ac+'__relative'].to_numpy();pc=ip[cc+'__asinh'].to_numpy();y=ip.true__E_P10_lux.to_numpy();local=[]
        for w in cfg['candidate_weights_on_A']:
            p=w*pa+(1-w)*pc;errors=[]
            for room in sorted(fitrooms):
                mask=ip.Room_ID.to_numpy()==room;met=metrics(y[mask],p[mask]);errors.append(met['mae'])
                roomscores.append({'outer_fold':k,'outer_held_room':held,'inner_validation_room':room,'weight_on_A':w,**met})
            row={'outer_fold':k,'held_room':held,'weight_on_A':w,'relative_candidate':ac,'asinh_candidate':cc,
                 'inner_mean_room_MAE_lux':float(np.mean(errors)),'inner_worst_room_MAE_lux':float(np.max(errors))}
            candidates.append(row);local.append(row)
        best=min(local,key=lambda r:r['inner_mean_room_MAE_lux']);chosen.append(best)
        item=next(i for i in index if i['fold']==k and i['feature_set']=='base' and i['family']=='RandomForest')
        recipe={'experiment_id':cfg['experiment_id'],'fold':k,'held_room':held,'training_rooms':fitrooms,
                'weight_on_A':best['weight_on_A'],'relative_candidate':ac,'asinh_candidate':cc,
                'source_bundle':item['path'],'source_bundle_sha256':item['sha256'],'feature_set':cfg['feature_set'],
                'target':TARGET,'formula':cfg['prediction'],'selection_source':'six fitting rooms inner OOF only',
                'deployment_ready':False,'note':'Cross-validation fold recipe, not final all-seven-room model'}
        path=models/f'fold_{k:02d}_{held}.json';T.dump(path,recipe);recipes.append({'path':path.relative_to(ROOT).as_posix(),'sha256':T.sha(path)})
        print(f'Fold {k}: held {held}; selected A weight {best["weight_on_A"]}; inner MAE {best["inner_mean_room_MAE_lux"]:.6f}',flush=True)
    pd.DataFrame(roomscores).to_csv(out/'inner_weight_room_scores.csv',index=False)
    pd.DataFrame(candidates).to_csv(out/'inner_weight_summary.csv',index=False)
    pd.DataFrame(chosen).to_csv(out/'selected_weights.csv',index=False)
    T.dump(out/'recipe_index.json',recipes)
    # Only now are outer labels loaded for scoring the frozen choices.
    a=pd.read_parquet(src/'predictions/A_RandomForest.parquet');c=pd.read_parquet(src/'predictions/C_RandomForest.parquet')
    assert a[KEYS].equals(c[KEYS]) and np.array_equal(a.true__E_P10_lux,c.true__E_P10_lux)
    assert len(a)==122640 and not set(a.Room_ID)&set(cfg['protected_test_rooms'])
    p=a[T.META+['fold','true__E_P10_lux']].copy();p['pred__A_relative']=a.pred__E_P10_lux;p['pred__C_asinh']=c.pred__E_P10_lux
    p['pred__Nested_blend']=np.nan;p['weight_on_A']=np.nan
    for pick in chosen:
        mask=p.Room_ID==pick['held_room'];w=pick['weight_on_A'];p.loc[mask,'weight_on_A']=w
        p.loc[mask,'pred__Nested_blend']=w*p.loc[mask,'pred__A_relative']+(1-w)*p.loc[mask,'pred__C_asinh']
    assert not p.isna().any().any() and np.isfinite(p.select_dtypes('number')).all().all()
    p.to_parquet(out/'predictions/P10_blend_OOF.parquet',index=False,compression='zstd')
    results=[]
    for method in ['A_relative','C_asinh','Nested_blend']:
        for room,g in p.groupby('Room_ID'):
            results.append({'method':method,'room':room,'target':TARGET,'weight_on_A':float(g.weight_on_A.iloc[0]) if method=='Nested_blend' else (1.0 if method=='A_relative' else 0.0),
                            **metrics(g.true__E_P10_lux,g['pred__'+method])})
    fm=pd.DataFrame(results);fm.to_csv(out/'fold_metrics.csv',index=False)
    summary=fm.groupby('method').agg(room_count=('room','nunique'),mae_mean=('mae','mean'),mae_median=('mae','median'),mae_max=('mae','max'),
       rmse_mean=('rmse','mean'),bias_mean=('bias','mean'),r2_median=('r2','median'),r2_min=('r2','min'),asinh_mae_mean=('asinh_mae','mean')).reset_index()
    summary.to_csv(out/'summary_metrics.csv',index=False)
    aa=fm[fm.method=='A_relative'].set_index('room').sort_index();bb=fm[fm.method=='Nested_blend'].set_index('room').sort_index();delta=bb.mae-aa.mae;tol=cfg['promotion_rule']['numerical_tolerance_lux']
    comparison=pd.DataFrame({'room':aa.index,'A_MAE_lux':aa.mae.to_numpy(),'blend_MAE_lux':bb.mae.to_numpy(),'MAE_change_lux':delta.to_numpy(),
                            'A_bias_lux':aa.bias.to_numpy(),'blend_bias_lux':bb.bias.to_numpy(),'weight_on_A':bb.weight_on_A.to_numpy()})
    comparison.to_csv(out/'paired_room_comparison.csv',index=False)
    gain=float(1-bb.mae.mean()/aa.mae.mean());nonworse=int((delta<=tol).sum());worst=float(bb.mae.max()-aa.mae.max())
    passed=gain>=.10 and nonworse>=5 and worst<=tol
    decision={'passes_development_gate':passed,'macro_MAE_reduction_fraction':gain,'rooms_improved':int((delta < -tol).sum()),
              'rooms_tied':int((delta.abs()<=tol).sum()),'rooms_worse':int((delta>tol).sum()),'rooms_nonworse':nonworse,'worst_room_MAE_increase_lux':worst,
              'recommended_status':'limited_development_candidate' if passed else 'retain_A_baseline_stop_expanding_weights_without_new_information',
              'automatic_replacement':False,'final_refit':False,'TEST_unsealed':False,'rule':cfg['promotion_rule']}
    T.dump(out/'candidate_decision.json',decision)
    assert all(T.sha(ROOT/path)==h for path,h in hashes.items())
    run.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
               weight_selection_tasks=7,inner_room_scores=len(roomscores),weight_candidate_scores=len(candidates),outer_scoring_rows=len(p),
               outer_method_room_metrics=len(results),fold_recipes=len(recipes),input_files_unchanged=True)
    T.dump(out/'run_manifest.json',run)
    print(summary.to_string(index=False));print(comparison.to_string(index=False));print(decision)

if __name__=='__main__':main()
