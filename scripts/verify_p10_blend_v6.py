"""Independently verify the frozen grouped weight choices and fold inference recipes."""
import numpy as np
import pandas as pd
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits
import train_cross_room_v5 as T
import run_p10_blend_v6 as B

ROOT=T.ROOT;OUT=ROOT/'reports/experiment_v6'

def stats(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);e=p-y;s=np.sum(np.square(y-y.mean()))
    return {'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),
            'r2':float(1-np.sum(e*e)/s) if s>0 else None,'asinh_mae':float(np.abs(np.arcsinh(p)-np.arcsinh(y)).mean())}

def compare(saved,computed):
    for k,v in computed.items():
        if v is None:assert pd.isna(saved[k]) or saved[k] is None
        else:assert np.isclose(saved[k],v,rtol=1e-11,atol=1e-11),(k,saved[k],v)

def main():
    cfg=T.read(B.CONFIG);run=T.read(OUT/'run_manifest.json');src=ROOT/cfg['frozen_source_run']
    assert run['status']=='completed' and run['new_regression_estimator_fits']==0 and run['weight_selection_tasks']==7
    assert T.sha(ROOT/'scripts/run_p10_blend_v6.py')==run['script_sha256']
    assert all(T.sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    inner=pd.read_csv(OUT/'inner_weight_room_scores.csv');candidates=pd.read_csv(OUT/'inner_weight_summary.csv');picked=pd.read_csv(OUT/'selected_weights.csv')
    folds=T.read(src/'fold_splits.json');v5selected=pd.read_csv(src/'selections.csv');v5scores=pd.read_csv(src/'inner_candidate_scores.csv')
    assert len(inner)==210 and len(candidates)==35 and len(picked)==7
    for fold in folds:
        k=fold['fold'];room=fold['held_room'];fit=set(fold['training_rooms'])
        assert len(fit)==6 and room not in fit and not fit&set(cfg['protected_test_rooms'])
        v5=v5selected[(v5selected.outer_fold==k)&(v5selected.feature_set=='base')&(v5selected.family=='RandomForest')]
        ac=v5.loc[v5.output=='relative','candidate'].item();cc=v5.loc[v5.output=='asinh','candidate'].item()
        for form,cid in [('relative',ac),('asinh',cc)]:
            s=v5scores[(v5scores.outer_fold==k)&(v5scores.feature_set=='base')&(v5scores.family=='RandomForest')&(v5scores.output==form)]
            assert s.loc[s.inner_mean_room_mae.idxmin(),'candidate']==cid
        ip=pd.read_parquet(src/'inner_predictions'/f'fold_{k:02d}_{room}__base__RandomForest.parquet')
        assert set(ip.Room_ID)==fit and room not in set(ip.Room_ID)
        for group in fold['inner']:
            assert set(ip.loc[ip.inner_fold==group['inner'],'Room_ID'])==set(group['validation_rooms'])
            assert not set(group['fitting_rooms'])&set(group['validation_rooms'])
        computed=[]
        for w in cfg['candidate_weights_on_A']:
            ps=ip[ac+'__relative'].to_numpy()*w+ip[cc+'__asinh'].to_numpy()*(1-w);ms=[]
            for vr,g in ip.groupby('Room_ID'):
                met=stats(g.true__E_P10_lux,ps[g.index.to_numpy()]);ms.append(met['mae'])
                s=inner[(inner.outer_fold==k)&(inner.inner_validation_room==vr)&(inner.weight_on_A==w)].iloc[0];compare(s,met)
            score=float(np.mean(ms));computed.append((w,score))
            saved=candidates[(candidates.outer_fold==k)&(candidates.weight_on_A==w)].iloc[0]
            assert np.isclose(saved.inner_mean_room_MAE_lux,score,rtol=1e-12)
        winner=min(computed,key=lambda r:r[1]);s=picked[picked.outer_fold==k].iloc[0]
        assert s.weight_on_A==winner[0] and s.held_room==room and s.relative_candidate==ac and s.asinh_candidate==cc
    p=pd.read_parquet(OUT/'predictions/P10_blend_OOF.parquet');a=pd.read_parquet(src/'predictions/A_RandomForest.parquet');c=pd.read_parquet(src/'predictions/C_RandomForest.parquet')
    assert len(p)==122640 and p[T.KEYS].equals(a[T.KEYS]) and p[T.KEYS].equals(c[T.KEYS])
    assert not set(p.Room_ID)&set(cfg['protected_test_rooms'])
    assert np.array_equal(p.true__E_P10_lux,a.true__E_P10_lux) and np.array_equal(p.true__E_P10_lux,c.true__E_P10_lux)
    assert np.array_equal(p.pred__A_relative,a.pred__E_P10_lux) and np.array_equal(p.pred__C_asinh,c.pred__E_P10_lux)
    expected=p.weight_on_A*p.pred__A_relative+(1-p.weight_on_A)*p.pred__C_asinh
    assert np.array_equal(expected,p.pred__Nested_blend)
    fm=pd.read_csv(OUT/'fold_metrics.csv');assert len(fm)==21
    for method in ['A_relative','C_asinh','Nested_blend']:
        for room,g in p.groupby('Room_ID'):
            compare(fm[(fm.method==method)&(fm.room==room)].iloc[0],stats(g.true__E_P10_lux,g['pred__'+method]))
            assert g.weight_on_A.nunique()==1 and g.weight_on_A.iloc[0]==picked.loc[picked.held_room==room,'weight_on_A'].item()
    aa=fm[fm.method=='A_relative'].set_index('room').sort_index();bb=fm[fm.method=='Nested_blend'].set_index('room').sort_index();delta=bb.mae-aa.mae
    tol=cfg['promotion_rule']['numerical_tolerance_lux'];gain=float(1-bb.mae.mean()/aa.mae.mean());nonworse=int((delta<=tol).sum());worst=float(bb.mae.max()-aa.mae.max())
    gate=gain>=.10 and nonworse>=5 and worst<=tol;decision=T.read(OUT/'candidate_decision.json')
    assert decision['passes_development_gate']==gate and np.isclose(decision['macro_MAE_reduction_fraction'],gain)
    assert decision['rooms_nonworse']==nonworse and np.isclose(decision['worst_room_MAE_increase_lux'],worst)
    # Exercise saved recipes on feature-only data. Loader excludes TEST before numeric conversion.
    dev,skipped=T.load_development(T.read(ROOT/'configs/experiment_v5.json'));assert dev[T.KEYS].equals(p[T.KEYS])
    cols=T.read(ROOT/'configs/model_plan.json')['feature_sets'][cfg['feature_set']]
    index=T.read(OUT/'recipe_index.json');reloadmax=0.
    with threadpool_limits(limits=cfg['thread_limit']):
        for item in index:
            assert T.sha(ROOT/item['path'])==item['sha256'];recipe=T.read(ROOT/item['path'])
            assert recipe['held_room'] not in recipe['training_rooms'] and not set(recipe['training_rooms'])&set(cfg['protected_test_rooms'])
            assert recipe['weight_on_A']==picked.loc[picked.held_room==recipe['held_room'],'weight_on_A'].item()
            ix=dev.index[dev.Room_ID==recipe['held_room']].to_numpy()[np.linspace(0,17519,31,dtype=int)]
            features=dev.loc[ix,cols];assert not set(features.columns)&{'Room_ID','E_mean_lux','E_P10_lux'}
            got=B.predict_recipe(recipe,features);stored=p.loc[ix,'pred__Nested_blend'].to_numpy()
            err=float(np.max(np.abs(got-stored)));reloadmax=max(reloadmax,err);assert err<1e-8
    assert all(T.sha(ROOT/path)==h for path,h in run['input_hashes'].items())
    T.dump(OUT/'verification.json',{'status':'passed','inner_room_scores_recomputed':210,'candidate_mean_scores_recomputed':35,
      'weight_choices_recomputed':7,'underlying_per_formulation_candidate_choices_checked':14,'outer_method_room_metrics_recomputed':21,
      'outer_blend_rows_reconstructed':len(p),'development_gate_recomputed':True,'fold_recipes_reloaded':len(index),'inference_samples_per_recipe':31,
      'recipe_prediction_max_abs_difference_lux':reloadmax,'inference_received_only_13_features_no_room_ID_or_true_targets':True,
      'new_regression_estimator_fits':0,'TEST_rows_discarded_before_numeric_conversion_during_recipe_verification':skipped,'TEST_predictions':0,'TEST_scores':0,
      'frozen_inputs_unchanged':True,'script_sha256':T.sha(__file__),'finished_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'Saved-prediction arithmetic, grouped selection, gate and recipe inference checks. No independent final TEST or full-seven-room deployment model.'})
    print(f'Verified 210 inner scores, 35 candidate scores, 7 weight choices, 122640 blended rows, 21 room metrics and 7 feature-only recipes. Gate passed: {gate}.')

if __name__=='__main__':main()
