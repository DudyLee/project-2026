"""Recompute nested-CV evidence and evaluate four-angle policies, TRAIN only."""
from __future__ import annotations
import json
from pathlib import Path
import runpy
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/train_nested_v3.py'),run_name='v3_evaluation_import')
TARGETS=M['TARGETS'];KEYS=M['KEYS'];BASE=M['BASE']
OUT=ROOT/'reports/experiment_v3'

def read_json(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,obj):M['dump'](p,obj)

def daylight(a):return (a[:,:,2]+a[:,:,3])/2
def feasible(a):return (a[:,:,2]>=0.5)&(a[:,:,3]>=0.5)
def deficit(a):return np.maximum(0,0.5-a[:,:,2])+np.maximum(0,0.5-a[:,:,3])
def objective(a,policy,scale):
    if policy=='PV_priority':return a[:,:,4]
    if policy=='Daylight_priority':return daylight(a)
    if policy=='Balanced':return 0.5*daylight(a)+0.5*a[:,:,4]/scale
    raise ValueError(policy)

def choose(a,policy,scale):
    """No labels are needed when a is a model prediction cube."""
    if policy!='Constrained_PV':return np.argmax(objective(a,policy,scale),axis=1)
    f=feasible(a);ds=deficit(a)
    eligible=np.where(f.any(axis=1)[:,None],f,np.isclose(ds,ds.min(axis=1)[:,None],rtol=0,atol=1e-12))
    return np.argmax(np.where(eligible,a[:,:,4],-np.inf),axis=1)

def training_fixed(frame,policy,scale,angles):
    values=[]
    for angle in angles:
        a=frame.loc[frame.PVSD_Angle_deg.eq(angle),TARGETS].to_numpy()[:,None,:]
        values.append((float(feasible(a).mean()),float(a[:,:,4].mean())) if policy=='Constrained_PV' else (float(objective(a,policy,scale).mean()),))
    return max(range(len(angles)),key=lambda k:values[k])

def check_regression(cfg,d,x,preds):
    fm=pd.read_csv(OUT/'fold_metrics.csv');inner=pd.read_csv(OUT/'inner_room_metrics.csv')
    scores=pd.read_csv(OUT/'inner_candidate_scores.csv');selections=pd.read_csv(OUT/'selections.csv')
    splits=read_json(OUT/'fold_splits.json');count=0
    for split in splits:
        assert split['held_room'] not in split['training_rooms'] and len(split['training_rooms'])==6
        seen=[]
        for ins in split['inner']:
            assert len(ins['fit_rooms'])==4 and len(ins['validation_rooms'])==2
            assert not set(ins['fit_rooms'])&set(ins['validation_rooms'])
            assert set(ins['fit_rooms']+ins['validation_rooms'])==set(split['training_rooms'])
            seen+=ins['validation_rooms']
        assert sorted(seen)==sorted(split['training_rooms'])
    regroup=inner.groupby(['outer_fold','family','candidate','target'],sort=False).mae.mean()
    for r in scores.to_dict('records'):
        expected=regroup.loc[(r['outer_fold'],r['family'],r['candidate'],r['target'])]
        assert np.isclose(expected,r['inner_mean_room_mae'],rtol=1e-10,atol=1e-10)
    for r in selections.to_dict('records'):
        pool=scores[(scores.outer_fold==r['outer_fold'])&(scores.target==r['target'])]
        if r['method']!='Nested_select':pool=pool[pool.family==r['method']]
        winner=pool.sort_values('inner_mean_room_mae',kind='stable').iloc[0]
        assert winner.family==r['selected_family'] and winner.candidate==r['selected_candidate']
    for r in fm.to_dict('records'):
        p=preds[r['method']];g=p.loc[p.Room_ID.eq(r['room'])]
        met=BASE['metrics'](g['true__'+r['target']].to_numpy(),g['pred__'+r['target']].to_numpy())
        for k,v in met.items():assert np.isclose(v,r[k],rtol=1e-9,atol=1e-9)
        count+=1
    bundles=0
    with threadpool_limits(limits=cfg['thread_limit']):
        for family in cfg['families']:
            for path in sorted((ROOT/cfg['model_output']/family).glob('*.joblib')):
                b=joblib.load(path);room=b['held_room'];idx=d.index[d.Room_ID.eq(room)].to_numpy()[np.linspace(0,17519,17,dtype=int)]
                assert set(b['train_rooms'])==set(d.Room_ID.unique())-{room}
                assert set(b['train_rooms']).isdisjoint(cfg['protected_test_rooms'])
                yp=M['predict_bundle'](b,x.loc[idx])
                for t in TARGETS:assert np.allclose(yp[t],preds[family].loc[idx,'pred__'+t],rtol=1e-9,atol=1e-9)
                bundles+=1
    assert count==175 and bundles==21
    for r in selections[selections.method.eq('Nested_select')].to_dict('records'):
        idx=d.index[d.Room_ID.eq(r['held_room'])]
        assert np.array_equal(preds['Nested_select'].loc[idx,'pred__'+r['target']],preds[r['selected_family']].loc[idx,'pred__'+r['target']])
    return {'outer_metric_rows_recomputed':count,'inner_candidate_scores_recomputed':len(scores),'selection_rows_recomputed':len(selections),
            'outer_bundles_reloaded':bundles,'reload_samples_per_bundle':17,'grouped_inner_and_outer_disjointness':True,'P10_prediction_api_accepts_X_only':True}

def regression_details(cfg,preds):
    tolerance=[];strat=[]
    for method,p in preds.items():
        for target in TARGETS:
            for room,g in p.groupby('Room_ID',sort=True):
                error=np.abs(g['pred__'+target].to_numpy()-g['true__'+target].to_numpy())
                for threshold in cfg['tolerance_scenarios'][target]:
                    tolerance.append({'method':method,'target':target,'room':room,'absolute_tolerance':threshold,'within_tolerance_fraction':float(np.mean(error<=threshold))})
            for by in ['Month','Hour','PVSD_Angle_deg']:
                for val,g in p.groupby(by):strat.append({'method':method,'target':target,'group_by':by,'group_value':int(val),**BASE['metrics'](g['true__'+target].to_numpy(),g['pred__'+target].to_numpy())})
    tol=pd.DataFrame(tolerance);tol.to_csv(OUT/'tolerance_by_room.csv',index=False)
    tol.groupby(['method','target','absolute_tolerance'],sort=False).agg(within_tolerance_fraction=('within_tolerance_fraction','mean'),worst_room_fraction=('within_tolerance_fraction','min')).reset_index().to_csv(OUT/'tolerance_summary.csv',index=False)
    pd.DataFrame(strat).to_csv(OUT/'stratified_metrics.csv',index=False)
    return len(tol)

def decision_evaluation(cfg,d,preds):
    decision=cfg['decision'];angles=np.array(decision['angles']);policies=decision['policies']
    decision_dir=OUT/'decisions';decision_dir.mkdir(exist_ok=True)
    rows=[];scales=[];angle_rows=[];checks={'selected_metric_relookup_checks':0,'oracle_zero_regret_checks':0,'decision_hourly_rows':0,'room_policy_strategy_groups':0}
    for room in sorted(d.Room_ID.unique()):
        train=d.loc[d.Room_ID.ne(room)];pv=train.loc[train.PV_AC_kWh.gt(0),'PV_AC_kWh'].to_numpy();scale=float(np.quantile(pv,0.95));assert scale>0
        scales.append({'room':room,'training_rooms':sorted(train.Room_ID.unique().tolist()),'pv_scale_q95_positive_kWh':scale})
        g=d.loc[d.Room_ID.eq(room)].sort_values(['HOY_0based','PVSD_Angle_deg'])
        n=len(g)//4;assert n==4380 and np.array_equal(g.PVSD_Angle_deg.to_numpy().reshape(n,4),np.tile(angles,(n,1)))
        true=g[TARGETS].to_numpy().reshape(n,4,5)
        times=g.iloc[::4][['Room_ID','Month','Day','Hour','DOY_1based','HOY_0based']].reset_index(drop=True)
        assert times.HOY_0based.is_monotonic_increasing and times.HOY_0based.is_unique
        prediction_cubes={m:p.loc[p.Room_ID.eq(room)].sort_values(['HOY_0based','PVSD_Angle_deg'])[['pred__'+t for t in TARGETS]].to_numpy().reshape(n,4,5) for m,p in preds.items()}
        hours=times.Hour.to_numpy();consecutive=(np.diff(times.DOY_1based)==0)&(np.diff(hours)==1)
        pv_ceiling=true[:,:,4].max(axis=1);oracle_feasible=feasible(true).any(axis=1)
        room_output=[]
        for policy in policies:
            oracle=choose(true,policy,scale)
            choices={f'Fixed_{angle}':np.full(n,k,dtype=int) for k,angle in enumerate(angles)}
            fixed_idx=training_fixed(train,policy,scale,angles);choices['Train_selected_fixed']=np.full(n,fixed_idx,dtype=int)
            choices['Clock_rule']=np.where((hours>=10)&(hours<=15),2,0)
            choices.update({m:choose(a,policy,scale) for m,a in prediction_cubes.items()});choices['Oracle']=oracle
            oracle_values=true[np.arange(n),oracle];best_deficit=deficit(true).min(axis=1)
            for strategy,idx in choices.items():
                chosen=true[np.arange(n),idx];selected_angle=angles[idx]
                actual_feasible=(chosen[:,2]>=0.5)&(chosen[:,3]>=0.5)
                pv_loss=pv_ceiling-chosen[:,4];assert np.all(pv_loss>=-1e-12)
                if policy!='Constrained_PV':
                    scores=objective(true,policy,scale);loss=scores.max(axis=1)-scores[np.arange(n),idx]
                    near=loss<=decision['near_optimal_tolerances'][policy]+1e-12
                    exact=loss<=1e-12;loss_unit='kWh' if policy=='PV_priority' else 'dimensionless_score'
                else:
                    ds=deficit(true);selected_ds=ds[np.arange(n),idx]
                    loss=selected_ds-best_deficit;loss_unit='summed_fraction_shortfall'
                    pv_gap=oracle_values[:,4]-chosen[:,4]
                    near=np.where(oracle_feasible,actual_feasible&(pv_gap<=decision['near_optimal_tolerances'][policy]+1e-12),
                                  (loss<=decision['near_optimal_tolerances']['Daylight_priority']+1e-12)&(pv_gap<=decision['near_optimal_tolerances'][policy]+1e-12))
                    exact=np.where(oracle_feasible,actual_feasible&(np.abs(pv_gap)<=1e-12),(loss<=1e-12)&(np.abs(pv_gap)<=1e-12))
                assert np.all(loss>=-1e-12)
                if strategy=='Oracle':assert np.max(np.abs(loss))<1e-12 and near.all() and exact.all();checks['oracle_zero_regret_checks']+=1
                # Independent row-key relookup, rather than trusting cube reshaping alone.
                selected_keys=pd.DataFrame({'HOY_0based':times.HOY_0based,'PVSD_Angle_deg':selected_angle})
                lookup=selected_keys.merge(g[['HOY_0based','PVSD_Angle_deg']+TARGETS],on=['HOY_0based','PVSD_Angle_deg'],validate='one_to_one',sort=False)
                assert np.array_equal(lookup[TARGETS].to_numpy(),chosen);checks['selected_metric_relookup_checks']+=1
                conditional=oracle_feasible&actual_feasible
                feasible_pv_shortfall=np.maximum(oracle_values[:,4]-chosen[:,4],0)
                row={'room':room,'policy':policy,'strategy':strategy,'hours':n,'pv_scale_kWh':scale,'selected_fixed_angle':int(angles[fixed_idx]) if strategy=='Train_selected_fixed' else None,
                     'pv_retained_hours_kWh':float(chosen[:,4].sum()),'pv_ceiling_kWh':float(pv_ceiling.sum()),'pv_loss_vs_energy_oracle_kWh':float(pv_loss.sum()),
                     'pv_loss_vs_energy_oracle_fraction':float(pv_loss.sum()/pv_ceiling.sum()) if pv_ceiling.sum()>0 else 0,
                     'F300_mean':float(chosen[:,2].mean()),'FUDI_mean':float(chosen[:,3].mean()),'E_mean_lux_mean':float(chosen[:,0].mean()),'P10_lux_mean':float(chosen[:,1].mean()),
                     'constraint_satisfaction_fraction':float(actual_feasible.mean()),'oracle_feasible_fraction':float(oracle_feasible.mean()),
                     'missed_feasible_hour_fraction':float((oracle_feasible&~actual_feasible).sum()/oracle_feasible.sum()) if oracle_feasible.any() else None,
                     'policy_loss_mean':float(loss.mean()),'policy_loss_sum':float(loss.sum()),'policy_loss_unit':loss_unit,
                     'near_optimal_fraction':float(near.mean()),'optimal_set_hit_fraction':float(exact.mean()),'exact_angle_match_fraction':float((idx==oracle).mean()),
                     'within_day_switch_count':int(((np.diff(idx)!=0)&consecutive).sum()),
                     'feasible_pair_PV_shortfall_kWh':float(feasible_pv_shortfall[conditional].sum()) if conditional.any() else None,
                     'feasible_pair_hours':int(conditional.sum())}
                rows.append(row)
                for angle in angles:
                    angle_rows.append({'room':room,'policy':policy,'strategy':strategy,'angle_deg':int(angle),'selected_fraction':float(np.mean(selected_angle==angle))})
                table=times.copy();table['policy']=policy;table['strategy']=strategy;table['angle_deg']=selected_angle
                table['oracle_angle_deg']=angles[oracle];table['policy_loss']=loss;table['near_optimal']=near;table['constraint_satisfied']=actual_feasible;table['pv_loss_vs_energy_oracle_kWh']=pv_loss
                for j,t in enumerate(TARGETS):table['actual__'+t]=chosen[:,j]
                room_output.append(table)
        p=pd.concat(room_output,ignore_index=True);assert len(p)==4380*4*12 and p.Room_ID.eq(room).all()
        p.to_parquet(decision_dir/f'{room}.parquet',index=False,compression='zstd');checks['decision_hourly_rows']+=len(p)
    detail=pd.DataFrame(rows);assert len(detail)==336
    detail.to_csv(OUT/'decision_by_room.csv',index=False)
    numeric=[c for c in detail.select_dtypes('number').columns if c not in ['hours','selected_fixed_angle']]
    summary=detail.groupby(['policy','strategy','policy_loss_unit'],sort=False)[numeric].mean().reset_index()
    summary['room_count']=7;summary.to_csv(OUT/'decision_summary.csv',index=False)
    angle_table=pd.DataFrame(angle_rows);angle_table.to_csv(OUT/'angle_usage_by_room.csv',index=False)
    angle_table.groupby(['policy','strategy','angle_deg'],sort=False).selected_fraction.mean().reset_index().to_csv(OUT/'angle_usage_summary.csv',index=False)
    dump(OUT/'decision_training_scales.json',scales)
    dump(OUT/'decision_protocol.json',{'configuration_sha256':M['digest'](ROOT/'configs/experiment_v3.json'),'decision':decision,
         'primary_loss_units':{'PV_priority':'hourly kWh regret','Daylight_priority':'D score regret','Balanced':'balanced score regret','Constrained_PV':'summed fractional constraint shortfall regret; report missed feasibility and conditional PV loss separately'},
         'constraint_near_optimal':'For oracle-feasible hours require actual feasibility and PV gap<=1e-4 kWh; otherwise shortfall regret<=daylight tolerance .005 and PV gap<=1e-4 kWh.',
         'aggregation':'Seven rooms equally weighted; reported PV kWh are per-room sums over the 4380 retained hours, then averaged over rooms. Not an all-8760-hour energy claim.',
         'decision_target_scope':'Policies directly use F300, FUDI and PV only. E_mean and P10 remain regression outcomes and are looked up for the selected angle, but are not terms in these predeclared decision objectives.',
         'interpretation':'Oracle and all scores use held-out simulation labels only after strategies are frozen. Model choices use predicted performance only. Fixed and clock schedules do not use held-out labels. No control-system dynamics, night transitions or actuator penalties.'})
    checks['room_policy_strategy_groups']=len(detail)
    assert checks['decision_hourly_rows']==1471680 and checks['oracle_zero_regret_checks']==28
    return checks

def main():
    started=time.perf_counter();cfg=read_json(ROOT/'configs/experiment_v3.json');run=read_json(OUT/'run_manifest.json');assert run['status']=='completed'
    assert M['digest'](ROOT/'configs/experiment_v3.json')==run['config_sha256']
    plan=read_json(ROOT/'configs/model_plan.json');d,skipped=BASE['load_train'](ROOT/cfg['source'],plan);x=BASE['encode_inputs'](d,plan['feature_sets'][cfg['feature_set']])
    preds={m:pd.read_parquet(OUT/'predictions'/f'{m}.parquet') for m in ['Frozen_HistGB',*cfg['families'],'Nested_select']}
    for m,p in preds.items():
        assert p[KEYS].equals(d[KEYS]) and len(p)==122640 and set(p.Room_ID).isdisjoint(cfg['protected_test_rooms'])
        for t in TARGETS:
            assert np.array_equal(p['true__'+t],d[t]) and np.isfinite(p['pred__'+t]).all() and p['pred__'+t].ge(0).all()
            if t.startswith('F_'):assert p['pred__'+t].le(1).all()
    verification=check_regression(cfg,d,x,preds);print('Regression and model-reload checks passed.',flush=True)
    verification['tolerance_room_rows']=regression_details(cfg,preds)
    verification.update(decision_evaluation(cfg,d,preds));print('Four-angle decision evaluation and row-key relookup passed.',flush=True)
    assert M['digest'](ROOT/cfg['source'])==run['source_sha256']
    assert all(M['digest'](ROOT/p)==h for p,h in run['reference_hashes'].items())
    verification.update(status='passed',source_unchanged=True,reference_files_unchanged=True,TEST_predictions=0,TEST_scores=0,TEST_rows_discarded=skipped,
                        analysis_script_sha256={name:M['digest'](ROOT/'scripts'/name) for name in ['evaluate_nested_v3.py','report_nested_v3.py']},
                        scope='Pipeline and saved-model consistency verification; development evaluation, not independent field validation or a full training rerun',
                        completed_utc=datetime.now(timezone.utc).isoformat(),evaluation_seconds=time.perf_counter()-started)
    dump(OUT/'verification.json',verification);print(json.dumps(verification,indent=2),flush=True)

if __name__=='__main__':main()
