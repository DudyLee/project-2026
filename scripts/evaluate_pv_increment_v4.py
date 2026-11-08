"""Independently verify v4 fits/calibration and score saved four-angle choices."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import runpy
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/experiment_v4'
BASE=runpy.run_path(str(ROOT/'scripts/train_logo_v1.py'),run_name='v4_evaluation_shared')
PV='PV_AC_kWh';KEYS=['Room_ID','HOY_0based','PVSD_Angle_deg'];ANGLES=np.array([0,30,60,90]);EPS=1e-12
TARGETS=['E_mean_lux','E_P10_lux','F_300_fraction','F_UDI_fraction',PV]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def truth_baseline_merge(frame):
    """Independent label construction by a checked relational merge."""
    left=frame[['Room_ID','HOY_0based','PVSD_Angle_deg',PV]].copy();left['row_order']=np.arange(len(left))
    ref=left[left.PVSD_Angle_deg.eq(0)][['Room_ID','HOY_0based',PV]].rename(columns={PV:'baseline_PV'})
    j=left.merge(ref,on=['Room_ID','HOY_0based'],how='left',sort=False,validate='many_to_one').sort_values('row_order')
    assert len(j)==len(left) and j.baseline_PV.notna().all()
    return j[PV].to_numpy()-j.baseline_PV.to_numpy()

def independent_choice(gain,threshold):
    if threshold is None:return np.zeros(len(gain),dtype=int)
    # Stable descending sort is an independent tie-preserving implementation.
    ranked=np.argsort(-gain,axis=1,kind='stable')
    top=ranked[:,0];best=np.take_along_axis(gain,top[:,None],axis=1)[:,0]
    top[best<=threshold]=0
    return top

def verify_training(cfg,run,d,x,p):
    index=json.loads((OUT/'model_index.json').read_text());splits=json.loads((OUT/'fold_splits.json').read_text())
    selections=json.loads((OUT/'threshold_selections.json').read_text());score_csv=pd.read_csv(OUT/'inner_threshold_room_scores.csv')
    summary_csv=pd.read_csv(OUT/'inner_threshold_summary.csv');inner={}
    count=0;choice_checks=0;threshold_rows=0
    for split in splits:
        fold=split['outer_fold'];held=split['held_room'];fit_rooms=split['training_rooms']
        assert len(fit_rooms)==6 and held not in fit_rooms and not set(fit_rooms)&set(cfg['protected_test_rooms'])
        seen=[]
        for ins in split['inner']:
            assert len(ins['fit_rooms'])==4 and len(ins['validation_rooms'])==2
            assert set(ins['fit_rooms']).isdisjoint(ins['validation_rooms'])
            assert set(ins['fit_rooms']+ins['validation_rooms'])==set(fit_rooms)
            seen+=ins['validation_rooms']
        assert sorted(seen)==fit_rooms
        f=pd.read_parquet(OUT/'predictions'/f'inner_OOF_outer_{fold:02d}_{held}.parquet');inner[fold]=f
        assert len(f)==105120 and set(f.Room_ID)==set(fit_rooms) and held not in set(f.Room_ID)
        expected=d[d.Room_ID.isin(fit_rooms)]
        assert f[KEYS].reset_index(drop=True).equals(expected[KEYS].reset_index(drop=True)) and np.array_equal(f.true_PV,expected[PV])
        for variant in ['absolute','delta']:
            values=[]
            for threshold in cfg['thresholds']:
                net=[]
                for room in fit_rooms:
                    g=f[f.Room_ID.eq(room)].sort_values(['HOY_0based','PVSD_Angle_deg'])
                    gm=g['pred_'+variant+'_gain'].to_numpy().reshape(4380,4);truth=g.true_PV.to_numpy().reshape(4380,4)
                    if variant=='absolute':
                        raw=g.raw_absolute_PV.to_numpy().reshape(4380,4);bounded=np.maximum(raw,0)
                        assert np.array_equal(bounded,g.pred_absolute_PV.to_numpy().reshape(4380,4))
                        # Native XGBoost outputs are float32. Inner gains were
                        # subtracted at that precision, then stored in float64 columns.
                        native=bounded.astype(np.float32)
                        assert np.array_equal((native-native[:,0:1]).astype(np.float64),gm)
                    assert (gm[:,0]==0).all()
                    idx=independent_choice(gm,threshold['kWh']);gain=truth[np.arange(4380),idx]-truth[:,0];value=float(gain.sum());net.append(value)
                    old=score_csv[(score_csv.outer_fold==fold)&(score_csv.variant==variant)&(score_csv.threshold_label==threshold['label'])&(score_csv.validation_room==room)].iloc[0]
                    assert np.isclose(value,old.net_gain_kWh,rtol=1e-9,atol=1e-9);threshold_rows+=1
                average=float(np.mean(net));values.append(average)
                old=summary_csv[(summary_csv.outer_fold==fold)&(summary_csv.variant==variant)&(summary_csv.threshold_label==threshold['label'])].iloc[0]
                assert np.isclose(average,old.mean_room_gain_kWh,rtol=1e-9,atol=1e-9)
            best=max(values);winner=[i for i,v in enumerate(values) if abs(v-best)<=1e-12][-1]
            saved=next(r for r in selections if r['outer_fold']==fold and r['variant']==variant)
            assert saved['threshold_label']==cfg['thresholds'][winner]['label'];assert saved['selection_rooms']==fit_rooms;choice_checks+=1
    with threadpool_limits(limits=cfg['thread_limit']):
        for row in index:
            path=ROOT/row['path'];assert sha(path)==row['sha256'];b=joblib.load(path)
            assert b['held_room'] not in b['fit_rooms'] and set(b['fit_rooms']).isdisjoint(b['validation_rooms'])
            assert set(b['fit_rooms']).isdisjoint(cfg['protected_test_rooms'])
            fitting=d[d.Room_ID.isin(b['fit_rooms'])]
            target=fitting[PV].to_numpy() if b['variant']=='absolute' else truth_baseline_merge(fitting)[fitting.PVSD_Angle_deg.ne(0).to_numpy()]
            assert len(target)==b['n_fit'] and hashlib.sha256(np.ascontiguousarray(target,dtype=np.float64).tobytes()).hexdigest()==b['fit_label_sha256']
            valid=d[d.Room_ID.isin(b['validation_rooms'])]
            samples=valid.index.to_numpy()[np.linspace(0,len(valid)-1,17,dtype=int)]
            if b['variant']=='absolute':actual=b['estimator'].predict(x.loc[samples])
            else:
                actual=np.zeros(len(samples));nonzero=d.loc[samples,'PVSD_Angle_deg'].ne(0).to_numpy();actual[nonzero]=b['estimator'].predict(x.loc[samples[nonzero]])
            if b['inner_fold'] is None:
                expected=p.loc[samples,'pred_delta_gain'].to_numpy()
            else:
                f=inner[b['outer_fold']].set_index(KEYS)
                field='raw_absolute_PV' if b['variant']=='absolute' else 'pred_delta_gain'
                lookup=pd.MultiIndex.from_frame(d.loc[samples,KEYS]);expected=f.reindex(lookup)[field].to_numpy()
            assert np.allclose(actual,expected,rtol=1e-9,atol=1e-9);count+=1
    assert count==49 and choice_checks==14 and threshold_rows==420
    assert np.array_equal(truth_baseline_merge(d),p.true_delta_PV)
    return {'new_models_reloaded':count,'training_label_hashes_independently_reconstructed':count,'reload_validation_samples_per_model':17,
            'inner_room_threshold_scores_recomputed':threshold_rows,'inner_candidate_mean_scores_recomputed':70,'threshold_choices_independently_recomputed':choice_checks,
            'grouped_room_disjointness_checked':True,'delta_labels_paired_within_fitting_room_and_hour':True,'delta_zero_anchor_checked':True,'delta_inference_uses_X_and_angles_only':True}

def evaluate_decisions(cfg,p):
    sel=json.loads((OUT/'threshold_selections.json').read_text());selections={(r['outer_held_room'],r['variant']):r for r in sel}
    decisions=[];metrics=[];forecast=[];segments=[]
    for room in sorted(p.Room_ID.unique()):
        g=p[p.Room_ID.eq(room)].sort_values(['HOY_0based','PVSD_Angle_deg']);n=4380;ii=np.arange(n)
        truth={t:g['true__'+t].to_numpy().reshape(n,4) for t in TARGETS};y=truth[PV];true_gain=y-y[:,0:1]
        absolute=g.pred_absolute_gain.to_numpy().reshape(n,4);delta=g.pred_delta_gain.to_numpy().reshape(n,4);hist=g.pred_frozen_HistGB_gain.to_numpy().reshape(n,4)
        assert all((a[:,0]==0).all() for a in [absolute,delta,hist])
        a_sel=selections[(room,'absolute')];d_sel=selections[(room,'delta')]
        specs=[('Fixed_0',None,None,'always_zero'),('Frozen_HistGB',hist,0,'0Wh'),('Absolute_XGBoost',absolute,0,'0Wh'),
               ('Absolute_XGBoost_guarded',absolute,a_sel['threshold_kWh'],a_sel['threshold_label']),('Delta_XGBoost',delta,0,'0Wh'),
               ('Delta_XGBoost_guarded',delta,d_sel['threshold_kWh'],d_sel['threshold_label']),('Oracle',true_gain,0,'0Wh')]
        oracle=y.max(axis=1);opportunity=oracle-y[:,0]>.0001+EPS
        meta=g.iloc[::4][['Room_ID','Month','Day','Hour','DOY_1based','HOY_0based','fold']].reset_index(drop=True)
        consecutive=(np.diff(meta.DOY_1based)==0)&(np.diff(meta.Hour)==1)
        for method,gains,threshold,label in specs:
            idx=np.zeros(n,dtype=int) if method=='Fixed_0' else independent_choice(gains,threshold)
            actual=y[ii,idx];gain=actual-y[:,0];regret=oracle-actual;switched=idx!=0;benefit=gain>EPS;harm=gain<-EPS
            assert np.all(regret>=-EPS)
            row={'room':room,'method':method,'threshold_label':label,'threshold_kWh':threshold,'hours':n,'PV_kWh':float(actual.sum()),'fixed_zero_PV_kWh':float(y[:,0].sum()),
                 'net_gain_kWh':float(gain.sum()),'gain_fraction_vs_zero':float(gain.sum()/y[:,0].sum()),'oracle_PV_kWh':float(oracle.sum()),'oracle_headroom_kWh':float((oracle-y[:,0]).sum()),'regret_kWh':float(regret.sum()),
                 'benefit_kWh':float(np.maximum(gain,0).sum()),'harm_kWh':float(np.maximum(-gain,0).sum()),'switch_hours':int(switched.sum()),
                 'beneficial_switch_hours':int((switched&benefit).sum()),'harmful_switch_hours':int((switched&harm).sum()),'neutral_switch_hours':int((switched&~benefit&~harm).sum()),
                 'switch_fraction':float(switched.mean()),'opportunity_hours':int(opportunity.sum()),'captured_opportunity_hours':int((opportunity&(gain>.0001+EPS)).sum()),
                 'within_day_switch_count':int(((np.diff(idx)!=0)&consecutive).sum()),
                 'F300_mean':float(truth['F_300_fraction'][ii,idx].mean()),'FUDI_mean':float(truth['F_UDI_fraction'][ii,idx].mean()),
                 'F300_change_pp':float(100*(truth['F_300_fraction'][ii,idx]-truth['F_300_fraction'][:,0]).mean()),
                 'FUDI_change_pp':float(100*(truth['F_UDI_fraction'][ii,idx]-truth['F_UDI_fraction'][:,0]).mean())}
            assert np.isclose(row['net_gain_kWh'],row['benefit_kWh']-row['harm_kWh'],rtol=1e-9,atol=1e-9)
            metrics.append(row);table=meta.copy();table['method']=method;table['threshold_label']=label;table['angle_deg']=ANGLES[idx];table['true_PV_kWh']=actual
            table['fixed_zero_PV_kWh']=y[:,0];table['gain_vs_zero_kWh']=gain;table['oracle_PV_kWh']=oracle;table['regret_kWh']=regret
            for t in TARGETS[:-1]:table['actual__'+t]=truth[t][ii,idx]
            decisions.append(table)
            for by in ['Month','Hour']:
                for value in sorted(meta[by].unique()):
                    mask=meta[by].to_numpy()==value
                    segments.append({'room':room,'method':method,'group_by':by,'group_value':int(value),'net_gain_kWh':float(gain[mask].sum()),'regret_kWh':float(regret[mask].sum()),'hours':int(mask.sum())})
        for method,pred_abs,pred_gain in [('Frozen_HistGB',g.pred_frozen_HistGB_PV.to_numpy().reshape(n,4),hist),('Absolute_XGBoost',g.pred_absolute_XGBoost_PV.to_numpy().reshape(n,4),absolute),('Delta_XGBoost',None,delta)]:
            forecast.append({'room':room,'method':method,'prediction_target':'PV_angle_difference','n':n*3,**BASE['metrics'](true_gain[:,1:].ravel(),pred_gain[:,1:].ravel())})
            if pred_abs is not None:forecast.append({'room':room,'method':method,'prediction_target':'Absolute_PV','n':n*4,**BASE['metrics'](y.ravel(),pred_abs.ravel())})
    dec=pd.concat(decisions,ignore_index=True);dec.to_parquet(OUT/'decisions.parquet',index=False,compression='zstd')
    fm=pd.DataFrame(metrics);fm.to_csv(OUT/'decision_by_room.csv',index=False);pd.DataFrame(segments).to_csv(OUT/'decision_segments.csv',index=False)
    pd.DataFrame(forecast).to_csv(OUT/'forecast_metrics_by_room.csv',index=False)
    pd.DataFrame(forecast).groupby(['method','prediction_target'],sort=False).agg(room_count=('room','nunique'),mae_mean=('mae','mean'),rmse_mean=('rmse','mean'),r2_median=('r2','median')).reset_index().to_csv(OUT/'forecast_summary.csv',index=False)
    rows=[]
    for method,group in fm.groupby('method',sort=False):
        gain=group.net_gain_kWh;sw=int(group.switch_hours.sum());opp=int(group.opportunity_hours.sum());gain_sum=float(gain.sum());base_sum=float(group.fixed_zero_PV_kWh.sum());head=float(group.oracle_headroom_kWh.sum())
        rows.append({'method':method,'room_count':len(group),'mean_gain_kWh':float(gain.mean()),'median_gain_kWh':float(gain.median()),'min_gain_kWh':float(gain.min()),'max_gain_kWh':float(gain.max()),
                     'rooms_improved':int((gain>1e-9).sum()),'rooms_tied':int((np.abs(gain)<=1e-9).sum()),'rooms_worse':int((gain<-1e-9).sum()),
                     'mean_PV_kWh':float(group.PV_kWh.mean()),'aggregate_gain_fraction_vs_zero':gain_sum/base_sum,'mean_room_gain_fraction_vs_zero':float(group.gain_fraction_vs_zero.mean()),
                     'mean_regret_kWh':float(group.regret_kWh.mean()),'aggregate_headroom_captured_fraction':gain_sum/head,'mean_benefit_kWh':float(group.benefit_kWh.mean()),'mean_harm_kWh':float(group.harm_kWh.mean()),
                     'switch_fraction':float(group.switch_hours.sum()/group.hours.sum()),'harmful_fraction_among_switches':float(group.harmful_switch_hours.sum()/sw) if sw else None,
                     'beneficial_fraction_among_switches':float(group.beneficial_switch_hours.sum()/sw) if sw else None,'opportunity_recall':float(group.captured_opportunity_hours.sum()/opp) if opp else None,
                     'mean_within_day_switch_count':float(group.within_day_switch_count.mean()),'mean_F300_change_pp':float(group.F300_change_pp.mean()),'mean_FUDI_change_pp':float(group.FUDI_change_pp.mean())})
    pd.DataFrame(rows).to_csv(OUT/'decision_summary.csv',index=False)
    # Independently re-lookup every selected physical output by the original row key.
    candidate_truth=p[KEYS+['true__'+t for t in TARGETS]].rename(columns={'PVSD_Angle_deg':'angle_deg'})
    joined=dec.merge(candidate_truth,on=['Room_ID','HOY_0based','angle_deg'],validate='many_to_one',sort=False)
    assert np.array_equal(joined.true_PV_kWh,joined['true__'+PV])
    for t in TARGETS[:-1]:assert np.array_equal(joined['actual__'+t],joined['true__'+t])
    count=0
    for (room,method),g in dec.groupby(['Room_ID','method']):
        row=fm[(fm.room==room)&(fm.method==method)].iloc[0]
        assert len(g)==4380 and np.isclose(g.gain_vs_zero_kWh.sum(),row.net_gain_kWh,rtol=1e-9,atol=1e-9)
        assert np.isclose(g.regret_kWh.sum(),row.regret_kWh,rtol=1e-9,atol=1e-9);count+=1
    assert count==49 and len(dec)==214620
    assert np.all(fm.loc[fm.method.eq('Fixed_0'),'net_gain_kWh']==0) and np.max(np.abs(fm.loc[fm.method.eq('Oracle'),'regret_kWh']))<EPS
    return {'decision_room_method_metrics_recomputed':count,'decision_hourly_rows_checked':len(dec),'all_selected_target_values_relooked_up_by_key':True,'fixed_zero_zero_gain_and_Oracle_zero_regret_checked':True}

def main():
    started=time.perf_counter();cfg=json.loads((ROOT/'configs/experiment_v4.json').read_text());run=json.loads((OUT/'run_manifest.json').read_text());assert run['status']=='completed'
    plan=json.loads((ROOT/'configs/model_plan.json').read_text());d,skipped=BASE['load_train'](ROOT/cfg['source'],plan);x=BASE['encode_inputs'](d,plan['feature_sets'][cfg['feature_set']])
    p=pd.read_parquet(OUT/'predictions/outer_PV_candidates.parquet');assert p[KEYS].equals(d[KEYS]) and set(p.Room_ID).isdisjoint(cfg['protected_test_rooms'])
    for t in TARGETS:assert np.array_equal(p['true__'+t],d[t])
    verified=verify_training(cfg,run,d,x,p);print('All 49 model reloads, fit-label hashes and 14 threshold choices verified.',flush=True)
    verified.update(evaluate_decisions(cfg,p))
    assert all(sha(ROOT/path)==value for path,value in run['input_hashes'].items())
    verified.update(status='passed',TEST_predictions=0,TEST_scores=0,TEST_rows_discarded_before_numeric_conversion=skipped,raw_and_frozen_inputs_unchanged=True,
                    evaluation_script_sha256=sha(Path(__file__)),finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
                    scope='Pipeline, pairing and saved-model verification; seven previously examined development rooms. No full training rerun or independent final TEST/field evaluation.')
    dump(OUT/'verification.json',verified);print(json.dumps(verified,indent=2),flush=True)
    print(pd.read_csv(OUT/'decision_summary.csv')[['method','mean_gain_kWh','min_gain_kWh','rooms_improved','rooms_tied','rooms_worse','mean_FUDI_change_pp']].to_string(index=False),flush=True)

if __name__=='__main__':main()
