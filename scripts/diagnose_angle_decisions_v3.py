"""Read-only diagnosis of frozen TRAIN predictions. No fitting or new policies."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'reports/experiment_v3'
OUT=ROOT/'reports/decision_diagnosis_v3'
TARGETS=['E_mean_lux','E_P10_lux','F_300_fraction','F_UDI_fraction','PV_AC_kWh']
METHODS=['Frozen_HistGB','HistGB','RandomForest','XGBoost','Nested_select']
POLICIES=['PV_priority','Daylight_priority','Balanced']
ANGLES=np.array([0,30,60,90]);EPS=1e-12

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rate(x):return float(np.mean(x))
def mean_if(x,mask):return float(np.mean(x[mask])) if np.any(mask) else None
def daylight(a):return (a[:,:,2]+a[:,:,3])/2
def feasible(a):return (a[:,:,2]>=.5)&(a[:,:,3]>=.5)
def deficit(a):return np.maximum(0,.5-a[:,:,2])+np.maximum(0,.5-a[:,:,3])
def score(a,policy,s):return a[:,:,4] if policy=='PV_priority' else daylight(a) if policy=='Daylight_priority' else .5*daylight(a)+.5*a[:,:,4]/s
def cchoose(a):
    f=feasible(a);ds=deficit(a)
    eligible=np.where(f.any(axis=1)[:,None],f,np.isclose(ds,ds.min(axis=1)[:,None],rtol=0,atol=EPS))
    return np.argmax(np.where(eligible,a[:,:,4],-np.inf),axis=1)

def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    if (OUT/'analysis.json').exists():raise RuntimeError('Diagnosis already completed; preserve it.')
    paths=[INPUT/'predictions'/f'{m}.parquet' for m in METHODS]+[ROOT/'configs/experiment_v3.json',INPUT/'decision_by_room.csv',INPUT/'decision_training_scales.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    cfg=json.loads((ROOT/'configs/experiment_v3.json').read_text());scales={r['room']:r['pv_scale_q95_positive_kWh'] for r in json.loads((INPUT/'decision_training_scales.json').read_text())}
    preds={m:pd.read_parquet(INPUT/'predictions'/f'{m}.parquet') for m in METHODS};base=preds['Frozen_HistGB']
    rooms=sorted(base.Room_ID.unique());assert rooms==['BM223','BM432','BM441','BM522','BM608','LM608','LM703']
    key=['Room_ID','HOY_0based','PVSD_Angle_deg']
    for p in preds.values():
        assert p[key].equals(base[key]);assert not set(p.Room_ID)&set(cfg['protected_test_rooms'])
        for t in TARGETS:assert np.array_equal(p['true__'+t],base['true__'+t])
    saved=pd.read_csv(INPUT/'decision_by_room.csv');saved_checks=0
    profiles=[];data_profiles=[];headroom=[];terms=[];decomp=[];pairs=[];constraints=[];segments=[]
    for room in rooms:
        g=base[base.Room_ID.eq(room)].sort_values(['HOY_0based','PVSD_Angle_deg'])
        a=g[['true__'+t for t in TARGETS]].to_numpy().reshape(4380,4,5);n=len(a);idx=np.arange(n);s=scales[room]
        meta=g.iloc[::4][['HOY_0based','Month','Hour']].reset_index(drop=True)
        cube={m:p[p.Room_ID.eq(room)].sort_values(['HOY_0based','PVSD_Angle_deg'])[['pred__'+t for t in TARGETS]].to_numpy().reshape(n,4,5) for m,p in preds.items()}
        D=daylight(a);P=a[:,:,4];Fn=a[:,:,2:4]
        all_equal=lambda v:(v.max(axis=1)-v.min(axis=1))<=EPS
        best=lambda v:v[:,0]>=v.max(axis=1)-EPS
        data_profiles.append({'room':room,'F300_zero_best_fraction':rate(best(a[:,:,2])),'FUDI_zero_best_fraction':rate(best(a[:,:,3])),'PV_zero_best_fraction':rate(best(P)),
            'zero_weakly_dominates_all_F300_FUDI_PV_fraction':rate((Fn[:,0:1,:]>=Fn-EPS).all(axis=(1,2))&(P[:,0]>=P.max(axis=1)-EPS)),
            'F300_all_four_zero_fraction':rate((a[:,:,2]==0).all(axis=1)),'FUDI_all_four_zero_fraction':rate((a[:,:,3]==0).all(axis=1)),
            'PV_all_four_zero_fraction':rate((P==0).all(axis=1)),'D_all_four_equal_fraction':rate(all_equal(D)),
            'FUDI_0_vs_30_nonzero_change_fraction':rate(np.abs(a[:,1,3]-a[:,0,3])>EPS),
            'positive_daylight_positive_PV_range_correlation':float(np.corrcoef(np.ptp(D,axis=1),np.ptp(P/s,axis=1))[0,1])})
        for k,angle in enumerate(ANGLES):
            profiles.append({'room':room,'angle':int(angle),'PV_retained_hours_kWh':float(P[:,k].sum()),'F300_mean':float(a[:,k,2].mean()),'FUDI_mean':float(a[:,k,3].mean()),
                'D_mean':float(D[:,k].mean()),'E_mean_lux_mean':float(a[:,k,0].mean()),'P10_lux_mean':float(a[:,k,1].mean()),'feasible_fraction':rate(feasible(a)[:,k])})
        for policy in POLICIES:
            v=score(a,policy,s);v0=v[:,0];oracle=v.max(axis=1);tol=cfg['decision']['near_optimal_tolerances'][policy]
            gain=oracle-v0;opt=np.abs(v-oracle[:,None])<=EPS;span=np.ptp(v,axis=1);margin=np.sort(v,axis=1)[:,-1]-np.sort(v,axis=1)[:,-2]
            headroom.append({'room':room,'policy':policy,'zero_in_optimal_set_fraction':rate(opt[:,0]),'zero_unique_best_fraction':rate(opt[:,0]&(opt.sum(axis=1)==1)),
                'zero_tied_best_fraction':rate(opt[:,0]&(opt.sum(axis=1)>1)),'all_four_equal_fraction':rate(span<=EPS),'span_within_existing_near_tolerance_fraction':rate(span<=tol+EPS),
                'zero_near_optimal_fraction':rate(gain<=tol+EPS),'material_opportunity_fraction':rate(gain>tol+EPS),'mean_oracle_gain_over_zero':float(gain.mean()),
                'sum_oracle_gain_over_zero':float(gain.sum()),'mean_best_second_gap':float(margin.mean()),'median_best_second_gap':float(np.median(margin)),
                'near_tolerance':tol,'PV_oracle_gain_over_zero_fraction':float(gain.sum()/P[:,0].sum()) if policy=='PV_priority' else None})
            if policy=='Balanced':
                for k in range(1,4):
                    dt=.5*(D[:,k]-D[:,0]);pt=.5*(P[:,k]-P[:,0])/s;total=dt+pt
                    terms.append({'room':room,'angle':int(ANGLES[k]),'daylight_delta_term_mean':float(dt.mean()),'PV_delta_term_mean':float(pt.mean()),
                        'daylight_delta_term_mean_abs':float(np.abs(dt).mean()),'PV_delta_term_mean_abs':float(np.abs(pt).mean()),
                        'opposing_sign_fraction':rate(dt*pt < -EPS),'daylight_term_larger_abs_fraction':rate(np.abs(dt)>np.abs(pt)+EPS),
                        'both_terms_nonpositive_fraction':rate((dt<=EPS)&(pt<=EPS)),
                        'PV_gain_canceled_by_daylight_fraction':rate((pt>EPS)&(total<-EPS)),
                        'positive_total_gain_fraction':rate(total>EPS)})
            for method,p in cube.items():
                vp=score(p,policy,s);choice=np.argmax(vp,axis=1);chosen=v[idx,choice];actual_gain=chosen-v0
                switched=choice!=0;benefit=actual_gain>EPS;harm=actual_gain<-EPS;meaningful=gain>tol+EPS
                fixed_regret=oracle-v0;model_regret=oracle-chosen
                saved_row=saved[(saved.room==room)&(saved.policy==policy)&(saved.strategy==method)].iloc[0]
                assert np.isclose(model_regret.mean(),saved_row.policy_loss_mean,rtol=1e-10,atol=1e-12);saved_checks+=1
                benefit_sum=float(actual_gain[benefit].sum());harm_sum=float(-actual_gain[harm].sum())
                assert np.isclose(model_regret.sum()-fixed_regret.sum(),harm_sum-benefit_sum,rtol=1e-9,atol=1e-10)
                row={'room':room,'policy':policy,'method':method,'switch_fraction':rate(switched),'beneficial_switch_fraction_all_hours':rate(switched&benefit),
                    'harmful_switch_fraction_all_hours':rate(switched&harm),'neutral_switch_fraction_all_hours':rate(switched&~benefit&~harm),
                    'beneficial_fraction_among_switches':mean_if(benefit.astype(float),switched),'harmful_fraction_among_switches':mean_if(harm.astype(float),switched),
                    'fixed_regret_mean':float(fixed_regret.mean()),'model_regret_mean':float(model_regret.mean()),'model_minus_fixed_regret':float((model_regret-fixed_regret).mean()),
                    'benefit_sum':benefit_sum,'harm_sum':harm_sum,'net_gain_over_zero_sum':benefit_sum-harm_sum,
                    'captured_material_opportunity_fraction':mean_if((actual_gain>tol+EPS).astype(float),meaningful),'material_opportunity_hours':int(meaningful.sum()),
                    'material_harmful_switch_fraction':rate(actual_gain < -tol-EPS),'predicted_all_four_equal_fraction':rate(np.ptp(vp,axis=1)<=EPS),
                    'mean_predicted_span':float(np.ptp(vp,axis=1).mean()),'mean_true_span':float(span.mean()),
                    'chosen_gain_prediction_mae':float(np.abs((vp[idx,choice]-vp[:,0])-actual_gain).mean())}
                decomp.append(row)
                for label,vals,pp,delta_tol in [('F300',a[:,:,2],p[:,:,2],.05),('FUDI',a[:,:,3],p[:,:,3],.05),('PV',P,p[:,:,4],.0001),('policy_score',v,vp,tol)]:
                    truth_delta=vals[:,1:]-vals[:,0:1];pred_delta=pp[:,1:]-pp[:,0:1];mask=np.abs(truth_delta)>delta_tol+EPS
                    pairs.append({'room':room,'policy':policy,'method':method,'target':label,'absolute_value_mae':float(np.abs(pp-vals).mean()),
                                  'angle_delta_mae':float(np.abs(pred_delta-truth_delta).mean()),'meaningful_delta_tolerance':delta_tol,
                                  'meaningful_delta_pair_fraction':rate(mask),'sign_correct_on_meaningful_deltas':mean_if((np.sign(pred_delta)==np.sign(truth_delta)).astype(float),mask)})
                for by in ['Month','Hour']:
                    for val in sorted(meta[by].unique()):
                        mask=meta[by].to_numpy()==val
                        segments.append({'room':room,'policy':policy,'method':method,'group_by':by,'value':int(val),'hours':int(mask.sum()),
                                         'fixed_regret_mean':float(fixed_regret[mask].mean()),'model_regret_mean':float(model_regret[mask].mean()),
                                         'model_minus_fixed_regret':float((model_regret-fixed_regret)[mask].mean())})
        tf=feasible(a);available=tf.any(axis=1);f0=tf[:,0];c_oracle=cchoose(a)
        for method,p in cube.items():
            pred_f=feasible(p);choice=cchoose(p);sel_true=tf[idx,choice];sel_pred=pred_f[idx,choice]
            both=f0&sel_true;true_gain=P[idx,choice]-P[:,0]
            constraints.append({'room':room,'method':method,'true_any_feasible_fraction':rate(available),'true_no_feasible_fraction':rate(~available),
                'zero_feasible_fraction':rate(f0),'pred_any_feasible_fraction':rate(pred_f.any(axis=1)),'selected_true_feasible_fraction':rate(sel_true),
                'predicted_feasible_but_actually_not_fraction':rate(sel_pred&~sel_true),'predicted_no_feasible_but_true_available_fraction':rate(~pred_f.any(axis=1)&available),
                'lost_feasibility_vs_zero_fraction':rate(f0&~sel_true),'gained_feasibility_vs_zero_fraction':rate(~f0&sel_true),
                'PV_net_gain_on_common_feasible_hours_kWh':float(true_gain[both].sum()),'common_feasible_hours':int(both.sum()),
                'PV_positive_gain_on_common_feasible_hours_kWh':float(true_gain[both& (true_gain>0)].sum()),'PV_loss_on_common_feasible_hours_kWh':float(-true_gain[both&(true_gain<0)].sum()),
                'zero_covers_available_fraction':float(f0.sum()/available.sum()) if available.any() else None})
    frames={name:pd.DataFrame(rows) for name,rows in [('angle_profiles',profiles),('data_profiles',data_profiles),('headroom',headroom),('balanced_terms',terms),('switch_decomposition',decomp),('pairwise_errors',pairs),('constraint_diagnosis',constraints),('error_segments',segments)]}
    for name,f in frames.items():f.to_csv(OUT/f'{name}.csv',index=False)
    for name,groups in [('angle_profiles',['angle']),('headroom',['policy']),('balanced_terms',['angle']),('switch_decomposition',['policy','method']),('pairwise_errors',['policy','method','target']),('constraint_diagnosis',['method'])]:
        f=frames[name];numbers=[c for c in f.select_dtypes('number').columns if c not in groups]
        f.groupby(groups,sort=False)[numbers].mean().reset_index().to_csv(OUT/f'{name}_summary.csv',index=False)
    frames['data_profiles'].select_dtypes('number').mean().to_csv(OUT/'data_profile_macro_means.csv',header=['mean_over_7_rooms'])
    assert saved_checks==105 and all(sha(ROOT/p)==h for p,h in hashes.items())
    dump(OUT/'analysis.json',{'status':'ANALYZED','scope':'Retrospective diagnosis using only frozen TRAIN OOF predictions and existing rules. No alternative policy, threshold or weight was evaluated.',
         'models_fitted':0,'model_predictions_generated':0,'hyperparameters_tuned':0,'TEST_rows_read':0,'TEST_predictions':0,'TEST_scores':0,'raw_source_reopened':False,
         'rooms':rooms,'input_hashes':hashes,'inputs_unchanged':True,'existing_decision_rows_reproduced':105,'switch_regret_identity_checks':105,
         'elapsed_seconds':time.perf_counter()-started,'analysis_script_sha256':sha(Path(__file__)),
         'interpretation_limits':['Seven development rooms previously examined; descriptive evidence, not a causal attribution percentage','Four fixed angle simulations, one building and weather year','Numerical ties use 1e-12; material-score thresholds reuse v3 predeclared near-optimal tolerances','Constrained_PV primary saved loss measures daylight deficit only; separate feasibility and conditional PV are required'],
         'statistical_fallacy_scan':{'coverage':'11/11 reviewed','Simpson_aggregation':'Report room-level and macro results; kWh and mean relative percentages can rank strategies differently','Ecological_inference':'No claim for all 476 rooms or real-world operation','Selection_Berkson':'Only seven selected representative development rooms; population representativeness unverified','Collider_control':'No new covariate controls or causal model used','Base_rate':'Report optimal-angle frequency, ties and no-feasible base rates','Regression_to_mean':'Frozen references retained; no causal before-after claim','Survivorship':'All seven TRAIN rooms and all retained hours included; omitted hours remain out of scope','Look_elsewhere':'All five saved methods and all existing policies reported, no significance tests','Forking_paths':'Post-hoc descriptive diagnosis explicitly labeled; no alternate objectives evaluated','Correlation_causation':'Mechanistic score decomposition distinguished from unverified simulation physics','Reverse_causality':'No causal inference from predictive correlations'}})
    for name in ['angle_profiles_summary','headroom_summary','switch_decomposition_summary','constraint_diagnosis_summary']:
        print(name,flush=True);print(pd.read_csv(OUT/f'{name}.csv').to_string(index=False),flush=True)
    print('Diagnosis completed: zero fits, zero new model predictions, no TEST or raw CSV read.',flush=True)

if __name__=='__main__':main()
