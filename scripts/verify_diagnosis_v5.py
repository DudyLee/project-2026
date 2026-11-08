"""Recompute diagnostic identities and all grouping/angle summaries; zero fitting."""
import itertools
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import train_cross_room_v5 as T
import diagnose_cross_room_v5 as D

ROOT=T.ROOT;OUT=ROOT/'reports/cross_room_diagnosis_v5'

def main():
    cfg=T.read(ROOT/'configs/diagnosis_v5.json');run=T.read(OUT/'run_manifest.json')
    assert run['status']=='completed' and run['new_model_fits']==0 and run['TEST_predictions']==0
    assert T.sha(ROOT/'scripts/diagnose_cross_room_v5.py')==run['script_sha256']
    assert all(T.sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    summary=pd.read_csv(OUT/'room_error_structure.csv');strata=pd.read_csv(OUT/'stratified_errors.csv')
    angles=pd.read_csv(OUT/'angle_delta_errors.csv');pairs=pd.read_csv(OUT/'angle_pair_order.csv')
    decomp=pd.read_csv(OUT/'P10_error_decomposition.csv');comp=pd.read_csv(OUT/'P10_model_complementarity.csv')
    allpreds={m:pd.read_parquet(ROOT/'reports/experiment_v5/predictions'/f'{m}.parquet') for m in cfg['methods']}
    assert len(summary)==126 and len(strata)==5040 and len(angles)==378 and len(pairs)==252 and len(decomp)==28
    for method,p in allpreds.items():
        assert len(p)==122640 and not set(p.Room_ID)&set(cfg['protected_test_rooms'])
        for room,g in p.groupby('Room_ID'):
            for target in cfg['targets']:
                y=g['true__'+target].to_numpy();yp=g['pred__'+target].to_numpy();err=yp-y
                r=summary[(summary.method==method)&(summary.room==room)&(summary.target==target)].iloc[0]
                assert np.isclose(r.mae,np.abs(err).mean(),rtol=1e-11)
                assert np.isclose(r.rmse**2,r.bias**2+r.centered_error_RMSE_lux**2,rtol=1e-11)
                assert np.isclose(r.bias,err.mean(),rtol=1e-11,atol=1e-11)
                sub=strata[(strata.method==method)&(strata.room==room)&(strata.target==target)]
                for axis,s in sub.groupby('axis'):
                    assert s.n.sum()==len(y)
                    assert np.isclose((s.mae*s.n).sum()/s.n.sum(),r.mae,rtol=1e-11)
                    assert np.isclose((s.bias*s.n).sum()/s.n.sum(),r.bias,rtol=1e-11,atol=1e-11)
                    assert np.isclose((s.rmse**2*s.n).sum()/s.n.sum(),r.rmse**2,rtol=1e-11)
                ypvt=g.pivot(index='HOY_0based',columns='PVSD_Angle_deg',values='true__'+target)
                ppvt=g.pivot(index='HOY_0based',columns='PVSD_Angle_deg',values='pred__'+target)
                for angle in [30,60,90]:
                    e=(ppvt[angle]-ppvt[0])-(ypvt[angle]-ypvt[0]);a=angles[(angles.method==method)&(angles.room==room)&(angles.target==target)&(angles.angle==angle)].iloc[0]
                    assert np.isclose(a.delta_mae_lux,np.abs(e).mean(),rtol=1e-11,atol=1e-11)
                for threshold in cfg['angle_pair_material_gaps_lux'][target]:
                    a=pairs[(pairs.method==method)&(pairs.room==room)&(pairs.target==target)&(pairs.true_gap_threshold_lux==threshold)].iloc[0]
                    masks=[];correct=[];errors=[]
                    for aa,bb in itertools.combinations([0,30,60,90],2):
                        t=(ypvt[aa]-ypvt[bb]).to_numpy();pred=(ppvt[aa]-ppvt[bb]).to_numpy();mask=np.abs(t)>threshold
                        masks.append(mask);correct.append((np.sign(t)==np.sign(pred))&mask);errors.append(np.abs(pred-t)[mask])
                    n=sum(v.sum() for v in masks);assert a.eligible_pairs==n
                    if n:
                        assert np.isclose(a.correct_order_fraction,sum(v.sum() for v in correct)/n,rtol=1e-11)
                        assert np.isclose(a.eligible_pair_delta_mae_lux,np.concatenate(errors).mean(),rtol=1e-11)
    count=0;identity_max=0.
    for rel in run['component_files']:
        q=pd.read_parquet(OUT/rel);method,room=(OUT/rel).stem.split('__')
        assert q.Room_ID.nunique()==1 and q.Room_ID.iloc[0]==room and room not in cfg['protected_test_rooms']
        p=allpreds[method].query('Room_ID == @room').reset_index(drop=True)
        assert q[T.KEYS].reset_index(drop=True).equals(p[T.KEYS])
        assert np.array_equal(q.saved_P10_prediction,p.pred__E_P10_lux)
        e=q.saved_P10_prediction-q.true_P10
        reconstructed=q.term_mean_lux+q.term_ratio_lux+q.term_interaction_lux+q.native_arithmetic_residual_lux
        maxerr=float(np.max(np.abs(e-reconstructed)));assert maxerr<1e-9;identity_max=max(identity_max,maxerr)
        assert np.allclose(q.true_ratio,q.true_P10/(q.true_mean+1),rtol=1e-14)
        assert np.allclose(q.ORACLE_true_mean_substitution,q.ratio_prediction*(q.true_mean+1),rtol=1e-14)
        assert np.allclose(q.ORACLE_true_ratio_substitution,q.true_ratio*(q.paired_mean_prediction+1),rtol=1e-14)
        s=decomp[(decomp.method==method)&(decomp.room==room)].iloc[0]
        for field,values in [('P10_MAE_lux',e),('ORACLE_true_mean_MAE_lux',q.ORACLE_true_mean_substitution-q.true_P10),('ORACLE_true_ratio_MAE_lux',q.ORACLE_true_ratio_substitution-q.true_P10)]:
            assert np.isclose(s[field],np.abs(values).mean(),rtol=1e-11)
        count+=len(q)
    for r in comp.to_dict('records'):
        a=allpreds['A_'+r['family']]
        a=a[a.Room_ID==r['room']];c=allpreds['C_'+r['family']];c=c[c.Room_ID==r['room']]
        aa=np.abs(a.pred__E_P10_lux-a.true__E_P10_lux);ac=np.abs(c.pred__E_P10_lux-c.true__E_P10_lux)
        assert np.isclose(r['ORACLE_pointwise_best_MAE_lux'],np.minimum(aa,ac).mean(),rtol=1e-11)
        assert r['ORACLE_pointwise_best_MAE_lux']<=min(r['A_MAE_lux'],r['C_MAE_lux'])+1e-9
    handoff=T.read(OUT/'handoff_manifest.json');assert all(T.sha(ROOT/p)==h for p,h in handoff['files'].items())
    T.dump(OUT/'verification.json',{'status':'passed','room_summaries_checked':126,'stratified_group_reaggregation_checks':126*6,
        'angle_delta_metrics_recomputed':378,'angle_pair_metrics_recomputed':252,'P10_component_rows_checked':count,
        'P10_error_identity_max_abs_lux':identity_max,'counterfactual_rows_checked':2*count,'complementarity_room_rows_checked':len(comp),
        'all_source_models_predictions_unchanged':True,'new_model_fits':0,'TEST_predictions':0,'TEST_scores':0,
        'counterfactuals_are_explanatory_only_not_deployable_scores':True,'script_sha256':T.sha(__file__),
        'finished_utc':datetime.now(timezone.utc).isoformat(),'scope':'Numerical diagnostic reconstruction and artifact checks; not new predictive validation or source simulation verification.'})
    print(f'Verified 126 room summaries, 756 group totals, 630 angle summaries and {count} P10 decomposition rows; zero fits, TEST sealed.')

if __name__=='__main__':main()
