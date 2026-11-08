"""Read existing TRAIN data and saved C models only. No fitting or TEST inference."""
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import mean_exposure_v9 as V
from train_mean_exposure_v9 import idx,sha

D=V.D;ROOT=V.ROOT
OUT=ROOT/'reports/diagnosis_BM432_LM608_v9'


def main():
    OUT.mkdir(parents=True,exist_ok=True);cfg=D.read(V.CONFIG)
    d,audit,profiles=V.load(cfg);x,q=V.features(d,cfg)
    paths=[ROOT/cfg['source'],ROOT/cfg['bim_source'],ROOT/'models/trial_v0_1/model.joblib',ROOT/'reports/experiment_v9/fold_metrics.csv',ROOT/'reports/experiment_v9/stratified_metrics.csv']
    paths += [ROOT/f'models/experiment_v9/room_{r}.joblib' for r in ['BM432','LM608']]
    paths += [ROOT/f'reports/experiment_v9/outer_predictions/room_{r}.parquet' for r in ['BM432','LM608']]
    frozen={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    profiles['true_mean_lux']=d.groupby('Room_ID').E_mean_lux.mean()
    profiles['true_mean_divided_by_q']=profiles.true_mean_lux/profiles.q
    profiles.to_csv(OUT/'development_scales.csv')
    angles=d.groupby(['Room_ID','PVSD_Angle_deg']).E_mean_lux.mean().unstack()
    angles.to_csv(OUT/'angle_true_means.csv');angles.div(angles[0],axis=0).to_csv(OUT/'angle_ratios_to_zero.csv')
    fm=pd.read_csv(ROOT/'reports/experiment_v9/fold_metrics.csv');sm=pd.read_csv(ROOT/'reports/experiment_v9/stratified_metrics.csv')
    fm[fm.room.isin(['BM432','LM608'])].to_csv(OUT/'pair_metrics.csv',index=False)
    sm[sm.room.isin(['BM432','LM608'])].to_csv(OUT/'pair_stratified_metrics.csv',index=False)
    supports=[];splits=[];modelchecks=[];matched=[]
    with threadpool_limits(limits=4):
        for room in ['BM432','LM608']:
            bundle=joblib.load(ROOT/f'models/experiment_v9/room_{room}.joblib');model=bundle['estimators']['C']
            fit=idx(d,bundle['job']['fit_rooms']);held=idx(d,[room]);z=x['F2']
            assert room not in bundle['job']['fit_rooms'] and not set(bundle['job']['fit_rooms'])&set(cfg['excluded_test_rooms'])
            pred,raw=V.predict(model,z.loc[held],q[held],'C')
            saved=pd.read_parquet(ROOT/f'reports/experiment_v9/outer_predictions/room_{room}.parquet')
            np.testing.assert_allclose(pred,saved.pred__C,rtol=1e-10,atol=1e-8)
            np.testing.assert_allclose(raw,saved.raw__C,rtol=1e-10,atol=1e-8)
            for method in ['A','B','C','T1','S1']:
                for key,value in D.base.metrics(saved.true__E_mean_lux,saved['pred__'+method]).items():
                    np.testing.assert_allclose(value,fm[(fm.protocol=='LeaveOneRoomOut')&(fm.room==room)&(fm.method==method)][key].iloc[0],rtol=1e-10,atol=1e-8)
            leaves_fit=np.asarray(model.apply(z.loc[fit]),dtype=int);leaves_held=np.asarray(model.apply(z.loc[held]),dtype=int)
            weights=q[fit]/q[fit].mean();fit_rooms=d.loc[fit,'Room_ID'].to_numpy()
            result={r:np.zeros(2) for r in bundle['job']['fit_rooms']}
            for tree in range(leaves_fit.shape[1]):
                a,b=leaves_fit[:,tree],leaves_held[:,tree];n=int(max(a.max(),b.max()))+1
                counts=np.bincount(a,minlength=n);mass=np.bincount(a,weights=weights,minlength=n);freq=np.bincount(b,minlength=n)
                used=freq>0;assert (counts[used]>0).all() and (mass[used]>0).all()
                for r in result:
                    member=fit_rooms==r
                    rc=np.bincount(a[member],minlength=n);rw=np.bincount(a[member],weights=weights[member],minlength=n)
                    result[r] += [np.dot(freq[used],rc[used]/counts[used])/len(held),np.dot(freq[used],rw[used]/mass[used])/len(held)]
            for r,value in result.items():
                value/=leaves_fit.shape[1]
                supports.append({'held_room':room,'training_room':r,'unweighted_shared_leaf_fraction':float(value[0]),'weighted_shared_leaf_fraction':float(value[1]),
                    'training_row_fraction':float((fit_rooms==r).mean()),'training_weight_fraction':float(weights[fit_rooms==r].sum()/weights.sum())})
            assert np.allclose(np.sum(list(result.values()),axis=0),1)
            trees=model.get_booster().trees_to_dataframe();g=trees[trees.Feature!='Leaf']
            for feature,h in g.groupby('Feature'):
                name=z.columns[int(feature[1:])] if feature.startswith('f') and feature[1:].isdigit() else feature
                assert name in z.columns
                splits.append({'held_room':room,'feature':name,'native_feature_name':feature,'split_count':len(h),'sum_training_gain':float(h.Gain.sum())})
            modelchecks.append({'held_room':room,'fit_rooms':bundle['job']['fit_rooms'],'C_depth':bundle['selected']['C'],
                'reproduced_prediction_rows':len(pred),'native_normalized_prediction_mean':float((pred/q[held]).mean()),
                'native_normalized_truth_mean':float((d.loc[held,V.TARGET].to_numpy()/q[held]).mean())})
            g=fm[(fm.protocol=='LeaveOneRoomOut')&(fm.room==room)].set_index('method')
            matched.append({'room':room,'A_MAE':float(g.loc['A','mae']),'B_MAE':float(g.loc['B','mae']),'C_MAE':float(g.loc['C','mae']),
                'C_minus_B_MAE':float(g.loc['C','mae']-g.loc['B','mae']),'C_minus_S1_MAE':float(g.loc['C','mae']-g.loc['S1','mae'])})
    pd.DataFrame(supports).to_csv(OUT/'C_leaf_cooccurrence.csv',index=False)
    pd.DataFrame(splits).to_csv(OUT/'C_split_features.csv',index=False)
    pd.DataFrame(matched).to_csv(OUT/'matched_method_comparisons.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in frozen.items())
    overall=fm[(fm.protocol=='LeaveOneRoomOut')&(fm.method=='C')].set_index('room').mae
    other_sum=float(overall.drop(['BM432','LM608']).sum())
    result={'status':'completed_verified_existing_model_diagnostics','completed_utc':datetime.now(timezone.utc).isoformat(),
        'estimator_fits':0,'TEST_numeric_rows':0,'TEST_predictions':0,'TEST_scores':0,'TRAIN_rows':len(d),'TEST_rows_skipped':52560,
        'saved_model_prediction_rows_reproduced':35040,'leaf_assignment_calls_on_existing_models':4,'protected_hashes_unchanged':True,'input_hashes':frozen,
        'matched_comparisons':matched,'model_checks':modelchecks,'C_other_five_MAE_sum':other_sum,
        'macro_if_BM432_120_LM608_40_others_as_C':(other_sum+120+40)/7,
        'q_ratio_BM432_to_LM608':float(profiles.loc['BM432','q']/profiles.loc['LM608','q']),
        'true_mean_ratio_BM432_to_LM608':float(profiles.loc['BM432','true_mean_lux']/profiles.loc['LM608','true_mean_lux']),
        'limits':['Leaf cooccurrence is a structural overlap diagnostic averaged over validation rows and trees, not additive prediction attribution or causal influence.',
            'B/C use the same input/loss family but depth selection and transformed-label/weight geometry differ; this is not a pure one-variable causal test.',
            'Normalized target means and observed angle ratios are retrospective diagnostics, never proposed input labels or room calibration constants.',
            'Seven development rooms are repeatedly examined; upstream physical provenance remains unverified.']}
    D.dump(OUT/'analysis.json',result)
    print(pd.DataFrame(supports).round(4).to_string(index=False))
    print(pd.DataFrame(modelchecks).to_string(index=False))
    print('No fits; all recorded metrics and stored C predictions reproduced. Conditional macro budget:',result['macro_if_BM432_120_LM608_40_others_as_C'])


if __name__=='__main__':main()
