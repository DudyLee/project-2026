"""Post-v5 diagnostic analysis; no model fit, no TEST target conversion or scoring."""
from __future__ import annotations
import copy
import itertools
import json
from pathlib import Path
import sys
import time
from datetime import datetime,timezone

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_cross_room_v5 as T

ROOT=T.ROOT;CONFIG=ROOT/'configs/diagnosis_v5.json';KEYS=T.KEYS

def corr(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else None

def summary(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);e=p-y;mse=float(np.mean(e*e));bias=float(e.mean())
    return T.extra_metrics(y,p)|{'n':len(y),'actual_mean_lux':float(y.mean()),'prediction_mean_lux':float(p.mean()),
        'pearson_r':corr(y,p),'bias_squared_fraction_of_MSE':bias*bias/mse if mse>0 else None,
        'one_sided_error_fraction':abs(bias)/float(np.mean(np.abs(e))) if np.any(e) else None,
        'overprediction_fraction':float((e>1e-9).mean()),'underprediction_fraction':float((e < -1e-9).mean()),
        'centered_error_RMSE_lux':float(np.sqrt(np.mean(np.square(e-bias)))),
        'actual_p95_lux':float(np.quantile(y,.95)),'absolute_error_p95_lux':float(np.quantile(np.abs(e),.95))}

def load_train(cfg):
    plan=copy.deepcopy(T.read(ROOT/'configs/model_plan.json'));plan['targets']=cfg['targets']
    base=plan['feature_sets'][cfg['base_feature_set']]
    plan['feature_sets']={'base':base,'auxiliary':['Sensor_Count','Floor_0based','Window_Count','Window_Area_Nominal_m2']}
    return T.BASE['load_train'](ROOT/cfg['source'],plan)

def main():
    start=time.perf_counter();cfg=T.read(CONFIG);out=ROOT/cfg['output']
    for p in [out,out/'components',out/'figures']:p.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():raise RuntimeError('Diagnosis already started; preserve its evidence.')
    assert Path(sys.prefix).resolve()==(ROOT/'.conda/envs/pvsd-ml-v1').resolve()
    inputs=[CONFIG,ROOT/cfg['source'],ROOT/'scripts/train_cross_room_v5.py',ROOT/'reports/experiment_v5/verification.json',ROOT/'reports/experiment_v5/model_index.json']
    modelindex=T.read(ROOT/'reports/experiment_v5/model_index.json')
    inputs += [ROOT/r['path'] for r in modelindex]
    inputs += [ROOT/'reports/experiment_v5/predictions'/f'{m}.parquet' for m in cfg['methods']]
    hashes={p.relative_to(ROOT).as_posix():T.sha(p) for p in inputs}
    assert T.read(ROOT/'reports/experiment_v5/verification.json')['status']=='passed'
    d,skipped=load_train(cfg);assert not set(d.Room_ID)&set(cfg['protected_test_rooms'])
    x=T.features(d,cfg)
    run={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'analysis_id':cfg['analysis_id'],
         'script_sha256':T.sha(__file__),'input_hashes':hashes,'train_rows':len(d),'train_rooms':sorted(d.Room_ID.unique().tolist()),
         'TEST_rows_skipped_before_numeric_conversion':skipped,'TEST_predictions':0,'TEST_scores':0,'new_model_fits':0,'new_simulations':0,'scope':cfg['scope']}
    T.dump(out/'run_manifest.json',run);T.dump(out/'config_snapshot.json',cfg)
    d['DNI_band']=pd.cut(d.EPW_DNI_Wh_m2,cfg['DNI_edges_Wh_m2'],labels=['0','0_to_200','200_to_500','above_500'],include_lowest=True).astype(str)
    d['GHI_band']=pd.cut(d.EPW_GHI_Wh_m2,cfg['GHI_edges_Wh_m2'],labels=['0_to_50','50_to_200','200_to_500','above_500'],include_lowest=True).astype(str)
    d['Altitude_band']=pd.cut(d.Solar_Altitude_mid_deg,cfg['altitude_edges_deg'],labels=['below_or_at_horizon','0_to_20','20_to_50','above_50'],include_lowest=True).astype(str)
    assert not d[['DNI_band','GHI_band','Altitude_band']].isin(['nan']).any().any()
    preds={m:pd.read_parquet(ROOT/'reports/experiment_v5/predictions'/f'{m}.parquet') for m in cfg['methods']}
    for p in preds.values():
        assert p[KEYS].equals(d[KEYS]) and not set(p.Room_ID)&set(cfg['protected_test_rooms'])
        for target in cfg['targets']:assert np.array_equal(p['true__'+target],d[target])
    rooms=[];strata=[];profiles=[];angledeltas=[];anglepairs=[]
    for method,p in preds.items():
        for room,ix in d.groupby('Room_ID',sort=True).groups.items():
            for target in cfg['targets']:
                y=p.loc[ix,'true__'+target].to_numpy();yp=p.loc[ix,'pred__'+target].to_numpy()
                rooms.append({'method':method,'room':room,'target':target,**summary(y,yp)})
                joined=d.loc[ix,T.META+['DNI_band','GHI_band','Altitude_band']].copy();joined['truth']=y;joined['prediction']=yp
                for axis in cfg['strata']:
                    for label,g in joined.groupby(axis,sort=True):
                        met=summary(g.truth,g.prediction)
                        strata.append({'method':method,'room':room,'target':target,'axis':axis,'group':str(label),**met})
                        if axis in ['Month','Hour','PVSD_Angle_deg']:
                            profiles.append({'method':method,'room':room,'target':target,'axis':axis,'group':int(label),
                                             'truth_mean':met['actual_mean_lux'],'pred_mean':met['prediction_mean_lux'],'MAE_lux':met['mae'],'bias_lux':met['bias']})
                true=joined.pivot(index='HOY_0based',columns='PVSD_Angle_deg',values='truth').sort_index(axis=1)
                pred=joined.pivot(index='HOY_0based',columns='PVSD_Angle_deg',values='prediction').sort_index(axis=1)
                assert len(true)==4380 and list(true.columns)==[0,30,60,90]
                for angle in [30,60,90]:
                    delta=true[angle]-true[0];dp=pred[angle]-pred[0]
                    e=dp-delta
                    angledeltas.append({'method':method,'room':room,'target':target,'angle':angle,'n':len(delta),
                       'delta_mae_lux':float(np.abs(e).mean()),'delta_rmse_lux':float(np.sqrt(np.square(e).mean())),
                       'delta_bias_lux':float(e.mean()),'delta_pearson_r':corr(delta,dp),'true_mean_delta_lux':float(delta.mean()),
                       'pred_mean_delta_lux':float(dp.mean()),'true_mean_abs_delta_lux':float(np.abs(delta).mean())})
                for threshold in cfg['angle_pair_material_gaps_lux'][target]:
                    eligible=0;correct=0;errsum=0.;opposite=0
                    for a,b in itertools.combinations([0,30,60,90],2):
                        delta=true[a].to_numpy()-true[b].to_numpy();dp=pred[a].to_numpy()-pred[b].to_numpy();mask=np.abs(delta)>threshold
                        eligible+=int(mask.sum());correct+=int((np.sign(delta[mask])==np.sign(dp[mask])).sum());opposite+=int((delta[mask]*dp[mask]<0).sum())
                        errsum+=float(np.abs(dp[mask]-delta[mask]).sum())
                    anglepairs.append({'method':method,'room':room,'target':target,'true_gap_threshold_lux':threshold,
                        'eligible_pairs':eligible,'all_pairs':4380*6,'eligible_fraction':eligible/(4380*6),
                        'correct_order_fraction':correct/eligible if eligible else None,'opposite_order_fraction':opposite/eligible if eligible else None,
                        'eligible_pair_delta_mae_lux':errsum/eligible if eligible else None})
        print(f'Room, environment and angle diagnostics complete: {method}',flush=True)
    for name,records in [('room_error_structure',rooms),('stratified_errors',strata),('temporal_profiles',profiles),('angle_delta_errors',angledeltas),('angle_pair_order',anglepairs)]:
        pd.DataFrame(records).to_csv(out/f'{name}.csv',index=False)
    decomps=[];componentfiles=[];maxidentity=0.;reloadmax=0.
    with threadpool_limits(limits=cfg['thread_limit']):
        for item in modelindex:
            bundle=joblib.load(ROOT/item['path']);held=item['room'];fs=item['feature_set'];family=item['family'];cid=bundle['selected_candidates']['relative']
            assert held not in bundle['train_rooms'] and not set(bundle['train_rooms'])&set(cfg['protected_test_rooms'])
            ix=d.index[d.Room_ID==held];xx=x[fs].loc[ix]
            mn=np.maximum(np.expm1(bundle['estimators'][cid+'__mean_log'].predict(xx)),0)
            rn=np.maximum(bundle['estimators'][cid+'__ratio'].predict(xx),0)
            native=(rn*(mn+1)).astype(float);mh=mn.astype(float);rh=rn.astype(float)
            m=d.loc[ix,'E_mean_lux'].to_numpy();y=d.loc[ix,'E_P10_lux'].to_numpy();r=y/(m+1)
            method=('A' if fs=='base' else 'B')+'_'+family;stored=preds[method].loc[ix,'pred__E_P10_lux'].to_numpy()
            diff=float(np.max(np.abs(native-stored)));reloadmax=max(reloadmax,diff);assert diff<1e-8
            dm=mh-m;dr=rh-r;term_mean=r*dm;term_ratio=(m+1)*dr;term_interaction=dm*dr;mathpred=rh*(mh+1)
            native_residual=stored-mathpred;identity=(term_mean+term_ratio+term_interaction+native_residual)-(stored-y)
            identitymax=float(np.max(np.abs(identity)));maxidentity=max(maxidentity,identitymax);assert identitymax<1e-9
            cfmean=rh*(m+1);cfratio=r*(mh+1)
            q=d.loc[ix,T.META].copy();q['true_mean']=m;q['true_P10']=y;q['true_ratio']=r;q['paired_mean_prediction']=mh;q['ratio_prediction']=rh
            q['saved_P10_prediction']=stored;q['term_mean_lux']=term_mean;q['term_ratio_lux']=term_ratio;q['term_interaction_lux']=term_interaction
            q['native_arithmetic_residual_lux']=native_residual;q['ORACLE_true_mean_substitution']=cfmean;q['ORACLE_true_ratio_substitution']=cfratio
            path=out/'components'/f'{method}__{held}.parquet';q.to_parquet(path,index=False,compression='zstd');componentfiles.append(path.relative_to(out).as_posix())
            decomps.append({'method':method,'room':held,'paired_mean_candidate':cid,'standalone_mean_candidate':bundle['selected_candidates']['mean'],
                 'actual_P10_mean_lux':float(y.mean()),'paired_mean_MAE_lux':float(np.abs(dm).mean()),'paired_mean_bias_lux':float(dm.mean()),
                 'ratio_MAE':float(np.abs(dr).mean()),'ratio_bias':float(dr.mean()),'actual_ratio_mean':float(r.mean()),'predicted_ratio_mean':float(rh.mean()),
                 'P10_MAE_lux':float(np.abs(stored-y).mean()),'P10_bias_lux':float((stored-y).mean()),
                 'mean_term_signed_mean_lux':float(term_mean.mean()),'ratio_term_signed_mean_lux':float(term_ratio.mean()),
                 'interaction_term_signed_mean_lux':float(term_interaction.mean()),
                 'mean_term_mean_absolute_lux':float(np.abs(term_mean).mean()),'ratio_term_mean_absolute_lux':float(np.abs(term_ratio).mean()),
                 'interaction_term_mean_absolute_lux':float(np.abs(term_interaction).mean()),
                 'ORACLE_true_mean_MAE_lux':float(np.abs(cfmean-y).mean()),'ORACLE_true_ratio_MAE_lux':float(np.abs(cfratio-y).mean()),
                 'native_arithmetic_residual_max_abs_lux':float(np.abs(native_residual).max()),'identity_max_abs_error':identitymax})
    pd.DataFrame(decomps).to_csv(out/'P10_error_decomposition.csv',index=False)
    comp=[]
    for family in ['RandomForest','XGBoost']:
        a=preds['A_'+family];c=preds['C_'+family]
        for room,ix in d.groupby('Room_ID').groups.items():
            y=d.loc[ix,'E_P10_lux'].to_numpy();pa=a.loc[ix,'pred__E_P10_lux'].to_numpy();pc=c.loc[ix,'pred__E_P10_lux'].to_numpy()
            ea=pa-y;ec=pc-y;aa=np.abs(ea);ac=np.abs(ec)
            comp.append({'family':family,'room':room,'A_MAE_lux':float(aa.mean()),'C_MAE_lux':float(ac.mean()),
                         'C_better_fraction':float((ac<aa-1e-9).mean()),'A_better_fraction':float((aa<ac-1e-9).mean()),
                         'opposite_error_sign_fraction':float((ea*ec<0).mean()),'error_pearson_r':corr(ea,ec),
                         'ORACLE_pointwise_best_MAE_lux':float(np.minimum(aa,ac).mean()),
                         'ORACLE_available_reduction_vs_A_lux':float(aa.mean()-np.minimum(aa,ac).mean())})
    pd.DataFrame(comp).to_csv(out/'P10_model_complementarity.csv',index=False)
    geometry=d.groupby('Room_ID',sort=True)[['Room_Area_m2','Glass_Area_m2','Window_Count','Window_Area_Nominal_m2','Sensor_Count','Floor_0based','Orientation_X_model','Orientation_Y_model','Solar_Obstruction_Ratio']].first().reset_index()
    geometry['sensor_count_per_floor_m2']=geometry.Sensor_Count/geometry.Room_Area_m2
    geometry.to_csv(out/'TRAIN_geometry_and_sensor_metadata.csv',index=False)
    assert all(T.sha(ROOT/p)==h for p,h in hashes.items())
    run.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
               existing_model_bundles_reloaded=len(modelindex),new_component_inference_rows=28*17520,
               component_files=componentfiles,reload_max_absolute_difference_lux=reloadmax,
               algebraic_identity_max_absolute_error_lux=maxidentity,input_files_unchanged=True)
    T.dump(out/'run_manifest.json',run)
    print(f'Completed: {len(rooms)} room summaries; {len(strata)} stratified groups; {len(decomps)} P10 decompositions; zero fits/TEST operations.',flush=True)

if __name__=='__main__':main()
