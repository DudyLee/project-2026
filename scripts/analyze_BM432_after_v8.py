"""TRAIN-only descriptive analysis of saved v8 predictions; zero fitting/inference."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import directional_v8 as D

ROOT=D.ROOT
out=ROOT/'reports/diagnosis_BM432_after_v8'
out.mkdir(parents=True,exist_ok=True)
cfg=D.read(D.CONFIG)
d,skipped=D.base.load_train(ROOT/cfg['source'],D.read(ROOT/'configs/model_plan.json'))
source=ROOT/'reports/experiment_v8/outer_predictions/room_BM432.parquet'
p=pd.read_parquet(source).merge(d[D.KEYS+['EPW_DNI_Wh_m2']],on=D.KEYS,validate='one_to_one')
assert len(p)==17520 and p.Room_ID.eq('BM432').all()
p['truth_bin']=pd.cut(p.true__E_mean_lux,[-1,100,300,1000,np.inf],labels=['<=100','100-300','300-1000','>1000'])
p['DNI_bin']=pd.cut(p.EPW_DNI_Wh_m2,[-1,0,200,500,np.inf],labels=['0','0-200','200-500','>500'])
rows=[]
for dimension in ['truth_bin','DNI_bin','PVSD_Angle_deg','Hour']:
    for value,g in p.groupby(dimension,observed=True):
        for method in ['T0','T1','S0','S1']:
            y=g.true__E_mean_lux;pred=g['pred__'+method+'__E_mean_lux'];err=pred-y
            total=(p['pred__'+method+'__E_mean_lux']-p.true__E_mean_lux).abs().sum()
            rows.append({'dimension':dimension,'value':str(value),'method':method,'n':len(g),'row_fraction':len(g)/len(p),
                'true_mean':float(y.mean()),'pred_mean':float(pred.mean()),'mae':float(err.abs().mean()),'bias':float(err.mean()),
                'absolute_error_share':float(err.abs().sum()/total),'underprediction_fraction':float((err<0).mean())})
pd.DataFrame(rows).to_csv(out/'BM432_stratified_errors.csv',index=False)
cols=['Room_ID','Room_Area_m2','Glass_Area_m2','Window_Count','WFR_window_to_floor','Solar_Obstruction_Ratio',
      'Orientation_X_model','Orientation_Y_model','Annual_Incident_Shaded_kWh_m2','Annual_Incident_Unshaded_kWh_m2','Sensor_Count']
seen={}
with (ROOT/cfg['source']).open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['Sample_Role']!='TRAIN' or r['Room_ID'] in seen:continue
        seen[r['Room_ID']]={c:r[c] if c=='Room_ID' else float(r[c]) for c in cols}
profiles=pd.DataFrame(seen.values()).set_index('Room_ID')
assert set(profiles.index)==set(cfg['development_rooms'])
assert np.allclose(profiles.Solar_Obstruction_Ratio,1-profiles.Annual_Incident_Shaded_kWh_m2/profiles.Annual_Incident_Unshaded_kWh_m2)
profiles['annual_exposure_index_q']=profiles.Glass_Area_m2/profiles.Room_Area_m2*profiles.Annual_Incident_Shaded_kWh_m2/100
assert profiles.annual_exposure_index_q.gt(0).all()
for stat in ['mean','median','max']:profiles['E_mean_'+stat]=d.groupby('Room_ID').E_mean_lux.agg(stat)
profiles.to_csv(out/'development_profiles.csv')
fm=pd.read_csv(ROOT/'reports/experiment_v8/fold_metrics.csv')
s=fm[(fm.protocol=='LeaveOneRoomOut')&(fm.method=='S1')&(fm.target=='E_mean_lux')].set_index('room')
budget=7*50-s.loc[s.index!='BM432','mae'].sum()
bright=p.true__E_mean_lux.gt(300);error=(p.pred__S1__E_mean_lux-p.true__E_mean_lux).abs()
result={'status':'descriptive_analysis_only','estimator_fits':0,'new_predictions':0,'TEST_numeric_rows':0,'TEST_rows_skipped':skipped,
        'source_sha256':D.base.sha256(ROOT/cfg['source']),'prediction_source':source.relative_to(ROOT).as_posix(),'prediction_sha256':D.base.sha256(source),
        'S1_mean_room_MAE':float(s.mae.mean()),'S1_BM432_MAE':float(s.loc['BM432','mae']),
        'S1_other_six_MAE_sum':float(s.loc[s.index!='BM432','mae'].sum()),
        'BM432_MAE_needed_for_macro_50_if_other_rooms_unchanged':float(budget),
        'required_BM432_reduction_fraction':float(1-budget/s.loc['BM432','mae']),
        'BM432_rows_above_300_fraction':float(bright.mean()),'BM432_above_300_absolute_error_share':float(error[bright].sum()/error.sum()),
        'S1_BM432_underprediction_fraction':float((p.pred__S1__E_mean_lux<p.true__E_mean_lux).mean()),
        'limits':'Target-based bins are retrospective diagnostics only, never prediction-time routing. Budget is arithmetic, not an expected gain. Annual radiation q is a proposed empirical predictor scale, not a measured daylight coefficient.'}
D.dump(out/'analysis.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
