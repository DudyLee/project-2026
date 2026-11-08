"""Read-only audit of Tower A simulation/BIM/EPW inputs. No model fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'reports' / 'data_audit'
OUT.mkdir(parents=True, exist_ok=True)
hp = RAW / 'SEND_ML_BIM_EPW_07-18.csv'
bp = RAW / 'SEND_BIM_476Rooms.csv'
d = pd.read_csv(hp)
b = pd.read_csv(bp)
targets = ['E_mean_lux', 'E_P10_lux', 'F_300_fraction', 'F_UDI_fraction', 'PV_AC_kWh']
weather = [c for c in d if c.startswith('EPW_') or c.startswith('Solar_Altitude') or c.startswith('Solar_Azimuth')]
static = [c for c in b if c != 'Room_ID'] + ['Floor_0based', 'Sensor_Count']
rooms = d.drop_duplicates('Room_ID').sort_values('Room_ID')
train = d[d.Sample_Role.eq('TRAIN')]
tr = rooms[rooms.Sample_Role.eq('TRAIN')]
report = {
    'sources': [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [hp,bp]],
    'hourly_shape': list(d.shape), 'bim_shape': list(b.shape),
    'columns': list(d.columns), 'dtypes': d.dtypes.astype(str).to_dict(),
    'missing_hourly': d.isna().sum().astype(int).to_dict(),
    'missing_bim': b.isna().sum().astype(int).to_dict(),
    'nonfinite_numeric': int((~np.isfinite(d.select_dtypes('number').to_numpy())).sum()),
    'duplicate_rows': int(d.duplicated().sum()),
    'duplicate_case_time_keys': int(d.duplicated(['Room_ID','PVSD_Angle_deg','HOY_0based']).sum()),
    'bim_duplicate_room_ids': int(b.Room_ID.duplicated().sum()),
    'roles': {str(k): {'rooms': int(g.Room_ID.nunique()),'rows':len(g)} for k,g in d.groupby('Sample_Role')},
    'bim_roles': b.Sample_Role.value_counts().astype(int).to_dict(),
    'angles': sorted(d.PVSD_Angle_deg.unique().tolist()),
    'hours': sorted(d.Hour.unique().tolist()),
    'per_case': d.groupby(['Room_ID','PVSD_Angle_deg']).agg(rows=('HOY_0based','size'),unique_hoy=('HOY_0based','nunique')).reset_index().to_dict('records'),
    'static_variation_within_room': d.groupby('Room_ID')[static].nunique().max().astype(int).to_dict(),
    'weather_variation_within_hoy': d.groupby('HOY_0based')[weather].nunique().max().astype(int).to_dict(),
    'train_room_static_unique_values': tr[[c for c in static if c!='Sample_Role']].nunique().astype(int).to_dict(),
}
time = d[['Month','Day','Hour','DOY_1based','HOY_0based']].drop_duplicates().sort_values('HOY_0based')
dates = pd.to_datetime({'year':2001,'month':time.Month,'day':time.Day})
report['time_checks'] = {
    'unique_time_rows': len(time),
    'doy_date_mismatches': int((dates.dt.dayofyear.to_numpy()!=time.DOY_1based.to_numpy()).sum()),
    'hoy_formula_mismatches': int((time.HOY_0based != (time.DOY_1based-1)*24+time.Hour).sum()),
    'first': time.iloc[0].to_dict(), 'last': time.iloc[-1].to_dict(),
    'expected_hoy_set': bool(set(time.HOY_0based) == {day*24+h for day in range(365) for h in range(7,19)}),
    'cases_with_different_time_set': sum(set(g.HOY_0based)!=set(time.HOY_0based) for _,g in d.groupby(['Room_ID','PVSD_Angle_deg'])),
}
joined = rooms.merge(b, on='Room_ID', how='left', suffixes=('_merged','_bim'),validate='one_to_one', indicator=True)
mismatch={}
for c in b.columns:
    if c=='Room_ID': continue
    a1, a2=joined[c+'_merged'],joined[c+'_bim']
    equal=np.isclose(a1,a2,atol=1e-8,rtol=1e-8,equal_nan=True) if pd.api.types.is_numeric_dtype(a1) else a1.eq(a2)
    mismatch[c]=int((~equal).sum())
report['bim_join']={'unmatched_rooms': joined.loc[joined['_merge'].ne('both'),'Room_ID'].tolist(),'mismatched_cells_by_column':mismatch}
report['range_checks']={}
for c in targets+weather:
    s=d[c]
    report['range_checks'][c]={'min':float(s.min()),'max':float(s.max()),'zeros':int(s.eq(0).sum()),'negative':int(s.lt(0).sum()),'unique':int(s.nunique())}
report['fraction_out_of_bounds']={c:int((d[c].lt(0)|d[c].gt(1)).sum()) for c in ['F_300_fraction','F_UDI_fraction']}
report['epw_sentinel_checks']={c:int(d[c].ge(limit).sum()) for c,limit in {'EPW_Outdoor_T_C':99.9,'EPW_RH_pct':999,'EPW_GHI_Wh_m2':9999,'EPW_DNI_Wh_m2':9999,'EPW_DHI_Wh_m2':9999,'EPW_WindSpeed_m_s':999,'EPW_WindDirection_deg':999}.items()}
w=d.drop_duplicates('HOY_0based')[['HOY_0based','Hour']+weather]
rad=w.EPW_GHI_Wh_m2-w.EPW_DHI_Wh_m2-w.EPW_DNI_Wh_m2*np.sin(np.deg2rad(w.Solar_Altitude_mid_deg)).clip(lower=0)
report['radiation_geometry_diagnostic']={'note':'Diagnostic only: mid-hour sun versus interval-integrated radiation; not proof of correct EPW join.','median_abs_residual_Wh_m2':float(rad.abs().median()),'p95_abs_residual_Wh_m2':float(rad.abs().quantile(.95)),'negative_altitude_hours':int(w.Solar_Altitude_mid_deg.lt(0).sum()),'positive_GHI_at_negative_altitude':int((w.Solar_Altitude_mid_deg.lt(0)&w.EPW_GHI_Wh_m2.gt(0)).sum())}
report['derived_checks']={'max_WFR_abs_error':float((d.WFR_window_to_floor-d.Window_Area_Nominal_m2/d.Room_Area_m2).abs().max()),'max_obstruction_abs_error':float((d.Solar_Obstruction_Ratio-(1-d.Annual_Incident_Shaded_kWh_m2/d.Annual_Incident_Unshaded_kWh_m2)).abs().max()),'max_orientation_norm_error':float((np.hypot(d.Orientation_X_model,d.Orientation_Y_model)-1).abs().max())}
report['train_target_stats']=train[targets].describe(percentiles=[.1,.5,.9,.99]).to_dict()
report['train_target_zero_fraction']=train[targets].eq(0).mean().to_dict()
report['train_angle_effect']={}
for c in targets:
    p=train.pivot(index=['Room_ID','HOY_0based'],columns='PVSD_Angle_deg',values=c)
    spread=p.max(axis=1)-p.min(axis=1)
    report['train_angle_effect'][c]={'median_max_minus_min':float(spread.median()),'mean_max_minus_min':float(spread.mean()),'fraction_four_angles_equal_at_1e-9':float(spread.le(1e-9).mean())}
cover={}
for c in ['Floor_0based','Room_Area_m2','Window_Count','Window_Area_Nominal_m2','WFR_window_to_floor','Orientation_X_model','Orientation_Y_model','Glass_Area_m2','Solar_Obstruction_Ratio']:
    lo,hi=tr[c].min(),tr[c].max()
    cover[c]={'train_min':float(lo),'train_max':float(hi),'building_rooms_outside_univariate_train_range':int((b[c].lt(lo)|b[c].gt(hi)).sum()) if c in b else None,'test_rooms_outside_univariate_train_range':rooms.loc[rooms.Sample_Role.eq('TEST')&(rooms[c].lt(lo)|rooms[c].gt(hi)),'Room_ID'].tolist()}
report['coverage'] = cover
roomcols=['Room_ID','Sample_Role','Floor_0based','Room_Area_m2','Window_Count','Window_Area_Nominal_m2','WFR_window_to_floor','Orientation_X_model','Orientation_Y_model','Glass_Area_m2','Solar_Obstruction_Ratio','Sensor_Count']
report['rooms']=rooms[roomcols].to_dict('records')
# Cross-room duplication of PV is a useful system-boundary diagnostic, using TRAIN only.
pv=train.pivot(index=['PVSD_Angle_deg','HOY_0based'],columns='Room_ID',values='PV_AC_kWh')
report['train_pv_identical_room_pairs']=[{'room_a':a,'room_b':bb} for i,a in enumerate(pv) for bb in pv.columns[i+1:] if np.allclose(pv[a],pv[bb],atol=1e-10,rtol=1e-8)]
schema=[]
for c in d:
    typ='target' if c in targets else 'weather_or_solar' if c in weather else 'identifier_or_split' if c in ['Room_ID','Sample_Role','Revit_Window_Element_IDs','Sensor_Count'] else 'time' if c in ['Month','Day','Hour','DOY_1based','HOY_0based'] else 'candidate_input'
    schema.append({'column':c,'role':typ,'dtype':str(d[c].dtype),'missing':int(d[c].isna().sum()),'unique':int(d[c].nunique())})
pd.DataFrame(schema).to_csv(OUT/'column_audit.csv',index=False,encoding='utf-8-sig')
rooms[roomcols].to_csv(OUT/'room_summary.csv',index=False,encoding='utf-8-sig')
(OUT/'data_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['columns','dtypes','per_case','train_target_stats','missing_hourly','missing_bim']},ensure_ascii=False,indent=2))
