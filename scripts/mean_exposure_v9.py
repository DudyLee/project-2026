"""Frozen v9 input-only features, TRAIN loader, source gate and grouped scoring."""
import csv
import copy
import numpy as np
import pandas as pd
import directional_v8 as D

ROOT=D.ROOT
CONFIG=ROOT/'configs/experiment_v9.json'
ANNUAL='Annual_Incident_Shaded_kWh_m2'
RAW=D.api.RAW_FEATURES+[ANNUAL]
METHODS=['A','B','C']
TARGET='E_mean_lux'


def load(cfg):
    plan=copy.deepcopy(D.read(ROOT/'configs/model_plan.json'))
    plan['feature_sets']={'raw':RAW,'audit':['Annual_Incident_Unshaded_kWh_m2']};plan['targets']=[TARGET]
    d,skipped=D.base.load_train(ROOT/cfg['source'],plan)
    assert skipped==52560 and set(d.Room_ID)==set(cfg['development_rooms'])
    cols=['Room_Area_m2','Glass_Area_m2',ANNUAL,'Annual_Incident_Unshaded_kWh_m2','Solar_Obstruction_Ratio']
    bim=[];excluded=0
    with (ROOT/cfg['bim_source']).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['Room_ID'] not in cfg['development_rooms']:excluded+=1;continue
            assert r['Sample_Role']=='TRAIN'
            bim.append({'Room_ID':r['Room_ID']}|{k:float(r[k]) for k in cols})
    b=pd.DataFrame(bim).set_index('Room_ID').sort_index()
    assert len(b)==7 and excluded==469 and b.index.is_unique
    assert d.groupby('Room_ID')[cols].nunique().eq(1).all().all()
    profiles=d.groupby('Room_ID')[cols].first().sort_index()
    np.testing.assert_allclose(profiles,b[cols],rtol=1e-10,atol=1e-9)
    np.testing.assert_allclose(profiles.Solar_Obstruction_Ratio,1-profiles[ANNUAL]/profiles.Annual_Incident_Unshaded_kWh_m2,rtol=1e-10,atol=1e-9)
    readme=(ROOT/'data/raw/SEND_README.txt').read_text(encoding='utf-8-sig')
    assert 'glass-area weighted prior annual solar radiation results' in readme
    q=features(d,cfg)[1];assert np.isfinite(q).all() and (q>0).all()
    audit={'status':'passed_for_exploratory_use','TRAIN_rows':len(d),'TEST_rows_skipped_before_numeric_conversion':skipped,
        'BIM_numeric_rooms':7,'BIM_rows_skipped_before_numeric_conversion':excluded,'BIM_hourly_match':True,
        'static_across_angles_and_hours':True,'obstruction_identity_verified':True,'positive_q':True,
        'source_definition':'Author README identifies these as glass-area weighted prior annual solar radiation results, separately from hourly illumination outputs.',
        'target_independence':'No illumination target used to derive q or any feature in this experiment; independence of original upstream computation relies on author handoff, not independent simulation-source inspection.',
        'limits':'EPW/time/north and original annual-radiation computation remain unverified; q is an empirical scale, not a measured physical daylight coefficient.'}
    profiles['q']=d.assign(q=q).groupby('Room_ID').q.first()
    return d,audit,profiles


def features(inputs,cfg):
    f1=D.features(inputs)['F1']
    q=(inputs.Glass_Area_m2/inputs.Room_Area_m2*inputs[ANNUAL]/cfg['q_reference_kWh_m2']).to_numpy(dtype=float)
    assert np.isfinite(q).all() and (q>0).all()
    f2=f1.copy();f2['q']=q;f2['q_DHI']=q*inputs.EPW_DHI_Wh_m2
    f2['q_horizontal_DNI']=q*inputs.EPW_DNI_Wh_m2*np.cos(np.deg2rad(inputs.Solar_Altitude_mid_deg))
    assert f1.shape[1]==21 and f2.shape[1]==24
    return {'F1':f1,'F2':f2},q


def training_arrays(d,q,indices,method):
    y=d.loc[indices,TARGET].to_numpy()
    if method=='C':return y/q[indices],q[indices]/q[indices].mean()
    return y,None


def predict(est,x,q,method):
    native=np.asarray(est.predict(x),dtype=float)
    raw=native*q if method=='C' else native
    assert np.isfinite(raw).all()
    return np.maximum(raw,0),raw


def predict_bundle(bundle,inputs,cfg):
    x,q=features(inputs,cfg)
    return {m:predict(bundle['estimators'][m],x[cfg['methods'][m]['features']],q,m) for m in METHODS}


def metrics(y,p,raw=None):
    y=np.asarray(y);p=np.asarray(p);ae=np.abs(p-y)
    result=D.base.metrics(y,p)|{'n':len(y),'true_mean':float(y.mean()),'pred_mean':float(p.mean()),
        'within_50_lux':float((ae<=50).mean()),'absolute_error_P90':float(np.quantile(ae,.9)),
        'underprediction_fraction':float((p<y).mean())}
    if raw is not None:result|={'raw_MAE':float(np.abs(np.asarray(raw)-y).mean()),'raw_negative_fraction':float((np.asarray(raw)<0).mean())}
    return result


def summarize(fm,cfg):
    rows=[]
    for (protocol,method),g in fm.groupby(['protocol','method'],sort=False):
        assert len(g)==7 and g.room.nunique()==7
        worst=g.sort_values(['mae','room'],ascending=[False,True]).iloc[0]
        rows.append({'protocol':protocol,'method':method,'macro_MAE':float(g.mae.mean()),'worst_MAE':float(worst.mae),'worst_room':worst.room,
            'BM432_MAE':float(g[g.room.eq('BM432')].mae.iloc[0]),'macro_RMSE':float(g.rmse.mean()),'macro_bias':float(g.bias.mean()),
            'macro_within_50_lux':float(g.within_50_lux.mean()),'macro_room_P90_error':float(g.absolute_error_P90.mean()),'min_R2':float(g.r2.min())})
    summary=pd.DataFrame(rows);comparisons=[];gates=[];eps=cfg['gate']['numerical_tolerance']
    def values(protocol,method):return fm[(fm.protocol==protocol)&(fm.method==method)].set_index('room').mae.sort_index()
    for method in METHODS:
        a=values('LeaveOneRoomOut',method);b=values('LeaveOneRoomOut','T1');s=values('LeaveOneRoomOut','S1')
        oa=values('LeaveOneOrientationOut',method);ob=values('LeaveOneOrientationOut','T1')
        flags={'macro_below_50':bool(a.mean()<50),'BM432_at_most_120':bool(a['BM432']<=120+eps),
            'worst_nonworse_than_S1':bool(a.max()<=s.max()+eps),'five_rooms_nonworse_than_T1':bool((a<=b+eps).sum()>=5),
            'orientation_macro_nonworse_than_T1':bool(oa.mean()<=ob.mean()+eps),'orientation_worst_nonworse_than_T1':bool(oa.max()<=ob.max()+eps)}
        gates.append({'method':method,'rooms_nonworse_than_T1':int((a<=b+eps).sum()),'rooms_nonworse_than_S1':int((a<=s+eps).sum()),
            'reaches_primary_targets':flags['macro_below_50'] and flags['BM432_at_most_120'],'passes_all':all(flags.values())}|flags)
        for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
            for reference in cfg['references']:
                x,y=values(protocol,method),values(protocol,reference)
                for room in x.index:comparisons.append({'protocol':protocol,'method':method,'reference':reference,'room':room,
                    'reference_MAE':float(y[room]),'candidate_MAE':float(x[room]),'change_lux':float(x[room]-y[room]),'worse':bool(x[room]>y[room]+eps)})
    passing=[r['method'] for r in gates if r['passes_all']]
    ranked=summary[(summary.protocol=='LeaveOneRoomOut')&summary.method.isin(passing)].sort_values(['worst_MAE','macro_MAE','method'])
    decision={'selected_candidate':ranked.iloc[0].method if len(ranked) else None,'passing_candidates':passing,
        'deployed_model_changed':False,'full_seven_refit':False,'scope':'Repeated development validation, not independent final testing.'}
    return summary,pd.DataFrame(gates),pd.DataFrame(comparisons),decision
