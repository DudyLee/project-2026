"""PVSD v8-v11 numerical methods, extracted from the verified research workflow.

One regression/parameter choice per output. No test-informed selection.
"""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

STUDY=read(ROOT/'configs/study.json')
CFG={v:STUDY for v in range(8,12)}
TREE=read(ROOT/'configs/tree_parameters.json')
TARGETS=['E_mean_lux','E_P10_lux','F_300_fraction','F_UDI_fraction','PV_AC_kWh']
META=['Room_ID','Month','Day','DOY_1based','Hour','HOY_0based','PVSD_Angle_deg']
KEYS=['Room_ID','DOY_1based','Hour','PVSD_Angle_deg']
RAW_FEATURES=['Room_Area_m2','Glass_Area_m2','Orientation_X_model','Orientation_Y_model',
    'Solar_Obstruction_Ratio','DOY_1based','Hour','PVSD_Angle_deg','EPW_GHI_Wh_m2',
    'EPW_DNI_Wh_m2','EPW_DHI_Wh_m2','Solar_Altitude_mid_deg','Solar_Azimuth_mid_deg']
RAW=RAW_FEATURES+['Annual_Incident_Shaded_kWh_m2']
SELECT=[]
START=0

def idx(d,rooms):return d.index[d.Room_ID.isin(rooms)].to_numpy()
def bound(y,t):return np.clip(y,0,1) if t.startswith('F_') else np.maximum(y,0)
def check_time():
    if START and time.perf_counter()-START>STUDY['hard_timeout_seconds']:
        raise TimeoutError('Study runtime limit exceeded')

def load_data(source,role,columns):
    """Discard the other split before retaining or converting any numeric fields."""
    rooms=STUDY['development_rooms'] if role=='TRAIN' else STUDY['excluded_test_rooms']
    other=STUDY['excluded_test_rooms'] if role=='TRAIN' else STUDY['development_rooms']
    records=[];skipped=0
    with Path(source).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        assert set(columns+['Sample_Role'])<=set(reader.fieldnames)
        for row in reader:
            assert row['Sample_Role'] in ['TRAIN','TEST']
            if row['Sample_Role']!=role:
                assert row['Room_ID'] in other;skipped+=1;continue
            assert row['Room_ID'] in rooms
            records.append({c:row[c] for c in columns})
    d=pd.DataFrame(records)
    for c in columns:
        if c!='Room_ID':d[c]=pd.to_numeric(d[c],errors='raise')
    assert len(d)==len(rooms)*STUDY['rows_per_room']
    assert skipped==len(other)*STUDY['rows_per_room']
    assert set(d.Room_ID)==set(rooms) and not d.duplicated(KEYS).any()
    assert np.isfinite(d.select_dtypes('number').to_numpy()).all()
    assert d.groupby(['Room_ID','PVSD_Angle_deg']).size().eq(4380).all()
    assert set(d.PVSD_Angle_deg)==set(STUDY['angles'])
    return d

def features(d):
    f0=pd.DataFrame(index=d.index)
    for c in RAW_FEATURES:
        if c=='DOY_1based':
            phase=2*np.pi*(d[c]-1)/365
            f0['DOY_sin']=np.sin(phase);f0['DOY_cos']=np.cos(phase)
        elif c=='Solar_Azimuth_mid_deg':
            phase=np.deg2rad(d[c]);f0['Solar_Azimuth_sin']=np.sin(phase);f0['Solar_Azimuth_cos']=np.cos(phase)
        else:f0[c]=d[c].astype(float)
    f1=f0.copy();az=np.deg2rad(d.Solar_Azimuth_mid_deg);alt=np.deg2rad(d.Solar_Altitude_mid_deg)
    u=d.EPW_DNI_Wh_m2*np.cos(alt)*np.sin(az);v=d.EPW_DNI_Wh_m2*np.cos(alt)*np.cos(az)
    for name,value in zip(['DNI_u','DNI_v','X_DNI_u','X_DNI_v','Y_DNI_u','Y_DNI_v'],
                         [u,v,d.Orientation_X_model*u,d.Orientation_X_model*v,d.Orientation_Y_model*u,d.Orientation_Y_model*v]):f1[name]=value
    q=(d.Glass_Area_m2/d.Room_Area_m2*d.Annual_Incident_Shaded_kWh_m2/STUDY['q_reference_kWh_m2']).to_numpy(dtype=float)
    assert np.isfinite(q).all() and (q>0).all()
    f2=f1.copy();f2['q']=q;f2['q_DHI']=q*d.EPW_DHI_Wh_m2
    f2['q_horizontal_DNI']=q*d.EPW_DNI_Wh_m2*np.cos(np.deg2rad(d.Solar_Altitude_mid_deg))
    b=f2.copy();angle=d.PVSD_Angle_deg/90;ratio=d.Glass_Area_m2/d.Room_Area_m2
    for name,value in {'angle_r':angle*ratio,'angle2_r':angle*angle*ratio,'angle_q':angle*q,
                       'angle2_q':angle*angle*q,'angle_DNI':angle*d.EPW_DNI_Wh_m2,'angle_DHI':angle*d.EPW_DHI_Wh_m2}.items():b[name]=value
    assert [f0.shape[1],f1.shape[1],f2.shape[1],b.shape[1]]==[15,21,24,30]
    return {'F0':f0,'F1':f1,'F2':f2,'B':b},q

def preprocessing(cfg):
    settings=cfg['spline'].copy();columns=settings.pop('columns')
    return Pipeline([('basis',ColumnTransformer([('spline',SplineTransformer(**settings),columns)],
        remainder='passthrough',verbose_feature_names_out=True)),('scale',StandardScaler())])

def outer_splits():
    rooms=STUDY['development_rooms'];jobs=[];protocols=[]
    for r in rooms:
        jid='room_'+r;jobs.append(dict(id=jid,fit_rooms=[x for x in rooms if x!=r],held_rooms=[r]))
        protocols.append(dict(protocol='LeaveOneRoomOut',fold=r,job_id=jid))
    for direction,held in STUDY['orientation_groups'].items():
        jid='room_'+held[0] if len(held)==1 else 'orientation_'+direction
        if len(held)>1:jobs.append(dict(id=jid,fit_rooms=[x for x in rooms if x not in held],held_rooms=held))
        protocols.append(dict(protocol='LeaveOneOrientationOut',fold=direction,job_id=jid))
    return jobs,protocols

def metrics(y,p):
    error=p-y;denominator=float(np.sum((y-np.mean(y))**2))
    return dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
        r2=float(1-np.sum(error**2)/denominator) if denominator>0 else None,bias=float(error.mean()))

def fit_fixed(spec,x,frame):
    y=frame[spec['target']].to_numpy()
    if spec['transform']=='relative':ys={'mean_log':np.log1p(frame.E_mean_lux.to_numpy()),'ratio':y/(frame.E_mean_lux.to_numpy()+1)}
    else:ys={'value':np.log1p(y) if spec['transform']=='log1p' else y}
    classes={'HistGB':HistGradientBoostingRegressor,'RandomForest':RandomForestRegressor,'XGBoost':XGBRegressor}
    estimators={}
    for k,label in ys.items():
        estimator=classes[spec['family']](**spec['params']);estimator.fit(x,label);estimators[k]=estimator
    return {'spec':spec,'estimators':estimators},None

def specs(v):
    if v==8:
        return {m:dict(feature=f'F{m[1]}',family='frozen_tree' if m[0]=='T' else 'ridge',normalized=False,parameters=[None] if m[0]=='T' else [.01,.1]) for m in ['T0','T1','S0','S1']} | {'S1_direct':dict(feature='F1',family='ridge',normalized=False,parameters=[.01,.1])}
    if v==9:return {m:dict(feature='F1' if m=='A' else 'F2',family='xgb_l1',normalized=m=='C',parameters=[3,5]) for m in ['A','B','C']}
    if v==10:return {m:dict(feature='F2',family='xgb_log' if 'X' in m else 'ridge',normalized=m.endswith('q'),parameters=[3,5] if 'X' in m else [.01,.1]) for m in ['v10_X0','v10_Xq','v10_S0','v10_Sq']}
    return {m:dict(feature='F2' if m[0]=='A' else 'B',family='ridge',normalized=m.endswith('q'),parameters=[.01,.1,1.]) for m in ['A0','Aq','B0','Bq']}

def labels(d,q,v,m,s,targets):
    if v==8:
        cols={}
        for t in targets:
            if t=='E_P10_lux' and m=='S1_direct':cols['p10_asinh']=np.arcsinh(d[t].to_numpy())
            elif t=='E_P10_lux':
                cols['mean_log']=np.log1p(d.E_mean_lux.to_numpy());cols['ratio']=d[t].to_numpy()/(d.E_mean_lux.to_numpy()+1)
            elif t=='E_mean_lux':cols['mean_log']=np.log1p(d[t].to_numpy())
            else:cols[t]=d[t].to_numpy()
    else:
        cols={t:d[t].to_numpy()/(q if s['normalized'] else 1) for t in targets}
        if v>=10:cols={t:np.log1p(y) for t,y in cols.items()}
    return list(cols),np.column_stack(list(cols.values()))

def restore(native,names,q,v,m,s,targets):
    z=dict(zip(names,np.asarray(native).reshape(len(q),len(names)).T)); result={}
    with np.errstate(over='raise',invalid='raise'):
        for t in targets:
            if v==8:
                if t=='E_mean_lux':raw=np.expm1(z['mean_log']);p=bound(raw,t)
                elif t=='E_P10_lux' and m=='S1_direct':raw=np.sinh(z['p10_asinh']);p=bound(raw,t)
                elif t=='E_P10_lux':
                    mean=np.expm1(z['mean_log']);raw=z['ratio']*(mean+1)
                    p=np.maximum(z['ratio'],0)*(np.maximum(mean,0)+1)
                else:raw=z[t];p=bound(raw,t)
            else:
                raw=np.expm1(z[t]) if v>=10 else z[t]
                raw=raw*(q if s['normalized'] else 1);p=bound(raw,t)
            assert np.isfinite(p).all() and np.isfinite(raw).all()
            result[t]=(p,raw)
    return result

def inner_splits(d,fit_rooms,stable):
    if stable:
        ss=[dict(id='room_'+r,protocol='room',fit_rooms=[x for x in fit_rooms if x!=r],held_rooms=[r]) for r in fit_rooms]
        for direction,rooms in CFG[11]['orientation_groups'].items():
            held=[r for r in fit_rooms if r in rooms];fit=[r for r in fit_rooms if r not in held]
            if held and len(fit)>=2:ss.append(dict(id='direction_'+direction,protocol='direction',fit_rooms=fit,held_rooms=held))
        return ss
    indices=idx(d,fit_rooms)
    return [dict(id=f'group_{k}',protocol='room',fit_rooms=sorted(d.loc[indices[a],'Room_ID'].unique()),held_rooms=sorted(d.loc[indices[b],'Room_ID'].unique())) for k,(a,b) in enumerate(GroupKFold(3).split(indices,groups=d.loc[indices,'Room_ID']))]

def native_fit(x,y,parameter,s,v,n,weights=None):
    check_time()
    if s['family']=='ridge':est=Ridge(alpha=parameter*n,solver='svd');est.fit(x,y);return est
    params=TREE['E_mean_lux']['params']|{'max_depth':int(parameter),'objective':'reg:absoluteerror' if v==9 else 'reg:squarederror'}
    est=[]
    for j in range(y.shape[1]):
        e=XGBRegressor(**params);e.fit(x,y[:,j],sample_weight=weights);est.append(e)
    return est

def native_predict(est,x):
    return np.column_stack([e.predict(x) for e in est]) if isinstance(est,list) else est.predict(x)

def logfit(v,m,jid,phase,fit_rooms,held_rooms,p,targets,n):
    assert not set(fit_rooms)&set(CFG[11]['excluded_test_rooms'])
    assert not set(fit_rooms)&set(held_rooms)

def train_cell(d,xx,q,v,m,s,targets,job):
    jid=job['id'];fit_rooms=job['fit_rooms'];a=idx(d,fit_rooms);x=xx[s['feature']]
    if s['family']=='frozen_tree':
        fitted={}
        for t in targets:
            check_time();fitted[t],_=fit_fixed(TREE[t],x.loc[a],d.loc[a])
            logfit(v,m,jid,'outer' if jid!='all_seven' else 'full_seven',fit_rooms,job['held_rooms'],None,[t],len(a))
        return dict(version=v,method=m,spec=s,targets=targets,trees=fitted,training_rooms=fit_rooms)
    names,y=labels(d,q,v,m,s,targets);local=[]
    for split in inner_splits(d,fit_rooms,v==11):
        ia,ib=idx(d,split['fit_rooms']),idx(d,split['held_rooms'])
        assert set(split['fit_rooms']+split['held_rooms'])<=set(fit_rooms)
        if s['family']=='ridge':
            prep=preprocessing(CFG[8]);za=prep.fit_transform(x.loc[ia]);zb=prep.transform(x.loc[ib])
            assert int(prep.named_steps['scale'].n_samples_seen_)==len(ia)
        else:za,zb=x.loc[ia],x.loc[ib]
        weights=q[ia]/q[ia].mean() if v==9 and s['normalized'] else None
        for p in s['parameters']:
            est=native_fit(za,y[ia],p,s,v,len(ia),weights)
            preds=restore(native_predict(est,zb),names,q[ib],v,m,s,targets)
            logfit(v,m,jid,'inner',split['fit_rooms'],split['held_rooms'],p,names,len(ia))
            for room in split['held_rooms']:
                mask=d.loc[ib,'Room_ID'].eq(room).to_numpy()
                for t in targets:
                    row=dict(version=v,method=m,job_id=jid,split_id=split['id'],protocol=split['protocol'],room=room,target=t,parameter=p,mae=float(np.abs(preds[t][0][mask]-d.loc[ib,t].to_numpy()[mask]).mean()))
                    local.append(row)
    chosen={}
    for t in targets:
        options=[]
        for p in s['parameters']:
            rr=[r['mae'] for r in local if r['target']==t and r['parameter']==p and r['protocol']=='room']
            oo=[r['mae'] for r in local if r['target']==t and r['parameter']==p and r['protocol']=='direction']
            score=max(np.mean(rr),np.mean(oo))+.25*max(max(rr),max(oo)) if v==11 and oo else np.mean(rr)+(.25*max(rr) if v==11 else 0)
            options.append((float(score),p))
        chosen[t]=min(options,key=lambda z:(z[0],-z[1] if s['family']=='ridge' else z[1]))[1]
        SELECT.append(dict(version=v,method=m,job_id=jid,target=t,parameter=chosen[t],options=[{'score':z[0],'parameter':z[1]} for z in options],fit_rooms=fit_rooms))
    # Selection is complete before outer predictions and all test access.
    if s['family']=='ridge':prep=preprocessing(CFG[8]);z=prep.fit_transform(x.loc[a])
    else:prep=None;z=x.loc[a]
    estimators={}
    for p in sorted(set(chosen.values())):
        weights=q[a]/q[a].mean() if v==9 and s['normalized'] else None
        estimators[p]=native_fit(z,y[a],p,s,v,len(a),weights)
        logfit(v,m,jid,'outer' if jid!='all_seven' else 'full_seven',fit_rooms,job['held_rooms'],p,names,len(a))
    return dict(version=v,method=m,spec=s,targets=targets,names=names,preprocessor=prep,estimators=estimators,chosen=chosen,training_rooms=fit_rooms)

def predict_cell(bundle,xx,q):
    v,m,s=bundle['version'],bundle['method'],bundle['spec'];x=xx[s['feature']]
    if 'trees' in bundle:
        result={}
        for t,fit in bundle['trees'].items():
            # Compute native output independently of clipping for diagnostics.
            ee=fit['estimators'];transform=fit['spec']['transform']
            if transform=='relative':
                mu=np.expm1(ee['mean_log'].predict(x));ratio=ee['ratio'].predict(x);raw=ratio*(mu+1);p=np.maximum(ratio,0)*(np.maximum(mu,0)+1)
            else:
                raw=np.asarray(ee['value'].predict(x),dtype=float)
                if transform=='log1p':raw=np.expm1(raw)
                p=bound(raw,t)
            result[t]=(p,raw)
        return result
    z=bundle['preprocessor'].transform(x) if bundle['preprocessor'] is not None else x
    options={p:restore(native_predict(e,z),bundle['names'],q,v,m,s,bundle['targets']) for p,e in bundle['estimators'].items()}
    return {t:options[bundle['chosen'][t]][t] for t in bundle['targets']}

def add_fusion(pred):
    for prefix in ['A','B']:
        pred['C'+prefix]={t:tuple(.5*(pred[prefix+'0'][t][i]+pred[prefix+'q'][t][i]) for i in [0,1]) for t in pred[prefix+'0']}

def score_frame(frame,v,protocol,fold,provenance):
    rows=[]
    for c in frame:
        if not c.startswith('pred__'):continue
        m,t=c[6:].split('__',1)
        for room,g in frame.groupby('Room_ID',sort=True):
            y,p=g['true__'+t].to_numpy(),g[c].to_numpy();raw=g['raw__'+m+'__'+t].to_numpy()
            rows.append(dict(version=v,protocol=protocol,fold=fold,room=room,method=m,target=t,provenance=provenance,
                **metrics(y,p),n=len(g),raw_mae=float(np.abs(raw-y).mean()),clip_fraction=float(np.mean(p!=raw))))
    return rows

def prediction_frame(d,v,job,pred,include_targets):
    f=d[META].copy()
    targets=sorted({t for values in pred.values() for t in values})
    if include_targets:
        for t in targets:f['true__'+t]=d[t].to_numpy()
    for m,vals in pred.items():
        for t,(p,raw) in vals.items():f['pred__'+m+'__'+t]=p;f['raw__'+m+'__'+t]=raw
    return f
