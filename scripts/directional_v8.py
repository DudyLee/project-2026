"""Shared deterministic features, transforms and scoring for approved v8."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.model_selection import GroupKFold
import pvsd_trial as api
import train_logo_v1 as base

CONFIG=ROOT/'configs/experiment_v8.json'
TARGETS=['E_mean_lux','E_P10_lux','F_UDI_fraction']
META=['Room_ID','Month','Day','DOY_1based','Hour','HOY_0based','PVSD_Angle_deg']
KEYS=['Room_ID','DOY_1based','Hour','PVSD_Angle_deg']
EXTRA=['DNI_u','DNI_v','X_DNI_u','X_DNI_v','Y_DNI_u','Y_DNI_v']


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float64).tobytes()).hexdigest()


def features(d):
    f0=base.encode_inputs(d,api.RAW_FEATURES)
    f1=f0.copy()
    az=np.deg2rad(d.Solar_Azimuth_mid_deg);alt=np.deg2rad(d.Solar_Altitude_mid_deg)
    u=d.EPW_DNI_Wh_m2*np.cos(alt)*np.sin(az);v=d.EPW_DNI_Wh_m2*np.cos(alt)*np.cos(az)
    for name,value in zip(EXTRA,[u,v,d.Orientation_X_model*u,d.Orientation_X_model*v,d.Orientation_Y_model*u,d.Orientation_Y_model*v]): f1[name]=value
    assert f0.shape[1]==15 and f1.shape[1]==21
    return {'F0':f0,'F1':f1}


def labels(d):
    return {'mean_log':np.log1p(d.E_mean_lux.to_numpy()),'ratio':d.E_P10_lux.to_numpy()/(d.E_mean_lux.to_numpy()+1),
            'udi':d.F_UDI_fraction.to_numpy(),'p10_asinh':np.arcsinh(d.E_P10_lux.to_numpy())}


def primitives(target,direct=False):
    if target=='E_mean_lux':return ['mean_log']
    if target=='E_P10_lux':return ['p10_asinh'] if direct else ['mean_log','ratio']
    return ['udi']


def preprocessing(cfg):
    s=cfg['spline'].copy();cols=s.pop('columns')
    return Pipeline([('basis',ColumnTransformer([('spline',SplineTransformer(**s),cols)],remainder='passthrough',verbose_feature_names_out=True)),
                     ('scale',StandardScaler())])


def preprocess_diagnostics(prep,x):
    basis=prep.named_steps['basis'];scaler=prep.named_steps['scale']
    return {'fit_rows':len(x),'raw_columns':list(x.columns),'expanded_columns':basis.get_feature_names_out().tolist(),
            'input_sha256':array_sha(x.to_numpy()),'constant_input_columns':[c for c in x if x[c].max()==x[c].min()],
            'scaler_n_samples_seen':int(scaler.n_samples_seen_),'scaler_mean':scaler.mean_.tolist(),'scaler_var':scaler.var_.tolist(),
            'spline_knots':[b.t.tolist() for b in basis.named_transformers_['spline'].bsplines_]}


def reconstruct(values,target,direct=False):
    with np.errstate(over='raise',invalid='raise'):
        if target=='E_mean_lux':
            raw=np.expm1(values['mean_log']);pred=np.maximum(raw,0);components={}
        elif target=='E_P10_lux' and not direct:
            mean_raw=np.expm1(values['mean_log']);ratio_raw=values['ratio']
            raw=ratio_raw*(mean_raw+1)
            pred=np.maximum(ratio_raw,0)*(np.maximum(mean_raw,0)+1)
            components={'paired_mean':np.maximum(mean_raw,0),'ratio':np.maximum(ratio_raw,0),
                        'component_clipped':((mean_raw<0)|(ratio_raw<0)).astype(float)}
        elif target=='E_P10_lux':
            raw=np.sinh(values['p10_asinh']);pred=np.maximum(raw,0);components={}
        else:
            raw=values['udi'];pred=np.clip(raw,0,1);components={}
    assert np.isfinite(raw).all() and np.isfinite(pred).all()
    return np.asarray(pred,dtype=float),np.asarray(raw,dtype=float),components


def predict_tree(fitted,x,target):
    est=fitted['estimators']
    if target=='E_P10_lux':values={k:est[k].predict(x) for k in ['mean_log','ratio']}
    else:values={'mean_log' if target=='E_mean_lux' else 'udi':est['value'].predict(x)}
    return reconstruct(values,target)


def predict_bundle(bundle,inputs,cfg):
    xx=features(inputs)
    results={}
    for method in ['T0','T1']:
        for target in TARGETS:
            results[(method,target)]=predict_tree(bundle['trees'][method][target],xx[cfg['methods'][method]['features']],target)
    for cell in ['S0','S1']:
        item=bundle['splines'][cell];z=item['preprocessor'].transform(xx[cfg['methods'][cell]['features']])
        entries=[(cell,t,False) for t in TARGETS]
        if cell=='S1':entries.append(('S1_direct','E_P10_lux',True))
        for method,target,direct in entries:
            lam=item['selected'][method+'::'+target]
            vals={p:item['estimators'][str(lam)+'::'+p].predict(z) for p in primitives(target,direct)}
            results[(method,target)]=reconstruct(vals,target,direct)
    return results


def metrics(y,p,raw,tolerance,target,components=None):
    y=np.asarray(y);p=np.asarray(p);raw=np.asarray(raw)
    violated=(raw<0)|((raw>1) if target=='F_UDI_fraction' else False)
    result={**base.metrics(y,p),'n':len(y),'true_mean':float(y.mean()),'pred_mean':float(p.mean()),
            'raw_mae':float(np.abs(raw-y).mean()),'raw_out_of_bounds_fraction':float(violated.mean()),
            'within_tolerance':float((np.abs(p-y)<=tolerance).mean()),'absolute_tolerance':tolerance}
    if components and 'component_clipped' in components:result['component_clipping_fraction']=float(np.mean(components['component_clipped']))
    else:result['component_clipping_fraction']=None
    return result


def splits(d,cfg):
    jobs=[];protocols=[]
    for room in cfg['development_rooms']:
        jid='room_'+room
        jobs.append({'id':jid,'fit_rooms':[r for r in cfg['development_rooms'] if r!=room],'held_rooms':[room]})
        protocols.append({'protocol':'LeaveOneRoomOut','fold':room,'job_id':jid})
    for direction,rooms in cfg['orientation_groups'].items():
        if len(rooms)==1:jid='room_'+rooms[0]
        else:
            jid='orientation_'+direction
            jobs.append({'id':jid,'fit_rooms':[r for r in cfg['development_rooms'] if r not in rooms],'held_rooms':rooms})
        protocols.append({'protocol':'LeaveOneOrientationOut','fold':direction,'job_id':jid})
    for job in jobs:
        fit=d.index[d.Room_ID.isin(job['fit_rooms'])].to_numpy();held=d.index[d.Room_ID.isin(job['held_rooms'])].to_numpy()
        assert not set(job['fit_rooms'])&set(job['held_rooms'])
        job['fit_rows']=len(fit);job['held_rows']=len(held);job['inner']=[]
        for i,(a,b) in enumerate(GroupKFold(3).split(fit,groups=d.loc[fit,'Room_ID']),1):
            job['inner'].append({'fold':i,'fit_rooms':sorted(d.loc[fit[a],'Room_ID'].unique()),'held_rooms':sorted(d.loc[fit[b],'Room_ID'].unique()),
                                  'fit_rows':len(a),'held_rows':len(b)})
    assert len(jobs)==9 and len(protocols)==10
    return jobs,protocols


def summarize(fold_metrics,cfg):
    rows=[]
    for (protocol,method,target),g in fold_metrics.groupby(['protocol','method','target'],sort=False):
        assert len(g)==7 and g.room.nunique()==7
        worst=g.sort_values(['mae','room'],ascending=[False,True]).iloc[0]
        rows.append({'protocol':protocol,'method':method,'target':target,'room_count':7,'macro_MAE':float(g.mae.mean()),'worst_MAE':float(g.mae.max()),
                     'worst_room':worst.room,'macro_RMSE':float(g.rmse.mean()),'median_R2':float(g.r2.median()),'min_R2':float(g.r2.min()),
                     'macro_bias':float(g.bias.mean()),'macro_within_tolerance':float(g.within_tolerance.mean()),
                     'raw_out_of_bounds_fraction':float(g.raw_out_of_bounds_fraction.mean())})
    summary=pd.DataFrame(rows);gates=[];paired=[];tol=cfg['gate']['numerical_tolerance']
    for method in ['T1','S0','S1','S1_direct']:
        for target in cfg['methods'][method]['targets']:
            ss={}
            for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
                a=fold_metrics[(fold_metrics.protocol==protocol)&(fold_metrics.method=='T0')&(fold_metrics.target==target)].set_index('room')
                b=fold_metrics[(fold_metrics.protocol==protocol)&(fold_metrics.method==method)&(fold_metrics.target==target)].set_index('room').loc[a.index]
                ss[protocol]=(a,b)
                for room in a.index:paired.append({'protocol':protocol,'method':method,'target':target,'room':room,'baseline_MAE':a.loc[room,'mae'],'candidate_MAE':b.loc[room,'mae'],'change':b.loc[room,'mae']-a.loc[room,'mae']})
            a,b=ss['LeaveOneRoomOut'];oa,ob=ss['LeaveOneOrientationOut']
            passed=bool(b.mae.max()<=a.mae.max()*.9+tol and b.mae.mean()<=a.mae.mean()+tol and (b.mae<=a.mae+tol).sum()>=5 and ob.mae.mean()<=oa.mae.mean()+tol and ob.mae.max()<=oa.mae.max()+tol)
            gates.append({'method':method,'target':target,'room_macro_MAE_change_pct':float((b.mae.mean()/a.mae.mean()-1)*100),
                          'room_worst_MAE_change_pct':float((b.mae.max()/a.mae.max()-1)*100),'room_nonworse_count':int((b.mae<=a.mae+tol).sum()),
                          'orientation_macro_MAE_change_pct':float((ob.mae.mean()/oa.mae.mean()-1)*100),
                          'orientation_worst_MAE_change_pct':float((ob.mae.max()/oa.mae.max()-1)*100),'passes_all':passed})
    decisions={}
    for target in TARGETS:
        eligible=[g['method'] for g in gates if g['target']==target and g['passes_all']]
        candidates=summary[(summary.protocol=='LeaveOneRoomOut')&(summary.target==target)&summary.method.isin(eligible)]
        if len(candidates):
            order={m:i for i,m in enumerate(['T1','S0','S1','S1_direct'])}
            candidates=candidates.assign(order=candidates.method.map(order)).sort_values(['worst_MAE','macro_MAE','order'])
            selected=candidates.iloc[0].method
        else:selected='T0'
        decisions[target]={'selected':selected,'passing_candidates':eligible,'promoted':selected!='T0'}
    return summary,pd.DataFrame(gates),pd.DataFrame(paired),decisions
