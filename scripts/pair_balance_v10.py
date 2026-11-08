"""Frozen v10 transforms, shared preprocessing and performance gates."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import mean_exposure_v9 as V9

D=V9.D;ROOT=V9.ROOT;RAW=V9.RAW;TARGET=V9.TARGET
CONFIG=ROOT/'configs/experiment_v10.json'
METHODS=['v10_X0','v10_Xq','v10_S0','v10_Sq']


def label(d,q,indices,normalized):
    y=d.loc[indices,TARGET].to_numpy()
    return np.log1p(y/q[indices] if normalized else y)


def inverse(z,q,normalized):
    with np.errstate(over='raise',invalid='raise'):
        raw=np.expm1(np.asarray(z,dtype=float))
        if normalized:raw=raw*q
    assert np.isfinite(raw).all()
    return np.maximum(raw,0),raw


def estimator(cfg,params,method,parameter,n):
    if cfg['methods'][method]['family']=='XGBoost':return XGBRegressor(**(params|{'max_depth':int(parameter)}))
    return Ridge(alpha=float(parameter)*n,solver=cfg['ridge']['solver'],fit_intercept=cfg['ridge']['fit_intercept'])


def choose(options,cfg,method):
    tree=cfg['methods'][method]['family']=='XGBoost'
    return min(options,key=lambda r:(r['macro_MAE'],r['parameter'] if tree else -r['parameter']))


def predict_bundle(bundle,inputs,cfg):
    xs,q=V9.features(inputs,cfg);x=xs['F2'];z=bundle['preprocessor'].transform(x)
    return {m:inverse(e.predict(x if cfg['methods'][m]['family']=='XGBoost' else z),q,cfg['methods'][m]['normalized']) for m,e in bundle['estimators'].items()}


def summarize(fm,cfg):
    rows=[];eps=cfg['gate']['numerical_tolerance']
    for (protocol,method),g in fm.groupby(['protocol','method'],sort=False):
        assert len(g)==7 and g.room.nunique()==7
        worst=g.sort_values(['mae','room'],ascending=[False,True]).iloc[0]
        rows.append({'protocol':protocol,'method':method,'macro_MAE':float(g.mae.mean()),'worst_MAE':float(worst.mae),'worst_room':worst.room,
            'BM432_MAE':float(g[g.room.eq('BM432')].mae.iloc[0]),'LM608_MAE':float(g[g.room.eq('LM608')].mae.iloc[0]),
            'macro_RMSE':float(g.rmse.mean()),'macro_bias':float(g.bias.mean()),'macro_within_50_lux':float(g.within_50_lux.mean()),
            'macro_room_P90_error':float(g.absolute_error_P90.mean()),'min_R2':float(g.r2.min()),'macro_raw_negative_fraction':float(g.raw_negative_fraction.mean())})
    summary=pd.DataFrame(rows);gates=[];paired=[]
    def errors(protocol,method):return fm[(fm.protocol==protocol)&(fm.method==method)].set_index('room').mae.sort_index()
    for method in METHODS:
        a=errors('LeaveOneRoomOut',method);s=errors('LeaveOneRoomOut','v8_S1');c=errors('LeaveOneRoomOut','v9_C')
        oa=errors('LeaveOneOrientationOut',method);ot=errors('LeaveOneOrientationOut','v8_T1');os=errors('LeaveOneOrientationOut','v8_S1')
        flags={'macro_below_50':bool(a.mean()<50),'BM432_at_most_120':bool(a.BM432<=120+eps),'LM608_at_most_40':bool(a.LM608<=40+eps),
            'worst_nonworse_than_C':bool(a.max()<=c.max()+eps),'five_rooms_nonworse_than_S1':bool((a<=s+eps).sum()>=5),
            'orientation_macro_nonworse_than_T1':bool(oa.mean()<=ot.mean()+eps),'orientation_worst_nonworse_than_T1':bool(oa.max()<=ot.max()+eps),
            'orientation_LM608_nonworse_than_S1':bool(oa.LM608<=os.LM608+eps)}
        gates.append({'method':method,'rooms_nonworse_than_S1':int((a<=s+eps).sum()),
            'reaches_primary_targets':all(flags[k] for k in ['macro_below_50','BM432_at_most_120','LM608_at_most_40']),
            'partial_progress':bool(a.mean()<s.mean() and a.BM432<=c.BM432+eps and a.LM608<=s.LM608+eps),'passes_all':all(flags.values())}|flags)
        for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
            for reference in cfg['references']:
                x,y=errors(protocol,method),errors(protocol,reference)
                for room in x.index:paired.append({'protocol':protocol,'method':method,'reference':reference,'room':room,'reference_MAE':float(y[room]),
                    'candidate_MAE':float(x[room]),'change_lux':float(x[room]-y[room]),'worse':bool(x[room]>y[room]+eps)})
    passing=[g['method'] for g in gates if g['passes_all']]
    rank=summary[(summary.protocol=='LeaveOneRoomOut')&summary.method.isin(passing)].copy()
    rank['order']=rank.method.map({m:i for i,m in enumerate(METHODS)});rank=rank.sort_values(['worst_MAE','macro_MAE','order'])
    decision={'selected_candidate':rank.iloc[0].method if len(rank) else None,'passing_candidates':passing,'deployed_model_changed':False,'full_seven_refit':False,
        'scope':'Repeated development evidence; absolute targets, partial progress and full stability pass reported separately.'}
    return summary,pd.DataFrame(gates),pd.DataFrame(paired),decision
