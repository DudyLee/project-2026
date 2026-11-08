"""Verify saved TRAIN-only predictions and generate offline experiment reports."""
from pathlib import Path
import json
import runpy
import hashlib
import math

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'experiment_v1'
TARGETS=['E_mean_lux','E_P10_lux','F_300_fraction','F_UDI_fraction','PV_AC_kWh']
LABELS={'E_mean_lux':'Mean illuminance','E_P10_lux':'P10 illuminance','F_300_fraction':'300 lux spatial fraction','F_UDI_fraction':'100-3000 lux spatial fraction','PV_AC_kWh':'Hourly PV AC energy'}
UNITS={'E_mean_lux':'lux','E_P10_lux':'lux','F_300_fraction':'percentage points','F_UDI_fraction':'percentage points','PV_AC_kWh':'kWh / hour record'}
FACTORS={t:100 if t.startswith('F_') else 1 for t in TARGETS}
FEATURES=['A_bim_time_angle','B_add_solar_weather']


def main():
    print('Report: verifying saved metrics, folds and models.',flush=True)
    run=json.loads((OUT/'run_manifest.json').read_text())
    assert run['status']=='completed' and run['completed_fits']==140 and run['model_file_count']==140
    assert run['test_predictions']==0 and run['test_scores']==0
    fm=pd.read_csv(OUT/'fold_metrics.csv')
    summary=pd.read_csv(OUT/'summary_metrics.csv')
    weather=pd.read_csv(OUT/'weather_feature_comparison.csv')
    pairs=pd.read_csv(OUT/'paired_room_comparison.csv')
    plan=json.loads((ROOT/'configs/model_plan.json').read_text())
    train_rooms=set(plan['validation']['development_rooms'])
    protected=set(plan['validation']['final_test_rooms'])
    assert set(fm.validation_room)==train_rooms and len(fm)==140
    module=runpy.run_path(str(ROOT/'scripts/train_logo_v1.py'),run_name='verification_import')
    frame,skipped=module['load_train'](ROOT/run['config']['source'],plan)
    verified_metrics=0
    verified_models=0
    for feature in FEATURES:
        x=module['encode_inputs'](frame,plan['feature_sets'][feature])
        for model in ['Ridge','HistGB']:
            oof=pd.read_parquet(OUT/'predictions'/f'{feature}__{model}.parquet')
            assert len(oof)==122640 and set(oof.Room_ID)==train_rooms
            assert not protected.intersection(oof.Room_ID)
            assert not oof.duplicated(['Room_ID','PVSD_Angle_deg','HOY_0based']).any()
            assert oof.fold.nunique()==7
            for room,g in oof.groupby('Room_ID'):
                assert len(g)==17520
                split=[s for s in run['folds'] if s['validation_room']==room][0]
                assert room not in split['training_rooms'] and len(split['training_rooms'])==6
                for target in TARGETS:
                    y=g['true__'+target].to_numpy()
                    pred=g['pred__'+target].to_numpy()
                    err=pred-y
                    ss_total=float(np.sum((y-y.mean())**2))
                    independent={'mae':float(np.abs(err).mean()),'rmse':math.sqrt(float(np.square(err).mean())),'r2':1-float(np.square(err).sum())/ss_total if ss_total>0 else None}
                    row=fm[(fm.feature_set==feature)&(fm.model==model)&(fm.target==target)&(fm.validation_room==room)].iloc[0]
                    for name,value in independent.items():
                        if value is not None:
                            assert np.isclose(value,row[name],rtol=1e-10,atol=1e-10),(feature,model,room,target,name)
                    verified_metrics+=1
                    bundle=joblib.load(ROOT/run['config']['model_output']/feature/model/f"fold_{split['fold']:02d}_{room}__{target}.joblib")
                    assert bundle['split']==split
                    if model=='Ridge':
                        train_mask=frame.Room_ID.isin(split['training_rooms'])
                        assert np.allclose(bundle['estimator'].named_steps['standardscaler'].mean_,x.loc[train_mask].mean().to_numpy(),atol=1e-10)
                    sample_indices=g.index[[0,len(g)//2,-1]]
                    saved=g.loc[sample_indices,'raw__'+target].to_numpy()
                    reloaded=bundle['estimator'].predict(x.loc[sample_indices])
                    assert np.allclose(saved,reloaded,rtol=1e-10,atol=1e-10)
                    verified_models+=1
    assert verified_metrics==140 and verified_models==140 and skipped==52560
    assert hashlib.sha256((ROOT/run['config']['source']).read_bytes()).hexdigest()==run['hashes']['source_sha256']
    checks={'status':'passed','metric_rows_recomputed':verified_metrics,'saved_models_reload_checked':verified_models,'test_rows_skipped':skipped,'test_predictions':0,'checks':['Seven disjoint room folds','TRAIN-only out-of-fold files, 122640 rows per model/feature set','Recomputed MAE, RMSE, R2 for all 140 fold-target results','Reloaded all 140 models and reproduced three saved raw predictions each','Ridge scaler means match only the six training rooms in every fold','Raw data SHA-256 unchanged'],'scope':'Pipeline reproducibility and split integrity; not evidence that predictive accuracy is sufficient'}
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Report: verification passed; creating figures.',flush=True)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    combos=[('Ridge',FEATURES[0]),('Ridge',FEATURES[1]),('HistGB',FEATURES[0]),('HistGB',FEATURES[1])]
    colors=['#b4b8bc','#6e777e','#a4c4d7','#28688d']
    fig,axes=plt.subplots(2,3,figsize=(15,8.3))
    for ax,target in zip(axes.flat,TARGETS):
        vals=[float(summary[(summary.model==m)&(summary.feature_set==f)&(summary.target==target)].mae_mean.iloc[0])*FACTORS[target] for m,f in combos]
        ax.bar(range(4),vals,color=colors,width=.66)
        ax.set_xticks(range(4),['Ridge\n8 inputs','Ridge\n13 inputs','HistGB\n8 inputs','HistGB\n13 inputs'])
        ax.set_title(LABELS[target],loc='left',fontweight='bold')
        ax.set_ylabel(f'Mean room MAE ({UNITS[target]})')
        ax.set_xlabel('Model and number of raw input fields')
        ax.set_ylim(0,max(vals)*1.22)
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:.4f}' if target=='PV_AC_kWh' else f'{v:.2f}',ha='center',fontsize=9)
    axes.flat[5].axis('off')
    axes.flat[5].text(0,.9,'Seven-room leave-one-room-out CV\n\nEach bar: unweighted mean MAE over 7 rooms.\nLower is better. Predictions are physically clipped.\n\n8 / 13 = raw input fields; encoded counts = 9 / 15.\nAll four angles and retained daytime hours included.\nThree TEST rooms were never predicted or scored.\n\nSource: saved out-of-fold simulation predictions.\nFull non-leap-year sequence, Hour 7..18.\nEPW matching is not independently verified.',va='top',linespacing=1.65)
    fig.suptitle('Tower A PVSD: baseline and weather-input comparison',fontsize=15,fontweight='bold',x=.055,ha='left')
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(OUT/'figures'/'mae_comparison.png',bbox_inches='tight')
    fig.savefig(OUT/'figures'/'mae_comparison.svg',bbox_inches='tight')
    print('Report: MAE overview saved.',flush=True)
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,8.3))
    room_order=sorted(train_rooms)
    for ax,target in zip(axes.flat,TARGETS):
        g=pairs[(pairs.model=='HistGB')&(pairs.target==target)].set_index('validation_room').loc[room_order]
        ax.plot(room_order,g.A_bim_time_angle*FACTORS[target],'-o',color='#888888',label='8 raw inputs',markersize=4)
        ax.plot(room_order,g.B_add_solar_weather*FACTORS[target],'-o',color='#28688d',label='13 raw inputs',markersize=4)
        ax.set_title(LABELS[target],loc='left',fontweight='bold')
        ax.set_xlabel('Held-out TRAIN room')
        ax.set_ylabel(f'MAE ({UNITS[target]})')
        if target=='E_P10_lux':ax.set_yscale('log');ax.set_ylabel('MAE (lux, logarithmic scale)')
        ax.tick_params(axis='x',rotation=35)
        ax.legend(frameon=False,fontsize=8)
    axes.flat[5].axis('off')
    axes.flat[5].text(0,.9,'HistGradientBoostingRegressor\n\nEach point is one room held out from training.\nLines connect rooms only to aid comparison.\nP10 uses a log scale due to large room differences.\n\nSource: paired_room_comparison.csv.\nFull non-leap-year sequence, Hour 7..18.\nNo TEST-room performance is included.',va='top',linespacing=1.7)
    fig.suptitle('Room-level errors reveal uneven generalization',fontsize=15,fontweight='bold',x=.055,ha='left')
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(OUT/'figures'/'room_mae_comparison.png',bbox_inches='tight')
    print('Report: room-level MAE figure saved.',flush=True)
    plt.close(fig)

    r=fm[(fm.model=='HistGB')&(fm.feature_set==FEATURES[1])].pivot(index='validation_room',columns='target',values='r2').loc[room_order,TARGETS]
    fig,ax=plt.subplots(figsize=(10,6.4))
    im=ax.imshow(np.clip(r.to_numpy(),-1,1),cmap='RdBu',vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(5),['E_mean','E_P10','F_300','F_UDI','PV_AC'])
    ax.set_yticks(range(7),room_order)
    ax.set_xlabel('Predicted performance metric')
    ax.set_ylabel('Held-out TRAIN room')
    for i in range(7):
        for j in range(5):
            value=r.iloc[i,j]
            ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if abs(value)>.6 else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Room R2 (color clipped to [-1, 1]; annotations show actual values)')
    ax.set_title('HistGB with 13 inputs: room R2 varies substantially',loc='left',fontweight='bold',pad=14)
    fig.text(.1,.01,'Source: TRAIN-only out-of-fold predictions, full annual sequence, Hour 7..18. Negative R2: worse squared error than that room\'s constant-mean reference. P10 near-zero variance can amplify negative R2.',fontsize=8,wrap=True)
    fig.tight_layout(rect=[0,.065,1,1])
    fig.savefig(OUT/'figures'/'room_r2.png',bbox_inches='tight')
    plt.close(fig)
    payload={'run':{'id':run['experiment_id'],'train_rows':run['train_rows'],'train_rooms':run['train_rooms'],'completed_fits':140,'elapsed_seconds':run['elapsed_seconds'],'test_predictions':0,'versions':run['versions']},'summary':summary.to_dict('records'),'weather':weather.to_dict('records'),'folds':fm.to_dict('records'),'verification':checks}
    (OUT/'report_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Verification passed: all 140 fold results recomputed, all 140 model files reload checked, TEST exclusion verified. Figures and report_data.json saved.')


if __name__=='__main__':main()
