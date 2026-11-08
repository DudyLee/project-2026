"""Independent verification and report for the approved P10 target experiments."""
from pathlib import Path
import hashlib
import json
import runpy

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/experiment_v2'
ORDER=['Original_HistGB_B','Unweighted_log1p_P10','Weighted_log1p_P10','Relative_P10']
LABELS={'Original_HistGB_B':'Original P10','Unweighted_log1p_P10':'log1p P10','Weighted_log1p_P10':'Weighted log1p','Relative_P10':'Relative P10'}


def main():
    cfg=json.loads((ROOT/'configs/experiment_v2.json').read_text())
    run=json.loads((OUT/'run_manifest.json').read_text())
    assert run['status']=='completed' and run['new_P10_fits']==14 and run['reused_mean_model_count']==7 and run['saved_model_count']==21
    base=runpy.run_path(str(ROOT/'scripts/train_logo_v1.py'),run_name='verify_v2')
    plan=json.loads((ROOT/'configs/model_plan.json').read_text())
    d,skipped=base['load_train'](ROOT/cfg['source'],plan)
    protected=set(cfg['protected_test_rooms'])
    cols=plan['feature_sets'][cfg['feature_set']]
    x=base['encode_inputs'](d,cols)
    fm=pd.read_csv(OUT/'fold_metrics.csv')
    summary=pd.read_csv(OUT/'summary_metrics.csv')
    weights=pd.read_csv(OUT/'training_room_weights.csv')
    splits=json.loads((OUT/'fold_splits.json').read_text())
    mean_metrics=pd.read_csv(OUT/'mean_stage_fold_metrics.csv')
    provenance=json.loads((OUT/'reused_mean_model_provenance.json').read_text())
    references={
        'Original_HistGB_B':pd.read_parquet(ROOT/'reports/experiment_v1/predictions/B_add_solar_weather__HistGB.parquet').rename(columns={'true__E_P10_lux':'truth','pred__E_P10_lux':'prediction'}),
        'Unweighted_log1p_P10':pd.read_parquet(ROOT/'reports/diagnosis_v1/predictions/HistGB_log1p__E_P10_lux.parquet'),
    }
    predictions={**references,**{method:pd.read_parquet(OUT/'predictions'/f'{method}.parquet') for method in cfg['p10_methods']}}
    mean_oof=pd.read_parquet(OUT/'predictions/Mean_log1p.parquet')
    assert set(mean_oof.Room_ID)==set(plan['validation']['development_rooms']) and len(mean_oof)==122640
    metric_checks=0
    for method,p in predictions.items():
        assert len(p)==122640 and not protected.intersection(p.Room_ID)
        assert set(p.Room_ID)==set(d.Room_ID)
        assert p[['Room_ID','PVSD_Angle_deg','HOY_0based']].equals(d[['Room_ID','PVSD_Angle_deg','HOY_0based']])
        assert np.array_equal(p.truth,d.E_P10_lux)
        assert not p.duplicated(['Room_ID','PVSD_Angle_deg','HOY_0based']).any()
        for room,g in p.groupby('Room_ID'):
            r=fm[(fm.method==method)&(fm.room==room)].iloc[0]
            delta=g.prediction.to_numpy()-g.truth.to_numpy()
            independent=[np.mean(np.abs(delta)),np.sqrt(np.mean(delta**2)),1-np.sum(delta**2)/np.sum((g.truth.to_numpy()-g.truth.mean())**2)]
            assert np.allclose(independent,[r.mae,r.rmse,r.r2],atol=1e-9,rtol=1e-9)
            metric_checks+=1
    verified_new_models=0
    with threadpool_limits(limits=4):
        for split in splits:
            fold,room=split['fold'],split['validation_room']
            vi=d.Room_ID.eq(room)
            ti=d.Room_ID.isin(split['training_rooms'])
            assert not (vi&ti).any() and vi.sum()==17520 and ti.sum()==105120
            assert len(split['training_rooms'])==6 and room not in split['training_rooms']
            pr=[r for r in provenance if r['fold']==fold][0]
            for path in [pr['source'],pr['archived']]:assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==pr['sha256']
            mean_bundle=joblib.load(ROOT/pr['archived'])
            assert mean_bundle['train_rooms']==split['training_rooms'] and mean_bundle['held_room']==room
            # This prediction receives only the encoded input columns, never true E_mean.
            mean_pred=np.maximum(np.expm1(mean_bundle['estimator'].predict(x.loc[vi])),0)
            assert np.allclose(mean_pred,mean_oof.loc[vi,'prediction'],atol=1e-9,rtol=1e-9)
            actual_mean_metrics=base['metrics'](d.loc[vi,'E_mean_lux'].to_numpy(),mean_pred)
            stored_mean=mean_metrics[mean_metrics.room.eq(room)].iloc[0]
            assert np.allclose([actual_mean_metrics[c] for c in ['mae','rmse','r2']],[stored_mean[c] for c in ['mae','rmse','r2']],atol=1e-9,rtol=1e-9)
            wi=weights[weights.fold.eq(fold)].set_index('fitting_room')
            assert set(wi.index)==set(split['training_rooms']) and room not in wi.index
            fit_means=d.loc[ti].groupby('Room_ID').E_P10_lux.mean()
            manual_raw={r:1/np.sqrt(1+fit_means[r]) for r in split['training_rooms']}
            manual_normalizer=float(d.loc[ti,'Room_ID'].map(manual_raw).mean())
            assert np.allclose(wi.loc[fit_means.index,'fitting_room_mean_P10'],fit_means)
            assert np.allclose(wi.loc[fit_means.index,'normalized_weight'],[manual_raw[r]/manual_normalizer for r in fit_means.index])
            assert np.isclose(np.average(wi.normalized_weight,weights=wi.training_rows),1)
            for method in cfg['p10_methods']:
                bundle=joblib.load(ROOT/cfg['model_output']/method/f'fold_{fold:02d}_{room}.joblib')
                assert bundle['training_rooms']==split['training_rooms'] and room not in bundle['training_rooms']
                assert bundle['raw_features']==cols and bundle['encoded_features']==list(x.columns)
                assert not set(bundle['raw_features']).intersection(plan['targets'])
                sample_ids=d.loc[vi].index[[0,8760,17519]]
                raw_component=bundle['estimator'].predict(x.loc[sample_ids])
                if method=='Weighted_log1p_P10':
                    raw=np.expm1(raw_component)
                    pred=np.maximum(raw,0)
                    assert set(bundle['weight_scales'])==set(split['training_rooms'])
                else:
                    mean_sample=np.maximum(np.expm1(mean_bundle['estimator'].predict(x.loc[sample_ids])),0)
                    raw=raw_component*(mean_sample+cfg['epsilon_lux'])
                    pred=np.maximum(raw_component,0)*(mean_sample+cfg['epsilon_lux'])
                    p=predictions[method]
                    assert np.allclose(p.loc[vi,'prediction'],p.loc[vi,'ratio_prediction']*(p.loc[vi,'predicted_mean']+cfg['epsilon_lux']),rtol=1e-10,atol=1e-10)
                    assert np.allclose(p.loc[vi,'predicted_mean'],mean_pred,rtol=1e-10,atol=1e-10)
                p=predictions[method]
                assert np.allclose(pred,p.loc[sample_ids,'prediction'],rtol=1e-9,atol=1e-9)
                assert np.allclose(raw,p.loc[sample_ids,'raw_prediction'],rtol=1e-9,atol=1e-9)
                verified_new_models+=1
    assert metric_checks==28 and verified_new_models==14 and skipped==52560
    assert hashlib.sha256((ROOT/cfg['source']).read_bytes()).hexdigest()==run['source_sha256']
    verification={'status':'passed','P10_metric_rows_recomputed':28,'mean_metric_rows_recomputed':7,'new_P10_models_reloaded':14,'reused_mean_models_hash_and_prediction_checked':7,'training_weight_folds_independently_recomputed':7,'relative_primary_reconstruction_uses_predicted_mean_only':True,'TEST_predictions':0,'TEST_scores':0,'source_unchanged':True,'scope':'Pipeline verification; all comparisons are exploratory development cross-validation'}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print('V2 verification passed: 28 P10 metric rows, 7 mean metric rows, 14 new models, 7 reused mean models, fold-only weights.',flush=True)

    pair=fm.pivot(index='room',columns='method',values='mae')[ORDER]
    paired=pd.read_csv(OUT/'paired_room_comparison.csv')
    decomposition=pd.read_csv(OUT/'ratio_error_decomposition.csv')
    mean_summary={'method':'Mean_log1p','mae_mean':float(mean_metrics.mae.mean()),'rmse_mean':float(mean_metrics.rmse.mean()),'focus_BM522_mae':float(mean_metrics.loc[mean_metrics.room.eq('BM522'),'mae'].iloc[0]),'scope':'Reused frozen six-room-fold models; not a full-seven-room deployment model'}
    original_delta=pair.Original_HistGB_B-pair.Relative_P10
    bm223_share=float(original_delta.loc['BM223']/original_delta.sum())
    stats=[]
    for room,g in predictions['Relative_P10'].groupby('Room_ID'):
        stats.append({'room':room,'true_P10_mean':float(g.truth.mean()),'predicted_P10_mean':float(g.prediction.mean()),'true_mean_illuminance':float(g.true_mean_for_evaluation_only.mean()),'predicted_mean_illuminance':float(g.predicted_mean.mean())})
    pd.DataFrame(stats).to_csv(OUT/'relative_room_prediction_means.csv',index=False)
    (OUT/'mean_stage_summary.json').write_text(json.dumps(mean_summary,indent=2),encoding='utf-8')
    payload={'summary':summary.to_dict('records'),'folds':fm.astype(object).where(pd.notna(fm),None).to_dict('records'),'paired':paired.to_dict('records'),'mean_summary':mean_summary,'mean_folds':mean_metrics.to_dict('records'),'decomposition':decomposition.to_dict('records'),'relative_room_means':stats,'bm223_share_of_total_MAE_reduction':bm223_share,'verification':verification}
    (OUT/'report_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':145})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    colors=['#b7bdc2','#858f97','#4f7d98','#28576f']
    ss=summary.set_index('method').loc[ORDER]
    for ax,col,title in [(axes[0],'mae_mean','Mean room MAE'),(axes[1],'mae_median','Median room MAE')]:
        bars=ax.bar(range(4),ss[col],color=colors,width=.65)
        ax.set_xticks(range(4),[LABELS[m] for m in ORDER],rotation=15,ha='right')
        ax.set_ylabel('P10 MAE (lux)')
        ax.set_xlabel('Target-handling method')
        ax.set_title(title,loc='left',fontweight='bold')
        ax.set_ylim(0,ss[col].max()*1.2)
        for bar,value in zip(bars,ss[col]):ax.text(bar.get_x()+bar.get_width()/2,value+ss[col].max()*.025,f'{value:.3f}',ha='center',fontsize=9)
    fig.suptitle('P10: relative-illuminance reconstruction is the strongest tested candidate',x=.07,ha='left',fontsize=13,fontweight='bold')
    fig.text(.07,.015,'Source: experiment_v2 TRAIN-only leave-one-room-out predictions; 7 rooms, full annual Hour 7..18, four angles. Means and medians are over room MAEs. No TEST scores.',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.92])
    fig.savefig(OUT/'figures/p10_summary.png',bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5.1))
    positions=np.arange(len(pair))
    width=.18
    for i,method in enumerate(ORDER):ax.bar(positions+(i-1.5)*width,pair[method],width=width,label=LABELS[method],color=colors[i])
    ax.set_xticks(positions,pair.index)
    ax.set_yscale('log')
    ax.set_xlabel('Held-out TRAIN room')
    ax.set_ylabel('P10 MAE (lux; logarithmic scale)')
    ax.set_title('Room-level results: BM223 improves, BM432 remains unresolved',loc='left',fontweight='bold',pad=14)
    ax.legend(frameon=False,ncol=4,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.01))
    ax.set_ylim(.05,200)
    fig.text(.075,.015,'Source: seven TRAIN room folds, all retained hours and angles. Log scale exposes dark-room tradeoffs. Weighted log1p uses weights computed only from fitting rooms.',fontsize=8)
    fig.tight_layout(rect=[0,.065,1,1])
    fig.savefig(OUT/'figures/p10_room_comparison.png',bbox_inches='tight')
    plt.close(fig)
    dec=decomposition.set_index('room').loc[['BM223','BM432','BM522']]
    fig,ax=plt.subplots(figsize=(10,4.5))
    positions=np.arange(3)
    ax.bar(positions-.18,dec.ratio_term_bias_lux,width=.35,label='Ratio estimation term',color='#607f95')
    ax.bar(positions+.18,dec.mean_term_bias_lux,width=.35,label='Mean prediction propagation term',color='#b28a65')
    ax.axhline(0,color='#666666',lw=.8)
    ax.set_xticks(positions,dec.index)
    ax.set_xlabel('Held-out TRAIN room')
    ax.set_ylabel('Signed mean error contribution (lux)')
    ax.set_title('Relative P10 error: components can cancel',loc='left',fontweight='bold')
    ax.legend(frameon=False,fontsize=9,loc='lower left')
    fig.text(.08,.012,'Retrospective error decomposition only; true held-out mean is not used for prediction. Signed terms sum to bias, not to MAE. Source: experiment_v2 saved outputs.',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,1])
    fig.savefig(OUT/'figures/ratio_error_components.png',bbox_inches='tight')
    plt.close(fig)
    print('Figures and structured V2 report data saved.',flush=True)


if __name__=='__main__':main()
