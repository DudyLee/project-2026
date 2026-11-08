"""Independently verify diagnostic outputs; build a focused evidence report."""
from pathlib import Path
import hashlib
import json
import runpy
import html

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/diagnosis_v1'
FOCUS=[('BM223','E_P10_lux'),('BM522','E_mean_lux')]
NAMES={'Original_HistGB_B':'Original HistGB','HistGB_log1p':'HistGB log1p','HistGB_absolute':'HistGB absolute loss','HistGB_add_floor_WFR':'Add floor + WFR','HistGB_without_obstruction':'Remove obstruction','ExtraTrees_B':'ExtraTrees'}
ORDER=list(NAMES)


def main():
    cfg=json.loads((ROOT/'configs/diagnosis_v1.json').read_text())
    run=json.loads((OUT/'run_manifest.json').read_text())
    assert run['status']=='completed' and run['cv_fits']==70 and run['known_room_fits']==2 and run['model_files']==72
    fm=pd.read_csv(OUT/'ablation_fold_metrics.csv')
    summary=pd.read_csv(OUT/'ablation_summary.csv')
    profiles=pd.read_csv(OUT/'train_room_profiles.csv')
    sensitivity=pd.read_csv(OUT/'static_feature_sensitivity.csv')
    template=pd.read_csv(OUT/'training_room_template_diagnostic.csv')
    error=pd.read_csv(OUT/'frozen_model_train_validation.csv')
    known=json.loads((OUT/'known_room_time_diagnostic.json').read_text())
    checks=json.loads((OUT/'train_data_checks.json').read_text())
    base=runpy.run_path(str(ROOT/'scripts/train_logo_v1.py'),run_name='diagnostic_verification')
    plan=json.loads((ROOT/'configs/model_plan.json').read_text())
    d,skipped=base['load_train'](ROOT/cfg['source'],plan)
    for c in ['Floor_0based','WFR_window_to_floor']:
        d[c]=d.Room_ID.map(profiles.set_index('Room_ID')[c])
    protected=set(cfg['protected_test_rooms'])
    assert len(fm)==84 and not set(fm.room).intersection(protected)
    verified_metrics,verified_models=0,0
    with threadpool_limits(limits=4):
        for variant in cfg['variants']:
            for target in cfg['targets']:
                p=pd.read_parquet(OUT/'predictions'/f'{variant}__{target}.parquet')
                assert len(p)==122640 and set(p.Room_ID)==set(cfg['development_rooms'])
                assert not p.duplicated(['Room_ID','PVSD_Angle_deg','HOY_0based']).any()
                assert not set(p.Room_ID).intersection(protected)
                assert np.array_equal(p.truth,d[target])
                for room,g in p.groupby('Room_ID'):
                    r=fm[(fm.variant==variant)&(fm.target==target)&(fm.room==room)].iloc[0]
                    diff=g.prediction.to_numpy()-g.truth.to_numpy()
                    mae=float(np.abs(diff).mean())
                    rmse=float(np.sqrt(np.square(diff).mean()))
                    r2=1-float(np.square(diff).sum())/float(np.square(g.truth.to_numpy()-g.truth.mean()).sum())
                    assert np.allclose([mae,rmse,r2],[r.mae,r.rmse,r.r2],rtol=1e-9,atol=1e-9)
                    verified_metrics+=1
                    fold=cfg['development_rooms'].index(room)+1
                    bundle=joblib.load(ROOT/cfg['model_output']/variant/target/f'fold_{fold:02d}_{room}.joblib')
                    assert room not in bundle['train_rooms'] and len(bundle['train_rooms'])==6
                    rows=g.index[[0,len(g)//2,-1]]
                    x=base['encode_inputs'](d.loc[rows],bundle['raw_features'])
                    pred=bundle['estimator'].predict(x)
                    if bundle['target_log1p']:pred=np.expm1(pred)
                    assert np.allclose(pred,p.loc[rows,'raw_prediction'],rtol=1e-9,atol=1e-9)
                    verified_models+=1
        for r in known:
            p=pd.read_parquet(OUT/'predictions'/f'known_room_month_holdout__{r["room"]}__{r["target"]}.parquet')
            assert len(p)==5856 and not set(p.Room_ID).intersection(protected)
            g=d[d.Room_ID.eq(r['room']) & d.Month.isin([3,6,9,12])]
            assert np.array_equal(p.truth,g[r['target']])
            assert np.isclose(np.abs(p.known_room_prediction-p.truth).mean(),r['known_room_mae'])
            assert np.isclose(np.abs(p.unseen_room_baseline_prediction-p.truth).mean(),r['unseen_room_baseline_mae_same_times'])
            assert not set(r['training_months']).intersection(r['validation_months'])
            bundle=joblib.load(ROOT/cfg['model_output']/f'known_room__{r["room"]}__{r["target"]}.joblib')
            ids=[0,len(g)//2,-1]
            reproduced=np.maximum(bundle['estimator'].predict(base['encode_inputs'](g.iloc[ids],bundle['raw_features'])),0)
            assert np.allclose(reproduced,p.known_room_prediction.iloc[ids],rtol=1e-9,atol=1e-9)
    assert verified_metrics==70 and verified_models==70
    assert hashlib.sha256((ROOT/cfg['source']).read_bytes()).hexdigest()==run['source_sha256']
    verification={'status':'passed','new_cv_metric_rows_recomputed':70,'new_cv_models_reloaded':70,'known_room_models_reloaded':2,'known_room_month_splits_disjoint':True,'TRAIN_rows_per_variant_target':122640,'test_predictions':0,'test_scores':0,'source_unchanged':True,'scope':'Pipeline verification; diagnostic model comparisons remain exploratory development results'}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':145})
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    raw=base['load_train'](ROOT/cfg['source'],plan)[0]
    for ax,(room,target) in zip(axes,FOCUS):
        p=pd.read_csv(OUT/f'{room}__{target}__baseline_predictions.csv')
        a=p.groupby('Hour')[[target,'prediction']].mean()
        reference=raw[raw.Room_ID.eq('BM432')].groupby('Hour')[target].mean()
        ax.plot(a.index,a[target],'-o',label=f'{room} actual',color='#285d77',markersize=3)
        ax.plot(a.index,a.prediction,'-s',label=f'{room} original prediction',color='#b06736',markersize=3)
        ax.plot(reference.index,reference,'--',label='BM432 actual (training room)',color='#7b848a')
        ax.set_title(f'{room}: {target}',loc='left',fontweight='bold')
        ax.set_xlabel('Source Hour label')
        ax.set_ylabel('Mean over all days and four angles (lux)')
        if room=='BM223':ax.set_yscale('log');ax.set_ylabel('Mean illuminance (lux; logarithmic scale)')
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Both failed transfers resemble the BM432 daylight curve',x=.06,ha='left',fontweight='bold',fontsize=14)
    fig.text(.06,.015,'Source: frozen TRAIN-only out-of-fold predictions and TRAIN simulations, annual Hour 7..18. Curve correlation uses matching angle/hour records; plot averages across days and angles.',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.93])
    fig.savefig(OUT/'figures'/'failure_fingerprint.png',bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5.2))
    for ax,(room,target) in zip(axes,FOCUS):
        r=fm[(fm.room.eq(room))&(fm.target.eq(target))].set_index('variant').loc[ORDER]
        bars=ax.bar(range(6),r.mae,color=['#a4a9ad']+['#55819c']*5)
        ax.set_xticks(range(6),[NAMES[v] for v in ORDER],rotation=30,ha='right',fontsize=8)
        ax.set_ylabel('Held-out room MAE (lux)')
        ax.set_xlabel('Controlled diagnostic variant')
        ax.set_title(f'{room}: {target}',loc='left',fontweight='bold')
        ax.set_ylim(0,r.mae.max()*1.18)
        for bar,value in zip(bars,r.mae):ax.text(bar.get_x()+bar.get_width()/2,value+r.mae.max()*.025,f'{value:.2f}',ha='center',fontsize=8)
    fig.suptitle('Model changes reduce some errors but do not resolve new-room transfer',x=.06,ha='left',fontweight='bold',fontsize=14)
    fig.text(.06,.012,'Source: diagnosis_v1, same seven TRAIN room folds. Variants were chosen after inspecting development failures; results are exploratory. No TEST rooms used.',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.92])
    fig.savefig(OUT/'figures'/'focus_ablation.png',bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,r in zip(axes,known):
        values=[r['unseen_room_baseline_mae_same_times'],r['known_room_mae']]
        bars=ax.bar([0,1],values,color=['#a4a9ad','#285d77'],width=.6)
        ax.set_xticks([0,1],['Unseen-room\noriginal model','Known-room\nmonth holdout'])
        ax.set_xlabel('Diagnostic training task (different information available)')
        ax.set_ylabel('MAE on identical held months (lux)')
        ax.set_title(f'{r["room"]}: {r["target"]}',loc='left',fontweight='bold')
        if r['room']=='BM223':ax.set_yscale('log');ax.set_ylabel('MAE (lux; logarithmic scale)')
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value*(1.2 if r['room']=='BM223' else 1.04),f'{value:.3f}',ha='center')
        if r['room']=='BM223':ax.set_ylim(.02,200)
        else:ax.set_ylim(0,max(values)*1.25)
    fig.suptitle('Within-room prediction is learnable; transfer across rooms is the bottleneck',x=.055,ha='left',fontsize=13,fontweight='bold')
    fig.text(.055,.01,'Diagnostic only: train on each focus room in months 1/2/4/5/7/8/10/11; hold out entire months 3/6/9/12 and all angles. These are NOT new-room validation scores.',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.92])
    fig.savefig(OUT/'figures'/'known_vs_unseen.png',bbox_inches='tight')
    plt.close(fig)
    focus_rows=fm[[any(row.room==room and row.target==target for room,target in FOCUS) for row in fm.itertuples()]].copy()
    focus_rows.to_csv(OUT/'focus_ablation_comparison.csv',index=False)
    payload={'room_profiles':profiles.to_dict('records'),'focus_errors':error.to_dict('records'),'template_diagnostic':template.to_dict('records'),'focus_ablations':focus_rows.to_dict('records'),'cv_summary':summary.to_dict('records'),'all_room_cv':fm.to_dict('records'),'known_room':known,'data_checks':checks,'verification':verification}
    (OUT/'report_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    def table(frame):return '<div class="scroll">'+frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')+'</div>'
    focus_table=focus_rows[['variant','room','target','mae','rmse','r2']].copy()
    profile_cols=['Room_ID','Room_Area_m2','Orientation_X_model','Orientation_Y_model','Solar_Obstruction_Ratio','Sensor_Count','E_P10_lux__mean','E_mean_lux__mean']
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BM223 与 BM522 诊断</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1120px;margin:32px auto;padding:0 22px;color:#263039;background:#fafbfc;line-height:1.8}h1{font-size:26px}h2{font-size:20px;margin-top:32px}h3{font-size:17px}p,li{max-width:1000px}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:9px 11px;border-bottom:1px solid #dbe1e5;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf1f3}.scroll{overflow:auto}img{max-width:100%;display:block;margin-top:16px}.evidence{border-left:3px solid #507e94;padding:5px 18px;background:#edf3f6}.caption{font-size:13px;color:#606e77}a{color:#255f7e}strong{font-weight:650}</style>
<h1>BM223 的 P10 与 BM522 的平均照度诊断</h1>
<p class="evidence"><strong>主要证据指向：训练房间覆盖不足、现有 BIM 特征不能稳定表达采光空间分布，以及树模型利用面积等特征区分少数房间。模型目标和算法也会影响误差，但简单换模型尚未解决问题。</strong></p>
<p>范围：仅 7 个 TRAIN 房间。新增 70 次留一房间拟合及 2 次已知房间整月留出诊断，未对 3 个 TEST 房间预测、评分或统计目标分布。原始数据、初版模型与结果均保留。</p>
<h2>BM223 的 P10 为什么失败</h2>
<p>真实 P10 平均 <strong>0.616 lux</strong>，原模型预测平均 <strong>54.208 lux</strong>，每条记录都高估，MAE 为 <strong>53.592 lux</strong>。真实值标准差只有 0.616 lux，因此巨大负 R² 会被小方差进一步放大；但绝对误差也远超真实尺度，不能仅归因于 R² 的计算方式。作为量级参照，恒为 0 的输出在该房间上的 MAE 为 0.616 lux，这不是给所有房间预测 0 的建议。</p>
<p><strong>覆盖问题已经直接确认：</strong>BM223 是 7 个训练房间中唯一的模型 +X 朝向。留出它后，剩余 6 个房间的 Orientation_X_model 全为 0。将冻结模型输入的 X 朝向从 1 改为 0 时，全部预测不变。这个验证折没有学到 +X 房间响应的样本。</p>
<p><strong>模型表现出面积分组的捷径：</strong>把诊断输入中的面积从 6.659 改为 7.763 m²，其他输入保持不变，P10 预测平均改变约 54.08 lux。该操作只是在探查模型，不能通过修改真实面积来修复预测。原预测与 BM432 同角度、同时间的真实 P10 相关系数为 <strong>0.9945</strong>；BM432 的 P10 均值是 53.683 lux。</p>
<h2>BM522 的平均照度为什么失败</h2>
<p>真实均值 <strong>152.811 lux</strong>，预测均值 <strong>389.718 lux</strong>；85.4% 的记录高估，MAE 为 <strong>246.265 lux</strong>。剩余 6 个训练房间上的拟合 MAE 为 15.709 lux，说明模型能拟合已有房间，却未成功迁移。</p>
<p>BM522 与 BM432 面积接近、玻璃面积相同，但模型朝向分别为 +Y 与 −Y。模型给 BM522 的预测曲线与 BM432 实际平均照度曲线相关系数为 <strong>0.9921</strong>。将面积从 6.443 改为 6.595 m² 时，预测全部不变；再改为 6.659 m²，预测平均绝对变化约 <strong>282.61 lux</strong>。这说明模型对少量房间的面积阈值依赖很强，方向和遮挡并未形成足够稳健的迁移规律。</p>
<img src="figures/failure_fingerprint.png" alt="两个失败预测与BM432模拟曲线对照">
<h2>数据是否出错</h2>
<p>目前未发现 CSV 层面的直接错误：无缺失或重复工况时间记录；比例乘以 Sensor_Count 与整数传感器数一致；平均照度、P10 和阈值比例之间满足所检查的非负照度约束；每个房间的 BIM 和传感器数量在各工况间固定。</p>
<p><strong>但还不能证明源数据完全正确。</strong>我们没有原始逐点传感器结果、网格与导出脚本，无法重算 P10 或确认传感器属于正确房间。BM223、BM432、BM522 的传感器数量分别为 74、40、66，虽然面积都约 6–7 m²，网格范围、分辨率或形状可能不同；数量不同本身并不证明错误。P10 的空间分布高度依赖房间深度、窗户位置、工作面及网格范围，这些没有被当前特征充分记录。</p>
<p>优先抽取同一天、同一小时、同一角度的三房间原始逐点 lux，检查房间 ID、网格坐标、工作面高度、采样间距、传感器是否越界或含室外点，并用相同的 numpy 线性百分位重算 P10；同时重算均值及 F_300、F_UDI。保留当前低 P10 记录，不能凭数值小就删除或改单位。</p>
<h2>更换模型与特征的对照</h2>
<p>五个方案均保持相同的 7 个留一房间折，分别预测 P10 和平均照度。只改目标变换、损失、既有 BIM 特征或算法中的一类设置；没有利用 TEST 做选择。</p>
<img src="figures/focus_ablation.png" alt="五项诊断方案对两个房间误差的影响">
__FOCUS_TABLE__
<p>ExtraTrees 将 BM223 P10 的 MAE 降到 18.787 lux，但仍远大于真实均值，且在 BM441、BM608、LM608、LM703 等低 P10 房间的误差变大，不能只根据总体均值下降就全局替换。加楼层和窗地比对两个问题基本没有帮助；移除遮挡使平均照度的 7 房间平均 MAE 恶化约 26%，不应把有用遮挡信息一概删掉。</p>
<p>log1p 目标变换将 BM522 平均照度 MAE 从 246.265 降到 149.983 lux；七房间平均 MAE 从 109.490 降到 95.921 lux，下降约 12.4%。这是可继续研究的候选，但该房间 R² 仍为 −4.19，且部分其他房间变差。逆变换也不保证无偏地估计原尺度条件均值。</p>
<h2>区分模型容量与跨房间信息不足</h2>
<p>对两个焦点房间各自用 8 个完整月份训练，留出 3/6/9/12 月及其全部角度；与原跨房间模型在完全相同的留出时刻比较：BM223 P10 的 MAE 从 53.264 变为 <strong>0.042 lux</strong>，BM522 平均照度从 257.353 变为 <strong>8.526 lux</strong>，R² 均约 0.98。</p>
<p><strong>这是不同任务的诊断，不是新房间泛化成绩。</strong>它使用了目标房间其他月份的模拟标签，只表明当前模型能学习同一房间随时间和天气变化的关系，不能据此宣称对未模拟房间已经准确，也不能证明原始网格或 EPW 对齐正确。</p>
<img src="figures/known_vs_unseen.png" alt="同一留出月份下跨房间模型与已知房间时间模型对照">
<h2>建议的改进顺序</h2>
<ol><li><strong>先核实采光源数据：</strong>完成上述三房间的逐点 lux、网格和导出核对；补充 EPW 与地理北处理说明。</li><li><strong>补能表达采光分布的特征：</strong>房间进深与宽度、长宽比、窗洞宽高与窗台高度、窗口位置、玻璃可见光透射率、反射率和工作面高度；所有房间使用统一定义。不要把传感器数或 Room_ID 当成预测房间响应的替代标识。</li><li><strong>优先补代表样本：</strong>若需坚持未见房间泛化，增加 +X 朝向的训练房间，以及与 BM522 相似的小面积、+Y、较低遮挡工况。不能把现有 TEST 房间移入训练来修补覆盖。</li><li><strong>模型上先保留可解释的小步改进：</strong>对平均照度继续研究 log1p 或稳健损失，报告逐房间 MAE 与 RMSE；P10 不建议仅靠加树、换 LLM 或挑一个平均分较好的模型解决。</li><li><strong>若可接受少量房间校准：</strong>可另立“少量目标房间模拟后校准”的任务，再设计独立的房间与时间验证。当前整月诊断没有验证少量校准所需样本量，不能直接套用其高分。</li></ol>
<h2>七个训练房间与整体对照</h2>__PROFILE_TABLE__<h3>七房间平均误差</h3>__SUMMARY_TABLE__
<p class="caption">数据来源：diagnosis_v1 的实际模拟输入、冻结模型、留出预测及控制实验。所有指标均在原 lux 单位评价。数据检查和全部 70 条新增 CV 指标、70 个新增 CV 模型及 2 个时间诊断模型的重载复核已通过。诊断方案是在看到开发失败后设计的，仍属于开发期探索；3 个 TEST 房间没有预测、评分或目标分布分析。</p>
<p class="caption">方法依据：<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html">R² 定义</a>；<a href="https://scikit-learn.org/stable/modules/compose.html#transforming-target-in-regression">目标变换与逆变换</a>。</p>
<p><a href="focus_ablation_comparison.csv">焦点房间对照</a> · <a href="ablation_fold_metrics.csv">全部逐折指标</a> · <a href="static_feature_sensitivity.csv">冻结模型敏感性</a> · <a href="train_data_checks.json">数据检查</a> · <a href="verification.json">复核记录</a></p></html>'''
    body=body.replace('__FOCUS_TABLE__',table(focus_table)).replace('__PROFILE_TABLE__',table(profiles[profile_cols])).replace('__SUMMARY_TABLE__',table(summary))
    (OUT/'report.html').write_text(body,encoding='utf-8')
    print('Diagnosis verified: 70 CV metrics, 72 model reloads, TRAIN-only outputs; figures and report.html saved.')


if __name__=='__main__':main()
