"""Present the frozen-result diagnosis and a clearly unapproved next-round draft."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/decision_diagnosis_v3'
METHODS=['Frozen_HistGB','HistGB','RandomForest','XGBoost','Nested_select']
def read(name):return pd.read_csv(OUT/f'{name}.csv')
def rec(f):return f.astype(object).where(pd.notna(f),None).to_dict('records')
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    audit=json.loads((OUT/'analysis.json').read_text());assert audit['models_fitted']==0
    profiles=read('angle_profiles_summary');headroom=read('headroom_summary');switch=read('switch_decomposition_summary');rooms=read('switch_decomposition')
    constraints=read('constraint_diagnosis');data_profiles=read('data_profiles');terms=read('balanced_terms_summary')
    for i,r in switch.iterrows():
        switch.loc[i,'pooled_harmful_fraction_among_switches']=r.harmful_switch_fraction_all_hours/r.switch_fraction if r.switch_fraction else None
        switch.loc[i,'pooled_beneficial_fraction_among_switches']=r.beneficial_switch_fraction_all_hours/r.switch_fraction if r.switch_fraction else None
    avail=constraints.drop_duplicates('room')
    f=read('headroom');pv=f[f.policy.eq('PV_priority')]
    baseline=profiles.loc[profiles.angle.eq(0),'PV_retained_hours_kWh'].item();potential=pv.sum_oracle_gain_over_zero.mean()
    balanced=headroom[headroom.policy.eq('Balanced')].iloc[0]
    stats={'joint_F300_FUDI_PV_weak_dominance_fraction':float(data_profiles.zero_weakly_dominates_all_F300_FUDI_PV_fraction.mean()),
           'F300_zero_best_fraction':float(data_profiles.F300_zero_best_fraction.mean()),'FUDI_zero_best_fraction':float(data_profiles.FUDI_zero_best_fraction.mean()),
           'balanced_zero_unique_best_fraction':float(balanced.zero_unique_best_fraction),'balanced_zero_near_optimal_fraction':float(balanced.zero_near_optimal_fraction),
           'balanced_material_opportunity_fraction':float(balanced.material_opportunity_fraction),'balanced_all_four_equal_fraction':float(balanced.all_four_equal_fraction),
           'PV_fixed_zero_mean_kWh':float(baseline),'PV_oracle_mean_gain_over_zero_kWh':float(potential),
           'PV_oracle_aggregate_gain_fraction':float(potential/baseline),'PV_material_opportunity_fraction':float(headroom.loc[headroom.policy.eq('PV_priority'),'material_opportunity_fraction'].item()),
           'constraint_true_no_feasible_fraction':float(avail.true_no_feasible_fraction.mean()),
           'zero_covers_available_pooled_fraction':float(avail.zero_feasible_fraction.sum()/avail.true_any_feasible_fraction.sum()),
           'daylight_headroom_share_BM223':float(f.loc[(f.policy=='Daylight_priority')&(f.room=='BM223'),'sum_oracle_gain_over_zero'].item()/f.loc[f.policy.eq('Daylight_priority'),'sum_oracle_gain_over_zero'].sum()),
           'balanced_headroom_share_BM223_LM608':float(f.loc[f.policy.eq('Balanced')&f.room.isin(['BM223','LM608']),'sum_oracle_gain_over_zero'].sum()/f.loc[f.policy.eq('Balanced'),'sum_oracle_gain_over_zero'].sum())}
    draft={'status':'PROPOSED_NOT_APPROVED_NOT_EXECUTED','question':'Can a model identify positive incremental PV generation relative to fixed 0 degrees in an unseen development room?',
        'scope':'Small PV-only diagnosis experiment; daylight metrics reported descriptively for chosen angles. Final comfort/balanced objective remains to be confirmed.',
        'data':'Existing TRAIN only, no new simulations, same 13 raw / 15 encoded features, no new physical direction features',
        'protected_TEST':['BM532','BM702','BMG26'],
        'comparators':['Fixed_0','Frozen_HistGB absolute PV reference','Existing XGBoost absolute PV reference','Absolute_XGBoost with inner-selected gain threshold','Delta_XGBoost without a positive margin (threshold 0)','Delta_XGBoost with inner-selected gain threshold'],
        'delta_target':'PV(room,time,angle)-PV(room,time,0 degrees), paired strictly within the same fitting room and hour; only angles 30/60/90 are fitted. Delta at 0 is fixed at zero.',
        'inference':'Use X only. True PV at 0 is never an inference input. Choose the largest predicted gain; switch only when it exceeds the selected threshold; otherwise keep 0.',
        'model_family':'XGBoost only, to isolate the response/decision change; freeze the per-outer-fold XGBoost parameters already selected from that fold\'s six fitting rooms in v3; no additional family or hyperparameter search',
        'threshold_grid_kWh_per_hour':[0,0.001,0.002,0.005,'infinity_keep_zero'],
        'threshold_selection':'Within each outer set of six rooms generate three-fold grouped OOF predictions. Choose each variant\'s threshold by maximum mean room PV gain over fixed 0; numerical ties choose the larger threshold. Never use current seven-room OOF predictions as cross-fold calibration data.',
        'validation':'Seven outer leave-one-room-out folds; inner three group folds, four fit and two validate; outer room only scores; same folds as v3.',
        'expected_new_primitive_fits':49,
        'fit_count_explanation':'7 outer folds x (3 inner absolute-PV fits + 3 inner delta-PV fits + 1 outer delta-PV fit); reuse saved outer absolute-XGBoost predictions',
        'primary_metrics':['Per-room net retained-hours PV gain over fixed 0 in kWh','Seven-room mean and median gain','Worst-room loss and count of rooms improved/tied/worse','Regret vs four-angle simulated oracle'],
        'secondary_metrics':['Beneficial/harmful/neutral switch rates','Missed positive-gain opportunities','PV MAE and angle-difference errors','F300/FUDI of selected angles','Within-day switch counts'],
        'guard_interpretation':'An empirically selected margin, not a guaranteed confidence bound or engineering switching cost',
        'decision_after_run':'Review whether positive gain is consistent across rooms; no automatic deployment, further experiment or TEST unsealing.',
        'parallel_nonexperimental_checks':['Request/inspect existing angle-definition diagrams or geometry export code when available','Confirm daylight priority, minimum acceptable spatial fractions and meaning of balanced weights before a combined-control follow-up'],
        'literature_support':[{'title':'Smart Predict, then Optimize','url':'https://arxiv.org/abs/1710.08005','use':'Supports distinguishing prediction loss and induced decision loss; the proposed simple delta/threshold pilot is our hypothesis, not a reproduced SPO implementation'},
                              {'title':'scikit-learn nested CV','url':'https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html','use':'Selection inside inner folds; assessment in outer folds'}]}
    proposal=ROOT/'configs/proposals';proposal.mkdir(exist_ok=True);dump(proposal/'decision_v4_draft.json',draft)
    dump(OUT/'decision_v4_draft.json',draft)
    interpretation=[
      'Data: fixed 0 is a strong baseline in the supplied simulation labels; numerical consistency alone cannot establish whether geometry angle definitions and simulation exports are physically correct.',
      'Objective: F300 and FUDI overlap and often move together in these rooms; the exploratory average D and training-quantile PV normalization do not encode confirmed user preferences. This does not mean daylight is always the dominant term.',
      'Model: net decision harm arises when losses from wrong departures exceed gains from useful departures; absolute regression MAE and decision ranking can disagree.',
      'Evaluation: compare room-level net gains and report tie/opportunity base rates. Near-optimal rates are easier when fixed 0 is usually strong. Constrained_PV saved primary regret omits the secondary PV objective.',
      'Causality: no numerical percentage attribution to data vs objective vs model is identified by this descriptive diagnosis; no alternate weights or threshold experiments have been run.'
    ]
    d={'stats':stats,'profiles':rec(profiles),'headroom':rec(headroom),'switch':rec(switch),'switch_rooms':rec(rooms),'terms':rec(terms),
       'constraints':rec(constraints),'data_profiles':rec(data_profiles),'rooms':audit['rooms'],'methods':METHODS,'interpretation':interpretation,'draft':draft,'audit':audit}
    dump(OUT/'report_data.json',d);dump(OUT/'findings.json',{'stats':stats,'interpretation':interpretation,'next_round_status':draft['status']})
    switch.to_csv(OUT/'switch_pooled_and_macro_summary.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(14,5));
    for axis,col,title,unit,mul in [(ax[0],'PV_retained_hours_kWh','PV generation','Mean room retained-hours PV (kWh)',1),(ax[1],'F300_mean','F300 spatial fraction','Mean spatial fraction (%)',100),(ax[2],'FUDI_mean','FUDI spatial fraction','Mean spatial fraction (%)',100)]:
        axis.bar(profiles.angle.astype(str),profiles[col]*mul,color='#446d85');axis.set_title(title,loc='left');axis.set_xlabel('PVSD angle (degrees)');axis.set_ylabel(unit)
    fig.suptitle('The simulated data already favor the fixed 0-degree baseline',x=.02,ha='left',fontsize=14)
    fig.text(.02,.01,'Source: frozen v3 TRAIN labels; seven rooms weighted equally; 4380 retained hourly records per angle and room. Not an all-8760-hour energy sum.',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(OUT/'figures/angle_profiles.png',dpi=160);plt.close(fig)
    pv_s=switch[switch.policy.eq('PV_priority')].set_index('method').loc[METHODS]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    y=np.arange(5);axes[0].barh(y,-pv_s.harm_sum,color='#b18b79',label='Loss from harmful switches');axes[0].barh(y,pv_s.benefit_sum,color='#446d85',label='Gain from beneficial switches');axes[0].set_yticks(y,METHODS);axes[0].invert_yaxis();axes[0].axvline(0,color='#65717a',linewidth=.7);axes[0].legend(loc='lower right',fontsize=8);axes[0].set_title('PV switching: gross gains and losses',loc='left');axes[0].set_xlabel('Mean room energy relative to fixed 0 (kWh)');axes[0].set_ylabel('Saved OOF model strategy')
    xgb=rooms[(rooms.policy=='PV_priority')&(rooms.method=='XGBoost')]
    axes[1].barh(xgb.room,xgb.net_gain_over_zero_sum,color='#446d85');axes[1].axvline(0,color='#65717a',linewidth=.7);axes[1].invert_yaxis();axes[1].set_title('XGBoost: room-level PV net gain',loc='left');axes[1].set_xlabel('Retained-hours energy gain over fixed 0 (kWh)');axes[1].set_ylabel('Held-out room')
    fig.suptitle('Models can gain in some rooms and lose overall',x=.02,ha='left',fontsize=14);fig.text(.02,.01,'Source: frozen v3 TRAIN decisions; existing PV-priority rule only. Negative net gain means fixed 0 generated more over the retained 4380 hours.',fontsize=8);fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(OUT/'figures/PV_switch_losses.png',dpi=160);plt.close(fig)
    payload=json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    canvas='''import {H1,H2,Text,Stack,Row,Table,Select,useState,useHostTheme} from "cursor/canvas";
const d=__DATA__;const f=(x:number|null|undefined,n=3)=>x==null?"不适用":x.toFixed(n);const pc=(x:number)=>f(x*100,2)+"%";
export default function AngleDiagnosis(){const theme=useHostTheme();const[policy,setPolicy]=useState("PV_priority");const[room,setRoom]=useState("BM608");
return <Stack gap={18} style={{padding:24,maxWidth:1250,color:theme.text.primary}}>
<H1>固定0°为何优于模型推荐</H1><Text weight="semibold">数据中的0°本来就强，现有目标放大了这一优势；模型在少数偏离0°的机会里又出现净损失。三个因素共同作用，不能只归因于算法。</Text>
<Row gap={30} wrap>{[["本次新增拟合","0"],["新模型预测","0"],["TEST读取 / 预测 / 评分","0 / 0 / 0"],["下一轮状态","草案，尚未批准"]].map(([k,v])=><div key={k}><Text size="small" tone="secondary">{k}</Text><Text weight="semibold">{v}</Text></div>)}</Row>
<H2>数据本身：固定0°是强基线</H2><Table headers={["角度","每房间保留时段PV kWh","平均F300 %","平均FUDI %"]} rows={d.profiles.map(r=>[r.angle+"°",f(r.PV_retained_hours_kWh,2),f(r.F300_mean*100,2),f(r.FUDI_mean*100,2)])}/>
<Text>在{pc(d.stats.joint_F300_FUDI_PV_weak_dominance_fraction)}的房间—小时里，0°的F300、FUDI和PV都不低于其他角度。平衡目标下0°唯一最优的比例为{pc(d.stats.balanced_zero_unique_best_fraction)}，全部四角度分数相同只有{pc(d.stats.balanced_all_four_equal_fraction)}；主要优势不能用并列处理解释。</Text>
<H2>目标函数：现有场景不代表最终偏好</H2><Text>D=(F300+FUDI)/2，两项空间比例有重叠；它们并非独立的采光和眩光目标。0.5D+0.5PV/s里的s是训练房间正PV的95分位数，各折约0.117至0.152 kWh。等权不等于工程价值相等。真实入射、眩光、冷负荷和控制成本没有在当前目标中得到验证。</Text>
<Text>约束场景有{pc(d.stats.constraint_true_no_feasible_fraction)}的保留小时四角度都不能同时满足F300≥50%、FUDI≥50%。此前主损失只算采光缺口，PV另列；不能将“4种场景主损失较小”表述为固定0°在完整分层目标上全面胜出。</Text>
<H2>模型：偏离0°时是否真的有收益</H2><Select value={policy} onChange={setPolicy} options={["PV_priority","Daylight_priority","Balanced"].map(p=>({value:p,label:p}))}/>
<Table headers={["方法","偏离0°比例","偏离后有害比例（汇总）","基线损失","模型损失","模型减基线损失"]} rows={d.switch.filter(r=>r.policy===policy).map(r=>[r.method,pc(r.switch_fraction),pc(r.pooled_harmful_fraction_among_switches??0),f(r.fixed_regret_mean,6),f(r.model_regret_mean,6),f(r.model_minus_fixed_regret,6)])}/>
<Text size="small" tone="secondary">损失单位：PV为逐时kWh，另两项为目标分数；不同目标不可混加。偏离后有害比例按全部房间的切换小时汇总。差值为正表示固定0°的损失更低。</Text>
<Row gap={12}><Text>逐房间检查</Text><Select value={room} onChange={setRoom} options={d.rooms.map(r=>({value:r,label:r}))}/></Row>
<Table headers={["方法","切换收益总和","切换损失总和","相对0°净收益"]} rows={d.switch_rooms.filter(r=>r.room===room&&r.policy===policy).map(r=>[r.method,f(r.benefit_sum),f(r.harm_sum),f(r.net_gain_over_zero_sum)])}/>
<Text>纯发电目标下，模拟Oracle相对固定0°每房间平均最多多{f(d.stats.PV_oracle_mean_gain_over_zero_kWh,2)} kWh，按总能量汇总约{pc(d.stats.PV_oracle_aggregate_gain_fraction)}。这只是四角度、保留时段的事后上限。平均MAE较小未必能选准少数有利的替代角度。</Text>
<H2>下一轮草案：尚未执行</H2><Text>建议先做纯PV小范围试验：沿用XGBoost参数，比较现有绝对PV预测与ΔPV=PV(候选角度)−PV(0°)预测，并增加仅在预测增益超过内层选择门槛时切换的规则。阈值候选为0、0.001、0.002、0.005 kWh及始终0°；预估49次拟合。外层仍留一房间，内层按房间分组选择门槛，三个TEST继续封存。</Text>
<Text>主要评价每房间净发电增益、最差房间损失、真正有收益的切换比例和相对Oracle损失。采光指标作为副结果列出。最终综合目标仍需确认使用需求、角度定义及采光约束，不自动降低阈值或改变权重来制造动态收益。</Text>
<Text size="small" tone="secondary">来源：Project_2026/reports/decision_diagnosis_v3；仅已有7个TRAIN房间的冻结预测及原决策规则。本次没有新策略回测或替代权重分析。数值重现与收益分解恒等式各检查105组，所有输入哈希未变。描述性分析不构成因果归因百分比或最终独立测试。</Text>
</Stack>}
'''.replace('__DATA__',payload)
    (OUT/'fixed-zero-diagnosis.canvas.tsx').write_text(canvas,encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定0°角度决策诊断</title><style>body{font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1240px;margin:28px auto;padding:0 24px 48px;color:#273944;background:#fafbfc}h1{font-size:26px}h2{font-size:20px;margin-top:30px}table{width:100%;border-collapse:collapse;background:white;font-size:13px}td,th{padding:10px;text-align:right;white-space:nowrap;border-bottom:1px solid #d9e1e5}th{background:#edf2f4}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.lead{font-size:18px}.note{border-left:3px solid #52748a;background:#edf3f6;padding:8px 18px}small,.muted,footer{color:#60727e;font-size:13px}select{font:inherit;padding:6px 12px;margin:0 12px 12px 0;background:white;border:1px solid #9cabb6}img{display:block;max-width:100%;margin:18px 0}a{color:#2b607c}.links{display:flex;gap:20px;flex-wrap:wrap}footer{border-top:1px solid #d9e1e5;margin-top:28px;padding-top:14px}</style>
<h1>固定0°为什么比模型推荐更好</h1><p class="lead">模拟数据中的0°本来就强，目标场景与此一致；模型在少数偏离0°的机会里又出现净损失。三者共同作用，不能只归因于某一种算法。</p><p class="note"><strong>本次只分析：</strong>新拟合0、新模型预测0、TEST读取/预测/评分0。本报告使用冻结的7个TRAIN房间预测；没有修改目标、阈值或权重，没有执行下一轮试验。</p>
<h2>1. 数据本身已经偏向0°</h2><div class="scroll"><table><thead><tr><th>角度</th><th>每房间保留时段PV kWh</th><th>平均F300 %</th><th>平均FUDI %</th></tr></thead><tbody id="profiles"></tbody></table></div><p id="dominance"></p><img src="figures/angle_profiles.png" alt="四个角度的真实模拟发电和采光空间比例">
<p>这是现有标签的结构，不是数据出错的证明。0°的几何含义、PV板法线、面积/容量、遮挡几何和时间对齐仍缺少来源确认。当前不能把0°直接解释成全开、水平或无遮挡，也不能据此断言模拟错误。</p>
<h2>2. 目标函数与可行性限制</h2><p>采光分数为 <code>D=(F300+FUDI)/2</code>。300–3000 lux的空间同时被两项计入；100–300 lux仅进入FUDI，大于3000 lux仍进入F300。因此这个分数并不是独立的采光、眩光和热舒适综合评价。两项比例不足以唯一重建过亮空间比例或眩光。</p><p>平衡分数为 <code>0.5D+0.5PV/s</code>，s是六个训练房间正PV的95分位数，各折约0.117–0.152 kWh；0.5:0.5不自动代表工程价值相等。下面是候选角度相对0°的两项分数变化，按房间与保留小时平均。不同角度的主导项并不相同，不能简单判断为采光权重总是过高。</p><div class="scroll"><table><thead><tr><th>候选角度</th><th>平均采光项变化</th><th>平均PV项变化</th><th>PV增加被采光减少抵消的小时比例</th></tr></thead><tbody id="terms"></tbody></table></div><p id="constraints"></p>
<p class="note"><strong>评价口径修正：</strong>Constrained_PV的保存主损失只统计采光缺口，PV是另列的次级结果。之前“四种场景固定0°更好”的说法应限定为相应保存指标，不能解释为固定0°在完整的“先满足采光，再最大化PV”目标上全面更好。后续需将可行性与同等可行条件下的PV分别列为主评价。</p>
<h2>3. 模型的损失来自哪些切换</h2><label for="policy">目标</label> <select id="policy"><option>PV_priority</option><option>Daylight_priority</option><option>Balanced</option></select><div class="scroll"><table><thead><tr><th>方法</th><th>偏离0°比例</th><th>偏离后有害比例</th><th>0°平均损失</th><th>模型平均损失</th><th>模型减基线损失</th></tr></thead><tbody id="switch"></tbody></table></div><p class="muted">损失单位：PV为逐时kWh，另两项为目标分数。偏离后有害比例以全部房间的偏离小时汇总；它不等于逐房间条件比例的平均。正差值说明固定0°更好。</p><p><code>模型损失 − 固定0°损失 = 有害切换损失 − 有益切换收益</code>。这个恒等式已对105个房间—目标—模型组合核对。</p><label for="room">留出房间</label> <select id="room"></select><div class="scroll"><table><thead><tr><th>方法</th><th>切换收益总和</th><th>切换损失总和</th><th>相对0°净收益</th></tr></thead><tbody id="room-detail"></tbody></table></div><p class="muted">PV目标的总和单位为kWh；另外两项目标为分数×小时，不是发电量。</p><img src="figures/PV_switch_losses.png" alt="有益切换与有害切换的能量，以及XGBoost各房间的净收益"><p id="PVfinding"></p>
<h2>4. 能继续改善多少</h2><div class="scroll"><table><thead><tr><th>目标</th><th>0°唯一最优</th><th>0°并列最优</th><th>四角度全相同</th><th>0°近优比例</th><th>超出既定容限的改善机会</th></tr></thead><tbody id="headroom"></tbody></table></div><p id="potential"></p><p class="muted">“近优容限”沿用v3预设值：PV 0.0001 kWh，采光和平衡0.005分数。本次未搜索新容限。机会指模拟Oracle相对0°的收益大于上述容限，不能解释为已确认的工程收益门槛。</p>
<h2>5. 建议下一轮：先做纯PV的小范围检验</h2><p><strong>状态：待确认，尚未批准、尚未执行。</strong>纯PV有明确的kWh评价，也保留一定动态收益空间，适合先检验模型能否识别正确切换时刻。最终综合控制仍需确认采光偏好与约束。</p><ol><li>沿用现有XGBoost参数，对照绝对PV预测与增量目标 <code>ΔPV=PV(候选角度)−PV(0°)</code>。只拟合30°/60°/90°，0°增量固定为0；推断时不输入真实PV(0°)。</li><li>比较直接选择与带门槛选择：预测增益超过门槛才切换，否则维持0°。门槛候选为0、0.001、0.002、0.005 kWh及始终维持0°；它们是经验增益门槛，不是置信区间。</li><li>外层仍7折留一房间；门槛只在六个训练房间内的3折分组OOF选择。不能直接拼用当前7个房间的外层OOF来给每个外层折调门槛，因为其他折模型可能见过该外层房间。</li><li>主指标改为逐房间净发电增益、最差房间损失、收益方向正确的切换比例和Oracle损失；MAE作为辅助。预估新增49次拟合，保留冻结绝对预测参考。</li><li>三个TEST房间保持封存，不追加模拟。另核对已有角度定义和几何导出信息，再确认采光目标；太阳入射特征继续等待必要来源信息。</li></ol>
<p>本方案是根据当前诊断提出的假设，没有保证Δ模型或增益门槛一定更好。若固定0°仍胜出，应保留这个结果。预测损失与决策损失的区分可参见 <a href="https://arxiv.org/abs/1710.08005">Smart Predict, then Optimize</a>；这里提议的是简单增量/门槛试验，并未实现或验证论文的SPO方法。内外层职责参照 <a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html">scikit-learn嵌套验证说明</a>。</p>
<div class="links"><a href="analysis.json">分析审计</a><a href="headroom.csv">逐房间机会</a><a href="switch_decomposition.csv">切换收益分解</a><a href="pairwise_errors.csv">角度差异预测误差</a><a href="constraint_diagnosis.csv">可行性诊断</a><a href="decision_v4_draft.json">待确认方案JSON</a></div><footer>来源：experiment_v3冻结TRAIN预测，全部7房间、四角度和模拟年度Hour=7..18；没有读取原始含TEST的CSV。本次输入哈希未变，105条已有主指标重现、105组切换分解恒等式通过。所有结论为事后描述性诊断，不报告显著性，不将逐时记录当作独立房间样本，也不进行数据/目标/模型的因果贡献百分比分配。</footer>
<script>const d=__DATA__,$=id=>document.getElementById(id),f=(v,n=3)=>v==null?'不适用':Number(v).toFixed(n),pc=v=>f(v*100,2)+'%',rows=a=>a.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('');$('profiles').innerHTML=rows(d.profiles.map(r=>[r.angle+'°',f(r.PV_retained_hours_kWh,2),f(r.F300_mean*100,2),f(r.FUDI_mean*100,2)]));$('dominance').textContent='在'+pc(d.stats.joint_F300_FUDI_PV_weak_dominance_fraction)+'的房间—小时里，0°的F300、FUDI和PV都不低于其他三个角度。F300单项0°最优或并列最优比例为'+pc(d.stats.F300_zero_best_fraction)+'，FUDI为'+pc(d.stats.FUDI_zero_best_fraction)+'。';$('terms').innerHTML=rows(d.terms.map(r=>[r.angle+'°',f(r.daylight_delta_term_mean,5),f(r.PV_delta_term_mean,5),pc(r.PV_gain_canceled_by_daylight_fraction)]));$('constraints').textContent='约束场景下，'+pc(d.stats.constraint_true_no_feasible_fraction)+'的保留小时没有任何可行角度。把所有房间的可行小时合并计算，固定0°覆盖其中'+pc(d.stats.zero_covers_available_pooled_fraction)+'；该条件比例不等于逐房间条件比例的平均。';$('headroom').innerHTML=rows(d.headroom.map(r=>[r.policy,pc(r.zero_unique_best_fraction),pc(r.zero_tied_best_fraction),pc(r.all_four_equal_fraction),pc(r.zero_near_optimal_fraction),pc(r.material_opportunity_fraction)]));$('potential').textContent='纯PV的四角度Oracle最多平均每房间多'+f(d.stats.PV_oracle_mean_gain_over_zero_kWh,2)+' kWh，相对于各房间总能量合并的固定0°基线约'+pc(d.stats.PV_oracle_aggregate_gain_fraction)+'。平衡目标超出既定容限的改善机会只占'+pc(d.stats.balanced_material_opportunity_fraction)+'的房间—小时；其Oracle增益约'+pc(d.stats.balanced_headroom_share_BM223_LM608)+'集中于BM223和LM608。';const xp=d.switch.find(r=>r.method==='XGBoost'&&r.policy==='PV_priority');$('PVfinding').textContent='XGBoost纯PV策略平均每房间获得'+f(xp.benefit_sum,2)+' kWh有益切换收益，同时损失'+f(xp.harm_sum,2)+' kWh，净收益'+f(xp.net_gain_over_zero_sum,2)+' kWh。它比固定0°略差，不能表述为所有房间都差。原HistGB的绝对PV MAE更低，但平均净收益约−7.61 kWh，说明MAE与角度决策质量可以不一致。';d.rooms.forEach(r=>{const o=document.createElement('option');o.value=r;o.textContent=r;$('room').appendChild(o)});$('room').value='BM608';function render(){const p=$('policy').value,r=$('room').value;$('switch').innerHTML=rows(d.switch.filter(x=>x.policy===p).map(x=>[x.method,pc(x.switch_fraction),pc(x.pooled_harmful_fraction_among_switches),f(x.fixed_regret_mean,6),f(x.model_regret_mean,6),f(x.model_minus_fixed_regret,6)]));$('room-detail').innerHTML=rows(d.switch_rooms.filter(x=>x.room===r&&x.policy===p).map(x=>[x.method,f(x.benefit_sum),f(x.harm_sum),f(x.net_gain_over_zero_sum)]))}['room','policy'].forEach(id=>$(id).addEventListener('change',render));render();</script></html>'''.replace('__DATA__',payload)
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2));print('Diagnosis views and unapproved proposal saved; no training or policy experiment executed.')

if __name__=='__main__':main()
