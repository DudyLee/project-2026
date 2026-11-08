"""Build scientific figures and offline interactive views from verified v3 outputs."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/experiment_v3'
METHODS=['Frozen_HistGB','HistGB','RandomForest','XGBoost','Nested_select']
TARGETS=['E_mean_lux','E_P10_lux','F_300_fraction','F_UDI_fraction','PV_AC_kWh']
LABELS={'E_mean_lux':'Mean illuminance','E_P10_lux':'P10 illuminance','F_300_fraction':'F300 spatial fraction','F_UDI_fraction':'FUDI spatial fraction','PV_AC_kWh':'PV AC energy'}
UNITS={t:'percentage points' if t.startswith('F_') else 'kWh / hourly record' if t=='PV_AC_kWh' else 'lux' for t in TARGETS}
FACTOR={t:100 if t.startswith('F_') else 1 for t in TARGETS}

def records(frame):return frame.astype(object).where(pd.notna(frame),None).to_dict('records')
def save_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def figures(summary,folds,decisions):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    fig,axes=plt.subplots(2,3,figsize=(15,8));palette=['#9aabb6','#708998','#476e84','#2b526a','#17384e']
    for ax,t in zip(axes.flat,TARGETS):
        vals=[summary.loc[(summary.method==m)&(summary.target==t),'mae_mean'].item()*FACTOR[t] for m in METHODS]
        ax.barh(np.arange(5),vals,color=palette);ax.set_yticks(np.arange(5),METHODS);ax.invert_yaxis();ax.set_title(LABELS[t],loc='left');ax.set_xlabel('Mean room MAE ('+UNITS[t]+')');ax.set_ylabel('Model / selection procedure')
        for i,v in enumerate(vals):ax.text(v,i,f' {v:.4g}',va='center',fontsize=9)
        ax.set_xlim(0,max(vals)*1.22)
    axes.flat[-1].axis('off');axes.flat[-1].text(0,0.8,'Outer: 7 leave-one-room-out folds\nInner: 3 folds grouped by room\n13 raw inputs; 15 encoded columns\nSame target handling across families\n3 TEST rooms remain sealed',va='top',linespacing=1.8)
    fig.suptitle('V3 development comparison: original-unit prediction errors',x=0.02,ha='left',fontsize=15)
    fig.text(0.02,0.015,'Source: experiment_v3 saved OOF predictions. Simulated year, Hour 7..18, four angles. Rooms weighted equally; no final TEST.',fontsize=9)
    fig.tight_layout(rect=[0,0.05,1,0.95]);fig.savefig(OUT/'figures/regression_comparison.png');plt.close(fig)
    rooms=sorted(folds.room.unique());fig,axes=plt.subplots(1,2,figsize=(14,6.5))
    for ax,t in zip(axes,['E_mean_lux','E_P10_lux']):
        values=np.array([[folds.loc[(folds.method==m)&(folds.target==t)&(folds.room==r),'mae'].item() for m in METHODS] for r in rooms])
        ratios=values/values[:,0,None];im=ax.imshow(ratios,cmap='Blues',vmin=0,vmax=max(2,float(np.max(ratios))))
        ax.set_xticks(np.arange(5),METHODS,rotation=25,ha='right');ax.set_yticks(np.arange(7),rooms)
        ax.set_xlabel('Model / selection procedure');ax.set_ylabel('Held-out room');ax.set_title(LABELS[t]+': annotated MAE (lux)',loc='left')
        for i in range(7):
            for j in range(5):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if ratios[i,j]>im.norm.vmax*.6 else '#162e3d')
        fig.colorbar(im,ax=ax,label='MAE / frozen HistGB room MAE (lower is better)',shrink=.72)
    fig.suptitle('Room-level tradeoffs: average improvement can hide failures',x=.02,ha='left',fontsize=15)
    fig.text(.02,.015,'Source: experiment_v3 fold_metrics.csv. Seven TRAIN rooms, all retained annual hours and four observed angles. Color is normalized; text is lux.',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(OUT/'figures/room_errors.png');plt.close(fig)
    strategies=['Train_selected_fixed']+METHODS
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    for ax,policy in zip(axes,['PV_priority','Daylight_priority','Balanced']):
        mult=1 if policy=='PV_priority' else 100
        values=[decisions.loc[(decisions.policy==policy)&(decisions.strategy==s),'policy_loss_mean'].item()*mult for s in strategies]
        ax.barh(np.arange(len(strategies)),values,color=['#a5b1b8']+palette);ax.set_yticks(np.arange(len(strategies)),strategies);ax.invert_yaxis();ax.set_title(policy,loc='left')
        ax.set_xlabel('Mean hourly regret (kWh)' if policy=='PV_priority' else 'Mean regret (score x 100)');ax.set_ylabel('Strategy');ax.set_xlim(0,max(max(values),1e-9)*1.25);ax.xaxis.set_major_locator(MaxNLocator(4))
        for i,v in enumerate(values):ax.text(v,i,f' {v:.3g}',va='center',fontsize=9)
    fig.suptitle('Angle decisions: loss relative to the four-angle simulated oracle',x=.02,ha='left',fontsize=15)
    fig.text(.02,.032,'Source: experiment_v3 decision_summary.csv. Seven rooms weighted equally, 4380 retained hourly decisions per room. Lower regret is better.',fontsize=9)
    fig.text(.02,.008,'Time-only Clock_rule is in the full table; its regret is 0.01376 kWh (PV), 0.08610 (daylight) and 0.08490 (balanced).',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(OUT/'figures/decision_regret.png');plt.close(fig)

def build_views(data):
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    canvas='''import {H1,H2,Text,Stack,Row,Table,Select,useState,useHostTheme} from "cursor/canvas";
const d=__DATA__;
const methods=["Frozen_HistGB","HistGB","RandomForest","XGBoost","Nested_select"];
const labels:Record<string,string>={E_mean_lux:"平均照度",E_P10_lux:"P10照度",F_300_fraction:"300 lux达标空间比例",F_UDI_fraction:"适宜照度空间比例",PV_AC_kWh:"PV发电量"};
const f=(x:number|null|undefined,n=3)=>x==null?"不适用":x.toFixed(n);
export default function NestedComparison(){
 const theme=useHostTheme();const [target,setTarget]=useState("E_P10_lux");const [room,setRoom]=useState("BM223");const [policy,setPolicy]=useState("Balanced");
 const factor=target.startsWith("F_")?100:1;const unit=target.startsWith("F_")?"百分点":target==="PV_AC_kWh"?"kWh / 小时记录":"lux";const digits=target==="PV_AC_kWh"?5:3;
 const summary=methods.map(m=>{const r=d.summary.find(x=>x.method===m&&x.target===target)!;return [m,f(r.mae_mean*factor,digits),f(r.rmse_mean*factor,digits),f(r.mae_max*factor,digits),f(r.r2_median),f(r.mae_change_vs_frozen_pct,1)+"%"];});
 const focus=methods.map(m=>{const r=d.folds.find(x=>x.method===m&&x.target===target&&x.room===room)!;return [m,f(r.mae*factor,digits),f(r.rmse*factor,digits),f(r.r2),f(r.bias*factor,digits)];});
 const decision=d.decision_summary.filter(x=>x.policy===policy).map(r=>[r.strategy,f(r.policy_loss_mean,5),f(r.near_optimal_fraction*100,1)+"%",f(r.constraint_satisfaction_fraction*100,1)+"%",f(r.pv_retained_hours_kWh,2),f(r.pv_loss_vs_energy_oracle_fraction*100,1)+"%"]);
 return <Stack gap={18} style={{padding:24,maxWidth:1320,color:theme.text.primary}}>
 <H1>第三轮：模型对照与四角度决策验证</H1><Text weight="semibold">{d.headline}</Text>
 <Row gap={28} wrap>{[["外层验证","7个房间"],["内层调参","3折按房间分组"],["实际拟合",String(d.run.fit_counts.inner+d.run.fit_counts.outer)],["TEST预测 / 评分","0 / 0"]].map(([k,v])=><div key={k}><Text tone="secondary" size="small">{k}</Text><Text weight="semibold">{v}</Text></div>)}</Row>
 <Text>保留冻结HistGB参考。三类模型各比较两组参数，输入和目标处理一致：均照度log1p、相对P10、三项直接回归。Nested_select在内层同时选择模型家族和参数，外层房间只评分。P10始终用预测均照度重建。</Text>
 <H2>性能预测：7个房间等权汇总</H2><Select value={target} onChange={setTarget} options={d.targets.map(t=>({value:t,label:labels[t]}))}/>
 <Text size="small" tone="secondary">MAE与RMSE单位：{unit}。变化率为相对冻结HistGB的MAE变化，负值表示降低。R²不是准确率百分比。</Text>
 <Table headers={["方法","平均MAE","平均RMSE","最差房间MAE","房间R²中位数","MAE变化"]} rows={summary}/>
 <Row gap={12}><Text>查看留出房间</Text><Select value={room} onChange={setRoom} options={d.rooms.map(r=>({value:r,label:r}))}/></Row>
 <Table headers={["方法","MAE","RMSE","R²","偏差"]} rows={focus}/>
 <H2>角度决策：模拟最优选择作为事后参考</H2><Select value={policy} onChange={setPolicy} options={d.policies.map(p=>({value:p,label:p}))}/>
 <Text>PV_priority损失单位为逐时kWh；Daylight_priority与Balanced为目标分数；Constrained_PV为采光约束缺口损失。不同策略的损失单位不同，不能合并。约束场景还必须同时查看采光满足率。评分直接使用F300、FUDI和PV；平均照度与P10用于预测评价，不直接进入本轮角度评分。</Text>
 <Table headers={["策略","平均逐时目标损失","近优比例","采光约束满足率","每房间保留时段PV kWh","相对纯发电最优损失"]} rows={decision}/>
 <H2>误差容限：探索性场景</H2><Table headers={["方法","绝对容限 "+unit,"容限内预测比例","最差房间比例"]} rows={d.tolerances.filter(r=>r.target===target).map(r=>[r.method,f(r.absolute_tolerance*factor),f(r.within_tolerance_fraction*100,1)+"%",f(r.worst_room_fraction*100,1)+"%"])}/>
 <Text>容限、两项采光比例均不低于50%的约束，以及平衡目标的等权重均为本轮预设探索场景，未被确认是工程验收标准。固定角度由训练房间选择，Clock_rule为预设时间表；Oracle使用模拟真实值，只是事后上限。</Text>
 <H2>本轮判断与结果边界</H2><Text>{d.conclusion}</Text><Text>这些7个开发房间已用于前几轮分析，嵌套调参不能消除既往方法设计的影响。未追加模拟、未使用三个TEST房间、未拟合最终全7房间部署模型。新太阳入射特征继续等待方向与时间定义。评价对象仍是CSV中的模拟值，而非建筑实测。</Text>
 <Text size="small" tone="secondary">来源：Project_2026/reports/experiment_v3；模拟年度Hour=7..18、四个固定角度。175条外层指标、210条内层候选分数、21个模型包及全部336组决策已复核。PV总量仅覆盖4380个保留小时。日内切换统计不含过夜或执行器损耗。</Text>
 </Stack>;
}
'''.replace('__DATA__',encoded)
    (OUT/'nested-model-decisions.canvas.tsx').write_text(canvas,encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PVSD 第三轮模型与角度决策实验</title>
<style>body{font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#24333d;background:#fafbfc;max-width:1280px;margin:28px auto;padding:0 24px 48px}h1{font-size:26px}h2{font-size:20px;margin-top:34px}.lead{font-size:18px;max-width:1050px}.stats{display:flex;gap:32px;flex-wrap:wrap;border-block:1px solid #d9e0e5;padding:16px 0}.stats small{display:block;color:#61717e}.stats strong{font-size:20px}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:9px 11px;text-align:right;border-bottom:1px solid #d9e0e5;white-space:nowrap}th{background:#edf1f4}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.note{border-left:3px solid #52748a;background:#edf3f6;padding:8px 18px}.muted,footer{color:#5e6f7c;font-size:13px}select{font:inherit;padding:6px 12px;margin:0 12px 12px 0;background:white;border:1px solid #a9b6bf}img{display:block;max-width:100%;margin:22px 0}a{color:#2b607c}.links{display:flex;gap:20px;flex-wrap:wrap}footer{margin-top:32px;padding-top:16px;border-top:1px solid #d9e0e5}code{font-size:13px}</style>
<h1>第三轮：模型对照与四角度决策验证</h1><p class="lead" id="headline"></p>
<div class="stats"><div><small>外层留一房间</small><strong>7折</strong></div><div><small>内层按房间分组</small><strong>3折</strong></div><div><small>实际模型拟合</small><strong id="fits"></strong></div><div><small>TEST预测 / 评分</small><strong>0 / 0</strong></div></div>
<h2>本轮方法</h2><p>输入均为13个原始特征、编码后15列；保留已有HistGB参考。HistGB、Random Forest和XGBoost各比较两组预设参数。每个外层折留下1个完整房间，其余6个房间内进行三折分组验证，每次4个拟合、2个验证。按原尺度房间平均MAE选择参数，外层房间只评分。<code>Nested_select</code>同时在内层选择模型家族和参数。</p>
<p>平均照度采用log1p；P10预测相对值 <code>r=P10/(E_mean+1 lux)</code>，再用预测均照度还原。每个P10候选的均照度和比值模型成对评价，其均照度组件可能与单独输出的最优均照度模型不同。F300、FUDI、PV直接回归。主评分保留物理截断，并保存各组件原始预测的越界比例。</p>
<h2>性能预测结果</h2><label for="target">目标指标</label> <select id="target"></select><span id="unit" class="muted"></span>
<div class="scroll"><table><thead><tr><th>方法</th><th>平均MAE</th><th>平均RMSE</th><th>最差房间MAE</th><th>R²中位数</th><th>MAE相对冻结参考变化</th></tr></thead><tbody id="summary"></tbody></table></div>
<p class="muted">先按每个房间计算指标，再对7个房间等权平均。负变化表示误差降低；R²不是准确率百分比。</p>
<img src="figures/regression_comparison.png" alt="五项指标的跨房间平均预测误差比较">
<h2>逐房间表现</h2><label for="room">留出房间</label> <select id="room"></select><div class="scroll"><table><thead><tr><th>方法</th><th>MAE</th><th>RMSE</th><th>R²</th><th>偏差</th></tr></thead><tbody id="focus"></tbody></table></div>
<img src="figures/room_errors.png" alt="平均照度与P10各房间MAE及相对参考的比值">
<h2>四角度决策验证</h2><p>每个房间、每个保留小时只在0°、30°、60°、90°中选择。模型策略只看预测性能；事后从CSV查回所选角度的真实模拟性能。固定角度由训练房间决定，时间规则预先固定。Oracle按真实模拟结果选择，是事后性能上限。本轮评分直接使用F300、FUDI和PV，平均照度与P10不直接进入评分；其预测误差改善不必然改善这些决策目标。</p>
<ul><li><strong>PV_priority：</strong>最大化PV。</li><li><strong>Daylight_priority：</strong>最大化 <code>D=(F300+FUDI)/2</code>。</li><li><strong>Balanced：</strong>最大化 <code>0.5D+0.5PV/s</code>；s为六个训练房间正PV的95分位数，所有策略共用，不从留出房间计算。</li><li><strong>Constrained_PV：</strong>优先在F300与FUDI均≥0.5的角度中最大化PV；无可行角度则最小化两项比例的总缺口，再最大化PV。</li></ul>
<label for="policy">决策场景</label> <select id="policy"></select><div class="scroll"><table><thead><tr><th>策略</th><th>平均逐时目标损失</th><th>近优比例</th><th>采光约束满足率</th><th>每房间保留时段PV kWh</th><th>相对纯发电最优损失</th></tr></thead><tbody id="decisions"></tbody></table></div>
<p class="muted">PV场景的目标损失单位是逐时kWh；采光和平衡场景为分数；约束场景为总采光缺口损失，不同场景不能合并。约束场景必须同时看可行性；不满足采光约束的高发电量不能视作有效改进。</p>
<img src="figures/decision_regret.png" alt="三种目标场景下推荐角度与模拟最优角度的目标损失">
<h2>按房间检查角度决策</h2><div class="scroll"><table><thead><tr><th>策略</th><th>平均逐时目标损失</th><th>近优比例</th><th>约束满足率</th><th>错失可行小时比例</th><th>日内切换次数</th></tr></thead><tbody id="decision-room"></tbody></table></div>
<h2>容限内预测比例</h2><p>以下为结果产生前固定的多档绝对误差容限，属于探索性场景，尚不是工程验收标准。比例容限的显示单位为百分点。</p><div class="scroll"><table><thead><tr><th>方法</th><th>绝对误差容限</th><th>容限内预测比例</th><th>最差房间比例</th></tr></thead><tbody id="tolerances"></tbody></table></div>
<h2>如何使用本轮结果</h2><div class="note"><p id="conclusion"></p><p>这些7个房间已经用于前几轮优化，嵌套验证仅隔离本轮参数和家族选择，不能抹去之前的方法设计历史。三个保留测试房间仍未预测、评分、调参或校准。当前结果属于开发证据，没有拟合最终全7房间部署模型。</p><p>坐标相对地理北的旋转、百叶角度定义、EPW身份与时间对齐仍未确认；本轮未增加相关物理角度特征，未追加模拟。误差比较对象为模拟CSV，不是实测。采光阈值、目标等权重和Clock_rule均为预设场景，不能直接解释为建筑规范或用户偏好。</p><p>PV总量仅求和现有4380个小时；两项照度比例的年度平均不是年度DA/UDI。日内切换统计不包含过夜、执行器损耗、控制延迟或连续动态模拟。</p></div>
<h2>文件与复现证据</h2><div class="links"><a href="summary_metrics.csv">预测汇总CSV</a><a href="fold_metrics.csv">逐房间指标</a><a href="inner_candidate_scores.csv">内层候选分数</a><a href="selections.csv">各折选择</a><a href="decision_summary.csv">决策汇总</a><a href="decision_by_room.csv">逐房间决策</a><a href="angle_usage_summary.csv">角度选择分布</a><a href="tolerance_summary.csv">容限统计</a><a href="verification.json">复核记录</a><a href="run_manifest.json">运行记录</a><a href="config_snapshot.json">固定配置</a></div>
<p class="muted">方法依据：<a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html">scikit-learn嵌套交叉验证说明</a>、<a href="https://xgboost.readthedocs.io/en/release_3.1.0/parameter.html">XGBoost官方参数文档</a>。算法参数、包版本和安装报告均存于项目内。</p>
<footer>来源：Project_2026/reports/experiment_v3；7个TRAIN房间、模拟年度全部Hour=7..18、四个角度。175条外层指标、210条内层候选分数、140条选择记录及21个模型包已独立核对；336组角度决策均按原始键重新查值，Oracle损失检查为零。原始数据与冻结预测文件哈希不变。</footer>
<script>const d=__DATA__;const methods=['Frozen_HistGB','HistGB','RandomForest','XGBoost','Nested_select'];const labels={E_mean_lux:'平均照度',E_P10_lux:'P10照度',F_300_fraction:'300 lux达标空间比例',F_UDI_fraction:'适宜照度空间比例',PV_AC_kWh:'PV发电量'};const fmt=(x,n=3)=>x==null?'不适用':Number(x).toFixed(n);const pct=x=>x==null?'不适用':fmt(x*100,1)+'%';const renderRows=rows=>rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');const $=id=>document.getElementById(id);function options(id,values,names){values.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=names?names[v]:v;$(id).appendChild(o)})}options('target',d.targets,labels);options('room',d.rooms);options('policy',d.policies);$('target').value='E_P10_lux';$('policy').value='Balanced';$('headline').textContent=d.headline;$('conclusion').textContent=d.conclusion;$('fits').textContent=d.run.fit_counts.inner+d.run.fit_counts.outer;
function render(){const t=$('target').value,room=$('room').value,policy=$('policy').value,k=t.startsWith('F_')?100:1,digits=t==='PV_AC_kWh'?5:3,unit=t.startsWith('F_')?'百分点':t==='PV_AC_kWh'?'kWh / 小时记录':'lux';$('unit').textContent='MAE、RMSE与容限单位：'+unit;
$('summary').innerHTML=renderRows(methods.map(m=>{const r=d.summary.find(x=>x.method===m&&x.target===t);return[m,fmt(r.mae_mean*k,digits),fmt(r.rmse_mean*k,digits),fmt(r.mae_max*k,digits),fmt(r.r2_median),fmt(r.mae_change_vs_frozen_pct,1)+'%']}));
$('focus').innerHTML=renderRows(methods.map(m=>{const r=d.folds.find(x=>x.method===m&&x.target===t&&x.room===room);return[m,fmt(r.mae*k,digits),fmt(r.rmse*k,digits),fmt(r.r2),fmt(r.bias*k,digits)]}));
$('decisions').innerHTML=renderRows(d.decision_summary.filter(r=>r.policy===policy).map(r=>[r.strategy,fmt(r.policy_loss_mean,5),pct(r.near_optimal_fraction),pct(r.constraint_satisfaction_fraction),fmt(r.pv_retained_hours_kWh,2),pct(r.pv_loss_vs_energy_oracle_fraction)]));
$('decision-room').innerHTML=renderRows(d.decision_rooms.filter(r=>r.policy===policy&&r.room===room).map(r=>[r.strategy,fmt(r.policy_loss_mean,5),pct(r.near_optimal_fraction),pct(r.constraint_satisfaction_fraction),pct(r.missed_feasible_hour_fraction),r.within_day_switch_count]));
$('tolerances').innerHTML=renderRows(d.tolerances.filter(r=>r.target===t).map(r=>[r.method,fmt(r.absolute_tolerance*k),pct(r.within_tolerance_fraction),pct(r.worst_room_fraction)]));}
['target','room','policy'].forEach(id=>$(id).addEventListener('change',render));render();</script></html>'''.replace('__DATA__',encoded)
    (OUT/'report.html').write_text(html,encoding='utf-8')

def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='passed'
    run=json.loads((OUT/'run_manifest.json').read_text());summary=pd.read_csv(OUT/'summary_metrics.csv');folds=pd.read_csv(OUT/'fold_metrics.csv');decisions=pd.read_csv(OUT/'decision_summary.csv')
    frozen=folds.loc[folds.method.eq('Frozen_HistGB'),['room','target','mae']].rename(columns={'mae':'frozen_mae'})
    paired=folds.merge(frozen,on=['room','target'],validate='many_to_one');paired['mae_change_vs_frozen']=paired.mae-paired.frozen_mae
    paired['mae_change_vs_frozen_pct']=100*paired.mae_change_vs_frozen/paired.frozen_mae
    paired.to_csv(OUT/'paired_room_comparison.csv',index=False)
    for i,r in summary.iterrows():
        ref=summary.loc[(summary.method=='Frozen_HistGB')&(summary.target==r.target),'mae_mean'].item();summary.loc[i,'mae_change_vs_frozen_pct']=100*(r.mae_mean-ref)/ref
        summary.loc[i,'rooms_improved_vs_frozen']=int(paired.loc[(paired.method==r.method)&(paired.target==r.target),'mae_change_vs_frozen'].lt(-1e-12).sum())
    summary.to_csv(OUT/'comparison_vs_frozen.csv',index=False)
    figures(summary,folds,decisions)
    nested=summary[summary.method.eq('Nested_select')];wins=int(nested.mae_change_vs_frozen_pct.lt(0).sum())
    headline='照度预测有改进，但自动选型仍不稳定；本轮角度推荐尚未优于训练选出的固定0°基线。'
    fixed=decisions.loc[decisions.strategy.eq('Train_selected_fixed')]
    model_decisions=decisions.loc[decisions.strategy.isin(METHODS)]
    fixed_wins=sum(fixed.loc[fixed.policy.eq(p),'policy_loss_mean'].item()<model_decisions.loc[model_decisions.policy.eq(p),'policy_loss_mean'].min() for p in run['config']['decision']['policies'])
    fixed_angles='、'.join(str(int(a))+'°' for a in pd.read_csv(OUT/'decision_by_room.csv').query("strategy == 'Train_selected_fixed'").selected_fixed_angle.unique())
    balance_near=fixed.loc[fixed.policy.eq('Balanced'),'near_optimal_fraction'].item()
    max_feasible=decisions.loc[(decisions.policy=='Constrained_PV')&(decisions.strategy=='Oracle'),'constraint_satisfaction_fraction'].item()
    oracle_zero=pd.read_csv(OUT/'angle_usage_summary.csv').query("policy == 'Balanced' and strategy == 'Oracle' and angle_deg == 0").selected_fraction.item()
    conclusion=f'更换算法有明确但不一致的收益：XGBoost平均照度MAE为82.43 lux，Random Forest的P10为9.75 lux；PV仍以冻结HistGB参考误差最低。内层自动选家族的方案只改善2/5项指标，BM522均照度选到RF后误差反而升至242.25 lux，说明少量房间下选择不稳定。角度决策中，各折训练选出的固定角度均为{fixed_angles}，在{fixed_wins}/4种目标的平均损失上低于所有模型策略。Balanced的模拟最优选择有{oracle_zero:.2%}的小时选0°（并列取较小角度），固定0°的近优比例为{balance_near:.2%}；不能把模型相似的高近优比例单独当作动态推荐的成功。预设采光约束下，即使四角度模拟Oracle也只有{max_feasible:.2%}的保留小时可行；这一低比例受到候选方案和阈值本身的限制。当前不全局替换模型，也不宣称动态推荐已优于固定角度。'
    focus=paired.loc[((paired.room.eq('BM223')|paired.room.eq('BM432'))&paired.target.eq('E_P10_lux'))|(paired.room.eq('BM522')&paired.target.eq('E_mean_lux'))]
    focus.to_csv(OUT/'focus_room_comparison.csv',index=False)
    data={'headline':headline,'conclusion':conclusion,'run':{'fit_counts':run['fit_counts'],'elapsed_seconds':run['elapsed_seconds'],'versions':run['versions']},'targets':TARGETS,'rooms':sorted(folds.room.unique().tolist()),'policies':run['config']['decision']['policies'],
          'summary':records(summary),'folds':records(folds),'decision_summary':records(decisions),'decision_rooms':records(pd.read_csv(OUT/'decision_by_room.csv')),'tolerances':records(pd.read_csv(OUT/'tolerance_summary.csv')),'verification':verification}
    save_json(OUT/'report_data.json',data);build_views(data)
    save_json(OUT/'candidate_decision.json',{'status':'development_comparison_complete_no_deployment_promotion','nested_selection_targets_with_lower_MAE_than_frozen':wins,'nested_target_results':records(nested),'TEST_used':False,'final_refit_performed':False,'interpretation':conclusion,
             'decision':'Keep frozen baselines and all v3 candidates; review prediction and decision outcomes together before approving another round. No automatic unsealing of TEST.',
             'material_passport':{'type':'code_experiment_result','source':'simulation CSV and saved out-of-fold model predictions','analysis_scope':'TRAIN-only nested grouped development comparison','verification':'metric and saved-model consistency verified; no complete independent retraining rerun','limitations':['Only seven previously examined development rooms','Single building and weather sequence','Unconfirmed weather and geometry provenance','Scenario decision thresholds are not engineering standards']}})
    print(headline);print(summary[['method','target','mae_mean','mae_change_vs_frozen_pct']].to_string(index=False));print('HTML, canvas and 3 scientific figures generated.')

if __name__=='__main__':main()
