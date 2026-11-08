"""Verified v4 result interpretation, scientific plots, HTML and canvas."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/experiment_v4'
METHODS=['Fixed_0','Frozen_HistGB','Absolute_XGBoost','Absolute_XGBoost_guarded','Delta_XGBoost','Delta_XGBoost_guarded','Oracle']
NAMES={'Fixed_0':'固定0°','Frozen_HistGB':'原HistGB','Absolute_XGBoost':'绝对PV XGBoost','Absolute_XGBoost_guarded':'绝对PV＋门槛','Delta_XGBoost':'增量PV XGBoost','Delta_XGBoost_guarded':'增量PV＋门槛','Oracle':'模拟四角度最优'}

def records(f):return f.astype(object).where(pd.notna(f),None).to_dict('records')
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def scientific_figures(summary,byroom):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    s=summary.set_index('method').loc[METHODS];values=s.mean_gain_kWh.to_numpy();colors=['#9daeb8' if abs(v)<1e-12 else '#446f87' if v>0 else '#a98371' for v in values]
    fig,ax=plt.subplots(figsize=(13,6));ax.barh(np.arange(7),values,color=colors);ax.set_yticks(np.arange(7),METHODS);ax.invert_yaxis();ax.axvline(0,color='#63717b',linewidth=.8)
    ax.set_xlabel('Mean room PV gain over fixed 0 degrees (kWh)');ax.set_ylabel('Strategy / retrospective reference');ax.set_xlim(min(values)*1.2,max(values)*1.28)
    for i,v in enumerate(values):ax.annotate(f'{v:+.3f}',(v,i),xytext=(5 if v>=0 else -5,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center')
    fig.suptitle('PV increment / threshold pilot: guarded gains are limited',x=.02,ha='left',fontsize=15)
    fig.text(.02,.01,'Source: verified experiment_v4 OOF decisions. Seven TRAIN rooms weighted equally; 4380 retained hours per room, four angles. Oracle uses simulation labels.',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(OUT/'figures/net_PV_gain.png',dpi=160);plt.close(fig)
    methods=['Absolute_XGBoost','Absolute_XGBoost_guarded','Delta_XGBoost','Delta_XGBoost_guarded'];rooms=sorted(byroom.room.unique())
    values=byroom.pivot(index='room',columns='method',values='net_gain_kWh').loc[rooms,methods].to_numpy()
    scale=max(abs(values.min()),abs(values.max()),1);fig,ax=plt.subplots(figsize=(11,7.5));im=ax.imshow(values,cmap='RdBu',norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale),aspect='auto')
    ax.set_xticks(np.arange(4),methods,rotation=20,ha='right');ax.set_yticks(np.arange(7),rooms);ax.set_xlabel('PV strategy');ax.set_ylabel('Held-out development room')
    for i in range(7):
        for j in range(4):ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',color='white' if abs(values[i,j])>.6*scale else '#18303e')
    fig.colorbar(im,ax=ax,label='Net retained-hours PV gain vs fixed 0 (kWh)',shrink=.8)
    fig.suptitle('Room-level gains: LM703 dominates unguarded delta losses',x=.02,ha='left',fontsize=14)
    fig.text(.02,.01,'Source: experiment_v4 decision_by_room.csv. All seven rooms retained; zero means equal to fixed 0, not a missing value.',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(OUT/'figures/room_net_gain.png',dpi=160);plt.close(fig)
    methods=['Absolute_XGBoost','Absolute_XGBoost_guarded','Delta_XGBoost','Delta_XGBoost_guarded'];ss=s.loc[methods]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for ax,column,title,unit in [(axes[0],'mean_gain_kWh','PV net gain','Mean room retained-hours gain (kWh)'),(axes[1],'mean_FUDI_change_pp','FUDI spatial-fraction change','Mean change vs fixed 0 (percentage points)')]:
        ax.barh(np.arange(4),ss[column],color='#446f87');ax.set_yticks(np.arange(4),methods);ax.invert_yaxis();ax.axvline(0,color='#63717b',linewidth=.8);ax.set_title(title,loc='left');ax.set_xlabel(unit);ax.set_ylabel('Strategy')
    fig.suptitle('PV-only optimization also changes daylight performance',x=.02,ha='left',fontsize=14)
    fig.text(.02,.01,'Source: verified experiment_v4 selected-angle simulation values. FUDI is an hourly spatial fraction averaged over retained hours, not annual DA/UDI.',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(OUT/'figures/PV_daylight_effect.png',dpi=160);plt.close(fig)

def views(data):
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    canvas='''import {H1,H2,Text,Stack,Row,Table,Select,useState,useHostTheme} from "cursor/canvas";
const d=__DATA__;const names:Record<string,string>=__NAMES__;const f=(v:number|null|undefined,n=3)=>v==null?"不适用":v.toFixed(n);
const pc=(v:number|null|undefined)=>v==null?"无切换":f(v*100,2)+"%";
export default function PVIncrementV4(){const theme=useHostTheme();const[method,setMethod]=useState("Absolute_XGBoost_guarded");const[room,setRoom]=useState("BM441");
return <Stack gap={18} style={{padding:24,maxWidth:1320,color:theme.text.primary}}>
<H1>第四轮：PV增量预测与切换门槛</H1><Text weight="semibold">{d.headline}</Text>
<Row gap={28} wrap>{[["新增模型拟合","49"],["内层门槛选择","14组"],["TEST预测 / 评分","0 / 0"],["验证范围","7个开发房间"]].map(([k,v])=><div key={k}><Text tone="secondary" size="small">{k}</Text><Text weight="semibold">{v}</Text></div>)}</Row>
<Text>每折冻结v3已在6个拟合房间内选出的XGBoost参数。内层3折按房间分组，以净发电增益选择0、1、2、5 Wh或始终0°；外层房间只评分。ΔPV模型拟合相对0°的有符号变化，0°为已知零锚点；真实PV(0°)未用于推断。</Text>
<H2>净发电收益：每房间4380个保留小时</H2>
<Table headers={["策略","平均净增益 kWh","中位数 kWh","最差房间 kWh","改善 / 持平 / 变差","总能量增益 %","FUDI变化 pp"]} rows={d.summary.map(r=>[names[r.method],f(r.mean_gain_kWh),f(r.median_gain_kWh),f(r.min_gain_kWh),r.rooms_improved+" / "+r.rooms_tied+" / "+r.rooms_worse,pc(r.aggregate_gain_fraction_vs_zero),f(r.mean_FUDI_change_pp)])}/>
<Text size="small" tone="secondary">增益相对固定0°。平均先逐房间计算再等权平均；总能量增益率由7个房间能量求和后计算，不等于房间相对增益率的平均。Oracle是事后上限。</Text>
<H2>逐房间结果</H2><Select value={method} onChange={setMethod} options={d.methods.map(m=>({value:m,label:names[m]}))}/>
<Table headers={["留出房间","选定门槛","净增益 kWh","有益切换收益 kWh","有害切换损失 kWh","切换小时数","FUDI变化 pp"]} rows={d.roomscores.filter(r=>r.method===method).map(r=>[r.room,r.threshold_label,f(r.net_gain_kWh),f(r.benefit_kWh),f(r.harm_kWh),r.switch_hours,f(r.FUDI_change_pp)])}/>
<H2>门槛怎样选择</H2><Select value={room} onChange={setRoom} options={d.rooms.map(r=>({value:r,label:r}))}/>
<Table headers={["预测方式","候选门槛","内层平均净增益 kWh","内层最差房间 kWh","是否选定"]} rows={d.inner.filter(r=>r.outer_held_room===room).map(r=>[r.variant,r.threshold_label,f(r.mean_room_gain_kWh),f(r.min_room_gain_kWh),d.selections.some(s=>s.outer_held_room===room&&s.variant===r.variant&&s.threshold_label===r.threshold_label)?"是":""] )}/>
<Text>增量模型7折都选始终0°，最终无切换、无收益。这是退回基线，不能视为学会了动态控制。绝对模型3折选5 Wh，4折选始终0°；外层只有BM441产生实际收益。</Text>
<H2>本轮结论</H2><Text>{d.conclusion}</Text>
<Text>BM441中绝对预测加门槛多发3.294 kWh，同时F300下降2.318个百分点、FUDI下降5.222个百分点。此轮只优化PV，尚不能作为综合采光与发电推荐。Δ模型直接选择虽有4/7房间增益，LM703损失75.137 kWh，整体仍为负。</Text>
<H2>预测误差与决策收益</H2><Table headers={["预测模型","评价量","房间平均MAE Wh","房间平均RMSE Wh"]} rows={d.forecasts.map(r=>[names[r.method],r.prediction_target,f(r.mae_mean*1000),f(r.rmse_mean*1000)])}/>
<Text>角度差异误差仅评价30°/60°/90°相对0°的变化。Δ模型没有独立绝对PV预测，所以不计算其绝对PV MAE。门槛改变决策，不改变基础预测误差。</Text>
<Text size="small" tone="secondary">来源：Project_2026/reports/experiment_v4，模拟年度Hour=7..18，四角度，7个TRAIN房间。49个新模型与训练标签哈希、420条门槛房间分数、14次门槛选择及214620条决策已复核。原始数据和冻结参考哈希未变。无最终全7房间部署拟合或最终TEST评分；7个房间已被多轮开发使用，当前为探索性证据。</Text>
</Stack>}
'''.replace('__DATA__',encoded).replace('__NAMES__',json.dumps(NAMES,ensure_ascii=False))
    (OUT/'PV-increment-threshold.canvas.tsx').write_text(canvas,encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第四轮PV增量与切换门槛</title><style>body{font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#273944;background:#fafbfc;max-width:1300px;margin:28px auto;padding:0 24px 48px}h1{font-size:26px}h2{font-size:20px;margin-top:32px}.lead{font-size:18px}.stats{display:flex;gap:32px;flex-wrap:wrap;padding:16px 0;border-block:1px solid #d9e0e5}.stats small{display:block;color:#61717d}.stats strong{font-size:20px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}td,th{padding:9px 11px;text-align:right;white-space:nowrap;border-bottom:1px solid #d9e0e5}th{background:#edf2f4}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.note{border-left:3px solid #52748a;background:#edf3f6;padding:8px 18px}.muted,footer{font-size:13px;color:#60727f}select{font:inherit;padding:6px 12px;margin:0 12px 12px 0;border:1px solid #9cabb6;background:white}img{display:block;max-width:100%;margin:20px 0}a{color:#2b607c}.links{display:flex;gap:18px;flex-wrap:wrap}footer{border-top:1px solid #d9e0e5;margin-top:30px;padding-top:14px}</style>
<h1>第四轮：PV增量预测与切换门槛</h1><p class="lead" id="headline"></p><div class="stats"><div><small>新增模型拟合</small><strong>49次</strong></div><div><small>外层 / 内层验证</small><strong>7折 / 3折</strong></div><div><small>内层门槛选择</small><strong>14组</strong></div><div><small>TEST预测 / 评分</small><strong>0 / 0</strong></div></div>
<h2>实际执行的方案</h2><p>保持13个原始输入、编码后15列，冻结v3各外层折在6个拟合房间内选出的XGBoost参数。内层每次4个房间拟合、2个房间验证，以净发电增益选择0、1、2、5 Wh或始终0°门槛；外层留下的房间只评分。新增21个内层绝对PV模型、21个内层ΔPV模型、7个外层ΔPV模型；外层绝对预测复用并核对v3模型。</p><p>训练目标 <code>ΔPV=PV(候选角度)−PV(0°)</code>只在同一拟合房间、同一小时配对，拟合30°/60°/90°，负增量保留。推断仅接收输入与角度，0°增量固定为0，真实PV(0°)没有作为输入。选择最大预测增量，仅当它严格超过选定门槛才切换。</p>
<h2>相对固定0°的净发电收益</h2><div class="scroll"><table><thead><tr><th>策略</th><th>平均净增益 kWh</th><th>中位数 kWh</th><th>最差房间 kWh</th><th>改善/持平/变差</th><th>总能量增益率</th><th>FUDI变化 pp</th></tr></thead><tbody id="summary"></tbody></table></div><p class="muted">每个房间求和4380个保留小时，再对7个房间等权平均。总能量增益率按七个房间的总发电量计算，不等于房间相对增益率的平均。正值表示多发电，Oracle为事后四角度上限。</p><img src="figures/net_PV_gain.png" alt="各策略相对固定0度的平均净发电收益">
<h2>逐房间结果</h2><label for="method">策略</label> <select id="method"></select><div class="scroll"><table><thead><tr><th>房间</th><th>门槛</th><th>净增益 kWh</th><th>有益切换收益 kWh</th><th>有害切换损失 kWh</th><th>偏离0°小时数</th><th>F300变化 pp</th><th>FUDI变化 pp</th></tr></thead><tbody id="roomscores"></tbody></table></div><img src="figures/room_net_gain.png" alt="各房间净发电收益及LM703的增量模型损失">
<h2>门槛选择证据</h2><label for="room">外层留出房间</label> <select id="room"></select><div class="scroll"><table><thead><tr><th>预测方式</th><th>候选门槛</th><th>内层平均净增益 kWh</th><th>内层最差房间 kWh</th><th>选定</th></tr></thead><tbody id="inner"></tbody></table></div><p>绝对模型在BM441、BM522、LM608这三折选5 Wh，其余4折选始终0°。外层BM522、LM608的预测没有触发该门槛，所以实际仍保持0°。Δ模型7折都选择始终0°，最终没有动态切换、没有净收益。</p><p class="muted">门槛选择只使用当前外层6个训练房间新生成的分组OOF。没有拼用跨全部7房间的既有外层OOF校准。参数选择和门槛选择重复使用该开发集合，内层不是独立校准样本；外层房间始终被排除。</p>
<h2>本轮结果如何解释</h2><div class="note"><p id="conclusion"></p><p>无门槛Δ模型在4/7房间有正收益，LM703的−75.137 kWh使总体平均变差；所有房间均保留，不能删除该房间后宣布有效。带门槛Δ模型与固定0°完全相同，这表示校准选择了退回基线。</p><p>绝对PV加门槛在BM441多发3.294 kWh，但F300平均下降2.318个百分点、FUDI下降5.222个百分点。本轮没有优化采光，也没有计入执行器成本和控制动态；不能将纯PV小收益当作综合控制成功。</p></div><img src="figures/PV_daylight_effect.png" alt="纯PV策略的发电收益和FUDI空间比例变化">
<h2>预测误差</h2><div class="scroll"><table><thead><tr><th>模型</th><th>评价量</th><th>房间平均MAE Wh</th><th>房间平均RMSE Wh</th><th>房间R²中位数</th></tr></thead><tbody id="forecast"></tbody></table></div><p>角度差异误差只评价30°/60°/90°相对0°的变化，避免零锚点稀释误差。Δ模型没有独立绝对PV预测，不能用真实PV(0°)还原预测后宣称绝对PV精度。门槛只改变决策，不改变基本预测误差。</p>
<h2>当前决定与边界</h2><p>保留“绝对PV＋门槛”作为开发候选，收益尚不足以支持普遍优势；暂不推进当前Δ模型。下一步优先核对LM703的角度响应、0°定义和采光/发电目标；任何进一步训练方案先与用户确认。三个TEST房间继续封存，没有新增模拟或最终全7房间部署拟合。</p><p class="muted">这些7个房间已经参与多轮开发，当前结果仍是探索性开发证据。天气身份与时刻对应、方向与几何定义仍有来源限制。所有PV总量只覆盖现有4380小时，F300/FUDI是空间比例的时间平均，日内切换不包含过夜、执行器损耗或连续控制动态。</p>
<div class="links"><a href="decision_summary.csv">决策汇总</a><a href="decision_by_room.csv">逐房间指标</a><a href="threshold_selections.json">所选门槛</a><a href="inner_threshold_summary.csv">内层候选分数</a><a href="forecast_summary.csv">预测误差</a><a href="delta_angle_profiles.csv">Δ响应概况</a><a href="verification.json">复核记录</a><a href="run_manifest.json">运行记录</a><a href="config_snapshot.json">执行配置</a></div><footer>来源：Project_2026/reports/experiment_v4；模拟年度Hour=7..18、四角度、7个TRAIN房间。49个新模型和训练标签哈希、420条门槛房间分数、70条候选汇总、14次选择及214620条决策已核对。原始文件及冻结参考哈希未变；TEST预测与评分为0。代码与环境仍存于Project_2026。</footer>
<script>const d=__DATA__,names=__NAMES__,$=id=>document.getElementById(id),f=(v,n=3)=>v==null?'不适用':Number(v).toFixed(n),pc=v=>v==null?'无切换':f(v*100,2)+'%',rows=a=>a.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');$('headline').textContent=d.headline;$('conclusion').textContent=d.conclusion;$('summary').innerHTML=rows(d.summary.map(r=>[names[r.method],f(r.mean_gain_kWh),f(r.median_gain_kWh),f(r.min_gain_kWh),r.rooms_improved+' / '+r.rooms_tied+' / '+r.rooms_worse,pc(r.aggregate_gain_fraction_vs_zero),f(r.mean_FUDI_change_pp)]));$('forecast').innerHTML=rows(d.forecasts.map(r=>[names[r.method],r.prediction_target,f(r.mae_mean*1000),f(r.rmse_mean*1000),f(r.r2_median)]));d.methods.forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=names[m];$('method').appendChild(o)});d.rooms.forEach(r=>{const o=document.createElement('option');o.value=r;o.textContent=r;$('room').appendChild(o)});$('method').value='Absolute_XGBoost_guarded';$('room').value='BM441';function render(){const m=$('method').value,r=$('room').value;$('roomscores').innerHTML=rows(d.roomscores.filter(x=>x.method===m).map(x=>[x.room,x.threshold_label,f(x.net_gain_kWh),f(x.benefit_kWh),f(x.harm_kWh),x.switch_hours,f(x.F300_change_pp),f(x.FUDI_change_pp)]));$('inner').innerHTML=rows(d.inner.filter(x=>x.outer_held_room===r).map(x=>[x.variant,x.threshold_label,f(x.mean_room_gain_kWh),f(x.min_room_gain_kWh),d.selections.some(s=>s.outer_held_room===r&&s.variant===x.variant&&s.threshold_label===x.threshold_label)?'是':'']))}['method','room'].forEach(id=>$(id).addEventListener('change',render));render();</script></html>'''.replace('__DATA__',encoded).replace('__NAMES__',json.dumps(NAMES,ensure_ascii=False))
    (OUT/'report.html').write_text(html,encoding='utf-8')

def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='passed'
    summary=pd.read_csv(OUT/'decision_summary.csv');byroom=pd.read_csv(OUT/'decision_by_room.csv');forecasts=pd.read_csv(OUT/'forecast_summary.csv')
    run=json.loads((OUT/'run_manifest.json').read_text());inner=pd.read_csv(OUT/'inner_threshold_summary.csv');selections=json.loads((OUT/'threshold_selections.json').read_text())
    p=pd.read_parquet(OUT/'predictions/outer_PV_candidates.parquet');profile=[]
    for (room,angle),g in p.groupby(['Room_ID','PVSD_Angle_deg'],sort=True):
        profile.append({'room':room,'angle':int(angle),'true_delta_mean_kWh':float(g.true_delta_PV.mean()),'pred_delta_mean_kWh':float(g.pred_delta_gain.mean()),
                        'true_gain_over_0_1Wh_fraction':float(g.true_delta_PV.gt(.0001).mean()),'pred_positive_gain_fraction':float(g.pred_delta_gain.gt(0).mean())})
    pd.DataFrame(profile).to_csv(OUT/'delta_angle_profiles.csv',index=False)
    headline='绝对PV加门槛获得小幅且集中的收益；增量模型尚未改善，带门槛后全部退回固定0°。'
    a=summary.set_index('method').loc['Absolute_XGBoost_guarded'];delta=summary.set_index('method').loc['Delta_XGBoost']
    conclusion=f'绝对PV＋门槛平均每房间净增益{a.mean_gain_kWh:.3f} kWh，总能量增益率{a.aggregate_gain_fraction_vs_zero:.3%}，中位数为0；全部收益来自BM441，其余6房间持平。Δ模型无门槛平均净增益{delta.mean_gain_kWh:.3f} kWh，最差LM703为{delta.min_gain_kWh:.3f} kWh；带门槛版本7折都选择保持0°。因此本轮支持用内层校准减少错误切换，但没有证明Δ目标本身改善泛化，也没有建立稳定的动态综合控制优势。'
    data={'headline':headline,'conclusion':conclusion,'methods':METHODS,'rooms':sorted(byroom.room.unique().tolist()),'summary':records(summary),'roomscores':records(byroom),'forecasts':records(forecasts),
          'inner':records(inner),'selections':selections,'delta_profiles':profile,'verification':verification,'run':{'elapsed_seconds':run['elapsed_seconds'],'fits':run['fits'],'versions':run['versions']}}
    dump(OUT/'report_data.json',data);scientific_figures(summary,byroom);views(data)
    dump(OUT/'candidate_decision.json',{'status':'development_only_no_deployment','absolute_plus_threshold':'Keep as a limited candidate; all positive gain from one of seven rooms, median gain zero, FUDI cost in BM441',
         'current_delta_model':'Do not promote this fitted formulation/parameter setting; unguarded large LM703 loss, guarded policy identical to fixed 0',
         'interpretation':conclusion,'TEST_used':False,'new_simulations':0,'final_seven_room_refit':False,'further_experiments':'Require user confirmation after reviewing this completed round',
         'scope':run['scope']})
    dump(OUT/'report_build_manifest.json',{'report_script_sha256':sha(Path(__file__)),'artifacts':{n:sha(OUT/n) for n in ['report.html','PV-increment-threshold.canvas.tsx','report_data.json','candidate_decision.json']},'canvas_native_compilation_performed':False})
    print(conclusion);print('V4 report, embedded-data canvas, three scientific figures and candidate status saved.')

if __name__=='__main__':main()
