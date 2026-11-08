"""Build a self-contained report, interactive canvas, and scientific figures for v7."""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_target_specific_v7 as V

T=V.T;ROOT=V.ROOT;OUT=ROOT/'reports/experiment_v7'
DISPLAY=[
 {'target':'E_mean_lux','label':'平均照度','english':'Mean illuminance','factor':1,'unit':'lux'},
 {'target':'E_P10_lux','label':'P10照度','english':'P10 illuminance','factor':1,'unit':'lux'},
 {'target':'F_300_fraction','label':'≥300 lux空间比例','english':'F300 spatial fraction','factor':100,'unit':'pp'},
 {'target':'F_UDI_fraction','label':'100–3000 lux空间比例','english':'FUDI spatial fraction','factor':100,'unit':'pp'},
 {'target':'PV_AC_kWh','label':'逐时PV发电量','english':'Hourly PV energy','factor':1000,'unit':'Wh'},
]


def records(frame):return json.loads(frame.to_json(orient='records',force_ascii=False,double_precision=15))


def main():
    cfg=T.read(V.CONFIG);run=T.read(OUT/'run_manifest.json');verify=T.read(OUT/'verification.json')
    assert run['status']=='completed' and verify['status']=='passed'
    s=pd.read_csv(OUT/'summary_metrics.csv');f=pd.read_csv(OUT/'fold_metrics.csv');g=pd.read_csv(OUT/'development_gate.csv')
    selections=pd.read_csv(OUT/'selections.csv');decision=T.read(OUT/'candidate_decision.json')
    scoreboard=[];facts=[]
    for meta in DISPLAY:
        target=meta['target'];z=s[s.target==target].set_index('method')
        a=z.loc['Retained_reference'];b=z.loc['Worst_guarded'];gate=g[(g.target==target)&(g.method=='Worst_guarded')&(g.reference=='Retained_reference')].iloc[0]
        passed=decision['targets'][target]['passes_both_references']
        row={'target':target,'reference_macro_MAE':a.mae_mean,'candidate_macro_MAE':b.mae_mean,
             'reference_worst_MAE':a.mae_max,'candidate_worst_MAE':b.mae_max,
             'reference_worst_room':a.worst_room,'candidate_worst_room':b.worst_room,
             'macro_MAE_change_pct':100*gate.macro_MAE_change_fraction,
             'worst_MAE_change_pct':-100*gate.worst_MAE_reduction_fraction,
             'rooms_improved':int(gate.rooms_improved),'rooms_tied':int(gate.rooms_tied),
             'rooms_nonworse':int(gate.rooms_nonworse),'passes_both_references':passed}
        scoreboard.append(row)
        factor=meta['factor'];unit='个百分点' if meta['unit']=='pp' else meta['unit']
        facts.append(f"{meta['label']}：平均MAE {a.mae_mean*factor:.3f} → {b.mae_mean*factor:.3f} {unit}；最差MAE {a.mae_max*factor:.3f} → {b.mae_max*factor:.3f} {unit}（{a.worst_room} → {b.worst_room}）；{int(gate.rooms_nonworse)}/7房间不恶化；{'通过' if passed else '未通过'}两套参考的筛选条件。")
    pd.DataFrame(scoreboard).to_csv(OUT/'scoreboard.csv',index=False)
    transfers=[]
    for target in V.TARGETS:
        for room in sorted(f.room.unique()):
            choices=selections[(selections.target==target)&(selections.held_room==room)].set_index('policy')
            errors=f[(f.target==target)&(f.room==room)].set_index('method')
            a=choices.loc['Target_baseline'];b=choices.loc['Worst_guarded']
            transfers.append({'target':target,'room':room,'baseline_candidate':a.candidate,'worst_candidate':b.candidate,
                'baseline_inner_worst_MAE':float(a.inner_worst_mae),'selected_inner_worst_MAE':float(b.inner_worst_mae),
                'inner_worst_MAE_change':float(b.inner_worst_mae-a.inner_worst_mae),
                'baseline_outer_MAE':float(errors.loc['Target_baseline','mae']),
                'selected_outer_MAE':float(errors.loc['Worst_guarded','mae']),
                'outer_MAE_change':float(errors.loc['Worst_guarded','mae']-errors.loc['Target_baseline','mae'])})
    transfer=pd.DataFrame(transfers);transfer.to_csv(OUT/'selection_transfer.csv',index=False)
    p10=transfer[(transfer.target=='E_P10_lux')&(transfer.room=='BM223')].iloc[0]
    mean=transfer[(transfer.target=='E_mean_lux')&(transfer.room=='BM522')].iloc[0]
    udi=transfer[(transfer.target=='F_UDI_fraction')&(transfer.room=='BM432')].iloc[0]
    diagnosis=[
        f"BM223的P10：内层最差房间MAE由{p10.baseline_inner_worst_MAE:.3f}降至{p10.selected_inner_worst_MAE:.3f} lux，选中{p10.worst_candidate}；但真正外层MAE从{p10.baseline_outer_MAE:.3f}升至{p10.selected_outer_MAE:.3f} lux。内层选择改善没有迁移到该留出房间。",
        f"BM522的平均照度：最差优先仍选择{mean.worst_candidate}，与逐目标原参考相同，外层MAE维持{mean.selected_outer_MAE:.3f} lux。本轮新增损失没有解决这个房间的问题。",
        f"BM432的FUDI：相对逐目标RF原参考，所选{udi.worst_candidate}使外层MAE从{udi.baseline_outer_MAE*100:.3f}升至{udi.selected_outer_MAE*100:.3f}个百分点。相对保留HistGB参考的总体平均改善，不能掩盖这个房间的恶化。",
        '这些结果支持“选择过程的跨房间迁移仍不稳定”的判断；不能据此证明全部原因都来自缺失几何，也不能证明所有其他模型方法都无效。当前13项输入与7房间覆盖的限制仍需单独核实。',
    ]
    passed=[m['label'] for m in DISPLAY if decision['targets'][m['target']]['passes_both_references']]
    headline=(f"最差误差优先方案有{len(passed)}/5项输出通过预设开发条件。" if passed else '最差误差优先方案没有输出同时通过全部开发条件。')
    protocol='固定13项输入、24个候选配置。外层7折留一房间；每折六个训练房间内部3折分组验证。Target_baseline沿用原候选；Mean_select按平均MAE选；Worst_guarded在平均MAE不比原候选高出5%的候选中，优先降低内层最大房间MAE。所有选项在外层评分前冻结。'
    gate_text='正式候选为Worst_guarded。相对Target_baseline和Retained_reference均须达到：最大房间MAE下降至少10%、平均MAE不增加、至少5/7房间不恶化。内层5%保护不能保证外层结果也满足条件；Mean_select是预先声明的对照，不在看过结果后改成主要候选。'
    limitations='七个房间已反复用于开发，本轮仍是开发证据。三个TEST房间未预测或评分。未新增几何、模拟或最终部署拟合。没有重新优化或评分角度决策，数值误差改善不直接等于推荐收益。当前EPW/时段/北向和网格来源限制仍存在。'
    data={'headline':headline,'facts':facts,'diagnosis':diagnosis,'selection_transfer':transfers,'protocol':protocol,'gate':gate_text,'limitations':limitations,
          'display':DISPLAY,'summary':records(s),'folds':records(f),'gates':records(g),'selections':records(selections),
          'scoreboard':scoreboard,'rooms':sorted(f.room.unique()),'methods':cfg['policies']+['Retained_reference'],
          'fit_counts':run['fit_counts'],'verification_refits':verify['bounded_reproducibility_estimator_refits']}
    T.dump(OUT/'report_data.json',data)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'savefig.facecolor':'white'})
    colors={'Retained_reference':'#8c989f','Mean_select':'#b77947','Worst_guarded':'#3b7896'}
    fig,axes=plt.subplots(1,2,figsize=(13,5.8));ys=np.arange(5)
    for ax,metric,title in zip(axes,['macro_MAE_change_fraction','worst_MAE_reduction_fraction'],['Mean room MAE change','Worst room MAE change']):
        for i,method in enumerate(['Mean_select','Worst_guarded']):
            rows=g[(g.method==method)&(g.reference=='Retained_reference')].set_index('target').loc[[m['target'] for m in DISPLAY]]
            values=rows[metric].to_numpy()*100*(-1 if metric=='worst_MAE_reduction_fraction' else 1)
            ax.barh(ys+(i-.5)*.3,values,height=.28,label=method,color=colors[method])
        ax.set_yticks(ys,[m['english'] for m in DISPLAY]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8)
        ax.set_xlabel('Change vs retained reference (%) — negative is better');ax.set_title(title);ax.grid(axis='x',alpha=.2)
    axes[1].legend(loc='best');fig.suptitle('V7 cross-room development: average error and worst-room error',fontsize=15)
    fig.text(.01,.015,'Source: v7 seven-room OOF, all four angles, Hour 7–18. Units normalized within target only; no TEST predictions.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(OUT/'figures/mean_and_worst_changes.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,12));rooms=data['rooms'];xp=np.arange(7)
    for ax,meta in zip(axes.flat,DISPLAY):
        for i,method in enumerate(['Retained_reference','Mean_select','Worst_guarded']):
            values=f[(f.method==method)&(f.target==meta['target'])].set_index('room').loc[rooms,'mae'].to_numpy()*meta['factor']
            ax.bar(xp+(i-1)*.24,values,width=.23,color=colors[method],label=method)
        ax.set_xticks(xp,rooms,rotation=35,ha='right');ax.set_title(meta['english']);ax.set_xlabel('Outer held-out room')
        ax.set_ylabel('MAE ('+meta['unit']+')'+(' — log scale' if meta['target']=='E_P10_lux' else ''))
        if meta['target']=='E_P10_lux':ax.set_yscale('log')
        ax.grid(axis='y',alpha=.2)
    axes.flat[-1].axis('off');handles,labels_=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels_,loc='upper center',ncol=3,bbox_to_anchor=(.5,.965))
    fig.suptitle('V7: every development room remains in the comparison',fontsize=16,y=.995)
    fig.text(.02,.015,'Source: v7 fold_metrics.csv. P10 uses a logarithmic y-axis; pp = percentage points, Wh = hourly energy. Lower is better.',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.94]);fig.savefig(OUT/'figures/room_errors.png',dpi=140);plt.close(fig)
    markdown=['# 第七轮：逐目标优化与最差房间误差','','## Material Passport','',
              '模拟代理模型开发结果；来源为TRAIN模拟标签和按房间留出预测。数值及模型重载核对通过，另完成7个回归器的有限重复拟合；不是全流程独立复跑。','',headline,'',protocol,'',gate_text,'']
    markdown += facts+['','## 选择为何未迁移','']+diagnosis+['',limitations,'','完整结果见report.html；数据见scoreboard.csv、fold_metrics.csv、development_gate.csv、selection_transfer.csv；配置见../../configs/experiment_v7.json。','']
    (OUT/'summary.md').write_text('\n\n'.join(markdown),encoding='utf-8')
    json_data=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    script='''const d=DATA;const fmt=(v,n=3)=>v==null?'—':Number(v).toFixed(n);
const table=(headers,rows)=>'<table><thead><tr>'+headers.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
const select=document.getElementById('target');select.innerHTML=d.display.map(x=>`<option value="${x.target}">${x.label}</option>`).join('');
function render(){const t=select.value,m=d.display.find(x=>x.target===t),k=m.factor;
document.getElementById('target-summary').innerHTML=table(['方法',`平均MAE (${m.unit})`,`最大MAE (${m.unit})`,'最差房间',`平均RMSE (${m.unit})`,'R²中位数','容限内比例'],d.summary.filter(r=>r.target===t).map(r=>[r.method,fmt(r.mae_mean*k),fmt(r.mae_max*k),r.worst_room,fmt(r.rmse_mean*k),fmt(r.r2_median),fmt(r.within_tolerance_mean*100,1)+'%']));
document.getElementById('room-table').innerHTML=table(['房间','保留参考MAE','逐目标原参考MAE','平均优先MAE','最差优先MAE','最差优先Bias'],d.rooms.map(room=>{const rs=d.folds.filter(r=>r.target===t&&r.room===room);const get=method=>rs.find(r=>r.method===method);return [room,...['Retained_reference','Target_baseline','Mean_select','Worst_guarded'].map(p=>fmt(get(p).mae*k)),fmt(get('Worst_guarded').bias*k)];}));
document.getElementById('selection-table').innerHTML=table(['外层房间','选择方式','所选模型','内层平均MAE','内层最差MAE'],d.selections.filter(r=>r.target===t).map(r=>[r.held_room,r.policy,r.candidate,fmt(r.inner_mean_mae*k),fmt(r.inner_worst_mae*k)]));
document.getElementById('unit-note').textContent=`以上误差单位：${m.unit}。容限仅为已有描述场景，不是工程验收标准。`;
}select.addEventListener('change',render);render();'''.replace('DATA',json_data,1)
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v7逐目标优化</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1220px;margin:auto;padding:30px;color:#24323a;background:#fafbfc;line-height:1.65}h1{font-size:29px}h2{font-size:22px;margin-top:35px}.lead{font-size:20px;font-weight:650;color:#245b75}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{text-align:left;padding:9px;border-bottom:1px solid #d9e0e3}th{background:#edf1f3}.scroll{overflow:auto}img{width:100%;height:auto;background:white}select{font:inherit;padding:7px;min-width:240px}small{color:#546671}a{color:#236483}.note{border-left:3px solid #98aab4;padding-left:16px}</style></head><body>
<h1>第七轮：逐目标优化与最差房间误差</h1><p class="lead">HEADLINE</p><p>2026-11-08 · 7个开发房间 · 13项输入 · 3个TEST房间封存</p>
<p>PROTOCOL</p><p class="note">GATE</p><h2>五项输出结果</h2>FACTS
<img src="figures/mean_and_worst_changes.png" alt="平均和最差房间MAE相对保留参考的变化；负值代表改善">
<h2>按输出查看模型与逐房间误差</h2><label for="target">输出指标：</label><select id="target"></select><p id="unit-note"></p><div class="scroll" id="target-summary"></div>
<p>Retained_reference为前轮保留参考；Target_baseline为同一目标原候选的内层选择，已重现v3；Mean_select和Worst_guarded使用同一完整候选池。</p><div class="scroll" id="room-table"></div>
<h2>所有房间完整保留</h2><img src="figures/room_errors.png" alt="五项输出七个留出房间的MAE；P10使用对数纵轴">
<h2>失败位置与解释边界</h2>DIAGNOSIS
<h2>选择过程</h2><p>每一行的候选只由该外层房间之外的六个房间决定。内层分数用于选择，不是独立性能估计。</p><div class="scroll" id="selection-table"></div>
<h2>核对与适用范围</h2><p>VERIFICATION</p><p>LIMITS</p><p>研究用例为模拟数据代理回归，不作实测精度或建筑规范合格声明。</p>
<h2>文件</h2><p><a href="scoreboard.csv">汇总表</a> · <a href="fold_metrics.csv">逐房间误差</a> · <a href="development_gate.csv">筛选条件</a> · <a href="selection_transfer.csv">内外层选择迁移</a> · <a href="verification.json">数值复核</a> · <a href="../../docs/experiment_v7_protocol.md">冻结方案</a> · <a href="../../configs/experiment_v7.json">运行配置</a> · <a href="../../docs/experiment_index.md">各轮实验索引</a></p><script>SCRIPT</script></body></html>'''
    verification_text=f"完成{run['fit_counts']['inner']}次内层和{run['fit_counts']['outer']}次外层拟合；7个外层模型包全部重载，核对1,839,600个预测值、1,008条内层房间分数、105次模型选择及140组外层指标。额外重复拟合7个回归器检查可重复性。"
    for token,value in [('HEADLINE',escape(headline)),('PROTOCOL',escape(protocol)),('GATE',escape(gate_text)),('FACTS',''.join('<p>'+escape(x)+'</p>' for x in facts)),('DIAGNOSIS',''.join('<p>'+escape(x)+'</p>' for x in diagnosis)),('VERIFICATION',escape(verification_text)),('LIMITS',escape(limitations)),('SCRIPT',script)]:html=html.replace(token,value)
    (OUT/'report.html').write_text(html,encoding='utf-8')
    canvas='''import {H1,H2,Text,Stack,Table,Select,useState,useHostTheme} from "cursor/canvas";
const d=DATA;
const fmt=(v:number|null,n=3)=>v==null?"—":v.toFixed(n);
export default function TargetSpecificV7(){const theme=useHostTheme();const[target,setTarget]=useState("E_mean_lux");const m=d.display.find(x=>x.target===target)!;const k=m.factor;return <Stack gap={16} style={{padding:24,maxWidth:1400,color:theme.text.primary}}>
<H1>逐目标优化：最差房间是否改善？</H1><Text weight="semibold">{d.headline}</Text><Text>2026-11-08 · 7个开发房间 · 3个TEST房间封存</Text>
{d.facts.map((s,i)=><Text key={i}>{s}</Text>)}<Text>{d.gate}</Text>
<H2>选择输出查看结果</H2><Select value={target} onChange={setTarget} options={d.display.map(x=>({value:x.target,label:x.label}))}/>
<Text>误差单位：{m.unit}；pp为百分点，Wh为逐小时能量。所有结果来自按房间留出预测。</Text>
<Table headers={["方法","平均MAE","最差MAE","最差房间","平均RMSE","R²中位数","容限内比例"]} rows={d.summary.filter(r=>r.target===target).map(r=>[r.method,fmt(r.mae_mean*k),fmt(r.mae_max*k),r.worst_room,fmt(r.rmse_mean*k),fmt(r.r2_median),fmt(r.within_tolerance_mean*100,1)+"%"])} />
<Text size="small" tone="secondary">Retained_reference为前轮保留参考；Target_baseline为原算法的内层参数选择；其余两种选择方式使用同一完整候选池。容限是描述场景，不是工程标准。</Text>
<H2>七个房间各自的误差</H2><Table headers={["房间","保留参考MAE","原候选MAE","平均优先MAE","最差优先MAE","最差优先Bias"]} rows={d.rooms.map(room=>{const rows=d.folds.filter(r=>r.target===target&&r.room===room);const get=(method:string)=>rows.find(r=>r.method===method)!;return [room,...["Retained_reference","Target_baseline","Mean_select","Worst_guarded"].map(p=>fmt(get(p).mae*k)),fmt(get("Worst_guarded").bias*k)];})}/>
<H2>每折的内层选择</H2><Table headers={["外层房间","选择方式","模型","内层平均MAE","内层最差MAE"]} rows={d.selections.filter(r=>r.target===target).map(r=>[r.held_room,r.policy,r.candidate,fmt(r.inner_mean_mae*k),fmt(r.inner_worst_mae*k)])}/>
<H2>失败位置与解释边界</H2>{d.diagnosis.map((s,i)=><Text key={i}>{s}</Text>)}
<Text>{d.protocol}</Text><Text>{d.limitations}</Text><Text size="small" tone="secondary">来源：Project_2026/reports/experiment_v7。内层{d.fit_counts.inner}次、外层{d.fit_counts.outer}次拟合；额外{d.verification_refits}次复核拟合。原始CSV与既有模型未修改。</Text></Stack>}
'''.replace('DATA',json_data,1)
    (OUT/'target-specific-worst-room.canvas.tsx').write_text(canvas,encoding='utf-8')
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.links=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a:self.ids.append(a['id'])
            if tag in ['a','img']:self.links.append(a.get('href',a.get('src')))
    parser=Parser();parser.feed(html);assert len(parser.ids)==len(set(parser.ids))
    assert all((OUT/link).is_file() for link in parser.links)
    check=OUT/'logs/report_inline_check.js';check.write_text(script,encoding='utf-8')
    result=subprocess.run([r'C:\Program Files\nodejs\node.exe','--check',str(check)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    T.dump(OUT/'presentation_verification.json',{'static_links_checked':len(parser.links),'duplicate_ids':0,
        'inline_JavaScript_syntax':'passed','browser_interaction':'not performed; browser previously disabled',
        'canvas_native_compilation':False,'figures':'awaiting visual review'})
    artifacts=['report.html','summary.md','report_data.json','scoreboard.csv','selection_transfer.csv','target-specific-worst-room.canvas.tsx',
               'figures/mean_and_worst_changes.png','figures/room_errors.png']
    T.dump(OUT/'report_build_manifest.json',{'script_sha256':T.sha(__file__),'artifacts':{p:T.sha(OUT/p) for p in artifacts}})
    print(headline)
    for fact in facts:print(fact)
    print('Report and interactive view written; static checks passed; two figures await visual review.')


if __name__=='__main__':main()
