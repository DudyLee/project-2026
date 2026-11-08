"""Render verified diagnostic findings, scientific figures and source handoff links."""
import html
import json
from pathlib import Path
import subprocess
from html.parser import HTMLParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import train_cross_room_v5 as T

ROOT=T.ROOT;OUT=ROOT/'reports/cross_room_diagnosis_v5'
def records(df):return json.loads(df.to_json(orient='records',force_ascii=False,double_precision=15))
def table(df):return df.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')

def main():
    run=T.read(OUT/'run_manifest.json');verify=T.read(OUT/'verification.json');assert verify['status']=='passed'
    rooms=pd.read_csv(OUT/'room_error_structure.csv');strata=pd.read_csv(OUT/'stratified_errors.csv');profiles=pd.read_csv(OUT/'temporal_profiles.csv')
    dec=pd.read_csv(OUT/'P10_error_decomposition.csv');pairs=pd.read_csv(OUT/'angle_pair_order.csv');comp=pd.read_csv(OUT/'P10_model_complementarity.csv')
    inventory=T.read(OUT/'source_inventory.json');cfg=T.read(ROOT/'configs/diagnosis_v5.json')
    def decomp(room):return dec[(dec.method=='A_RandomForest')&(dec.room==room)].iloc[0]
    b223=decomp('BM223');b432=decomp('BM432');b522=decomp('BM522')
    bm=rooms[(rooms.method=='A_XGBoost')&(rooms.room=='BM522')&(rooms.target=='E_mean_lux')].iloc[0]
    weather=strata[(strata.method=='A_XGBoost')&(strata.room=='BM522')&(strata.target=='E_mean_lux')&(strata.axis=='DNI_band')].set_index('group')
    ranks=pairs[(pairs.method=='A_XGBoost')&(pairs.room=='BM522')&(pairs.target=='E_mean_lux')&(pairs.true_gap_threshold_lux==10)].iloc[0]
    facts=[
      f'BM223的相对P10模型主要高估比例：真实逐时P10/(平均照度+1 lux)的均值为{b223.actual_ratio_mean:.4f}，预测为{b223.predicted_ratio_mean:.4f}。原P10 MAE {b223.P10_MAE_lux:.3f} lux；诊断中仅代入真实平均照度仍为{b223.ORACLE_true_mean_MAE_lux:.3f} lux，不能解决高估。',
      f'BM432存在两阶段误差抵消：原P10 MAE {b432.P10_MAE_lux:.3f} lux；诊断中仅代入真实平均照度反而为{b432.ORACLE_true_mean_MAE_lux:.3f} lux。不能把一个阶段变准确就等同于最终P10更准确。其P10配对均值模型为{b432.paired_mean_candidate}，独立均值输出模型为{b432.standalone_mean_candidate}，分解使用正确的配对模型。',
      f'BM522的XGBoost平均照度在{bm.overprediction_fraction*100:.1f}%记录中高估，平均偏差+{bm.bias:.2f} lux。DNI=0时MAE {weather.loc["0","mae"]:.2f} lux，DNI>500 Wh/m²时为{weather.loc["above_500","mae"]:.2f} lux。偏差随环境变化，不能仅视为一个固定常数误差。',
      f'BM522在真实平均照度角度差>10 lux的{int(ranks.eligible_pairs):,}个角度对中，排序正确率为{ranks.correct_order_fraction*100:.1f}%（覆盖全部角度对的{ranks.eligible_fraction*100:.1f}%），但角度差值MAE仍为{ranks.eligible_pair_delta_mae_lux:.2f} lux。此为单一指标排序诊断，不是综合推荐准确率。',
      '本地未找到与这批Tower A模拟建立来源对应的原始几何、网格、逐传感器照度或EPW。已准备7个TRAIN房间定位表、字段字典、待填模板和24个已有工况的原始传感器结果核对清单；没有要求新模拟，也没有向他人发送消息。'
    ]
    cautions=['以上真实均值/真实比例替换仅用于误差解释，使用了留出房间真值，不能作为模型改进后的验证分数。',
              '误差分解三项可相互抵消，不能把各项绝对值当作可相加的原因百分比。',
              '天气分层揭示当前CSV中的关联，不证明EPW身份、小时对齐或地理北约定正确。',
              'P10角度排序好不代表P10绝对值、F300/FUDI、PV或综合推荐已经可靠；四角度顺序稳定本身也可能使排序任务较容易。',
              '7个房间已用于多轮开发，所有本轮结果都是解释性开发分析；BM532、BM702、BMG26保持封存。']
    next_steps=[
       '优先补齐可追溯的房间进深、窗户位置/高度、材料与网格信息，解释BM223比例尺度和BM522直射/季节响应差异；提取已有几何无需新增全年模拟。',
       '如暂时拿不到几何，可另行确认一个有限的P10融合试验：仅用现有Random Forest A/C预测，权重从内层按房间选择，再由外层房间评分；不按已知房间名称指定模型。',
       '融合是低成本的候选排除试验，不能预期解决缺失的房间信息。本轮尚未选择权重或运行融合试验，具体草案已保存。']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    cases=[('BM223','E_P10_lux',['A_RandomForest','C_RandomForest']),('BM432','E_P10_lux',['A_RandomForest','C_RandomForest']),('BM522','E_mean_lux',['A_XGBoost','B_XGBoost'])]
    for axis,filename,xlabel in [('Month','monthly_profiles.png','Month'),('PVSD_Angle_deg','angle_profiles.png','PVSD angle (degrees)')]:
        fig,axs=plt.subplots(1,3,figsize=(13,4.4))
        for ax,(room,target,methods) in zip(axs,cases):
            q=profiles[(profiles.room==room)&(profiles.target==target)&(profiles.axis==axis)]
            base=q[q.method==methods[0]].sort_values('group')
            ax.plot(base.group,base.truth_mean,color='#29363e',lw=2,marker='o',ms=3,label='Simulation truth')
            for method,color in zip(methods,['#347f9f','#b47443']):
                g=q[q.method==method].sort_values('group');ax.plot(g.group,g.pred_mean,color=color,lw=1.7,label=method.replace('RandomForest','RF').replace('XGBoost','XGB'))
            ax.set_title(room+(' | P10' if target=='E_P10_lux' else ' | Mean illuminance'))
            ax.set_xlabel(xlabel);ax.set_ylabel('Mean illuminance metric (lux)');ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
            if axis=='PVSD_Angle_deg':ax.set_xticks([0,30,60,90])
        fig.suptitle('Frozen held-room predictions: scale and response',fontsize=13)
        fig.text(.01,.01,'Source: v5 OOF, TRAIN rooms only; Hour 7-18. Monthly profiles average hours and four angles; angle profiles average retained annual hours.',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(OUT/'figures'/filename,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4.4))
    for ax,room in zip(axs,cfg['focal_rooms']):
        r=decomp(room);vals=[r.P10_MAE_lux,r.ORACLE_true_mean_MAE_lux,r.ORACLE_true_ratio_MAE_lux]
        ax.bar([0,1,2],vals,color=['#657985','#b89a78','#678f89'])
        ax.set_xticks([0,1,2],['Original','True mean\nsubstitution','True ratio\nsubstitution']);ax.set_ylabel('P10 MAE (lux)');ax.set_xlabel('Diagnostic reconstruction');ax.set_title(room)
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.02,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(vals)*1.2)
    fig.suptitle('ORACLE substitutions of unavailable labels: explanation only',fontsize=13)
    fig.text(.01,.01,'Not deployable models or new validation scores. A_RandomForest; original paired mean and ratio models. Error cancellation can worsen a substitution.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.93));fig.savefig(OUT/'figures/P10_diagnostic_substitutions.png',bbox_inches='tight');plt.close(fig)
    payload={'facts':facts,'cautions':cautions,'next_steps':next_steps,'rooms':run['train_rooms'],'methods':cfg['methods'],
      'room_errors':records(rooms),'profiles':records(profiles),'strata':records(strata),'decomposition':records(dec),
      'angle_pairs':records(pairs),'complementarity':records(comp),'verification':verify,'inventory':inventory}
    T.dump(OUT/'report_data.json',payload)
    title='跨房间误差诊断与几何资料整理'
    md='# '+title+'\n\n## Material Passport\n\n- Mode: diagnostic analysis / source preparation\n- New fits: 0\n- Numerical checks: passed; original simulation provenance remains unverified\n\n'+'\n\n'.join(facts+cautions+next_steps)+'\n'
    (OUT/'summary.md').write_text(md,encoding='utf-8')
    focaldec=dec[(dec.method=='A_RandomForest')&(dec.room.isin(cfg['focal_rooms']))][['room','P10_MAE_lux','ORACLE_true_mean_MAE_lux','ORACLE_true_ratio_MAE_lux','mean_term_signed_mean_lux','ratio_term_signed_mean_lux','interaction_term_signed_mean_lux']]
    links=['room_error_structure.csv','stratified_errors.csv','angle_delta_errors.csv','angle_pair_order.csv','P10_error_decomposition.csv','P10_model_complementarity.csv','TRAIN_geometry_and_sensor_metadata.csv','source_inventory.json','verification.json','cross-room-error-diagnosis.canvas.tsx','../../docs/data_requests/geometry_and_simulation_sources.md','../../docs/data_requests/TRAIN_room_source_index.csv','../../docs/data_requests/geometry_field_dictionary.csv','../../docs/data_requests/existing_sensor_result_sample_requests.csv','../../configs/proposals/P10_blend_v6_draft.json']
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tower A 跨房间诊断</title><style>body{font:16px/1.65 "Microsoft YaHei",sans-serif;background:#f9fbfc;color:#253540;margin:0}main{max-width:1200px;margin:auto;padding:28px}h1{font-size:27px}h2{font-size:21px;margin-top:30px}.note{border-left:3px solid #367984;padding:10px 16px;background:#edf5f5}.meta{font-size:13px;color:#617481}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:8px;border-bottom:1px solid #dde4e7;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf1f4}.scroll{overflow-x:auto}img{width:100%}select{padding:8px;margin:8px}a{color:#266a8b}li{margin-bottom:9px}</style><main>'''
    doc+=f'<h1>{title}</h1><p class="meta">7个开发房间 · 9组已保存方法输出 · 新增拟合0 · TEST预测/评分0</p>'
    doc+='<h2>主要发现</h2><ol>'+''.join('<li>'+html.escape(s)+'</li>' for s in facts)+'</ol>'
    doc+='<h2>时间变化与角度响应</h2><img src="figures/monthly_profiles.png" alt="三个重点房间的真实与预测月平均照度指标"><img src="figures/angle_profiles.png" alt="三个重点房间四角度平均指标">'
    doc+='<h2>P10两阶段误差</h2><p class="note">真实均值或比例替换是不可部署的诊断。它们使用留出房间标签，不能作为新模型验证成绩。</p><img src="figures/P10_diagnostic_substitutions.png" alt="仅用于诊断的真实均值和真实比例替换"><div class="scroll">'+table(focaldec)+'</div>'
    doc+='<p>令m为真实平均照度、r=P10/(m+1)，则P10误差为 r×(m̂−m)+(m+1)×(r̂−r)+(m̂−m)×(r̂−r)，另保留原生浮点乘法残差。三项可互相抵消，绝对值不能相加为贡献百分比。r̂和m̂取自相对P10的配对候选。</p>'
    doc+='<h2>按房间与环境查看</h2><select id="room"></select><select id="target"><option value="E_P10_lux">P10</option><option value="E_mean_lux">平均照度</option></select><select id="method"></select><select id="axis"><option>DNI_band</option><option>Month</option><option>Hour</option><option>PVSD_Angle_deg</option><option>GHI_band</option><option>Altitude_band</option></select><p class="meta">DNI/GHI分组单位Wh/m²；太阳高度角单位度。分层是当前CSV中的关联，不证明天气来源与时间对齐正确。</p><div class="scroll"><table><thead><tr><th>分组</th><th>记录数</th><th>真实均值lux</th><th>预测均值lux</th><th>MAE lux</th><th>偏差lux</th><th>相关系数</th></tr></thead><tbody id="stratarows"></tbody></table></div>'
    doc+='<h2>资料与后续范围</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in next_steps)+'</ul>'
    doc+='<p>几何模板空白项表示未知，尚不可入模。原始逐传感器照度未取得，因此尚不能重新计算源数据P10。资料清单未发送给他人。融合草案标记为未批准、未执行。</p><h2>解释边界与检查</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in cautions)+'</ul>'
    doc+=f'<p>复核{verify["room_summaries_checked"]}个房间指标、{verify["stratified_group_reaggregation_checks"]}个分层汇总、630个角度指标及{verify["P10_component_rows_checked"]:,}条P10分解。原始数据、旧模型和预测文件哈希保持不变。本轮未重新拟合模型或解封TEST。</p><p class="meta">HTML脚本与链接静态检查；浏览器交互与原生Canvas编译未验证。</p>'
    doc+='<h2>下载与复核</h2><ul>'+''.join(f'<li><a href="{p}">{p}</a></li>' for p in links)+'</ul>'
    mini={'rooms':payload['rooms'],'methods':payload['methods'],'strata':payload['strata']}
    doc+='''<script>const d=PAYLOAD;const room=document.querySelector('#room'),method=document.querySelector('#method'),target=document.querySelector('#target'),axis=document.querySelector('#axis');d.rooms.forEach(v=>room.add(new Option(v,v)));d.methods.forEach(v=>method.add(new Option(v,v)));room.value='BM522';method.value='A_XGBoost';target.value='E_mean_lux';const f=v=>v==null?'—':Number(v).toFixed(3);function render(){document.querySelector('#stratarows').innerHTML=d.strata.filter(r=>r.room===room.value&&r.method===method.value&&r.target===target.value&&r.axis===axis.value).map(r=>'<tr>'+[r.group,r.n,f(r.actual_mean_lux),f(r.prediction_mean_lux),f(r.mae),f(r.bias),f(r.pearson_r)].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');}room.onchange=method.onchange=target.onchange=axis.onchange=render;render();</script></main></html>'''.replace('PAYLOAD',json.dumps(mini,ensure_ascii=False))
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    canvasdata={k:payload[k] for k in ['facts','cautions','next_steps','rooms','methods','room_errors','decomposition','angle_pairs','strata']}
    canvas='''import {H1,H2,Text,Stack,Row,Table,Select,useState,useHostTheme} from "cursor/canvas";
const d=PAYLOAD;const f=(x:number|null,n=3)=>x==null?"不适用":x.toFixed(n);
export default function ErrorDiagnosis(){const theme=useHostTheme();const[room,setRoom]=useState("BM223");const[target,setTarget]=useState("E_P10_lux");const[method,setMethod]=useState("A_RandomForest");const[axis,setAxis]=useState("DNI_band");
return <Stack gap={18} style={{padding:24,maxWidth:1300,color:theme.text.primary}}><H1>跨房间误差诊断与资料整理</H1>
<Text weight="semibold">BM223比例尺度高估；BM432存在误差抵消；BM522对强辐射的响应幅度失准。</Text><Text>重用v5留出预测与模型。新增拟合0，TEST预测/评分0。本轮是误差解释，不是新模型验证。</Text>
{d.facts.map((s,i)=><Text key={i}>{s}</Text>)}
<H2>逐房间误差结构</H2><Row gap={12} wrap><Select value={room} onChange={setRoom} options={d.rooms.map(v=>({value:v,label:v}))}/><Select value={target} onChange={setTarget} options={[{value:"E_P10_lux",label:"P10"},{value:"E_mean_lux",label:"平均照度"}]}/></Row>
<Table headers={["方法","真实均值lux","预测均值lux","MAE lux","偏差lux","相关系数","高估记录%"]} rows={d.room_errors.filter(r=>r.room===room&&r.target===target).map(r=>[r.method,f(r.actual_mean_lux),f(r.prediction_mean_lux),f(r.mae),f(r.bias),f(r.pearson_r),f(r.overprediction_fraction*100,1)])}/>
{target==="E_P10_lux"&&<><H2>相对P10：不可部署的真值替换诊断</H2><Text>两项替换使用留出房间真实标签，只解释误差，不是预测性能。误差项会相互抵消。</Text><Table headers={["方法","原P10 MAE lux","代入真实均值 MAE lux","代入真实比例 MAE lux","真实比例均值","预测比例均值"]} rows={d.decomposition.filter(r=>r.room===room).map(r=>[r.method,f(r.P10_MAE_lux),f(r.ORACLE_true_mean_MAE_lux),f(r.ORACLE_true_ratio_MAE_lux),f(r.actual_ratio_mean,4),f(r.predicted_ratio_mean,4)])}/></>}
<H2>环境与时间分层</H2><Row gap={12} wrap><Select value={method} onChange={setMethod} options={d.methods.map(v=>({value:v,label:v}))}/><Select value={axis} onChange={setAxis} options={["DNI_band","GHI_band","Altitude_band","Month","Hour","PVSD_Angle_deg"].map(v=>({value:v,label:v}))}/></Row>
<Table headers={["分组","记录数","真实均值lux","预测均值lux","MAE lux","偏差lux"]} rows={d.strata.filter(r=>r.room===room&&r.target===target&&r.method===method&&r.axis===axis).map(r=>[r.group,r.n,f(r.actual_mean_lux),f(r.prediction_mean_lux),f(r.mae),f(r.bias)])}/>
<Text size="small" tone="secondary">DNI/GHI单位Wh/m²，高度角单位度。天气分层不验证EPW身份、时区或小时对齐。相关系数高不代表误差小。</Text>
<H2>四角度排序，仅为单指标诊断</H2><Table headers={["真实差值门槛lux","参与角度对","覆盖%","排序正确%","角度差MAE lux"]} rows={d.angle_pairs.filter(r=>r.room===room&&r.target===target&&r.method===method).map(r=>[r.true_gap_threshold_lux,r.eligible_pairs,f(r.eligible_fraction*100,1),r.correct_order_fraction==null?"不适用":f(r.correct_order_fraction*100,1),f(r.eligible_pair_delta_mae_lux)])}/><Text>门槛为描述性场景，非工程容差。角度排序分数不是采光与发电综合推荐准确率。</Text>
<H2>后续方向与资料缺口</H2>{d.next_steps.map((s,i)=><Text key={i}>{s}</Text>)}<Text>7个TRAIN房间字段模板与24个已有工况核对清单已保存。尚未取得原始几何/网格，无新模拟、无消息外发。</Text>
<Text size="small" tone="secondary">来源：Project_2026/reports/cross_room_diagnosis_v5，模拟年度Hour 7–18、四角度。126个房间指标、630个角度指标、490560条P10分解已复核。7个开发房间已被多轮使用，三个TEST房间封存。</Text></Stack>}
'''.replace('PAYLOAD',json.dumps(canvasdata,ensure_ascii=False))
    (OUT/'cross-room-error-diagnosis.canvas.tsx').write_text(canvas,encoding='utf-8')
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.refs=[];self.ids=[];self.scripts=[];self.active=False
        def handle_starttag(self,t,a):
            a=dict(a)
            if 'id' in a:self.ids.append(a['id'])
            if t in ['a','img']:self.refs.append(a.get('href') or a.get('src'))
            if t=='script':self.active=True
        def handle_endtag(self,t):
            if t=='script':self.active=False
        def handle_data(self,s):
            if self.active:self.scripts.append(s)
    p=Parser();p.feed(doc);assert len(p.ids)==len(set(p.ids)) and len(p.scripts)==1
    assert all((OUT/r).is_file() for r in p.refs)
    js=ROOT/'.cache/diagnosis_v5_inline.js';js.write_text(p.scripts[0],encoding='utf-8')
    result=subprocess.run(['C:/Program Files/nodejs/node.exe','--check',str(js)],capture_output=True,text=True);assert result.returncode==0,result.stderr
    T.dump(OUT/'presentation_verification.json',{'HTML_relative_links_checked':len(p.refs),'duplicate_ids':0,'inline_script_syntax':'passed',
         'browser_interaction':'not performed; local browser use previously disabled','native_canvas_compilation':False,'figures':'awaiting visual review'})
    files=['report.html','summary.md','report_data.json','cross-room-error-diagnosis.canvas.tsx','figures/monthly_profiles.png','figures/angle_profiles.png','figures/P10_diagnostic_substitutions.png']
    T.dump(OUT/'report_build_manifest.json',{'script_sha256':T.sha(__file__),'artifacts':{p:T.sha(OUT/p) for p in files}})
    for fact in facts:print(fact)
    print('Report, canvas, three figures and static checks completed.')

if __name__=='__main__':main()
