"""Produce the verified v6 report, weight diagnostics and embedded-data canvas."""
import html
import json
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import train_cross_room_v5 as T

ROOT=T.ROOT;OUT=ROOT/'reports/experiment_v6'
NAMES={'A_relative':'原相对P10','C_asinh':'直接asinh P10','Nested_blend':'内层选择权重的融合'}
def rows(df):return json.loads(df.to_json(orient='records',force_ascii=False,double_precision=15))
def table(df):return df.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')

def main():
    cfg=T.read(ROOT/'configs/experiment_v6.json');run=T.read(OUT/'run_manifest.json');verify=T.read(OUT/'verification.json');decision=T.read(OUT/'candidate_decision.json')
    assert run['status']=='completed' and verify['status']=='passed'
    summary=pd.read_csv(OUT/'summary_metrics.csv');fm=pd.read_csv(OUT/'fold_metrics.csv');picked=pd.read_csv(OUT/'selected_weights.csv')
    inner=pd.read_csv(OUT/'inner_weight_summary.csv');paired=pd.read_csv(OUT/'paired_room_comparison.csv')
    s=summary.set_index('method');a=s.loc['A_relative'];b=s.loc['Nested_blend'];c=s.loc['C_asinh']
    transfer=[]
    for pick in picked.to_dict('records'):
        baseline=inner[(inner.outer_fold==pick['outer_fold'])&(inner.weight_on_A==1)].iloc[0]
        outer=paired[paired.room==pick['held_room']].iloc[0]
        transfer.append({'held_room':pick['held_room'],'weight_on_A':pick['weight_on_A'],'inner_A_MAE_lux':baseline.inner_mean_room_MAE_lux,
            'inner_selected_MAE_lux':pick['inner_mean_room_MAE_lux'],'inner_MAE_reduction_lux':baseline.inner_mean_room_MAE_lux-pick['inner_mean_room_MAE_lux'],
            'outer_A_MAE_lux':outer.A_MAE_lux,'outer_blend_MAE_lux':outer.blend_MAE_lux,'outer_MAE_reduction_lux':-outer.MAE_change_lux})
    transfer=pd.DataFrame(transfer);transfer.to_csv(OUT/'selection_transfer.csv',index=False)
    headline='融合达到预先约定的开发筛选标准，仍不是最终测试结果。' if decision['passes_development_gate'] else '融合没有提升跨房间P10效果，保留原相对P10基线。'
    findings=[f'房间平均P10 MAE：原相对方法{a.mae_mean:.3f} lux，直接asinh {c.mae_mean:.3f} lux，内层选择权重的融合{b.mae_mean:.3f} lux。融合相对原方法的MAE变化为{100*(b.mae_mean/a.mae_mean-1):+.2f}%。',
      f'融合在{decision["rooms_improved"]}个房间改善、{decision["rooms_tied"]}个持平、{decision["rooms_worse"]}个变差；最差房间MAE从{a.mae_max:.3f}升至{b.mae_max:.3f} lux。',
      'BM223与BM432的外层折都由内层选择了纯asinh（相对方法权重0）：前者改善，后者明显恶化。选择过程没有偷看外层标签，但六个训练房间内的偏好未能稳定迁移到新房间。',
      f'平均asinh误差从{a.asinh_mae_mean:.4f}降至{b.asinh_mae_mean:.4f}，但原单位MAE及逐房间标准未通过。本轮不因次要指标更好而改变预先确定的主指标。']
    for room in ['BM223','BM432','BM522']:
        r=paired[paired.room==room].iloc[0];findings.append(f'{room} P10 MAE：{r.A_MAE_lux:.3f} → {r.blend_MAE_lux:.3f} lux；相对P10权重{r.weight_on_A:.2f}。')
    protocol='每个外层折用6个房间训练、1个房间评分。复用v5内层3折预测及已在内层选定的RF参数，在1、0.75、0.5、0.25、0五个相对P10权重中，以6个验证房间平均MAE选择；完全相同时优先更大的相对P10权重。7个权重全部确定后才读取外层标签评分。'
    formula='融合P10 = w×相对P10预测 + (1−w)×直接asinh P10预测。同一折的w对所有小时和角度固定；不按房间名字、真实照度或已知最优方法路由。'
    gate='预设标准：相对A，平均MAE至少降低10%，至少5/7房间不恶化，最差房间MAE不增加。浮点持平容差为1e−9 lux；这是开发筛选条件，不是工程标准。'
    conclusion='本轮未通过筛选，继续保留A作为P10开发基准。按预定停止规则，暂停细化权重或增加按房间切换规则；下一步优先取得与原模拟一致的几何、网格或其他独立输入信息。' if not decision['passes_development_gate'] else '本轮仅列为有限开发候选；仍保留旧模型，不自动部署或解封TEST。'
    limitations='只有7个开发房间，且已经用于多轮方法设计。内层分数同时用于参数/权重选择，不能作为独立性能；外层分数仍属于开发证据。三个TEST房间未预测或评分。没有新模拟、没有全7房间最终重训，其他四项性能指标没有改变。'
    data={'headline':headline,'findings':findings,'protocol':protocol,'formula':formula,'gate':gate,'conclusion':conclusion,'limitations':limitations,
          'summary':rows(summary),'metrics':rows(fm),'weights':rows(picked),'inner':rows(inner),'paired':rows(paired),'transfer':rows(transfer),
          'rooms':paired.room.tolist(),'decision':decision,'verification':verify,'run':run}
    T.dump(OUT/'report_data.json',data)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,ax=plt.subplots(figsize=(11,4.8));positions=np.arange(len(paired));methods=['A_relative','C_asinh','Nested_blend'];colors=['#83939e','#477e9c','#b07848']
    for j,(method,color) in enumerate(zip(methods,colors)):
        vals=fm[fm.method==method].set_index('room').loc[paired.room,'mae'].to_numpy()
        ax.bar(positions+(j-1)*.24,vals,width=.22,color=color,label=method)
    ax.set_yscale('log');ax.set_xticks(positions,paired.room);ax.set_xlabel('Outer held-out room');ax.set_ylabel('P10 MAE (lux, logarithmic scale)')
    ax.set_title('P10 cross-room prediction: original, direct and nested blend');ax.legend(frameon=False,ncol=3,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.01,.01,'Source: v6 OOF. Seven development rooms; four angles and retained Hour 7-18. Lower is better; no TEST predictions.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'figures/room_MAE.png',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(13,6.3));axs=axs.ravel()
    for ax,(room,k) in zip(axs,zip(paired.room,range(1,8))):
        g=inner[inner.held_room==room].sort_values('weight_on_A');pick=picked[picked.held_room==room].iloc[0]
        ax.plot(g.weight_on_A,g.inner_mean_room_MAE_lux,color='#477e9c',marker='o',ms=3)
        ax.scatter([pick.weight_on_A],[pick.inner_mean_room_MAE_lux],color='#b07848',s=44,zorder=3,label='Selected')
        ax.set_title(f'Outer holdout: {room}');ax.set_xlabel('Weight on relative P10');ax.set_ylabel('Inner mean room MAE (lux)');ax.set_xticks([0,.5,1]);ax.grid(alpha=.15)
    axs[-1].axis('off');fig.suptitle('Weight selection uses six fitting rooms only',fontsize=13)
    fig.text(.01,.01,'Source: v6 inner_weight_summary.csv. Orange = selected. Inner scores select parameters/weights; they are not independent performance estimates.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(OUT/'figures/inner_weight_selection.png',bbox_inches='tight');plt.close(fig)
    display=summary.rename(columns={'method':'方法','room_count':'房间数','mae_mean':'平均MAE lux','mae_median':'MAE中位数lux','mae_max':'最差房间MAE lux','rmse_mean':'平均RMSE lux','bias_mean':'平均偏差lux','r2_median':'R²中位数','r2_min':'最差R²','asinh_mae_mean':'平均asinh误差'})
    links=['summary_metrics.csv','fold_metrics.csv','paired_room_comparison.csv','selected_weights.csv','inner_weight_summary.csv','inner_weight_room_scores.csv','selection_transfer.csv','candidate_decision.json','verification.json','recipe_index.json','config_snapshot.json','P10-nested-blend.canvas.tsx','../cross_room_diagnosis_v5/report.html','../../docs/data_requests/geometry_and_simulation_sources.md','../../exports/geometry_source_handoff_v1.zip']
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P10融合实验 v6</title><style>body{font:16px/1.65 "Microsoft YaHei",sans-serif;color:#263641;background:#f8fafb;margin:0}main{max-width:1200px;margin:auto;padding:30px}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.lead{font-size:20px;font-weight:600}.meta{font-size:13px;color:#62737e}.note{border-left:3px solid #487e8d;padding:10px 16px;background:#eef5f6}table{width:100%;border-collapse:collapse;font-size:13px;background:white}th,td{padding:8px;border-bottom:1px solid #dce4e8;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf2f4}.scroll{overflow-x:auto}img{width:100%}select{padding:8px;margin:8px 0}a{color:#256b91}li{margin-bottom:9px}</style><main><h1>第六轮：P10融合权重实验</h1>'''
    doc+=f'<p class="lead">{headline}</p><p class="meta">35组内层权重候选 · 7次权重选择 · 复用原树模型，无重复拟合 · TEST预测/评分0</p>'
    doc+='<ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in findings)+'</ul>'
    doc+='<h2>方法和验证</h2>'+''.join('<p>'+html.escape(s)+'</p>' for s in [protocol,formula])+'<p class="note">'+gate+'</p>'
    doc+='<h2>预测结果</h2><div class="scroll">'+table(display)+'</div><p class="meta">每个房间先计算指标，再等权平均。asinh误差使用固定1 lux尺度；R²不是准确率百分比。主指标是原单位MAE。</p><img src="figures/room_MAE.png" alt="三个方法的逐房间P10误差，lux，对数坐标">'
    doc+='<h2>内层改善是否迁移到外层</h2><div class="scroll">'+table(transfer)+'</div><p>内层MAE降低为正表示在六个训练房间的验证数据上有收益；外层MAE降低为负表示对新房间反而变差。这两列来自不同房间集合，不能把差值解释为单纯随机波动或因果效应。</p><img src="figures/inner_weight_selection.png" alt="七个外层折内部的五个候选权重分数">'
    doc+='<h2>逐折权重明细</h2><select id="room"></select><div class="scroll"><table><thead><tr><th>相对P10权重</th><th>内层平均MAE lux</th><th>内层最差MAE lux</th><th>是否选中</th></tr></thead><tbody id="weightrows"></tbody></table></div>'
    doc+='<h2>当前决定</h2><p class="note">'+conclusion+'</p><p>'+limitations+'</p>'
    doc+=f'<p>独立复核了{verify["inner_room_scores_recomputed"]}个内层房间分数、35个候选平均分、7次权重选择、122,640条融合预测及21条外层房间指标。7个折配置在不含Room_ID或真实目标的13项特征上重载推断通过。</p>'
    doc+='<p>models/experiment_v6保存的是各折融合配置，引用原v5模型，不是最终部署模型。原数据、模型及历史预测保持不变。旧提案保留原始未批准文本；本次批准与执行状态记录在新的配置和批准凭据中。</p><p class="meta">HTML链接和脚本静态验证；图表视觉检查；浏览器交互和原生Canvas编译未验证。</p>'
    doc+='<h2>文件</h2><ul>'+''.join(f'<li><a href="{p}">{p}</a></li>' for p in links)+'</ul>'
    payload={'rooms':data['rooms'],'inner':data['inner'],'selected':data['weights']}
    doc+='''<script>const d=PAYLOAD;const room=document.querySelector('#room');d.rooms.forEach(v=>room.add(new Option(v,v)));room.value='BM432';function render(){const w=d.selected.find(r=>r.held_room===room.value).weight_on_A;document.querySelector('#weightrows').innerHTML=d.inner.filter(r=>r.held_room===room.value).map(r=>'<tr>'+[r.weight_on_A,r.inner_mean_room_MAE_lux.toFixed(4),r.inner_worst_room_MAE_lux.toFixed(4),r.weight_on_A===w?'是':''].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');}room.onchange=render;render();</script></main></html>'''.replace('PAYLOAD',json.dumps(payload,ensure_ascii=False))
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    canvasdata={k:data[k] for k in ['headline','findings','protocol','formula','gate','conclusion','limitations','summary','paired','inner','weights','rooms','transfer']}
    canvas='''import {H1,H2,Text,Stack,Row,Table,Select,useState,useHostTheme} from "cursor/canvas";
const d=PAYLOAD;const f=(v:number|null,n=3)=>v==null?"不适用":v.toFixed(n);
export default function P10BlendV6(){const theme=useHostTheme();const[room,setRoom]=useState("BM432");return <Stack gap={18} style={{padding:24,maxWidth:1300,color:theme.text.primary}}>
<H1>第六轮：P10融合实验</H1><Text weight="semibold">{d.headline}</Text><Text>7次权重选择，35组内层候选。复用v5树模型，TEST预测/评分0。</Text>
{d.findings.slice(0,4).map((s,i)=><Text key={i}>{s}</Text>)}<Text>{d.formula}</Text>
<H2>七个房间等权平均结果</H2><Table headers={["方法","平均MAE lux","最差房间MAE lux","平均RMSE lux","平均asinh误差"]} rows={d.summary.map(r=>[r.method,f(r.mae_mean),f(r.mae_max),f(r.rmse_mean),f(r.asinh_mae_mean,4)])}/>
<Text>{d.gate}</Text><H2>逐房间结果</H2><Table headers={["房间","相对P10权重","原MAE lux","融合MAE lux","MAE变化lux","原偏差lux","融合偏差lux"]} rows={d.paired.map(r=>[r.room,r.weight_on_A,f(r.A_MAE_lux),f(r.blend_MAE_lux),f(r.MAE_change_lux),f(r.A_bias_lux),f(r.blend_bias_lux)])}/>
<Text size="small" tone="secondary">MAE变化=融合−原方法，负数表示改善。权重0是纯asinh，1是原相对P10；各折权重对所有小时/角度固定。</Text>
<H2>权重怎样选择</H2><Text>{d.protocol}</Text><Select value={room} onChange={setRoom} options={d.rooms.map(v=>({value:v,label:v}))}/>
<Table headers={["相对P10权重","内层平均MAE lux","内层最差MAE lux","已选"]} rows={d.inner.filter(r=>r.held_room===room).map(r=>[r.weight_on_A,f(r.inner_mean_room_MAE_lux),f(r.inner_worst_room_MAE_lux),d.weights.find(s=>s.held_room===room)!.weight_on_A===r.weight_on_A?"是":""])}/>
<Text size="small" tone="secondary">内层分数用于参数/权重选择，不是独立性能估计。选择时不读取外层真实P10。</Text>
<H2>内层收益与外层收益</H2><Table headers={["外层房间","内层MAE降低lux","外层MAE降低lux"]} rows={d.transfer.map(r=>[r.held_room,f(r.inner_MAE_reduction_lux),f(r.outer_MAE_reduction_lux)])}/>
<H2>当前决定</H2><Text>{d.conclusion}</Text><Text>{d.limitations}</Text><Text size="small" tone="secondary">来源：Project_2026/reports/experiment_v6。模拟年度Hour 7–18，四角度；210个内层房间分数、7次权重选择、122640条融合预测与7个特征输入推断配置已复核。模型配置为交叉验证折模型，未部署。</Text>
</Stack>}
'''.replace('PAYLOAD',json.dumps(canvasdata,ensure_ascii=False))
    (OUT/'P10-nested-blend.canvas.tsx').write_text(canvas,encoding='utf-8')
    (OUT/'summary.md').write_text('# P10融合实验 v6\n\n## Material Passport\n\n- Mode: approved experiment + verification\n- New regression fits: 0; weight selection tasks: 7\n- Numerical verification: passed; no final TEST\n\n'+headline+'\n\n'+'\n\n'.join(findings+[protocol,formula,gate,conclusion,limitations])+'\n',encoding='utf-8')
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
    parsed=Parser();parsed.feed(doc);assert len(parsed.ids)==len(set(parsed.ids)) and len(parsed.scripts)==1
    assert all((OUT/p).is_file() for p in parsed.refs)
    js=ROOT/'.cache/v6_report_inline.js';js.write_text(parsed.scripts[0],encoding='utf-8')
    check=subprocess.run(['C:/Program Files/nodejs/node.exe','--check',str(js)],capture_output=True,text=True);assert check.returncode==0,check.stderr
    T.dump(OUT/'presentation_verification.json',{'static_links_checked':len(parsed.refs),'duplicate_ids':0,'inline_JavaScript_syntax':'passed',
      'browser_interaction':'not performed; local browser use previously disabled','canvas_native_compilation':False,'figures':'awaiting visual review'})
    files=['report.html','summary.md','report_data.json','P10-nested-blend.canvas.tsx','selection_transfer.csv','figures/room_MAE.png','figures/inner_weight_selection.png']
    T.dump(OUT/'report_build_manifest.json',{'script_sha256':T.sha(__file__),'artifacts':{p:T.sha(OUT/p) for p in files}})
    print(headline)
    for s in findings[:4]:print(s)
    print('HTML, canvas, two figures and static checks completed.')

if __name__=='__main__':main()
