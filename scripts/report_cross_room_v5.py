"""Build a reproducible Chinese report, scientific figures and embedded-data canvas."""
from pathlib import Path
import html
import json
import re
import subprocess
from html.parser import HTMLParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import train_cross_room_v5 as T

ROOT=T.ROOT;OUT=ROOT/'reports/experiment_v5'
TARGET_LABELS={'E_mean_lux':'平均照度','E_P10_lux':'P10照度'}
PROFILES={'A':'现有特征＋相对P10','B':'增强特征＋相对P10','C':'现有特征＋直接asinh P10','D':'增强特征＋直接asinh P10'}

def rows(df):return json.loads(df.to_json(orient='records',force_ascii=False,double_precision=15))
def table(df):return df.to_html(index=False,border=0,escape=True,float_format=lambda x:f'{x:.4f}')

def main():
    run=T.read(OUT/'run_manifest.json');verify=T.read(OUT/'verification.json');cfg=T.read(T.CONFIG)
    assert run['status']=='completed' and verify['status']=='passed'
    summary=pd.read_csv(OUT/'summary_metrics.csv');fm=pd.read_csv(OUT/'fold_metrics.csv')
    gates=pd.read_csv(OUT/'promotion_criteria.csv');paired=pd.read_csv(OUT/'paired_room_changes.csv')
    details=pd.read_csv(OUT/'room_error_details.csv');support=pd.read_csv(OUT/'feature_support.csv')
    selections=pd.read_csv(OUT/'selections.csv');decision=T.read(OUT/'candidate_decision.json')
    roomnames=sorted(fm.room.unique());families=cfg['families'];targets=cfg['targets']
    passed=gates[gates.passes_development_rule]
    headline='本轮没有方案同时通过预设的三项开发筛选条件。' if passed.empty else f'本轮有{len(passed)}项“方案—目标”组合通过开发筛选，仍需检查逐房间局限。'
    findings=[]
    for target in targets:
        candidates=summary[(summary.target==target)&(summary.method!='Frozen_HistGB')]
        best=candidates.loc[candidates.mae_mean.idxmin()]
        findings.append(f'{TARGET_LABELS[target]}最低房间平均MAE为{best.mae_mean:.3f} lux（{best.method}）；这是本轮开发结果的事后描述，不是锁定模型的独立测试分数。')
    for room,target in [('BM223','E_P10_lux'),('BM522','E_mean_lux')]:
        g=fm[(fm.room==room)&(fm.target==target)&(fm.method!='Frozen_HistGB')];best=g.loc[g.mae.idxmin()]
        truth=details[(details.room==room)&(details.target==target)].actual_mean_lux.iloc[0]
        findings.append(f'{room}的{TARGET_LABELS[target]}：本轮最低MAE {best.mae:.3f} lux（{best.method}），真实均值{truth:.3f} lux。最优方法按本房间事后查看，不能据此给新房间选择模型。')
    def value(method,target,room=None):
        if room is None:return summary.loc[(summary.method==method)&(summary.target==target),'mae_mean'].item()
        return fm.loc[(fm.method==method)&(fm.target==target)&(fm.room==room),'mae'].item()
    ma=value('A_XGBoost','E_mean_lux');mb=value('B_XGBoost','E_mean_lux')
    pa=value('A_RandomForest','E_P10_lux');pc=value('C_RandomForest','E_P10_lux')
    p223a=value('A_RandomForest','E_P10_lux','BM223');p223c=value('C_RandomForest','E_P10_lux','BM223')
    p432a=value('A_RandomForest','E_P10_lux','BM432');p432c=value('C_RandomForest','E_P10_lux','BM432')
    findings.insert(0,f'新增特征的XGBoost平均照度MAE：{ma:.3f} → {mb:.3f} lux，下降{100*(1-mb/ma):.2f}%，6/7房间改善，仍未达到预先约定的10%幅度。')
    findings.insert(1,f'Random Forest直接asinh P10将BM223 MAE从{p223a:.3f}降至{p223c:.3f} lux（下降{100*(1-p223c/p223a):.1f}%），但BM432从{p432a:.3f}增至{p432c:.3f} lux；7房间平均MAE从{pa:.3f}增至{pc:.3f} lux（恶化{100*(pc/pa-1):.1f}%）。局部改善不能支持全局替换。')
    rule='预先约定：相对同一算法A组，房间平均MAE至少下降10%，至少5/7房间不恶化，最差房间MAE不增加。逐目标判断；这是开发筛选规则，不是工程验收标准。'
    limitation='只有7个开发房间，且已用于多轮方法设计。内层分组调参不能消除累计开发选择偏差。BM532、BM702、BMG26仍封存；本轮无新模拟、无角度决策优化、无全部7房间部署重训。'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    colors=['#a9b4bd','#437a99','#8d6f9e','#317d65']
    for ax,target in zip(axes,targets):
        for i,profile in enumerate(PROFILES):
            vals=[summary.loc[(summary.method==profile+'_'+fam)&(summary.target==target),'mae_mean'].item() for fam in families]
            ax.bar(np.arange(2)+(i-1.5)*.18,vals,width=.17,color=colors[i],label=profile)
        reference=summary.loc[(summary.method=='Frozen_HistGB')&(summary.target==target),'mae_mean'].item()
        ax.axhline(reference,color='#65717a',ls='--',lw=1,label='Frozen HistGB')
        ax.set_xticks([0,1],['Random Forest','XGBoost']);ax.set_xlabel('Model family');ax.set_ylabel('Mean room MAE (lux)')
        ax.set_title('Mean illuminance' if target=='E_mean_lux' else 'P10 illuminance');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[1].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.19),frameon=False)
    fig.suptitle('Cross-room prediction: seven outer held-room folds',fontsize=13)
    fig.text(.01,.01,'A: base/relative P10; B: enhanced/relative; C: base/asinh; D: enhanced/asinh. Mean: A=C, B=D. Source: v5 OOF.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(OUT/'figures/macro_MAE.png',bbox_inches='tight');plt.close(fig)
    methods=[p+'_'+fam for fam in families for p in ['B','C','D']]
    fig,axes=plt.subplots(1,2,figsize=(14,5.2))
    for ax,target in zip(axes,targets):
        mat=paired[paired.target==target].pivot(index='method',columns='room',values='mae_change_lux').reindex(index=methods,columns=roomnames)
        improvement=-mat.to_numpy();limit=max(1,float(np.max(np.abs(improvement))))
        im=ax.imshow(improvement,cmap='BrBG',vmin=-limit,vmax=limit,aspect='auto')
        for i in range(len(methods)):
            for j in range(len(roomnames)):
                v=improvement[i,j];label='0.0' if abs(v)<.05 else f'{v:+.1f}';ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if abs(v)>.6*limit else 'black')
        ax.set_xticks(range(len(roomnames)),roomnames,rotation=45,ha='right');ax.set_yticks(range(len(methods)),[m.replace('RandomForest','RF').replace('XGBoost','XGB') for m in methods])
        ax.set_xlabel('Outer held-out room');ax.set_ylabel('Variant / model family');ax.set_title('Mean illuminance' if target=='E_mean_lux' else 'P10 illuminance')
        fig.colorbar(im,ax=ax,label='MAE reduction vs same-family A (lux)')
    fig.suptitle('Paired room changes: positive = improvement',fontsize=13)
    fig.text(.01,.01,'Source: v5 paired_room_changes.csv; seven development rooms, four angles and retained Hour 7-18.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'figures/room_MAE_changes.png',bbox_inches='tight');plt.close(fig)
    data={'headline':headline,'findings':findings,'rule':rule,'limitation':limitation,'profiles':PROFILES,'targets':TARGET_LABELS,
          'rooms':roomnames,'summary':rows(summary),'metrics':rows(fm),'details':rows(details),'criteria':rows(gates),'paired':rows(paired),
          'support':rows(support),'selections':rows(selections),'verification':verify,'run':run,'config':cfg}
    T.dump(OUT/'report_data.json',data)
    desc='四组对照：A现有特征＋相对P10；B增强特征＋相对P10；C现有特征＋直接asinh P10；D增强特征＋直接asinh P10。平均照度都使用log1p，A/C完全共用平均照度预测，B/D也完全共用。'
    feat='增强特征仅增加Glass_Area_m2/Room_Area_m2、Window_Count、Window_Area_Nominal_m2/Window_Count。原始13项输入编码为15列，增强后为16项输入、18列编码。平均每窗面积不是窗户高度或位置；本轮没有提取尚缺的房间进深等几何信息。'
    validation='外层每次6个房间训练、1个房间评分。内层3折，每次4个房间训练、2个房间验证；每个模型家族沿用v3的两个参数候选，以内层6个验证房间的平均原单位MAE选参数。全部选择固定后才统一计算外层分数。'
    relation='相对P10：预测P10/(平均照度+1 lux)，推断时乘以该候选自己的预测平均照度+1 lux；直接P10：拟合asinh(P10/1 lux)，用sinh还原并截断负值。1 lux尺度事先固定，未用留出房间标签估计。两种方式推断时都不使用真实平均照度。'
    flat=summary[['method','target','mae_mean','mae_max','rmse_mean','r2_median','asinh_mae_mean']].rename(columns={'method':'方案','target':'目标','mae_mean':'平均MAE lux','mae_max':'最差房间MAE lux','rmse_mean':'平均RMSE lux','r2_median':'房间R²中位数','asinh_mae_mean':'平均asinh误差'})
    gateview=gates[['method','target','macro_mae_reduction_fraction','rooms_improved','rooms_tied','rooms_worse','worst_room_mae_change_lux','passes_development_rule']].copy()
    gateview['macro_mae_reduction_fraction']*=100
    gateview=gateview.rename(columns={'method':'方案','target':'目标','macro_mae_reduction_fraction':'平均MAE下降%','rooms_improved':'改善','rooms_tied':'持平','rooms_worse':'变差','worst_room_mae_change_lux':'最差MAE增加lux','passes_development_rule':'通过筛选'})
    links=['summary_metrics.csv','fold_metrics.csv','promotion_criteria.csv','paired_room_changes.csv','room_error_details.csv','feature_support.csv','selections.csv','verification.json','baseline_reproduction.json','config_snapshot.json','Cross-room-features-P10.canvas.tsx']
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第五轮：跨房间特征与P10预测</title>
<style>body{font:16px/1.65 "Microsoft YaHei",sans-serif;color:#24323b;background:#f8fafb;margin:0}main{max-width:1200px;margin:auto;padding:32px}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.lead{font-size:20px;font-weight:600}p{max-width:1050px}.meta{color:#61717e;font-size:14px}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:13px;width:100%;background:white}th,td{text-align:right;padding:8px 10px;border-bottom:1px solid #dde3e8}th:first-child,td:first-child{text-align:left}th{background:#edf2f5}img{width:100%;height:auto}select{padding:8px;margin:0 16px 16px 0}a{color:#1e6288}li{margin:6px 0}.note{border-left:3px solid #337b7b;padding:8px 16px;background:#eef6f6}</style>
<main><h1>第五轮：跨房间特征与P10预测</h1>'''
    doc+=f'<p class="lead">{headline}</p><p class="meta">7个开发房间 · 122,640条记录 · {sum(run["fit_counts"].values())}次模型拟合 · 3个TEST房间封存</p>'
    doc+='<ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in findings)+'</ul>'
    doc+='<h2>本轮改了什么</h2>'+''.join('<p>'+html.escape(s)+'</p>' for s in [desc,feat,relation,validation])
    doc+='<p class="note">'+html.escape(rule)+'</p><h2>房间平均误差</h2><img src="figures/macro_MAE.png" alt="各算法与四组方案的房间平均MAE，单位lux"><div class="scroll">'+table(flat)+'</div>'
    doc+='<p class="meta">MAE/RMSE越低越好，单位lux；R²不是准确率百分比。asinh误差使用固定1 lux尺度，仅用于比较量级差异。Frozen_HistGB为之前冻结的参考，筛选条件相对同算法A组计算。</p>'
    doc+='<h2>是否达到预先约定的改善</h2><div class="scroll">'+table(gateview)+'</div><p>平均照度C与A完全相同；B与D完全相同，不能把重复结果当成两份独立改善证据。最差房间MAE不增加，不代表每个房间都不恶化，需继续看下面的逐房间表。</p>'
    doc+='<h2>哪些房间改善或变差</h2><img src="figures/room_MAE_changes.png" alt="各房间相对同算法A组的MAE下降值，正值表示改善"><label>指标 <select id="target"></select></label><label>方案 <select id="method"></select></label><div class="scroll"><table><thead><tr><th>房间</th><th>真实均值 lux</th><th>预测均值 lux</th><th>MAE lux</th><th>偏差 lux</th><th>R²</th><th>asinh误差</th></tr></thead><tbody id="roomrows"></tbody></table></div>'
    doc+='<h2>模型能否看到相似房间</h2><p>feature_support.csv按每一外层折记录各静态特征的训练范围及留出值。BM223留出后，训练集没有＋X朝向样本。范围检查是覆盖提示，不是校准过的置信度；范围内也不等于有相似几何房间。</p>'
    doc+='<h2>验证与适用范围</h2><p>'+html.escape(limitation)+'</p>'
    doc+=f'<p>已独立重算{verify["outer_room_target_metrics_recomputed"]}条外层指标、{verify["inner_room_metrics_recomputed"]}条内层房间指标与{verify["inner_selections_recomputed"]}次选择；复核全部{verify["training_target_hashes_reconstructed"]}次拟合标签哈希，重载{verify["outer_bundles_reloaded"]}个模型包（{verify["outer_estimators_reloaded"]}个估计器）。A组与v3的预测逐项一致；TEST预测和评分均为0。</p>'
    doc+='<p>原始文件及历史参考哈希保持不变。没有自动替换旧模型。EPW身份/时间对齐、地理北旋转、百叶角度定义和传感器网格来源仍未核实。</p><p class="meta">图和数值已验证；本轮未进行浏览器交互验证或原生Canvas编译验证。</p>'
    doc+='<h2>数据与配置</h2><ul>'+''.join(f'<li><a href="{p}">{p}</a></li>' for p in links)+'</ul>'
    payload=json.dumps({'metrics':data['metrics'],'details':data['details'],'targets':TARGET_LABELS},ensure_ascii=False).replace('</','<\\/')
    doc+='''<script>const d=PAYLOAD;const t=document.querySelector('#target'),m=document.querySelector('#method');for(const [k,v] of Object.entries(d.targets))t.add(new Option(v,k));for(const v of [...new Set(d.metrics.map(x=>x.method))])m.add(new Option(v,v));m.value='A_RandomForest';t.value='E_P10_lux';const fmt=x=>x==null?'—':Number(x).toFixed(4);function render(){const rows=d.metrics.filter(x=>x.target===t.value&&x.method===m.value).map(r=>{const q=d.details.find(x=>x.room===r.room&&x.method===r.method&&x.target===r.target);return '<tr>'+[r.room,fmt(q.actual_mean_lux),fmt(q.predicted_mean_lux),fmt(r.mae),fmt(r.bias),fmt(r.r2),fmt(r.asinh_mae)].map(x=>'<td>'+x+'</td>').join('')+'</tr>';});document.querySelector('#roomrows').innerHTML=rows.join('');}t.onchange=m.onchange=render;render();</script></main></html>'''.replace('PAYLOAD',payload)
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    canvas_data={k:data[k] for k in ['headline','findings','rule','limitation','summary','metrics','details','criteria','rooms','targets','profiles']}
    canvas='''import {H1,H2,Text,Stack,Row,Table,Select,useState,useHostTheme} from "cursor/canvas";
const d=PAYLOAD;
const f=(x:number|null,n=3)=>x==null?"不适用":x.toFixed(n);
export default function CrossRoomV5(){const theme=useHostTheme();const[target,setTarget]=useState("E_P10_lux");const[method,setMethod]=useState("A_RandomForest");
const methods=Array.from(new Set(d.metrics.map(r=>r.method)));
return <Stack gap={18} style={{padding:24,maxWidth:1320,color:theme.text.primary}}>
<H1>第五轮：跨房间特征与P10预测</H1><Text weight="semibold">{d.headline}</Text>
<Text>四组对照：A现有特征＋相对P10；B增强特征＋相对P10；C现有特征＋直接asinh P10；D增强特征＋直接asinh P10。平均照度始终log1p，A=C、B=D。</Text>
<Text>增强特征：玻璃面积/地板面积、窗户数量、名义窗面积/窗户数量。两个算法均沿用v3的两个参数候选，在外层6个训练房间内部做3折分组选择。外层房间只评分。</Text>
{d.findings.map((s,i)=><Text key={i}>{s}</Text>)}
<H2>七个房间等权平均误差</H2><Select value={target} onChange={setTarget} options={Object.entries(d.targets).map(([value,label])=>({value,label}))}/>
<Table headers={["方案","平均MAE lux","最差房间MAE lux","平均RMSE lux","房间R²中位数","平均asinh误差"]} rows={d.summary.filter(r=>r.target===target).map(r=>[r.method,f(r.mae_mean),f(r.mae_max),f(r.rmse_mean),f(r.r2_median),f(r.asinh_mae_mean)])}/>
<Text size="small" tone="secondary">MAE/RMSE单位lux。R²不是准确率。asinh误差使用固定1 lux尺度。平均照度A/C、B/D是相同预测，不能重复计为独立证据。</Text>
<H2>相对同算法A组的筛选结果</H2><Text>{d.rule}</Text>
<Table headers={["方案","平均MAE下降%","改善 / 持平 / 变差","最差MAE增加lux","通过筛选"]} rows={d.criteria.filter(r=>r.target===target).map(r=>[r.method,f(r.macro_mae_reduction_fraction*100,2),r.rooms_improved+" / "+r.rooms_tied+" / "+r.rooms_worse,f(r.worst_room_mae_change_lux),r.passes_development_rule?"是":"否"])}/>
<H2>逐房间误差</H2><Select value={method} onChange={setMethod} options={methods.map(value=>({value,label:value}))}/>
<Table headers={["房间","真实均值lux","预测均值lux","MAE lux","偏差lux","R²","asinh误差"]} rows={d.metrics.filter(r=>r.target===target&&r.method===method).map(r=>{const q=d.details.find(x=>x.room===r.room&&x.target===target&&x.method===method)!;return [r.room,f(q.actual_mean_lux),f(q.predicted_mean_lux),f(r.mae),f(r.bias),f(r.r2),f(r.asinh_mae)];})}/>
<Text>偏差为预测减真实；正值表示平均高估。BM223留出后，训练集没有＋X朝向样本。范围检查不是置信度估计。</Text>
<H2>当前结论的边界</H2><Text>{d.limitation}</Text><Text>没有自动替换旧模型。当前报告是开发比较，不能宣称对全部476房间可靠。EPW/时间对齐、北向、百叶定义和传感器来源仍未确认。</Text>
<Text size="small" tone="secondary">来源：Project_2026/reports/experiment_v5；模拟年度Hour 7–18，四个角度。原始数据及历史参考哈希不变；TEST预测/评分为0。详见同目录verification.json。</Text>
</Stack>}
'''.replace('PAYLOAD',json.dumps(canvas_data,ensure_ascii=False,allow_nan=False))
    (OUT/'Cross-room-features-P10.canvas.tsx').write_text(canvas,encoding='utf-8')
    reportmd='# 跨房间特征与P10实验 v5\n\n## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n- Mode: run + validate\n- Verification: saved predictions/model reloads verified; no independent TEST\n\n'+headline+'\n\n'+'\n\n'.join(findings+[desc,feat,relation,validation,rule,limitation])+'\n'
    (OUT/'summary.md').write_text(reportmd,encoding='utf-8')
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.refs=[];self.ids=[];self.scripts=[];self.active=False
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a:self.ids.append(a['id'])
            if tag in ['a','img']:self.refs.append(a.get('href') or a.get('src'))
            if tag=='script':self.active=True
        def handle_endtag(self,tag):
            if tag=='script':self.active=False
        def handle_data(self,data):
            if self.active:self.scripts.append(data)
    parser=Parser();parser.feed(doc);assert len(parser.ids)==len(set(parser.ids)) and len(parser.scripts)==1
    assert all((OUT/p).is_file() for p in parser.refs)
    script=ROOT/'.cache/report_v5_inline.js';script.write_text(parser.scripts[0],encoding='utf-8')
    result=subprocess.run(['C:/Program Files/nodejs/node.exe','--check',str(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    T.dump(OUT/'presentation_verification.json',{'HTML_links_checked':len(parser.refs),'HTML_duplicate_ids':0,'inline_JavaScript_syntax':'passed',
           'browser_interaction_validation':'not performed; local browser use previously disabled','canvas_native_compilation_performed':False,
           'figure_visual_inspection':'pending','report_script_sha256':T.sha(__file__)})
    files=['report.html','report_data.json','summary.md','Cross-room-features-P10.canvas.tsx','figures/macro_MAE.png','figures/room_MAE_changes.png']
    T.dump(OUT/'report_build_manifest.json',{'report_script_sha256':T.sha(__file__),'artifacts':{p:T.sha(OUT/p) for p in files}})
    print(headline)
    for s in findings:print(s)
    print('HTML, two figures, canvas, summary and static checks completed.')

if __name__=='__main__':main()
