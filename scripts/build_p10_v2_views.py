"""Self-contained HTML and canvas views for verified P10 v2 results."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/experiment_v2'
data=json.loads((OUT/'report_data.json').read_text())
fm=pd.read_csv(OUT/'fold_metrics.csv')
# Preserve missing reference-only fields as null, never invented zero metrics.
data['folds']=fm.astype(object).where(pd.notna(fm),None).to_dict('records')
(OUT/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
encoded=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)
summary_by_method={r['method']:r for r in data['summary']}
selection={
    'status':'development_candidate_only',
    'mean_candidate':'HistGB_log1p_E_mean_lux',
    'P10_candidate':'Relative_P10',
    'weighted_log1p_selected':False,
    'reason':'Relative P10 has the lowest mean and median room MAE among the four frozen/predeclared methods; additional scale weighting yields little gain over ordinary log1p.',
    'relative_mean_room_MAE_lux':summary_by_method['Relative_P10']['mae_mean'],
    'relative_MAE_reduction_vs_original_pct':summary_by_method['Relative_P10']['mae_reduction_vs_Original_HistGB_B_pct'],
    'relative_rooms_improved_vs_original':summary_by_method['Relative_P10']['rooms_improved_vs_Original_HistGB_B'],
    'weighted_additional_MAE_reduction_vs_log1p_pct':summary_by_method['Weighted_log1p_P10']['mae_reduction_vs_Unweighted_log1p_P10_pct'],
    'BM223_share_of_total_MAE_reduction':data['bm223_share_of_total_MAE_reduction'],
    'unresolved':['BM223 remains far from its small true P10 scale','BM432 P10 remains severely underestimated','Relative reconstruction propagates mean-stage errors; BM522 shows partial error cancellation'],
    'test_rooms_used':False,
    'deployment_ready':False,
    'final_seven_room_refit_performed':False,
    'scope':'Selected using development CV after previous failures; no independent final-test performance claim.'
}
(OUT/'candidate_decision.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
canvas='''import {H1,H2,Text,Stack,Row,Select,Table,useState,useHostTheme} from "cursor/canvas";
const data=__DATA__;
const order=["Original_HistGB_B","Unweighted_log1p_P10","Weighted_log1p_P10","Relative_P10"];
const names:Record<string,string>={Original_HistGB_B:"原始 P10 回归",Unweighted_log1p_P10:"普通 log1p",Weighted_log1p_P10:"尺度加权 log1p",Relative_P10:"相对照度还原"};
const fmt=(v:number|null|undefined,n=3)=>v==null?"未计算":v.toFixed(n);
export default function P10ExperimentV2(){
 const theme=useHostTheme();const [room,setRoom]=useState("BM223");
 const rows=order.map(m=>{const r=data.summary.find(s=>s.method===m)!;return [names[m],fmt(r.mae_mean),fmt(r.mae_median),fmt(r.rmse_mean),fmt(r.mae_max),fmt(r.mae_reduction_vs_Original_HistGB_B_pct,1)+"%",r.rooms_improved_vs_Original_HistGB_B+" / 7"];});
 const roomRows=["BM223","BM432","BM441","BM522","BM608","LM608","LM703"].map(r=>[r,...order.map(m=>fmt(data.folds.find(f=>f.room===r&&f.method===m)!.mae))]);
 const focus=order.map(m=>{const r=data.folds.find(f=>f.room===room&&f.method===m)!;return [names[m],fmt(r.mae),fmt(r.rmse),fmt(r.r2,2),fmt(r.bias)];});
 const mean=data.mean_folds.find(r=>r.room===room)!;const dec=data.decomposition.find(r=>r.room===room)!;
 return <Stack gap={18} style={{padding:24,maxWidth:1200,color:theme.text.primary}}>
 <H1>P10 目标处理 第二轮实验</H1>
 <Text weight="semibold">相对照度方案更有希望：平均 MAE 下降约 25.7%，但 BM432 仍未解决。尺度加权相对普通 log1p 的额外收益仅约 0.35%。</Text>
 <Row gap={28} wrap>{[["TRAIN 留一房间","7 折"],["新拟合 P10 模型","14 个"],["复用均照度模型","7 个"],["TEST 预测 / 评分","0 / 0"]].map(([k,v])=><div key={k}><Text tone="secondary" size="small">{k}</Text><Text weight="semibold">{v}</Text></div>)}</Row>
 <Text>沿用 13 个原始输入，周期编码后 15 列，梯度提升树参数固定。平均照度使用已保存的 log1p 模型；所有模型均排除当前留出房间。相对照度模型只从输入预测 r，再用预测的平均照度还原 P10。</Text>
 <H2>7 个房间等权汇总 P10 单位 lux</H2>
 <Table headers={["方法","平均 MAE","MAE 中位数","平均 RMSE","最差房间 MAE","较原始 MAE 降低","改善房间数"]} rows={rows}/>
 <Text size="small" tone="secondary">这是房间指标的平均或中位数。R² 和绝对误差均保留原始定义，不做最终测试集选择。约 88% 的总 MAE 降低来自 BM223，平均改善不意味着所有房间已可靠。</Text>
 <H2>逐房间 P10 MAE</H2><Table headers={["留出房间",...order.map(m=>names[m])]} rows={roomRows}/>
 <Row gap={12}><Text>查看焦点房间</Text><Select value={room} onChange={setRoom} options={["BM223","BM432","BM441","BM522","BM608","LM608","LM703"].map(r=>({value:r,label:r}))}/></Row>
 <Table headers={["方法","MAE lux","RMSE lux","R²","偏差 lux"]} rows={focus}/>
 <H2>平均照度与误差传递</H2>
 <Text>平均照度 log1p 的 7 房间平均 MAE 为 {fmt(data.mean_summary.mae_mean)} lux。当前房间 {room} 的均照度 MAE 为 {fmt(mean.mae)} lux。</Text>
 <Table headers={["相对照度误差项绝对均值 lux","均照度传递项绝对均值 lux","实际 P10 MAE lux"]} rows={[[fmt(dec.ratio_estimation_term_mean_abs_lux),fmt(dec.mean_propagation_term_mean_abs_lux),fmt(dec.actual_mae_lux)]]}/>
 <Text size="small" tone="secondary">误差分解在预测后使用真实标签进行诊断；真实均照度没有用于预测。两项可能相互抵消，绝对均值不能相加为 MAE。BM522 中就存在这种抵消。</Text>
 <H2>本轮决定</H2>
 <Text>保留“平均照度 log1p + 相对照度 P10”作为下一步的主候选，不全局采用尺度加权。BM223 的 P10 MAE 虽降到约 22.49 lux，仍远大于真实均值；BM432 约 53.09 lux 的误差仍是主要瓶颈。当前模型尚不支持可靠的整栋楼推荐。</Text>
 <Text>本轮没有调参、追加模拟或使用最终 TEST。后续可在现有数据内继续研究相对照度的房间迁移与均照度误差传递，避免为单个房间设置标签导出的特例规则。</Text>
 <Text size="small" tone="secondary">来源：Project_2026/reports/experiment_v2。范围：7 个 TRAIN 房间、模拟年度全部 Hour=7..18、四个角度。28 条 P10 指标和 7 条均照度指标已重算；14 个新模型、7 个复用模型及各折权重已复核。EPW 对齐和原始网格的来源限制仍在。</Text>
 </Stack>;
}
'''.replace('__DATA__',encoded)
(OUT/'p10-target-handling.canvas.tsx').write_text(canvas,encoding='utf-8')
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P10 第二轮实验</title><style>body{font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#263039;background:#f9fafb;max-width:1180px;margin:32px auto;padding:0 24px 45px}h1{font-size:25px}h2{font-size:19px;margin-top:32px}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #d9e0e5;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf1f4}.scroll{overflow:auto}.stats{display:flex;gap:36px;flex-wrap:wrap;border-bottom:1px solid #d9e0e5;padding:18px 0}.stats small{display:block;color:#60717b}.stats strong{display:block;font-size:20px}.lead{font-size:18px}.note{padding:5px 18px;border-left:3px solid #547e94;background:#edf3f6}.muted{color:#61707a;font-size:13px}select{font:inherit;padding:6px 12px;border:1px solid #a8b4bd;background:white}img{max-width:100%;display:block;margin-top:20px}a{color:#285d79}.links{display:flex;flex-wrap:wrap;gap:20px}code{font-size:13px}footer{border-top:1px solid #d9e0e5;padding-top:14px;margin-top:28px}</style>
<h1>P10 目标处理 第二轮实验</h1><p class="lead">相对照度方案的平均 MAE 下降约 25.7%，优于尺度加权方案；但 BM432 仍未解决，BM223 也仍不够准确。</p>
<div class="stats"><div><small>TRAIN 留一房间</small><strong>7 折</strong></div><div><small>新拟合 P10 模型</small><strong>14 个</strong></div><div><small>复用均照度 log1p 模型</small><strong>7 个</strong></div><div><small>TEST 预测 / 评分</small><strong>0 / 0</strong></div></div>
<h2>本轮运行了什么</h2><p>固定 13 个原始输入和 HistGB 参数，周期编码后为 15 列。原始 P10 与普通 log1p 为冻结参考；新增尺度加权 log1p 和相对照度两种方法。平均照度复用前次相同折的 log1p 模型，逐一核对训练房间与模型哈希，不重复训练。</p>
<p><strong>尺度加权：</strong>各折仅根据 6 个拟合房间计算 <code>w ∝ 1/sqrt(1+该训练房间的平均 P10)</code>，将训练样本权重均值归一化为 1。留出房间的 P10 尺度不参与权重。</p>
<p><strong>相对照度：</strong>训练目标为 <code>r=P10/(E_mean+1 lux)</code>；预测时使用 <code>P10_hat=max(r_hat,0)×(E_mean_hat+1 lux)</code>。真实留出均照度没有传入模型或预测重建。r 不是 F_UDI 一类的空间比例，没有强制截为 0–1。</p>
<h2>P10 的 7 房间等权汇总</h2><div class="scroll"><table><thead><tr><th>方法</th><th>平均 MAE lux</th><th>MAE 中位数 lux</th><th>平均 RMSE lux</th><th>最差房间 MAE lux</th><th>较原始 MAE 降低</th><th>改善房间数</th></tr></thead><tbody id="summary"></tbody></table></div>
<p class="muted">主要指标均为原尺度 lux。房间等权平均，保留逐房间代价；开发折上的改进不是最终独立测试结果。约 <span id="share"></span> 的总体 MAE 降低来自 BM223。</p>
<img src="figures/p10_summary.png" alt="四种P10处理方案的房间平均与中位数误差">
<h2>逐房间 P10 MAE</h2><div class="scroll"><table><thead><tr><th>留出房间</th><th>原始 P10</th><th>普通 log1p</th><th>尺度加权 log1p</th><th>相对照度还原</th></tr></thead><tbody id="room-table"></tbody></table></div>
<img src="figures/p10_room_comparison.png" alt="各房间P10误差与处理方法的比较">
<h2>查看单个房间</h2><label for="room">留出房间</label> <select id="room"></select><div class="scroll"><table><thead><tr><th>方法</th><th>MAE lux</th><th>RMSE lux</th><th>R²</th><th>偏差 lux</th></tr></thead><tbody id="focus"></tbody></table></div>
<h2>均照度与相对照度误差传递</h2><p>沿用的平均照度 log1p 模型平均 MAE 为 <span id="mean-mae"></span> lux。下表为预测完成后的误差分解：<code>P10_hat−P10 = (r_hat−r_true)×(E_mean_true+1) + r_hat×(E_mean_hat−E_mean_true)</code>。</p>
<p class="muted">分解时用真实标签只是为了分析误差，未用于模型预测。两项有正负且可以抵消，绝对项均值不能相加为 MAE。</p>
<div class="scroll"><table><thead><tr><th>房间</th><th>比值项绝对均值 lux</th><th>均照度传递项绝对均值 lux</th><th>实际 P10 MAE lux</th></tr></thead><tbody id="decomposition"></tbody></table></div>
<img src="figures/ratio_error_components.png" alt="相对照度模型误差项存在正负抵消">
<h2>如何解释本轮结果</h2><div class="note"><p>尺度加权只比普通 log1p 的平均 MAE 额外改善约 <strong>0.35%</strong>，BM223 基本不变，因此不作为首选。</p><p>相对照度的平均 MAE 为 <strong>14.59 lux</strong>，较原始模型下降 <strong>25.7%</strong>，较普通 log1p 下降 <strong>24.2%</strong>，7 个房间中有 6 个优于原始模型。BM441 比原始模型增加约 0.009 lux；相对于普通 log1p，BM441 和 LM608 仍有误差增加。</p><p>BM223 从 53.59 降至 <strong>22.49 lux</strong>，但仍远大于其真实均值约 0.62 lux。BM432 仍为 <strong>53.09 lux</strong>，没有实质解决。其误差主要来自比值模型，不能只靠优化均照度来修复。</p><p>BM522 的 P10 改善中存在“比值低估”与“均照度高估”的相互抵消，不能把这种抵消当作已经学对物理关系。</p></div>
<h2>本轮决定</h2><p>保留 <strong>均照度 log1p + 相对照度 P10</strong> 作为下一步主候选；不增加尺度加权的复杂度。此决定是开发阶段的候选选择，尚不支持部署或整栋楼准确性结论，也没有重训最终全 7 房间模型。</p><p>继续只用现有数据时，应重点研究相对照度的房间迁移与均照度误差传递，并报告所有 7 个房间的表现；不为 BM223 或 BM432 写使用真实输出的特例规则，不使用最终 TEST 反复试验。</p>
<div class="links"><a href="summary_metrics.csv">汇总 CSV</a><a href="fold_metrics.csv">逐折指标</a><a href="paired_room_comparison.csv">逐房间变化</a><a href="training_room_weights.csv">实际训练权重</a><a href="verification.json">复核记录</a><a href="run_manifest.json">运行记录</a></div>
<footer class="muted">来源：experiment_v2 实际保存的 TRAIN-only 预测和指标；模拟年度全部 Hour=7..18、四个固定角度。28 条 P10 指标和 7 条均照度指标已重算，14 个新模型、7 个均照度模型和各折权重已复核。原始文件 SHA-256 未变，3 个 TEST 房间没有预测、评分、调参或目标分布分析。EPW 来源、时间对应和原始网格仍有未确认事项。</footer>
<script>const data=__DATA__;const order=['Original_HistGB_B','Unweighted_log1p_P10','Weighted_log1p_P10','Relative_P10'];const names={Original_HistGB_B:'原始 P10',Unweighted_log1p_P10:'普通 log1p',Weighted_log1p_P10:'尺度加权 log1p',Relative_P10:'相对照度还原'};const rooms=['BM223','BM432','BM441','BM522','BM608','LM608','LM703'];const fmt=(v,n=3)=>v==null?'未计算':Number(v).toFixed(n);document.getElementById('summary').innerHTML=order.map(m=>{let r=data.summary.find(x=>x.method===m);return `<tr><td>${names[m]}</td><td>${fmt(r.mae_mean)}</td><td>${fmt(r.mae_median)}</td><td>${fmt(r.rmse_mean)}</td><td>${fmt(r.mae_max)}</td><td>${fmt(r.mae_reduction_vs_Original_HistGB_B_pct,1)}%</td><td>${r.rooms_improved_vs_Original_HistGB_B}/7</td></tr>`}).join('');document.getElementById('room-table').innerHTML=rooms.map(room=>`<tr><td>${room}</td>${order.map(m=>`<td>${fmt(data.folds.find(r=>r.room===room&&r.method===m).mae)}</td>`).join('')}</tr>`).join('');document.getElementById('decomposition').innerHTML=data.decomposition.map(r=>`<tr><td>${r.room}</td><td>${fmt(r.ratio_estimation_term_mean_abs_lux)}</td><td>${fmt(r.mean_propagation_term_mean_abs_lux)}</td><td>${fmt(r.actual_mae_lux)}</td></tr>`).join('');document.getElementById('share').textContent=fmt(data.bm223_share_of_total_MAE_reduction*100,1)+'%';document.getElementById('mean-mae').textContent=fmt(data.mean_summary.mae_mean,2);let select=document.getElementById('room');rooms.forEach(room=>{let o=document.createElement('option');o.value=room;o.textContent=room;select.appendChild(o)});function render(){let room=select.value;document.getElementById('focus').innerHTML=order.map(m=>{let r=data.folds.find(x=>x.room===room&&x.method===m);return `<tr><td>${names[m]}</td><td>${fmt(r.mae)}</td><td>${fmt(r.rmse)}</td><td>${fmt(r.r2,2)}</td><td>${fmt(r.bias)}</td></tr>`}).join('')}select.addEventListener('change',render);render();</script></html>'''.replace('__DATA__',encoded)
(OUT/'report.html').write_text(html,encoding='utf-8')
print('V2 offline HTML and canvas generated from verified results.')
