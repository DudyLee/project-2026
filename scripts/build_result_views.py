"""Build self-contained, offline views from verified experiment results."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'experiment_v1'


def main():
    d=json.loads((OUT/'report_data.json').read_text())
    compact={
        'summary':d['summary'], 'weather':d['weather'],
        'folds':[{k:r[k] for k in ['feature_set','model','target','validation_room','mae','rmse','r2','clipped_fraction','raw_mae']} for r in d['folds']],
    }
    data=json.dumps(compact,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    canvas='''import { H1, H2, Text, Stack, Row, Table, Select, useState, useHostTheme } from "cursor/canvas";
const data=__DATA__;
const targets=["E_mean_lux","E_P10_lux","F_300_fraction","F_UDI_fraction","PV_AC_kWh"];
const A="A_bim_time_angle", B="B_add_solar_weather";
function unit(t:string){return t.startsWith("F_")?"百分点":t==="PV_AC_kWh"?"kWh/小时记录":"lux";}
function scale(t:string){return t.startsWith("F_")?100:1;}
function fmt(v:number|null,n=3){return v===null?"未定义":v.toFixed(n);}
export default function ExperimentResults(){
  const theme=useHostTheme();
  const [model,setModel]=useState("HistGB");
  const [target,setTarget]=useState("PV_AC_kWh");
  const s=data.summary.filter(r=>r.model===model);
  const overview=targets.map(t=>{
    const a=s.find(r=>r.target===t&&r.feature_set===A)!;
    const b=s.find(r=>r.target===t&&r.feature_set===B)!;
    const c=data.weather.find(r=>r.target===t&&r.model===model)!;
    return [t,fmt(a.mae_mean*scale(t)),fmt(b.mae_mean*scale(t)),unit(t),fmt(c.mae_reduction_pct,1)+"%",c.rooms_improved+" / 7"];
  });
  const rooms=data.folds.filter(r=>r.model===model&&r.target===target&&r.feature_set===A).map(a=>{
    const b=data.folds.find(r=>r.model===model&&r.target===target&&r.feature_set===B&&r.validation_room===a.validation_room)!;
    return [a.validation_room,fmt(a.mae*scale(target)),fmt(b.mae*scale(target)),fmt(100*(a.mae-b.mae)/a.mae,1)+"%",fmt(b.r2,2),fmt(b.clipped_fraction*100,1)+"%"];
  });
  return <Stack gap={20} style={{padding:24,maxWidth:1100,color:theme.text.primary}}>
    <H1>Tower A PVSD 首轮建模结果</H1>
    <Text weight="semibold">含天气的梯度提升树降低了 PV 平均 MAE，采光跨房间泛化仍不稳定。</Text>
    <Row gap={30} wrap>
      {[['TRAIN 数据','122,640 条'],['验证方式','7 折留一房间'],['独立拟合','140 次'],['TEST 预测 / 评分','0 / 0']].map(([k,v])=><div key={k}><Text tone="secondary" size="small">{k}</Text><Text weight="semibold">{v}</Text></div>)}
    </Row>
    <Text tone="secondary">8 和 13 为原始输入字段数；日期和方位角编码后分别为 9 和 15 列。五项性能分别回归。所有结果来自开发交叉验证，尚未测试最终三个房间。</Text>
    <Row gap={12}><Text>查看模型</Text><Select value={model} onChange={setModel} options={[{value:"HistGB",label:"梯度提升树 HistGB"},{value:"Ridge",label:"岭回归 Ridge"}]} /></Row>
    <H2>加入天气前后 每个房间 MAE 的平均值</H2>
    <Table headers={["指标","8 输入 MAE","13 输入 MAE","单位","MAE 降低","改善房间数"]} rows={overview} />
    <Text size="small" tone="secondary">MAE 越低越好，正的降低百分比表示改善。对 7 个房间等权平均，比例误差换算为百分点。主结果对负预测截为 0，比例截为 0–1；完整 CSV 同时保留未截断结果。</Text>
    <H2>逐房间表现</H2>
    <Row gap={12}><Text>预测指标</Text><Select value={target} onChange={setTarget} options={targets.map(t=>({value:t,label:t}))} /><Text tone="secondary">MAE 单位：{unit(target)}</Text></Row>
    <Table headers={["留出房间","8 输入 MAE","13 输入 MAE","MAE 降低","13 输入 R²","预测截断比例"]} rows={rooms} />
    <Text size="small" tone="secondary">R² 为负表示该房间上的平方误差比其均值参考更大。P10 的部分房间真实值变化很小，会放大负 R²，因此须同时阅读绝对误差。</Text>
    <H2>解释与下一步</H2>
    <Text>梯度提升树加入天气后：PV MAE 由 0.01858 降至 0.01261 kWh，7 个房间均改善；平均照度 MAE 降低约 5%，但其平均 RMSE 反而上升。不能仅凭 MAE 降低就判定模型可部署。</Text>
    <Text>先检查 BM223 的 P10 和 BM522 的平均照度迁移误差，结合房间几何、窗口位置和模拟网格解释差异，再设计下一轮实验。保持最终三个 TEST 房间独立。EPW 来源和时间匹配仍待确认。</Text>
    <Text size="small" tone="secondary">来源：Project_2026/reports/experiment_v1 的逐折指标与保存预测。范围：模拟年度全部 Hour=7..18、四个角度。代码与配置、依赖锁定、原始数据校验值均已保存。此视图的所有数值来自本次实际运行。</Text>
  </Stack>;
}
'''.replace('__DATA__',data)
    (OUT/'experiment-results.canvas.tsx').write_text(canvas,encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tower A PVSD 首轮实验</title>
<style>:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#20252a;background:#f7f8f9}body{max-width:1180px;margin:36px auto;padding:0 24px 50px}h1{font-size:26px;margin-bottom:12px}h2{font-size:19px;margin-top:30px}.lead{font-size:18px;line-height:1.65}.stats{display:flex;gap:36px;flex-wrap:wrap;padding:20px 0;border-bottom:1px solid #d5dadd}.stats span{display:block;color:#64717a;font-size:13px}.stats strong{display:block;font-size:20px;margin-top:6px}p{line-height:1.75}.muted{color:#59656d;font-size:13px}select{font:inherit;padding:7px 10px;border:1px solid #a9b4ba;border-radius:4px;background:white}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{padding:12px;text-align:right;border-bottom:1px solid #dde2e5}th:first-child,td:first-child{text-align:left}th{color:#4c5961;background:#edf1f3}.tablewrap{overflow:auto}img{display:block;max-width:100%;margin-top:18px}a{color:#205d80}.note{border-left:3px solid #587f95;padding:3px 16px;background:#edf2f5}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.tabs{display:flex;gap:18px;margin:22px 0}footer{border-top:1px solid #d5dadd;margin-top:32px;padding-top:14px}.improve{font-weight:600;color:#285f65}</style>
<h1>Tower A PVSD 首轮建模结果</h1><p class="lead">含天气的梯度提升树降低了 PV 平均 MAE；采光在部分未见房间上的预测仍然不稳定。</p>
<div class="stats"><div><span>TRAIN 数据</span><strong>122,640 条</strong></div><div><span>验证方式</span><strong>7 折留一房间</strong></div><div><span>独立拟合</span><strong>140 次</strong></div><div><span>TEST 预测 / 评分</span><strong>0 / 0</strong></div></div>
<p>每次使用 6 个房间训练，剩余 1 个房间验证。8 个基础输入与 13 个含天气输入分别比较，周期编码后为 9 和 15 列。五个输出分别训练；固定参数，没有调参或全 7 房间最终重训。</p>
<div class="controls"><label for="model">模型</label><select id="model"><option value="HistGB">梯度提升树 HistGB</option><option value="Ridge">岭回归 Ridge</option></select></div>
<h2>加入天气前后的平均误差</h2><div class="tablewrap"><table><thead><tr><th>指标</th><th>8 输入 MAE</th><th>13 输入 MAE</th><th>单位</th><th>MAE 降低</th><th>改善房间</th></tr></thead><tbody id="summary"></tbody></table></div>
<p class="muted">MAE 为 7 个房间各自 MAE 的等权平均，越低越好。正的降低百分比表示改善。比例误差换算成百分点。主结果截断负预测为 0、比例为 0–1；未截断结果也已保存在 CSV 和 Parquet 文件中。</p>
<h2>逐房间结果</h2><div class="controls"><label for="target">指标</label><select id="target"></select><span id="unit" class="muted"></span></div><div class="tablewrap"><table><thead><tr><th>留出房间</th><th>8 输入 MAE</th><th>13 输入 MAE</th><th>MAE 降低</th><th>13 输入 RMSE</th><th>13 输入 R²</th><th>截断比例</th></tr></thead><tbody id="rooms"></tbody></table></div>
<p class="muted">R² 为负，表示该房间的预测平方误差超过以该房间均值为参考的误差。P10 在部分房间内变化很小，会放大负 R²；需要结合 MAE 和 RMSE 判断。</p>
<h2>目前可以得出的结论</h2><div class="note"><p>梯度提升树的 PV MAE 从 0.01858 降至 0.01261 kWh，下降 32.1%，7 个房间全部改善。平均照度 MAE 下降 5.0%，300 lux 空间比例误差下降 5.6%，适宜照度空间比例误差下降 9.0%。</p><p>这些是开发交叉验证结果。平均照度和 P10 的平均 RMSE 没有改善；BM223 的 P10 和 BM522 的平均照度仍存在明显迁移失败。尚不能判断整栋楼预测或动态角度推荐已可靠。</p></div>
<h2>图表</h2><img src="figures/mae_comparison.png" alt="两个模型和两组特征的五项指标平均绝对误差"><img src="figures/room_mae_comparison.png" alt="梯度提升树逐房间绝对误差比较"><img src="figures/room_r2.png" alt="含天气梯度提升树的逐房间R平方">
<h2>文件与下一步</h2><p>下一步先检查上述房间的几何、窗口位置、传感器网格及数据生成过程，再确定下一轮特征或目标处理。保持三个 TEST 房间独立，不根据它们调参。天气文件来源、小时对齐和地理北仍未确认。</p>
<div class="tabs"><a href="summary_metrics.csv">汇总指标 CSV</a><a href="fold_metrics.csv">逐折指标 CSV</a><a href="paired_room_comparison.csv">逐房间天气对照 CSV</a><a href="verification.json">复核记录</a><a href="run_manifest.json">运行记录</a></div>
<footer class="muted">来源：本目录内实际保存的逐折指标与 TRAIN-only 预测。时间范围：模拟非闰年全年 Hour=7..18，四个固定角度。原始数据未修改；EPW 匹配尚未独立验证。图表误差汇总不表示统计显著性或跨年份泛化。<br>复核通过：140 行指标独立重算，140 个模型文件重载与样本预测核对，标准化仅使用各折 6 个训练房间，原始文件 SHA-256 未变。</footer>
<script>const data=__DATA__;const targets=['E_mean_lux','E_P10_lux','F_300_fraction','F_UDI_fraction','PV_AC_kWh'];const A='A_bim_time_angle',B='B_add_solar_weather';const fmt=(v,n=3)=>v===null?'未定义':Number(v).toFixed(n);const factor=t=>t.startsWith('F_')?100:1;const units=t=>t.startsWith('F_')?'百分点':t==='PV_AC_kWh'?'kWh/小时记录':'lux';const sel=document.getElementById('target');targets.forEach(t=>{let o=document.createElement('option');o.value=t;o.textContent=t;sel.appendChild(o)});sel.value='PV_AC_kWh';function render(){const m=document.getElementById('model').value,t=sel.value;document.getElementById('summary').innerHTML=targets.map(k=>{let a=data.summary.find(r=>r.model===m&&r.target===k&&r.feature_set===A),b=data.summary.find(r=>r.model===m&&r.target===k&&r.feature_set===B),c=data.weather.find(r=>r.model===m&&r.target===k);return `<tr><td>${k}</td><td>${fmt(a.mae_mean*factor(k))}</td><td>${fmt(b.mae_mean*factor(k))}</td><td>${units(k)}</td><td class="improve">${fmt(c.mae_reduction_pct,1)}%</td><td>${c.rooms_improved}/7</td></tr>`}).join('');document.getElementById('unit').textContent='MAE / RMSE 单位：'+units(t);document.getElementById('rooms').innerHTML=data.folds.filter(r=>r.model===m&&r.target===t&&r.feature_set===A).map(a=>{let b=data.folds.find(r=>r.model===m&&r.target===t&&r.feature_set===B&&r.validation_room===a.validation_room);return `<tr><td>${a.validation_room}</td><td>${fmt(a.mae*factor(t))}</td><td>${fmt(b.mae*factor(t))}</td><td>${fmt(100*(a.mae-b.mae)/a.mae,1)}%</td><td>${fmt(b.rmse*factor(t))}</td><td>${fmt(b.r2,2)}</td><td>${fmt(b.clipped_fraction*100,1)}%</td></tr>`}).join('')}document.getElementById('model').addEventListener('change',render);sel.addEventListener('change',render);render();</script></html>'''.replace('__DATA__',data)
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print('Saved report.html and experiment-results.canvas.tsx with embedded verified results.')


if __name__=='__main__':main()
