"""Create the focused diagnostic canvas from verified local results."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/diagnosis_v1'
d=json.loads((OUT/'report_data.json').read_text())
payload={k:d[k] for k in ['focus_errors','focus_ablations','cv_summary','all_room_cv','known_room']}
template='''import { H1,H2,Text,Stack,Row,Select,Table,useState,useHostTheme } from "cursor/canvas";
const data=__DATA__;
const names:Record<string,string>={Original_HistGB_B:"原 HistGB",HistGB_log1p:"HistGB + log1p",HistGB_absolute:"HistGB + 绝对误差损失",HistGB_add_floor_WFR:"加入楼层和窗地比",HistGB_without_obstruction:"移除遮挡",ExtraTrees_B:"ExtraTrees"};
const order=Object.keys(names);
const fmt=(v:number,n=3)=>v.toFixed(n);
export default function Diagnosis(){
 const t=useHostTheme();
 const [room,setRoom]=useState("BM223");
 const target=room==="BM223"?"E_P10_lux":"E_mean_lux";
 const e=data.focus_errors.find(r=>r.room===room)!;
 const k=data.known_room.find(r=>r.room===room)!;
 const rows=order.map(v=>{const f=data.focus_ablations.find(r=>r.room===room&&r.variant===v)!;const s=data.cv_summary.find(r=>r.target===target&&r.variant===v)!;return [names[v],fmt(f.mae),fmt(f.rmse),fmt(f.r2,2),fmt(s.mae_mean)];});
 const otherRooms=["BM223","BM432","BM441","BM522","BM608","LM608","LM703"].map(r=>[r,...order.map(v=>fmt(data.all_room_cv.find(a=>a.room===r&&a.target===target&&a.variant===v)!.mae))]);
 return <Stack gap={18} style={{padding:24,maxWidth:1150,color:t.text.primary}}>
 <H1>BM223 与 BM522 的预测失败诊断</H1>
 <Text weight="semibold">主要问题是跨房间迁移：训练样本覆盖有限，模型过度依赖面积等区分少量房间的特征。当前 CSV 未发现直接错误，但原始网格尚未核实。</Text>
 <Row gap={28} wrap>{[["开发房间","7 个"],["新增 CV 拟合","70 次"],["时间诊断拟合","2 次"],["TEST 预测 / 评分","0 / 0"]].map(([a,b])=><div key={a}><Text size="small" tone="secondary">{a}</Text><Text weight="semibold">{b}</Text></div>)}</Row>
 <Row gap={12}><Text>焦点任务</Text><Select value={room} onChange={setRoom} options={[{value:"BM223",label:"BM223 的 P10"},{value:"BM522",label:"BM522 的平均照度"}]} /></Row>
 <H2>原模型的偏差</H2>
 <Table headers={["房间","真实均值 lux","预测均值 lux","训练 MAE lux","留出房间 MAE lux","R²"]} rows={[[room,fmt(e.true_mean),fmt(e.predicted_mean),fmt(e.training_mae),fmt(e.validation_mae),fmt(e.r2,2)]]}/>
 {room==="BM223"?<Text>BM223 是唯一 +X 训练房间。留出它后，其他 6 个房间 X 朝向全为 0；冻结模型对输入 X 从 1 改为 0 完全无反应。原预测与 BM432 同角度同时间的真实 P10 相关系数约 0.9945，预测幅度明显借用了另一房间的响应。</Text>:<Text>BM522 与 BM432 面积接近、玻璃面积相同，但模型朝向相反。预测与 BM432 的真实均照度相关系数约 0.9921。诊断时把面积从 6.443 改到 6.595 m²，预测不变；改到 6.659 m²，平均预测绝对变化约 282.61 lux，显示模型对面积阈值依赖过强。</Text>}
 <Text size="small" tone="secondary">输入替换仅用于模型敏感性检查，不能通过修改真实面积来修复模型；相关性也不证明物理因果。</Text>
 <H2>控制实验 原单位评价</H2>
 <Table headers={["方法","焦点 MAE lux","焦点 RMSE lux","焦点 R²","7 房间平均 MAE lux"]} rows={rows}/>
 <Text>log1p 能缓解 BM522 的均照度误差，但尚未解决迁移问题。ExtraTrees 降低 BM223 的 P10 误差，却显著损害其他低 P10 房间，不能只按总平均分替换。加楼层与窗地比基本无帮助。</Text>
 <H2>查看其他房间是否付出代价</H2>
 <Text size="small" tone="secondary">下表为 {target} 的逐房间 MAE，单位 lux，越低越好。所有行都是留出房间预测。</Text>
 <Table headers={["房间",...order.map(v=>names[v])]} rows={otherRooms}/>
 <H2>已知房间时间诊断</H2>
 <Table headers={["评价时刻","原跨房间 MAE lux","目标房间其他月训练 MAE lux","已知房间 R²"]} rows={[["3 / 6 / 9 / 12 月",fmt(k.unseen_room_baseline_mae_same_times),fmt(k.known_room_mae),fmt(k.known_room_r2,3)]]}/>
 <Text>该诊断使用目标房间另外 8 个完整月份的标签，表明模型能学习该房间的时间变化。这是不同任务，不能作为未见房间泛化成绩，也尚未证明少量校准样本足够。</Text>
 <H2>改进决定</H2>
 <Text>先核对 BM223、BM432、BM522 原始逐点 lux、传感器归属、工作面和网格，再补充房间进深、宽高、窗位置、透射率等能表达采光分布的特征。若追加模拟，优先补 +X 方向和与 BM522 相似的小房间，保持三个 TEST 独立。平均照度可继续研究 log1p；P10 先解决数据语义和覆盖，避免只加模型复杂度。</Text>
 <Text size="small" tone="secondary">来源：Project_2026/reports/diagnosis_v1。范围：7 个 TRAIN 房间、模拟年度 Hour=7..18、四个角度。诊断是在观察开发失败后设计的；所有 70 条新 CV 指标与 72 个新模型文件已复核。原始网格和 EPW 对应关系仍待核实。</Text>
 </Stack>;
}
'''
(OUT/'room-diagnosis.canvas.tsx').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)),encoding='utf-8')
print('Saved diagnostic canvas with verified values.')
