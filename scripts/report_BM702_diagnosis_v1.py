"""Write numerical diagnosis and register a proposed, unexecuted follow-up."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import pvsd_trial as api
import prepare_repository_snapshot as audit
OUT=ROOT/"reports/diagnosis_BM702_v1"


def read(p): return json.loads(p.read_text(encoding="utf-8"))


def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def main():
    run=read(OUT/"run_manifest.json");verification=read(OUT/"verification.json")
    assert verification["status"]=="passed"
    for relative,digest in run["input_hashes"].items():assert api.sha256(ROOT/relative)==digest,relative
    checks=pd.read_csv(OUT/"data_checks.csv")
    consistency=pd.read_csv(OUT/"daylight_consistency_checks.csv")
    strata=pd.read_csv(OUT/"BM702_stratified_errors.csv")
    native=pd.read_csv(OUT/"native_room_metrics.csv")
    focal=native[native.room=="BM702"].set_index("target")
    oracle=pd.read_csv(OUT/"P10_oracle_diagnostics.csv")
    attribution=pd.read_csv(OUT/"RF_ratio_room_attribution.csv")
    decomposition=read(OUT/"P10_decomposition.json")
    names={"E_mean_lux":"平均照度", "E_P10_lux":"P10", "F_UDI_fraction":"FUDI"}
    lines=["# BM702：平均照度、P10与FUDI误差诊断", "", "## Material Passport", "",
           "- 日期：2026-11-08。状态：ANALYZED；数值重载/代数分解/评分复核通过，原模拟网格尚不能核验。",
           "- 授权：用户要求重点诊断BM702，仅用现有数据给出下一轮方案，不追加模拟、不做可视化。",
           "- 使用7个TRAIN房间与已观察TEST房间BM702的现有数据；原始CSV读取时跳过另外两个TEST房间35,040行。没有新增模拟、模型拟合、调参、房间校准或模型替换。",
           "- 固定模型：trial 0.1.0；原数据与模型内容哈希保持不变。BM702结果已用于事后分析，不能继续当作未见最终测试。", "",
           "## 归因结论", "",
           "证据最充分的是：**训练房间缺少−X方向，当前特征与树模型的组合没有学会这一方向的日内响应；P10还叠加了相对照度分支的偏低和均值分支的日内误差。**目前没有发现CSV错行、BIM关联冲突、天气在房间间不一致或照度汇总明显违反必要约束的证据。缺少原始网格/时刻/几何来源，不能据此排除源数据问题，也不能把全部误差归因于单一因素。", "",
           "| 层面 | 直接证据 | 判断边界 |", "|---|---|---|",
           "| 数据整理 | 索引、完整性、BIM映射、天气一致性、比例计数及必要照度约束均通过 | 排除本轮可检验的明显错误；不等于原始模拟正确 |",
           "| 样本与特征 | 7个TRAIN只有3个朝向，−X没有样本；只有1个+X房间；面积/玻璃/遮挡不足以唯一描述光分布 | 朝向覆盖不足是确定事实；窗位/进深/网格影响未被量化 |",
           "| 模型响应 | BM702的X从−1改为0后，3项预测在数值精度内不变；日内高低顺序明显失配 | 直接说明当前模型不能区分这段X取值，不能证明全部误差只来自朝向 |",
           "| P10结构 | 比例低估、配套均值日内失准、两分支误差相互抵消或叠加 | 不适合仅加倍率或只替换一个分支 |", "",
           "## 1. 数据检查结果", "",
           f"共检查8个房间140,160行。{len(checks)}项内部一致性检查均未发现违反；另外逐房间核对7项照度汇总必要约束，{len(consistency)*7}项计数均为0。", "",
           "包括：每房间角度4,380小时、无缺失/重复/非有限值；Month/Day/DOY与HOY相对索引一致；房间静态属性跨小时/角度一致；逐时表和BIM表共同字段一致；同一HOY的全部天气与太阳位置值在保留房间/角度间一致。F300和FUDI乘Sensor_Count接近整数，且满足非负传感器照度应有的汇总下界。",
           "BM702有51个传感器，FUDI一个传感器对应约1.96个百分点。当前FUDI平均绝对误差27.04个百分点，约相当于13.79个传感器的空间比例差异；不是小数舍入就能解释。numpy线性P10在N=51时对应排序后的第6个点，但没有坐标/逐点照度，不能核查传感器布置是否合理。",
           "两张表来自同一导出链，彼此一致并非独立真值。EPW身份、Hour区间、北向、窗位和网格来源仍未确认。本轮没有证据支持把BM702标为坏数据后删除，也不根据预测错峰擅自平移天气数小时。", "",
           "## 2. 平均照度：日内形状预测错误", "",
           "BM702实际全年各小时平均照度在14点达到峰值；预测曲线在11点达到峰值。这里是按小时汇总的错峰，不能等同于已经发现源数据固定错了3小时。", "",
           "| 小时 | 真实均值 lux | 模型均值 lux | 偏差 lux |", "|---|---:|---:|---:|"]
    for h in [9,11,14,16]:
        r=strata[(strata.target=="E_mean_lux")&(strata.axis=="Hour")&(strata.group==str(h))].iloc[0]
        lines.append(f"| {h} | {r.true_mean:.2f} | {r.pred_mean:.2f} | {r.bias:+.2f} |")
    lines += ["", "| DNI条件（Wh/m²） | 平均照度MAE lux | 行数 | 占总绝对误差 |", "|---|---:|---:|---:|"]
    for key in ["zero","0_to_200","200_to_500","above_500"]:
        r=strata[(strata.target=="E_mean_lux")&(strata.axis=="DNI_band")&(strata.group==key)].iloc[0]
        lines.append(f"| {key} | {r.mae:.2f} | {int(r.n)} | {100*r.fraction_of_target_absolute_error:.2f}% |")
    lines += ["", "DNI=0时真实平均照度本来就较低，因此低MAE本身不能证明直射量是唯一原因；但强直射条件下的失准与上午/下午偏差反向共同指向方向相关响应迁移不足。",
              "TRAIN的X仅有0和1，BM702为−1。保留其余输入，只把X=−1换成X=0，均值和FUDI预测完全不变，P10最大差异约1.42×10⁻¹³。该探针的(0,0)不是物理朝向，只用于验证树分支行为，绝不是推荐的输入修正。", "",
              "## 3. P10：配套均值与比例都失准", "",
              "P10预测为r_hat×(配套RF平均照度+1)，配套RF均值不是独立输出的XGBoost均值。BM702真实P10均值59.74 lux，预测16.55 lux；94.07%的记录被低估。真实r=P10/(平均照度+1)的逐时均值为0.2695，预测r均值只有0.0959。", "",
              "追溯64棵RF的原始bootstrap样本和预测叶节点后，能够按训练房间精确重建比例预测：", "",
              "| 训练房间 | 对BM702比例预测的平均叶节点样本权重 |", "|---|---:|"]
    for r in attribution.sort_values("mean_leaf_weight",ascending=False).itertuples():
        lines.append(f"| {r.TRAIN_room} | {100*r.mean_leaf_weight:.2f}% |")
    lines += ["", "BM432占54.69%，其余45.31%来自平均相对P10很低的房间；具有较高相对P10、且面积/玻璃/遮挡数值相近的BM522权重为0。权重表示RF叶内训练样本对当前数值的贡献，不是物理因果重要性。",
              "下面只为诊断使用真实标签替换分支。这些oracle结果不能部署、不能当作下一版成绩：", "",
              "| 诊断方式 | P10 MAE lux | 含义 |", "|---|---:|---|"]
    meanings={"native_pair":("原模型","两分支同时预测"),"ORACLE_true_mean_only":("使用真实平均照度","比例分支仍有较大误差"),"ORACLE_true_ratio_only":("使用真实比例","均值的时序误差显现，MAE反而增加")}
    for r in oracle.itertuples():
        if r.diagnostic in meanings:
            label,note=meanings[r.diagnostic]
            lines.append(f"| {label} | {r.mae:.2f} | {note} |")
    lines += ["", "因此，不能只把比例放大或只改配套均值。相对方案有部分误差抵消；有符号分解和绝对误差不是可相加的贡献比例。需要同时改善输入表示，并以直接P10模型作有限对照。", "",
              "## 4. FUDI：房间内光分布没有成功迁移", "",
              "FUDI是独立的HistGB模型，不由平均照度或P10预测推导。BM702全年真实平均FUDI为47.83%，预测22.09%；13–17点真实均值61.19%，预测18.73%，该时段MAE42.59个百分点。四角度中，0°真实均值69.66%、预测25.22%，MAE44.65个百分点；不是仅个别角度有问题。",
              "同一HistGB在7个拟合房间上的FUDI MAE约1.33–4.00个百分点、R²约0.932–0.982。这些是训练内诊断分数，不能充当验证成绩，但说明模型能拟合已见房间，主要失败出现在跨房间迁移。均值模型本身仍有拟合残差，所以也不把全部误差简单归为过拟合。",
              "均值、P10与FUDI测量的是光分布的不同方面。只有面积、玻璃总面积、方向和年度遮挡，无法确认窗户位置、房间进深和传感器覆盖差异。BM702面积/玻璃/遮挡最接近BM432和BM522，但方向和网格不同。现有汇总数据不能唯一拆出<100与>3000 lux的空间比例，因此不能断定FUDI误差全部来自过暗或过亮。", "",
              "## 5. 下一轮方案", "",
              "详细可执行方案见[下一轮方案](../../docs/next_experiment_after_BM702.md)，**尚未开始训练**。核心是：仅用现有13项输入新增6个太阳条件与方向交互项，用当前树模型与低自由度样条＋Ridge构成2×2对照，分别检查特征和函数表示的作用；P10增加一个直接asinh对照，F300/PV暂不动。",
              "验证同时保留7折留一房间，并增加3折留一朝向组；所有预处理及参数选择在拟合房间内部完成。每个输出须满足平均MAE不增加、最差房间MAE至少下降10%、至少5/7房间不恶化，且方向压力验证不恶化。无法通过就保留原模型，不扩大参数搜索或按BM702成绩挑方案。",
              "只用现有数据可以检验这种有机制依据的改进，但不能承诺恢复未观察−X方向的全部规律。没有新方向样本和几何信息的限制仍在。BM702及另外两个已观察留出房间不加入本轮拟合；未来若查看改进后结果，必须称为复用/已观察留出评估。", "",
              "## 数值核验与文件", "",
              "原生输出重载与先前冻结测试结果一致；P10代数分解最大残差1.85×10⁻¹³，RF叶权重重建比例最大差异1.39×10⁻¹⁶。独立复核21个分层分区的加权误差和4个P10诊断评分，通过。模型和源数据哈希不变。", "",
              "- data_checks.csv / daylight_consistency_checks.csv：内部数据核对。",
              "- BM702_stratified_errors.csv：小时、月、角度、辐射和高度角分层。",
              "- orientation_branch_probe.csv：方向分支探针；非物理、非候选输入。",
              "- P10_components.parquet / P10_oracle_diagnostics.csv：配套均值与比例分解。",
              "- RF_ratio_room_attribution.csv：bootstrap叶节点训练房间归属。",
              "- native_room_metrics.csv：BM702留出结果及明确标记的TRAIN训练内诊断。",
              "- target_profiles.csv / room_profiles.csv / geometry_neighbours.csv：目标与房间特征对照。",
              "- verification.json / mechanical_verification.json：数值核验。所有文件均为CSV、JSON、Parquet或文本，没有图形。", ""]
    (OUT/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    fallacies=[
        ("Simpson / aggregation","CAUTION","Morning overprediction and afternoon underprediction are both reported; annual bias hides shape error."),
        ("Ecological fallacy","CAUTION","One focal room and seven training groups; time rows are not independent room samples."),
        ("Berkson / selection","CAUTION","BM702 selected after poor test results; diagnosis is post-holdout and representative-room sampling is limited."),
        ("Collider bias","NOT_APPLICABLE_TO_CLAIM","No causal regression adjustment; geometry distances and leaf weights are descriptive."),
        ("Base rate neglect","CHECKED","Low actual illuminance in DNI=0 subset noted; label distributions shown alongside MAE."),
        ("Regression to mean","CHECKED","No model improvement or new performance claim; oracle substitution not a deployable result."),
        ("Survivorship bias","CHECKED","All 17520 BM702 rows retained; no outlier deletion."),
        ("Look elsewhere","CAUTION","Strata are diagnostic; no model or correction chosen by best BM702 score."),
        ("Forking paths","NOTE","Exploratory analysis and resulting proposal explicitly marked; no candidate training this turn."),
        ("Correlation versus causation","CAUTION","Negative-X saturation is measured; its fraction of total physical error is not causally identified."),
        ("Reverse causality / leakage","CHECKED","Truth used only in clearly labeled oracle diagnostics; model inputs remain feature-only.")]
    dump(OUT/"interpretation_checks.json",{"coverage":"11/11","checks":[{"type":a,"assessment":b,"basis":c} for a,b,c in fallacies]})
    approval={"date_local":"2026-11-08","user_request":"帮我重点分析一下 BM702 的平均照度、P10 和 FUDI 为什么预测这么差，是数据、特征还是模型的问题？只用现有数据，给出下一轮提升跨房间预测性能的方案，不追加模拟，也不做可视化。",
              "scope":"Existing-data diagnosis and next-round proposal only","report":"reports/diagnosis_BM702_v1/summary.md",
              "verification":"reports/diagnosis_BM702_v1/verification.json","proposal":"docs/next_experiment_after_BM702.md","new_training_executed":False,"new_simulations":0,"new_visualizations":0}
    dump(ROOT/"docs/approvals/diagnosis_BM702_v1_approval.json",approval)
    state_path=ROOT/"docs/project_status.json";state=read(state_path)
    state.update({"latest_completed_analysis":"BM702_existing_data_diagnosis_v1","latest_analysis":approval["report"],
                  "latest_approval":"docs/approvals/diagnosis_BM702_v1_approval.json",
                  "pending_proposal":approval["proposal"],"pending_proposal_status":"Proposed only, not approved or executed as a training round; current user requested diagnosis and a plan.",
                  "next_priority":"Review bounded directional-interaction and model-representation proposal; preserve fixed model and acknowledge observed holdout feedback.",
                  "BM702_diagnosis":{"report":approval["report"],"verification":approval["verification"],"new_estimator_fits":0,"finding":"Negative-X coverage gap with model branch saturation, wrong diurnal response and coupled P10 errors; internal data checks pass but source geometry/time remains unverified."}})
    dump(state_path,state)
    registry_path=ROOT/"docs/model_evidence_registry.json";registry=read(registry_path)
    registry["BM702_post_holdout_diagnosis"]=state["BM702_diagnosis"]|{"proposal":approval["proposal"],"proposal_executed":False}
    dump(registry_path,registry)
    p=ROOT/"docs/experiment_index.md";text=p.read_text(encoding="utf-8")
    heading="## BM702事后诊断与下一轮方案"
    if heading not in text:
        p.write_text(text+"\n\n"+heading+"\n\n2026-11-08完成，仅用现有数据、0次拟合、无图形。内部数据检查通过；确认−X取值在当前树模型中与0分支数值等价，平均照度日内响应错位，P10均值与比例分支同时失准。详见[诊断结果](../reports/diagnosis_BM702_v1/summary.md)和[尚未执行的下一轮方案](next_experiment_after_BM702.md)。不是新的独立测试成绩。\n",encoding="utf-8")
    paths=[ROOT/"configs/diagnosis_BM702_v1.json",ROOT/"scripts/diagnose_BM702_v1.py",ROOT/"scripts/verify_BM702_diagnosis_v1.py",Path(__file__),
           ROOT/approval["proposal"],ROOT/"docs/approvals/diagnosis_BM702_v1_approval.json",state_path,registry_path,ROOT/"docs/experiment_index.md"]
    paths += [p for p in OUT.iterdir() if p.is_file() and p.name not in ["artifact_manifest.json","repository_preflight.json"]]
    inventory=[];findings=[]
    for p in sorted(set(paths)):
        relative=p.relative_to(ROOT).as_posix();assert p.stat().st_size<100*2**20
        inventory.append({"path":relative,"bytes":p.stat().st_size,"sha256":api.sha256(p)})
        if p.suffix.lower() in audit.TEXT:
            text=p.read_text(encoding="utf-8-sig",errors="replace")
            for name,pattern in audit.PATTERNS.items():
                for match in pattern.finditer(text):findings.append({"path":relative,"type":name,"line":text.count("\n",0,match.start())+1})
    assert not findings,findings
    dump(OUT/"repository_preflight.json",{"status":"passed","files":len(inventory),"credential_findings":findings})
    dump(OUT/"artifact_manifest.json",{"analysis_id":"BM702_existing_data_diagnosis_v1","files":inventory,"note":"Shared project status docs may evolve; immutable model/source/diagnostic hashes preserved."})
    print(f"Diagnosis and unexecuted plan recorded; {len(inventory)} files inventoried.")


if __name__=="__main__":main()
