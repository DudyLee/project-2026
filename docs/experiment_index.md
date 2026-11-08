# 实验与模型证据索引

<!-- v10-current -->
最新v10已完成252次拟合并核验。S0留房间达成平均<50和BM432≤120，但LM608及留朝向未达要求，全部候选未通过。[结果](../reports/experiment_v10/summary.md)。下文为历史轮次记录。

<!-- v9-current -->
最新完成v9平均照度实验，189次拟合并通过核验。达到平均MAE<50且BM432≤120的候选：无；通过全部稳定性条件：无。[完整报告](../reports/experiment_v9/summary.md)。下文保留各轮当时结论。

更新：2026-11-08。最新完成v8方向交互与模型表示实验，523次拟合、两套分组验证通过核验。平均照度有平均误差收益，全部候选仍未同时通过最差房间和稳定性条件；固定0.1模型保留。

**这里记录开发证据，不能据此宣称新房间可靠。** 七个房间已被多轮用于开发；BM532、BM702、BMG26已在0.1固定模型评估中被观察，但继续排除在训练/选模之外。下文历史轮次“封存”是当时状态。

| 阶段 | 内容 | 结果入口 |
|---|---|---|
| v1 | 8/13项特征与Ridge、HistGB基线 | [报告](../reports/experiment_v1/report.html) |
| v1诊断 | 误差归因、log1p与同房间时间验证；与跨房间分数分开解释 | [报告](../reports/diagnosis_v1/report.html) |
| v2 | 平均照度log1p与相对P10处理 | [报告](../reports/experiment_v2/report.html) |
| v3 | HistGB、Random Forest、XGBoost的按房间嵌套验证与角度评价 | [报告](../reports/experiment_v3/report.html) |
| v3决策诊断 | 固定0°的优势、可改进空间和切换损失 | [报告](../reports/decision_diagnosis_v3/report.html) |
| v4 | PV增量与内层选择切换门槛；收益局限于个别房间 | [报告](../reports/experiment_v4/report.html) |
| v5 | 房间特征增强与直接asinh P10；均未通过替换条件 | [报告](../reports/experiment_v5/report.html) |
| v5误差诊断 | 两阶段误差、环境响应与现有几何资料核对 | [报告](../reports/cross_room_diagnosis_v5/report.html) |
| v6 | 相对/直接P10融合；未通过条件，保留原参考 | [报告](../reports/experiment_v6/report.html) |
| v7 | 逐目标损失函数与最差房间优先选择 | [报告](../reports/experiment_v7/report.html) |
| v8 | 方向交互、树/样条＋Ridge、P10直接对照；留房间/留朝向 | [数值报告](../reports/experiment_v8/summary.md) |

## 当前比较参考

下表用于后续比较，不表示把各目标的事后最优算法拼成一个已验证的生产模型。F300、FUDI与PV列出冻结HistGB参考；v3的其他候选保留在对应报告。

| 目标 | 参考方法 | 7房间平均MAE | 单位 |
|---|---|---:|---|
| E_mean_lux | A_XGBoost | 82.433157 | lux |
| E_P10_lux | A_RandomForest | 9.747747 | lux |
| F_300_fraction | Frozen_HistGB | 0.071180 | fraction |
| F_UDI_fraction | Frozen_HistGB | 0.110359 | fraction |
| PV_AC_kWh | Frozen_HistGB | 0.012613 | kWh |

比例误差以0–1表示：例如0.07118相当于7.118个百分点。MAE不是准确率。逐房间差异和最差误差须同时查看。

## v6结论

Random Forest的相对P10与直接asinh P10按五个候选权重融合。每个外层折只在其余六个房间的内层OOF中选权重，外层房间只评分。原始单位平均MAE从9.748升至10.729 lux，恶化10.1%；2个房间改善、1个持平、4个恶化。没有通过预设条件，不替换原参考。

内层分数用于选模型和权重，不能作为独立效果证明；次要的asinh误差改善不能推翻主要MAE结论。按停止规则，暂停无新信息的更细权重或按房间路由搜索。

## 下一步所需资料

优先核对已有原始模型的房间进深、窗户位置、传感器网格，以及EPW/时间/北向约定。缺失几何是待验证解释，当前实验没有证明它必然能提高精度。只请求现有资料，不追加模拟。

- [资料请求说明](data_requests/geometry_and_simulation_sources.md)
- [资料包](../exports/geometry_source_handoff_v1.zip)
- [机器可读证据索引](model_evidence_registry.json)
- [当前状态](project_status.json)

历史草案保留创建时状态。v6的后续批准与完成以[批准记录](approvals/P10_blend_v6_approval.json)、[执行配置](../configs/experiment_v6.json)及[验证记录](../reports/experiment_v6/verification.json)为准。


## v7逐目标结果

最差误差优先方案没有输出同时通过全部开发条件。

平均照度：平均MAE 82.433 → 83.240 lux；最差MAE 231.642 → 231.642 lux（BM432 → BM432）；5/7房间不恶化；未通过两套参考的筛选条件。

P10照度：平均MAE 9.748 → 15.453 lux；最差MAE 29.275 → 48.956 lux（BM432 → BM223）；3/7房间不恶化；未通过两套参考的筛选条件。

≥300 lux空间比例：平均MAE 7.118 → 7.204 个百分点；最差MAE 15.686 → 14.837 个百分点（BM432 → BM432）；5/7房间不恶化；未通过两套参考的筛选条件。

100–3000 lux空间比例：平均MAE 11.036 → 9.566 个百分点；最差MAE 25.471 → 26.119 个百分点（BM432 → BM432）；3/7房间不恶化；未通过两套参考的筛选条件。

逐时PV发电量：平均MAE 12.613 → 12.945 Wh；最差MAE 23.809 → 23.809 Wh（LM608 → LM608）；4/7房间不恶化；未通过两套参考的筛选条件。

正式候选为Worst_guarded。相对Target_baseline和Retained_reference均须达到：最大房间MAE下降至少10%、平均MAE不增加、至少5/7房间不恶化。内层5%保护不能保证外层结果也满足条件；Mean_select是预先声明的对照，不在看过结果后改成主要候选。

原参考数值和v7候选分开保存；未执行全7房间部署训练、TEST评价或新的角度决策验证。详细误差与每折所选模型见[v7报告](../reports/experiment_v7/report.html)。


## 试用版0.1（全7开发房间固定模型）

2026-11-08按用户批准完成。沿用保留模型族，平均照度XGB深度3、相对P10 RF深度12、F300/FUDI/PV固定HistGB；42次开发参数选择拟合后，用7房间122,640行重训6个估计器。

[训练与接口验证](../reports/trial_v0_1/summary.md) · [使用说明](trial_v0_1_usage.md)。CSV/JSON/Python入口已验证，三个TEST房间继续封存。本轮软件检查和参数选择分数不等于最终独立泛化成绩。


## 固定0.1模型首次留出测试

2026-11-08按用户最新请求评估BM532、BM702、BMG26，共52,560行；0次拟合。MAE平均为111.07 lux、18.68 lux、6.53个百分点、12.03个百分点、0.00767 kWh/小时。五项最差房间均为BM702。详见[数值报告](../reports/holdout_trial_v0_1/summary.md)。三个房间仍不参加训练，但已不是未观察的测试集。


## BM702事后诊断与下一轮方案

2026-11-08完成，仅用现有数据、0次拟合、无图形。内部数据检查通过；确认−X取值在当前树模型中与0分支数值等价，平均照度日内响应错位，P10均值与比例分支同时失准。详见[诊断结果](../reports/diagnosis_BM702_v1/summary.md)和[尚未执行的下一轮方案](next_experiment_after_BM702.md)。不是新的独立测试成绩。


<!-- v8-current -->
## v8方向交互与跨朝向验证

仅7开发房间，完成523次估计器拟合。普通留房间：T1平均照度82.433→70.274 lux，7/7改善；S1为64.380 lux，但最差仍222.832 lux、仅改善3.80%。S1 FUDI平均11.036→9.513个百分点，仅2/7房间改善。P10直接对照平均12.344 lux，高于基线9.398 lux。0/10候选-目标组合通过全部冻结条件，模型0.1保持不变。

[完整两套验证与最差房间](../reports/experiment_v8/summary.md) · [核验记录](../reports/experiment_v8/verification.json)。本轮没有TEST预测/评分、模拟或可视化。


## v9平均照度与房间强度归一化

达到平均MAE<50且BM432≤120的候选：无；通过全部稳定性条件：无。

[两种验证及逐房间变化](../reports/experiment_v9/summary.md) · [输入来源核对](../reports/experiment_v9/input_source_audit.json) · [核验](../reports/experiment_v9/verification.json)。仅7开发房间，其他输出与0.1模型不变。


<!-- pair-diagnosis-v9 -->
## v9后BM432/LM608诊断（无训练）

C相对B同时降低两房间MAE，LM608退步在A/B已经出现，不能归因于归一化单一因素。q没有消除归一化亮度或角度响应差异。复核保存模型，0次拟合、0次TEST操作。[诊断报告](../reports/diagnosis_BM432_LM608_v9/summary.md) · [下一轮提议，尚未训练](next_experiment_pair_balance_v10.md)。


## v10四格log目标/归一化实验

同时达到50/120/40的候选：无；全部稳定性条件通过：无。S0普通留房间平均48.040、BM432 112.541、LM608 42.662；留朝向平均115.460、BM432 480.140、LM608 102.251。Sq留朝向平均68.018，但普通留房间尚未达标。无TEST使用或部署变更。
