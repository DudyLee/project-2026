# 实验与模型证据索引

更新：2026-11-08。最新完成v7逐目标优化。最差误差优先方案没有输出同时通过全部开发条件。 旧参考保留，新候选单独记录。

**这里记录开发证据，不是一套已经锁定并通过独立测试的最终模型。** 七个房间已被多轮用于开发；BM532、BM702、BMG26继续封存。

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
