# Tower A PVSD 项目

最新可用版本：**试用版0.1**，已用全部7个开发房间拟合固定模型，支持四角度×五指标CSV/JSON/Python预测。开始使用：[中文操作说明](docs/trial_v0_1_usage.md)；[试用版检查结果](reports/trial_v0_1/summary.md)。三个测试房间继续封存；此版本用于研究联调，尚未独立测试。

各轮开发结果：[实验索引](docs/experiment_index.md)。v7没有候选通过全部替换条件，试用版沿用保留方法。

本文件夹统一保存本项目的数据、分析脚本、建模配置和检查结果。后续下载和生成的项目文件均放在这里，并按用途分类。

## 文件位置

- `data/raw/`：从 Google Drive 下载的原始文件，保留原文件名和内容。
  - `SEND_ML_BIM_EPW_07-18.csv`：已合并 BIM、EPW 和太阳位置的逐小时模拟数据，175,200 行、36 列。
  - `SEND_BIM_476Rooms.csv`：476 个房间的 BIM 特征，476 行、14 列。
  - `SEND_README.txt`：作者原始说明。此说明仍引用旧逐时文件名，没有说明新增天气字段。
- `scripts/audit_data.py`：读取原始 CSV，重新生成数据检查结果。
- `configs/model_plan.json`：整体建模计划、按房间验证方式及待确认事项。第一轮交叉验证已完成，最终 TEST 尚未评价。
- `configs/experiment_v1.json`：本轮实际使用的固定参数和预处理规则。
- `environment.yml`：conda 环境依赖声明。
- `configs/conda-explicit-win64.txt`：本次实际安装的精确 conda 包版本和构建信息。
- `reports/data_audit/`：程序生成的检查结果。
  - `data_audit.json`：数量、缺失值、重复项、时间索引、BIM 连接、数值范围和特征覆盖检查。
  - `column_audit.csv`：字段清单、数据类型、缺失数、唯一值数及候选用途。
  - `room_summary.csv`：10 个模拟房间的特征和训练/测试划分。
- `docs/data_sources.json`：下载来源、原始文件 SHA-256 校验值及来源信息的核实状态。
- `reports/experiment_v1/`：第一轮交叉验证结果，优先打开 `report.html`。
- `models/experiment_v1/`：140 个交叉验证模型及其训练房间和输入字段。它们是各折模型，并非使用全部 7 个训练房间拟合的最终部署模型。
- `.conda/envs/pvsd-ml-v1/`：独立 conda 环境；`.conda/pkgs/` 为包缓存。

## 使用约定

原始文件存放于 `data/raw/`。清洗或特征工程应另存到后续创建的 `data/processed/`，保留原始数据。训练脚本放在 `scripts/`，模型配置放在 `configs/`，训练完成后的模型放在 `models/`，评价结果和图放在 `reports/`。原始 EPW 取得后放在 `data/raw/weather/`，记录具体文件名及来源。

路径均相对于 `Project_2026` 根目录。`audit_data.py` 根据脚本位置定位根目录，不依赖当前终端目录。重新运行会更新 `reports/data_audit/`，不会修改原始 CSV。

## 重新检查数据

在具备 `numpy` 和 `pandas` 的 Python 环境运行：

```powershell
python scripts/audit_data.py
```

## 已核实的范围

逐小时数据覆盖 10 个房间、0/30/60/90 度四个固定角度、全年每天 Hour=7..18 的 4,380 个时间索引。7 个 TRAIN 房间共 122,640 行，3 个 TEST 房间共 52,560 行。CSV 内部时间索引和 BIM 字段一致，没有缺失值或重复工况时间键。

保持原房间划分。在 7 个 TRAIN 房间内部做留一房间验证；最终 TEST 房间是 BM532、BM702、BMG26。训练特征对这三个房间的朝向、遮挡或楼层存在覆盖不足，需逐房间报告效果。数据检查不等于模型性能验证。

## 待确认的来源信息

用户当前无法确认 EPW 是否与原模拟使用同一文件、Hour 的区间含义，以及模型轴相对地理北的旋转角。

可以先进行探索性训练和按房间验证，但尚不能断言天气与模拟结果正确对应，也不能据此宣称跨天气年份泛化。太阳高度角和方位角的计算地点、时区、时间点及方位角约定也需要来源记录。

暂时保留原始 `Orientation_X_model`、`Orientation_Y_model`，不将其直接解释为地理方位，不计算依赖地理北转换的相对太阳角或入射角。

首轮比较 8 个基础输入与加入太阳位置和辐射后的 13 个原始输入，分别预测五项性能。新增天气特征是否有帮助由训练房间内部验证判断，不能用最终测试集反复挑选。

## 第一轮实验结果

本轮已完成 140 次拟合：2 个模型、2 组特征、5 个目标、7 个留一房间折。只使用 TRAIN 数据，TEST 没有预测或评分。周期编码后的输入列数为 9 和 15；Ridge 是标准化后的加性线性基线，本轮没有交互项；梯度提升树关闭早停，没有随机按行分出验证集。没有进行超参数搜索或最终全训练集重训。

梯度提升树加入天气后的平均房间 MAE：PV 从 0.01858 降至 0.01261 kWh；平均照度从 115.31 降至 109.49 lux；P10 从 19.90 降至 19.65 lux；F_300 从 7.54 降至 7.12 个百分点；F_UDI 从 12.12 降至 11.04 个百分点。PV 的 7 个房间全部改善。采光的跨房间效果仍不稳定，平均照度和 P10 的平均 RMSE 没有改善，部分房间 R² 为负，暂不支持部署结论。

`reports/experiment_v1/summary_metrics.csv` 为各房间等权平均的汇总；`fold_metrics.csv` 为 140 条逐折逐目标指标；`weather_feature_comparison.csv` 和 `paired_room_comparison.csv` 分别提供平均与逐房间天气对照；`stratified_metrics.csv` 提供按房间、月份、小时和角度的误差。

`predictions/` 内的 4 个 Parquet 文件各有 122,640 行，包含 TRAIN 的留出预测、真实值、未截断预测、房间、时间和角度。`__sample.csv` 只含前 100 行，便于查看字段，不能作为完整评价数据。主指标将负预测截为 0、两项比例截到 0–1；完整结果同时保留 raw 指标和截断比例。

`verification.json` 记录了对全部 140 条指标的独立重算、140 个模型重载预测复核、各折标准化均值检查、TEST 排除以及原始文件哈希检查。`run_manifest.json` 记录版本、参数、输入哈希、运行时间和折划分。模型 pickle/joblib 仅应从可信项目文件加载。

## 使用和复现

在项目根目录，用本机现有 Miniconda 的 PowerShell 方式激活环境：

```powershell
(& 'C:\Users\Test\miniconda3\python.exe' -m conda shell.powershell hook) | Out-String | Invoke-Expression
conda activate "$PWD\.conda\envs\pvsd-ml-v1"
```

如需在另一台 Windows 机器创建同版本环境：

```powershell
conda create --prefix .\.conda\envs\pvsd-ml-v1 --file .\configs\conda-explicit-win64.txt
```

若只需声明主要依赖，可使用 `conda env create --prefix .\.conda\envs\pvsd-ml-v1 --file environment.yml`。其中 matplotlib 无界面绘图使用 Agg；PowerShell 启动脚本也会设置 conda 的 DLL 路径。

本轮入口为 `scripts/run_experiment.ps1`，它依次运行训练、结果复核和视图生成，设置 1 小时硬超时并保存日志。训练脚本会拒绝覆盖已完成的本轮指标；新实验应先复制配置，使用新的实验 ID、报告目录和模型目录，保留本轮作为基线。该新实验配置还需通过参数接入或相应入口脚本指定，不能直接覆盖本轮运行。

在已激活环境中，仅重新生成报告可运行：

```powershell
python scripts/report_experiment_v1.py
python scripts/build_result_views.py
```

## BM223 与 BM522 的专项诊断

结果位于 `reports/diagnosis_v1/report.html`，配置为 `configs/diagnosis_v1.json`，运行入口为 `scripts/run_diagnosis.ps1`。沿用现有 conda 环境，新增 70 次两目标留一房间拟合和 2 次已知房间整月留出诊断，未对 TEST 房间预测、评分或分析目标分布。

`scripts/diagnose_rooms_v1.py` 负责 CSV 内部一致性、特征覆盖、冻结模型敏感性、五项控制实验与时间诊断。`scripts/report_diagnosis_v1.py` 独立重算指标、重载模型并生成报告和图；`scripts/build_diagnosis_view.py` 生成交互结果视图。所有新增模型位于 `models/diagnosis_v1/`，原版 `experiment_v1` 保留。

主要证据指向跨房间特征和样本支持不足，以及面积等特征被模型用作少数房间的分组代理。CSV 内部检查未发现直接错误；原始网格和 P10 导出仍需核实。目标变换或换模型能降低部分误差，但尚未支持全局替换模型。已知房间时间诊断使用目标房间其他月份的数据，其高分不能当成新房间泛化成绩。

在已激活环境中重新构建报告：

```powershell
python scripts/report_diagnosis_v1.py
python scripts/build_diagnosis_view.py
```

## P10 目标处理第二轮

本轮已完成，结果入口为 `reports/experiment_v2/report.html`，交互视图为同目录的 `p10-target-handling.canvas.tsx`。配置在 `configs/experiment_v2.json`，训练入口为 `scripts/run_p10_v2.ps1`，核心代码为 `scripts/train_p10_v2.py`。沿用独立 conda 环境与相同的 7 个 TRAIN 留一房间折。

新增 14 个 P10 模型：7 个按训练房间尺度加权的 log1p 模型，以及 7 个相对照度模型。均照度采用上轮的 log1p 方法；因输入、参数与验证折没有变化，本轮归档并复用了 7 个已有模型，逐一核对了训练房间、文件哈希和预测。没有重复拟合或使用 TEST。

尺度加权仅在每折 6 个训练房间中计算 `1/sqrt(1+该房间P10均值)`，归一化到训练样本平均权重为 1。相对照度目标为 `r=P10/(E_mean+1 lux)`，最终 P10 为非负 r 预测乘以预测均照度加 1 lux。真实留出均照度只用于评价及预测后的误差分析，没有用于主要预测重建。

P10 的 7 房间平均 MAE：原始模型 19.651 lux；普通 log1p 19.251 lux；尺度加权 log1p 19.183 lux；相对照度方案 14.591 lux。相对方案比原始模型降低约 25.7%，6 个房间改善；约 88% 的总体改善来自 BM223，其 MAE 从 53.592 降至 22.487 lux。BM432 仍有约 53.087 lux 的误差，尚未解决。尺度加权相对普通 log1p 只额外改善约 0.35%，不作为本轮首选。当前候选与限制记录在 `candidate_decision.json`，没有生成最终全 7 房间部署模型。

`models/experiment_v2/` 保存 14 个新 P10 模型和 7 个复用均照度模型；`predictions/` 内保存两组完整 P10 留出预测及均照度预测。`training_room_weights.csv` 记录每折每个拟合房间的实际权重。`ratio_error_decomposition.csv` 使用真实标签作预测后诊断，误差项可能抵消，其绝对值均值不能直接相加为 MAE。

`verification.json` 记录 28 条 P10 指标、7 条均照度指标的独立重算，14 个新模型重载、7 个复用模型及各折权重核对，确认主相对照度预测只使用预测均照度。TEST 预测和评分仍为 0。原始文件和前两轮结果均保留。

在已激活环境内重新构建本轮报告：

```powershell
python scripts/report_p10_v2.py
python scripts/build_p10_v2_views.py
```

## 第三轮：模型对照、嵌套选参与角度决策

本轮已于 2026-11-02 完成。报告入口为 `reports/experiment_v3/report.html`，交互视图为同目录 `nested-model-decisions.canvas.tsx`，固定配置在 `configs/experiment_v3.json`。结果仍是开发阶段证据，BM532、BM702、BMG26 三个最终测试房间没有预测、评分、调参或校准。

沿用项目独立 conda 环境，新增 XGBoost 3.1.3；通过 pip 的 `--python` 定向安装到该环境，使用 `--no-deps` 保持其余包不变。安装来源、wheel 哈希及版本记录见 `configs/xgboost-install-v3.json`、`python-packages-v3.txt` 和 `conda-explicit-v3-win64.txt`。conda explicit 文件不包含 pip 安装项；重建时还需在环境中安装 `xgboost==3.1.3`，或直接使用已更新的 `environment.yml`。

三类模型为 HistGB、Random Forest、XGBoost，各有两组预设参数，使用同一组13个原始输入、编码后15列。平均照度采用 log1p，相对 P10 由预测均照度重建，F300、FUDI和PV直接回归。保留前两轮已保存的 HistGB 五指标预测作为冻结参考，不重复拟合。

外层为7折留一房间；每折其余6个房间内采用3折 GroupKFold，每次4个拟合、2个验证。按内层验证房间等权平均、原尺度 MAE 选参。另有 Nested_select 在内层选择算法家族和参数。每个相对 P10 候选均包含自己的均照度和比值组件，整个预测过程没有使用留出房间真实均照度。单独的均照度输出可以选择其他组件。

训练耗时约547秒，共630次内层拟合、113次外层拟合；保存21个外层模型包、5组各122,640行的完整留出预测。内层模型未保存。全部175条外层指标、210条候选分数、140条选择记录和21个模型包已核对。训练入口 `scripts/run_nested_v3.ps1` 设2小时硬超时，拒绝覆盖已启动的本轮结果。核心脚本为 `train_nested_v3.py`，验证与决策脚本为 `evaluate_nested_v3.py`，视图脚本为 `report_nested_v3.py`。

相对冻结参考，各算法的收益不一致。XGBoost 的平均照度 MAE 为82.433 lux，较95.921降低14.1%；Random Forest 的P10 MAE为9.748 lux，较14.591降低33.2%。XGBoost 的F300 MAE为6.932个百分点；Random Forest的FUDI为8.764个百分点，但只在2/7房间改善，而XGBoost的FUDI为9.436个百分点、6/7房间改善。PV仍以冻结HistGB的0.012613 kWh MAE最低。Nested_select只在P10和FUDI上改善，其平均照度反而变为109.521 lux。

重点房间方面，RF将BM223 P10从22.487降到9.757 lux、BM432 P10从53.087降到29.275 lux；XGBoost将BM522平均照度从149.983降到108.545 lux，但内层自动选择在该房间选了RF，误差为242.251 lux。这说明小房间样本下模型选择仍不稳定，不能全局替换算法或仅报告各指标事后最优分数。

决策验证覆盖四个角度、四种预设目标：PV优先、采光空间比例优先、等权平衡、采光约束下PV最大化。采光分数为 `(F300+FUDI)/2`，PV用每折六个训练房间正值的95分位数归一化。约束场景要求F300和FUDI均不低于0.5。上述权重与阈值均为本轮预设探索场景，不是已认可的工程标准。E_mean与P10不直接参与本轮角度评分。

对照含四个固定角度、训练房间选定的固定角度、预设时间表、五组模型策略及模拟Oracle。各折训练选定的固定角度均为0°，在四种场景的平均主目标损失上均优于所有模型策略。平衡目标的模拟Oracle有91.34%的小时选0°（并列取较小角度），固定0°的近优比例为94.46%；不能把模型相似的高比例视为动态推荐成功。约束场景的Oracle仅7.47%的保留小时可行，低满足率不能全部归因于预测误差。

`decision_by_room.csv`、`decision_summary.csv`、`angle_usage_summary.csv` 和 `decisions/` 保存336组汇总及1,471,680条决策记录。每项所选性能均重新按Room_ID、HOY、角度查回原始模拟数据；28组Oracle的目标损失检查为零。PV总量仅覆盖4380个保留小时，两项采光比例的时间平均不是年度DA/UDI；切换次数只统计同一天的相邻保留小时，不含控制延迟、过夜或执行器损耗。

`tolerance_summary.csv`提供多档绝对误差容限内比例，属于探索场景。例如XGBoost均照度误差在±50 lux内的比例为68.1%，RF P10在±5 lux内为69.9%，均为房间等权平均；必须同时查看最差房间，不能据此宣称统一预测准确率。

结果解释与候选状态在 `candidate_decision.json`。当前保留冻结基线及新候选，不自动全局替换模型或解封TEST。新太阳入射特征继续等待地理北旋转、角度定义与时间对齐确认；本轮没有追加模拟、全7房间部署重训或最终测试。

仅重建本轮核对结果与报告，无需重新训练：

```powershell
& .\scripts\report_nested_v3.ps1
```

## 固定0°角度推荐专项诊断（分析完成，下一轮待确认）

入口为 `reports/decision_diagnosis_v3/report.html`，交互视图为 `fixed-zero-diagnosis.canvas.tsx`；分析脚本为 `scripts/diagnose_angle_decisions_v3.py`，视图生成脚本为 `scripts/build_angle_diagnosis_v3.py`。本次仅使用v3冻结的TRAIN留出预测与既有决策规则：新拟合0、新模型预测0、原始含TEST的CSV读取0、TEST预测和评分0。没有改变目标、权重、阈值或进行新策略回测。`analysis.json`保存输入哈希、105组原指标重现及切换收益分解恒等式检查。

在61.46%的房间—小时里，0°的F300、FUDI和PV均不低于其他三个角度；平衡目标下0°唯一最优88.11%、近优94.46%，超过既定近优容限的改善机会仅5.54%。纯PV则仍有事后Oracle增益：每房间平均11.38 kWh，相对合并总能量的固定0°基线约4.61%，只覆盖现有4380小时。XGBoost的PV切换平均带来2.35 kWh收益、2.55 kWh损失，净少0.20 kWh；BM608净多2.92 kWh，BM223净少5.77 kWh，不能把总体排序解释为所有房间的排序。

当前目标只是一组探索性场景。F300与FUDI重叠，等权平均不等于独立舒适指标；训练PV分位数归一化也不是已确认的工程权衡。采光约束下92.53%的保留小时没有可行角度，固定0°覆盖全部可行小时的99.26%（合并小时计算）。此前约束场景的主损失仅为采光缺口，PV另列，不能据此宣称0°在完整分层目标上全面胜出。

原草案位于 `configs/proposals/decision_v4_draft.json`，报告目录保存相同副本，保留了2026-11-02当时“待确认”的历史状态。用户于2026-11-07批准后，已按 `configs/experiment_v4.json` 完成下述第四轮；审批记录见 `docs/approvals/decision_v4_approval.json`。三个TEST房间继续封存，原草案、既有实验及原始数据保留。

## 第四轮：PV增量预测与切换门槛（2026-11-07完成）

结果入口为 `reports/experiment_v4/report.html`，旁侧交互视图为 `PV-increment-threshold.canvas.tsx`。执行配置是 `configs/experiment_v4.json`，训练入口 `scripts/run_pv_increment_v4.ps1`，核心脚本 `train_pv_increment_v4.py`；核对与报告脚本分别为 `evaluate_pv_increment_v4.py`、`report_pv_increment_v4.py`。

沿用项目conda环境和XGBoost 3.1.3，各折参数冻结为v3在其六个拟合房间中选出的参数，没有新增算法或参数搜索。新增49次拟合：21个内层绝对PV模型、21个内层增量PV模型、7个外层增量模型；外层绝对PV预测复用v3并重载核对。训练约14秒，所有49个新模型保存在 `models/experiment_v4/`。

增量标签为同一拟合房间、同一小时的 `PV(角度)-PV(0°)`，只拟合30°/60°/90°；负增量保留，0°增量固定为0。推断时仅使用原13个输入（编码15列）和角度，不传入真实PV(0°)。原始含TEST的CSV按角色先排除TEST，再作数值转换；TEST预测、评分仍为0。

外层仍为7折留一房间，每折六个拟合房间内部采用3折分组验证（4房间拟合、2房间验证），选择预测增益门槛0、1、2、5 Wh或始终0°。选择标准为六个内层验证房间的平均净发电增益；并列时选较大门槛。没有拼用全7房间的既有外层OOF进行校准。已保存7组各105,120行的内层OOF及122,640行完整外层候选预测。

相对固定0°，原绝对XGBoost平均每房间净增益为−0.198 kWh；加门槛后为+0.471 kWh，合并总能量增益率约0.191%，中位数0。全部正收益来自BM441（+3.294 kWh），其余六个房间持平。门槛在BM441、BM522、LM608三折选择5 Wh，其余四折选择始终0°；BM522和LM608的外层预测未触发5 Wh门槛。

无门槛增量模型平均净增益−10.318 kWh，4房间改善、3房间变差，LM703损失75.137 kWh。带门槛增量模型在7折均选择始终0°，外层完全退回固定基线。Δ角度差异MAE为0.017549 kWh，未优于绝对XGBoost差分的0.015177 kWh。Δ模型未产生独立绝对PV预测，不用真实PV(0°)重建后报告绝对PV精度。

BM441的纯PV收益同时使F300平均空间比例下降2.318个百分点、FUDI下降5.222个百分点。因此只保留“绝对PV＋门槛”作为有限开发候选，不宣称已建立稳定的动态综合控制收益；当前Δ模型暂不推进。所有增益只求和现有4380个保留小时，未计控制延迟、过夜转换或执行器损耗。三个TEST房间封存，没有追加模拟或最终全7房间部署拟合。

`verification.json`保存49个模型重载及训练标签哈希核对、420条门槛房间分数及14次选择独立重算、214,620条决策所选性能按原始键重新查值。内层XGBoost原始输出采用float32，差分复核保留其原始运算精度，确认精确重现。原始文件与冻结参考哈希不变。

`decision_summary.csv`与`decision_by_room.csv`分别为总体和逐房间决策指标；`forecast_summary.csv`为预测误差；`threshold_selections.json`和`inner_threshold_summary.csv`为门槛证据；`candidate_decision.json`记录当前结论。图像已人工检查；本地浏览器功能关闭，HTML交互实机验证未完成，静态脚本及链接检查另存 `presentation_verification.json`。

仅重新核对和生成报告，不重训：

```powershell
& .\scripts\report_pv_increment_v4.ps1
```


## 第五轮：跨房间特征与直接P10预测

用户批准后已完成600次拟合（504次内层、96次外层），28个外层模型包。报告入口为`reports/experiment_v5/report.html`，交互视图为同目录`Cross-room-features-P10.canvas.tsx`。

四组：A原13项特征＋相对P10；B增加3项房间特征＋相对P10；C原特征＋直接asinh P10；D增强特征＋直接asinh P10。平均照度均采用log1p，A/C与B/D分别共用相同平均照度输出。模型为Random Forest与XGBoost，每家族沿用v3的两个候选并在外层训练房间内部选择。HistGB作为冻结参考。

新增特征为玻璃面积/地板面积、窗户数量、名义窗面积/窗户数量；模型编码由15列增至18列。P10直接变换的尺度固定为1 lux。推断不使用留出房间真实照度。没有假设缺失的房间进深、窗户位置、北向或百叶几何。

本轮没有方案同时通过预设的三项开发筛选条件。

新增特征的XGBoost平均照度MAE：82.433 → 81.769 lux，下降0.81%，6/7房间改善，仍未达到预先约定的10%幅度。

Random Forest直接asinh P10将BM223 MAE从9.757降至1.479 lux（下降84.8%），但BM432从29.275增至44.361 lux；7房间平均MAE从9.748增至10.743 lux（恶化10.2%）。局部改善不能支持全局替换。

平均照度最低房间平均MAE为81.769 lux（B_XGBoost）；这是本轮开发结果的事后描述，不是锁定模型的独立测试分数。

P10照度最低房间平均MAE为9.748 lux（A_RandomForest）；这是本轮开发结果的事后描述，不是锁定模型的独立测试分数。

BM223的P10照度：本轮最低MAE 1.479 lux（C_RandomForest），真实均值0.616 lux。最优方法按本房间事后查看，不能据此给新房间选择模型。

BM522的平均照度：本轮最低MAE 108.370 lux（B_XGBoost），真实均值152.811 lux。最优方法按本房间事后查看，不能据此给新房间选择模型。

预设逐目标筛选：相对同算法A组，房间平均MAE下降至少10%、至少5/7房间不恶化、最差房间MAE不增加。所有组合均报告；最优分数属于开发结果，未自动替换旧模型。

配置`configs/experiment_v5.json`；协议`docs/experiment_v5_protocol.md`；批准记录`docs/approvals/cross_room_v5_approval.json`。训练入口`scripts/run_cross_room_v5.ps1`拒绝覆盖已开始的运行。已激活项目conda环境时，执行以下入口仅重算验证及报告，不训练新模型：

```powershell
.\scripts\report_cross_room_v5.ps1
```

结果目录包含内外层预测、候选选择、逐房间误差、特征覆盖、配对比较、开发筛选条件和验证记录；模型保存于`models/experiment_v5/`。原始数据和冻结参考不变，3个TEST房间继续封存，无最终全7房间部署重训。图已检查，HTML静态链接/脚本已验证；未执行浏览器交互或原生Canvas编译验证。


## 第五轮之后：跨房间误差诊断与资料整理

已按批准的下一步完成误差诊断与资料整理，新增模型拟合0。报告为`reports/cross_room_diagnosis_v5/report.html`，交互视图为同目录`cross-room-error-diagnosis.canvas.tsx`。

BM223的相对P10模型主要高估比例：真实逐时P10/(平均照度+1 lux)的均值为0.0090，预测为0.0944。原P10 MAE 9.757 lux；诊断中仅代入真实平均照度仍为9.920 lux，不能解决高估。

BM432存在两阶段误差抵消：原P10 MAE 29.275 lux；诊断中仅代入真实平均照度反而为45.261 lux。不能把一个阶段变准确就等同于最终P10更准确。其P10配对均值模型为depth12，独立均值输出模型为depth8，分解使用正确的配对模型。

BM522的XGBoost平均照度在85.9%记录中高估，平均偏差+103.28 lux。DNI=0时MAE 1.86 lux，DNI>500 Wh/m²时为223.08 lux。偏差随环境变化，不能仅视为一个固定常数误差。

BM522在真实平均照度角度差>10 lux的20,330个角度对中，排序正确率为100.0%（覆盖全部角度对的77.4%），但角度差值MAE仍为27.02 lux。此为单一指标排序诊断，不是综合推荐准确率。

本地未找到与这批Tower A模拟建立来源对应的原始几何、网格、逐传感器照度或EPW。已准备7个TRAIN房间定位表、字段字典、待填模板和24个已有工况的原始传感器结果核对清单；没有要求新模拟，也没有向他人发送消息。

真实均值/比例替换仅用于解释误差，不是可部署模型或新验证分数。角度排序属于单指标诊断，不能代替综合推荐评估。所有7个TRAIN房间完整保留，三个TEST房间没有预测或评分。

脚本：`scripts/diagnose_cross_room_v5.py`计算诊断；`scripts/prepare_geometry_handoff_v5.py`生成资料清单与模板；`scripts/verify_diagnosis_v5.py`独立复核；`scripts/report_diagnosis_v5.py`构建报告。入口`run_diagnosis_v5.ps1`拒绝覆盖既有分析，`report_diagnosis_v5.ps1`可重建数值复核及报告。

资料包：`exports/geometry_source_handoff_v1.zip`，包含`docs/data_requests/`和`data/templates/`对应文件。空白几何字段标记为MISSING_NOT_FOR_TRAINING；没有伪造缺失几何、取得原始网格或向他人发消息。

P10融合草案`configs/proposals/P10_blend_v6_draft.json`保留创建时的PROPOSED_NOT_APPROVED_NOT_EXECUTED历史状态。用户随后批准并已完成下述v6；当前状态以`docs/approvals/P10_blend_v6_approval.json`和`docs/project_status.json`为准。


## 第六轮：内层选权重的P10融合（2026-11-07完成）

入口`reports/experiment_v6/report.html`；交互视图`reports/experiment_v6/P10-nested-blend.canvas.tsx`。统一实验索引见`docs/experiment_index.md`，机器可读参考记录见`docs/model_evidence_registry.json`。

基于v5已冻结的Random Forest相对P10与直接asinh P10，按w×相对预测+(1−w)×直接预测融合。仅比较w=1、0.75、0.5、0.25、0；每个外层折在其余六个房间的3折分组内层OOF中选择原始单位平均MAE最低的权重，并列偏向较大w。七个权重先冻结，再加载外层答案评分。没有新回归器拟合，也没有按留出房间名称或真实标签路由。

P10的7房间平均MAE：原相对模型9.747747 lux，直接asinh 10.743254 lux，内层选权重融合10.729370 lux。融合比原模型恶化10.07%，2房间改善、1房间持平、4房间恶化。BM223从9.757降至1.479，BM432从29.275升至44.361 lux。融合没有通过预设的≥10%平均改善、≥5/7房间不恶化、最差房间误差不增加三项条件。

保留A_RandomForest相对P10作为比较参考。次要asinh误差较低不改变原始单位主指标的结论；按照停止规则，不在已看过的外层结果上继续细调权重或加入房间路由。其余四个预测目标未变，没有全7房间最终部署拟合。

独立数值复核通过：210条内层房间分数、35条权重汇总、7次权重选择、14次底层候选选择、122640行融合预测、21组外层误差和开发条件；7份推断配方各重载31个样本，只接收13项输入，无Room_ID和真实目标。三个TEST房间没有预测或评分，无新增模拟。

模型配方在`models/experiment_v6/`，引用并校验v5模型包。运行记录、预测、逐房间比较、图与报告在`reports/experiment_v6/`。两张图已目视检查，HTML静态链接和脚本检查通过；浏览器交互和原生Canvas编译未验证。

复核及重建报告：在Project_2026内执行`& .\scripts\report_p10_blend_v6.ps1`，不会训练新回归器。该命令会重置图像审阅状态；重新目视审阅后，再用项目conda Python执行`scripts/finalize_p10_blend_v6.py --figures-reviewed`生成最终归档。选择入口`run_p10_blend_v6.ps1`拒绝覆盖本次已完成运行。


## 第七轮：逐目标优化与最差房间误差（2026-11-08完成）

最差误差优先方案没有输出同时通过全部开发条件。

固定13项输入、24个候选配置。外层7折留一房间；每折六个训练房间内部3折分组验证。Target_baseline沿用原候选；Mean_select按平均MAE选；Worst_guarded在平均MAE不比原候选高出5%的候选中，优先降低内层最大房间MAE。所有选项在外层评分前冻结。

正式候选为Worst_guarded。相对Target_baseline和Retained_reference均须达到：最大房间MAE下降至少10%、平均MAE不增加、至少5/7房间不恶化。内层5%保护不能保证外层结果也满足条件；Mean_select是预先声明的对照，不在看过结果后改成主要候选。

平均照度：平均MAE 82.433 → 83.240 lux；最差MAE 231.642 → 231.642 lux（BM432 → BM432）；5/7房间不恶化；未通过两套参考的筛选条件。

P10照度：平均MAE 9.748 → 15.453 lux；最差MAE 29.275 → 48.956 lux（BM432 → BM223）；3/7房间不恶化；未通过两套参考的筛选条件。

≥300 lux空间比例：平均MAE 7.118 → 7.204 个百分点；最差MAE 15.686 → 14.837 个百分点（BM432 → BM432）；5/7房间不恶化；未通过两套参考的筛选条件。

100–3000 lux空间比例：平均MAE 11.036 → 9.566 个百分点；最差MAE 25.471 → 26.119 个百分点（BM432 → BM432）；3/7房间不恶化；未通过两套参考的筛选条件。

逐时PV发电量：平均MAE 12.613 → 12.945 Wh；最差MAE 23.809 → 23.809 Wh（LM608 → LM608）；4/7房间不恶化；未通过两套参考的筛选条件。

完成546次内层、81次外层拟合，7个外层模型包；验证另重复拟合7个回归器。全部五项目标原候选重现v3。独立复核1008条内层房间分数、168条候选汇总、105次选择、140组外层指标，重载核对1839600个预测值。

报告`reports/experiment_v7/report.html`，交互视图`target-specific-worst-room.canvas.tsx`；配置`configs/experiment_v7.json`；方案`docs/experiment_v7_protocol.md`；模型`models/experiment_v7/`；批准记录`docs/approvals/target_specific_v7_approval.json`。

训练入口`scripts/run_target_specific_v7.ps1`拒绝覆盖运行。复核/重建报告可用`scripts/report_target_specific_v7.ps1`，会执行7个回归器的有限重复拟合。图像审阅后用项目conda Python执行`scripts/finalize_target_specific_v7.py --figures-reviewed`更新归档。图已检查，HTML静态脚本/链接已通过；浏览器交互与原生Canvas编译未验证。

七个房间已反复用于开发，本轮仍是开发证据。三个TEST房间未预测或评分。未新增几何、模拟或最终部署拟合。没有重新优化或评分角度决策，数值误差改善不直接等于推荐收益。当前EPW/时段/北向和网格来源限制仍存在。

## 仓库归档与模型试用（2026-11-08）

本项目已准备为私有Git仓库，保存原始数据、代码、配置、各轮模型、预测与报告；conda环境和缓存不提交。文件换行保持原始字节，以便核对实验哈希。

初次归档时尚未创建全7房间模型和统一接口；后续按用户批准已完成[试用版0.1](docs/trial_v0_1_usage.md)。独立TEST评估仍未开展，三个测试房间继续封存。初次归档前的评估保留在[模型使用就绪度](docs/model_usage_readiness.md)。
