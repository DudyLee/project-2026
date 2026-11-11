# Tower A PVSD：机器学习方法与结果

基于建筑年度逐时模拟数据，输入房间BIM特征、时间、天气和候选百叶角度，分别预测五项采光及发电指标。本仓库提供方法代码、结果说明和试用版0.2预测接口，包含0.2及可回退的0.1模型包和小型使用示例；不包含完整原始数据、批量逐小时预测或开发日志。

## 文档与结果

- [方法说明](METHODS.md)：输入输出、特征、模型和验证方式。
- [性能对比](RESULTS.md)：三种验证的平均与最差房间MAE，以及各输出的模型建议。方法名链接至对应代码。
- [历史MAE总表](results/history_mae.csv)：v1–v11原版及本次补充实验。
- [v8–v11结果汇总](results/summary_metrics.csv)：258项指标，包括平均误差和最差房间误差。
- [逐房间结果](results/room_metrics.csv)：1,462条房间级指标，用于核对汇总结果。

当前训练代码覆盖v8–v11的17个五输出方案及v8的P10专项对照；v1–v7保留历史结果追溯，不保留中途开发脚本。2026-11-09的补充实验记为`v8+`至`v11+`，不改写原版当时的实验范围。

## 使用已训练模型进行预测

试用版0.2已整合五个独立模型：平均照度、F300和PV采用v8 T1，P10和FUDI采用v10 X0。接口复用已有权重，无需重新训练。仓库已包含运行所需的接口、模型权重和示例。首次使用，在已安装conda的终端运行：

```bash
git clone https://github.com/DudyLee/project-2026.git
cd project-2026
conda env create --prefix ./.conda/envs/pvsd-ml-v1 -f environment.yml
conda activate ./.conda/envs/pvsd-ml-v1
python -m pvsd_trial --input examples/trial_v0_2/input.csv --output outputs/trial_v0_2/my_predictions.csv
```

`environment.yml`固定了接口校验的库版本，请勿自行升级这些依赖。已有本地完整项目可直接使用原环境。

Windows下也可在项目根目录使用脚本，自动调用上述项目内conda环境：

```powershell
.\scripts\predict_trial.ps1 -InputPath .\examples\trial_v0_2\input.csv -OutputPath .\outputs\trial_v0_2\my_predictions.csv
```

每行输入代表一个房间的一个小时，接口自动计算0°、30°、60°、90°四个角度，返回四行结果。五个输出为 `E_mean_lux`、`E_P10_lux`（lux），`F_300_fraction`、`F_UDI_fraction`（0–1），以及 `PV_AC_kWh`（该小时的kWh电能）。当前接口返回性能，不自动推荐角度。

使用自己的数据时，复制[输入模板](examples/trial_v0_2/input_template.csv)，填写6项房间特征、2项时间和5项天气／太阳位置字段。0.2新增必需字段 `Annual_Incident_Shaded_kWh_m2`；完整字段、单位和要求见[完整使用说明](docs/trial_v0_2_usage.md)。输入中不需要填写角度或真实性能值。

如果希望通过 `Room_ID` 自动补齐六项房间特征，可指定自行准备的BIM表（不包含在仓库中）；时间和天气仍由输入提供：

```powershell
.\scripts\predict_trial.ps1 -InputPath .\examples\trial_v0_2\input_lookup.csv -BimPath .\data\raw\SEND_BIM_476Rooms.csv -OutputPath .\outputs\trial_v0_2\my_lookup.csv
```

CSV和JSON均支持。输出文件已存在时换一个文件名，或加 `-Overwrite`。在已激活项目环境的Python中，可用 `from pvsd_trial import TrialPredictor`，随后调用 `TrialPredictor().predict(records)`，其中 `records` 为DataFrame或字典列表。

回退到0.1使用原输入格式：

```powershell
.\scripts\predict_trial.ps1 -ModelVersion 0.1 -InputPath .\examples\trial_v0_1\input.csv -OutputPath .\outputs\trial_v0_1\my_rollback.csv
```

Python可用 `TrialPredictor(version='0.1')`。0.1保留原来的12项非角度输入和测试房间ID限制，不支持0.2新增的年度辐射字段或BIM关联。

0.2整合后与原模型在全部175,200条记录上的预测一致，见[一致性记录](results/trial_v0_2_verification.json)。运行 `python -m unittest discover -s tests -p test_trial.py` 可检查发布包的示例预测、BIM关联及0.1回退。五个模型使用7个开发房间训练，但选取方案参考了已观察测试成绩；其效果与适用限制见[性能对比](RESULTS.md)。

## 代码结构

```text
pvsd_trial.py                默认0.2的统一预测入口
models/trial_v0_2/           五个已训练模型及校验信息
models/trial_v0_1/           回退模型包
examples/                   输入模板与示例预期结果
docs/trial_v0_2_usage.md      输入、输出与调用说明
configs/study.json           固定房间划分、样条设置、源数据指纹
configs/tree_parameters.json 树模型固定参数
scripts/methods.py           特征、目标变换、分组调参、训练及预测
scripts/train.py             完整复现实验入口
scripts/check_results.py     无需数据即可核对公开结果
tests/test_methods.py        数值和数据划分检查
```

五个输出分别选参和拟合。Ridge计算中可共用矩阵分解，各输出的回归系数和损失仍独立。误差按各输出的单位分别报告。

## 核对结果与复现实验

环境与原实验一致：Python 3.11、scikit-learn 1.7.2、XGBoost 3.1.3。

```bash
conda activate ./.conda/envs/pvsd-ml-v1
python -m unittest discover -s tests
python scripts/check_results.py
```

重跑训练需要自行取得原始合并数据`SEND_ML_BIM_EPW_07-18.csv`，格式见[方法说明](METHODS.md)。输入文件保留在本地，Git不会跟踪。先只检查数据：

```bash
python scripts/train.py --source data/raw/SEND_ML_BIM_EPW_07-18.csv --preflight
```

执行两类开发验证并拟合全7房间模型：

```bash
python scripts/train.py --source data/raw/SEND_ML_BIM_EPW_07-18.csv --output runs/reproduction
```

如需复评已观察的三个测试房间，加`--evaluate-test`。所有模型先固定，再统一预测测试输入，最后才读取测试标签评分。可以用`--versions 8 11`指定版本；新运行必须使用不存在的输出目录。

运行生成的权重、结果和必要冻结记录均留在`runs/`等本地目录。默认不输出逐小时预测；研究复核确有需要时可加`--save-predictions`。这些文件均不提交到仓库。

代码已通过输入检查、数值与分组测试、结果表重算，并与存档模型核对了预测结果。公开CSV来自已完成的实验，完整资料保存在本地。重新运行时可能存在浮点舍入差异。
