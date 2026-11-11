# PVSD试用版0.2使用说明

默认接口为0.2。它复用已训练的五个回归器，没有重新训练。每条房间／小时输入自动展开为0°、30°、60°、90°，每个角度返回五项性能。模型用7个开发房间训练，方法按已观察测试集的平均MAE选定；已有测试成绩不能作为该组合新的独立验证。

|输出|方案|实际模型|
|---|---|---|
|E_mean_lux|v8 T1|F1方向交互特征，XGBoost＋log1p|
|E_P10_lux|v10 X0|F2含q特征，XGBoost＋log1p直接预测|
|F_300_fraction|v8 T1|F1，独立HistGB直接回归|
|F_UDI_fraction|v10 X0|F2，独立XGBoost＋log1p|
|PV_AC_kWh|v8 T1|F1，独立HistGB直接回归|

P10不再需要配套RF均值模型。接口只做性能预测，不选择推荐角度，不控制Revit或Grasshopper。

## 安装环境

克隆仓库后，在项目根目录执行：

```bash
conda env create --prefix ./.conda/envs/pvsd-ml-v1 -f environment.yml
conda activate ./.conda/envs/pvsd-ml-v1
python -m pvsd_trial --input examples/trial_v0_2/input.csv --output outputs/trial_v0_2/my_predictions.csv
```

仓库包含0.2和0.1模型包，不需要重新训练。输入完整字段时，不需要原始模拟数据。接口会校验numpy、pandas、scikit-learn、XGBoost、joblib的版本，环境声明已固定对应版本。已有完整本地项目可沿用原环境。

## 直接运行示例

在项目根目录的PowerShell中运行，脚本自动使用上述项目内conda环境：

```powershell
.\scripts\predict_trial.ps1 -InputPath .\examples\trial_v0_2\input.csv -OutputPath .\outputs\trial_v0_2\my_predictions.csv
```

示例输入包含7个开发房间各一个小时，输出28行。开发房间示例用于检查用法，不作为新房间效果证明。输出文件已存在时，换一个文件名，或显式加`-Overwrite`。CSV和JSON均可用，JSON输入须为对象数组，数值字段须使用数值类型。

如果PowerShell阻止执行脚本，可使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\predict_trial.ps1 -InputPath .\examples\trial_v0_2\input.json -OutputPath .\outputs\trial_v0_2\my_predictions.json
```

## 输入字段

每行对应一个房间的一个小时，提供13个非角度数值字段。可额外提供Room_ID、Request_ID文本标识。PVSD_Angle_deg由接口生成，不接受输入的真实输出列或其他未知字段。

|字段|含义／单位|要求|
|---|---|---|
|Room_Area_m2|房间面积，m²|大于0|
|Glass_Area_m2|玻璃面积，m²|大于0|
|Orientation_X_model|模型坐标中的朝向X|与Y组成单位向量|
|Orientation_Y_model|模型坐标中的朝向Y|与原模拟坐标一致|
|Solar_Obstruction_Ratio|遮挡比|0–1|
|Annual_Incident_Shaded_kWh_m2|年度受遮挡入射辐射，kWh/m²|大于0；0.2新增必需字段|
|DOY_1based|年内日序号|整数1–365|
|Hour|模拟小时标签|整数7–18|
|EPW_GHI_Wh_m2|逐小时水平总辐射，Wh/m²|非负|
|EPW_DNI_Wh_m2|逐小时法向直射辐射，Wh/m²|非负|
|EPW_DHI_Wh_m2|逐小时水平散射辐射，Wh/m²|非负|
|Solar_Altitude_mid_deg|太阳高度角，°|−90–90|
|Solar_Azimuth_mid_deg|太阳方位角，°|0–360|

复制`examples/trial_v0_2/input_template.csv`填写自己的数据。所有必需输入须有限且非空，不用0代替未知值。玻璃面积和年度受遮挡入射辐射为0时，无法满足现有q特征的适用条件，接口会拒绝。太阳位置、辐射量和小时约定应与原模拟一致；这些来源及模型北向仍有待完整核实。

## 按Room_ID关联本地BIM表

BIM表不随仓库发布，需自行提供。显式指定BIM表后，只需提供Room_ID、时间和五项天气／太阳位置字段；六个房间字段由表中关联。接口不自动查询天气、不猜测缺失数值。

```powershell
.\scripts\predict_trial.ps1 -InputPath .\examples\trial_v0_2\input_lookup.csv -BimPath .\data\raw\SEND_BIM_476Rooms.csv -OutputPath .\outputs\trial_v0_2\lookup_predictions.csv
```

BIM表Room_ID须唯一且能找到每个请求房间。若同时提供房间字段，接口会核对其与BIM表是否一致；冲突时报错，不覆盖用户值。不指定`-BimPath`时，Room_ID只作标识，不触发查询。

476个房间可在字段满足接口要求时调用模型，但只有10个模拟房间具有当前验证证据。其他房间的预测能力不能由接口运行成功推断。

## Python调用

使用项目conda环境，在项目根目录执行：

```python
import pandas as pd
from pvsd_trial import TrialPredictor

model = TrialPredictor()  # 默认0.2，启动时加载一次
conditions = pd.read_csv('examples/trial_v0_2/input.csv')
result = model.predict(conditions)  # 也接受list[dict]
result.to_csv('outputs/trial_v0_2/python_predictions.csv', index=False)
```

使用BIM关联：

```python
model = TrialPredictor(bim_path='data/raw/SEND_BIM_476Rooms.csv')
conditions = pd.read_csv('examples/trial_v0_2/input_lookup.csv')
result = model.predict(conditions, batch_size=5000)
```

直接CLI入口为`python -m pvsd_trial --input ... --output ...`，可加`--bim`、`--version 0.1`、`--batch-size`、`--overwrite`。使用已有项目环境即可，无需安装新环境。0.2沿用0.1的运行依赖；接口会核对主要库版本与模型文件哈希。

## 输出单位与标识

每条输入返回4行，按0/30/60/90°排列。Input_Row从1开始，保留Room_ID、Request_ID、DOY_1based、Hour及PVSD_Angle_deg。

|输出|单位|
|---|---|
|E_mean_lux、E_P10_lux|lux|
|F_300_fraction、F_UDI_fraction|0–1，0.70表示70%|
|PV_AC_kWh|该小时的kWh电能，乘1000才是Wh|

照度、PV限制为非负，比例限制为[0,1]。五个输出独立预测，范围约束不保证所有输出之间的物理一致性。

Model_Version为`0.2.0`，Model_SHA256标识精确模型包，Usage_Status为`RESEARCH_TRIAL_TEST_INFORMED_SELECTION`。Coverage_Flags描述输入覆盖：

- `DEVELOPMENT_ROOM`：属于7个拟合房间。
- `OBSERVED_TEST_ROOM`：属于BM532、BM702或BMG26；0.2允许这些已观察测试房间正常预测。
- `NEW_OR_UNSPECIFIED_ROOM`：不属于开发房间，或未填ID；可与OBSERVED_TEST_ROOM同时出现。
- `UNSEEN_ROOM_FEATURE_COMBINATION`、`UNSEEN_MODEL_ORIENTATION`：房间特征组合或朝向未见于训练房间。
- `OUTSIDE_TRAIN_RANGE:字段`：超出开发数据范围。
- `ROOM_ID_GEOMETRY_MISMATCH`：开发房间ID与所填房间特征不符。
- `PROVENANCE_UNCONFIRMED`：原始天气、时间、方位来源仍未完全核实。

这些提示不是置信区间，也不保证误差大小。

## 回退到0.1

0.1模型包和原接口实现均保留，回退不需要重训或复制模型：

```powershell
.\scripts\predict_trial.ps1 -ModelVersion 0.1 -InputPath .\examples\trial_v0_1\input.csv -OutputPath .\outputs\trial_v0_1\rollback_predictions.csv
```

```python
legacy = TrialPredictor(version='0.1')
```

0.1仍使用原来的12个非角度输入，不接受0.2新增的年度辐射字段或BIM查询。其原有测试房间ID拦截、输出状态文字均保持不变，属于历史接口行为。固定调用0.1可显式传`version='0.1'`，或使用`pvsd_trial_v0_1.TrialPredictor`。省略版本仍调用0.2。

## 模型包与验证

- `models/trial_v0_2/`：model.joblib、metadata.json、manifest.json，运行时保留全部三个文件。
- `pvsd_trial.py`：统一入口；`pvsd_trial_v0_1.py`、`pvsd_trial_v0_2.py`：各版本实现。
- `tests/test_trial.py`：无需原始数据的预测检查；`results/trial_v0_2_verification.json`：原完整数据一致性核对记录。
- `examples/trial_v0_2/expected_output.csv`：示例预期结果。

已核对全部175,200条记录：新接口与来源模型的五项预测最大绝对差均为0；与已保存的52,560条测试预测也完全一致。CSV/JSON、批处理与BIM关联结果一致，0.1回退与原示例在浮点精度内一致。验证过程禁用了相关模型的fit方法，新增训练次数为0。

模型包可离线运行；完整字段方式不需要原始数据库。BIM关联方式额外需要指定BIM表。只加载本项目可信的joblib模型文件。发布包直接复用已核验模型，未重新训练。
