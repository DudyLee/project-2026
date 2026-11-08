# Tower A PVSD 试用版 0.1 使用说明

本版本用全部7个开发房间的122,640行数据拟合，输入同一房间在某小时的条件，返回0°、30°、60°、90°各自的5项预测。模型已经固定，使用时不训练。三个测试房间BM532、BM702、BMG26继续封存，接口会拒绝这些Room_ID。

适合现阶段的演示、性能比较、Python/Grasshopper/Revit接口联调；新的房间、天气或真实运行准确性尚未独立验证。它不会自动推荐角度或控制设备。此前的开发交叉验证分数不等于本模型包的最终测试分数。

## 1. 在当前电脑运行

打开PowerShell，复制以下命令。无需激活conda，脚本自动使用项目现有环境。

```powershell
Set-Location 'C:\NewDisk\UST\Year 2\5220\Project_2026'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\predict_trial.ps1 -InputPath .\examples\trial_v0_1\input.csv -OutputPath .\outputs\trial_v0_1\my_predictions.csv
```

输入示例是7个开发房间各一个小时，共7行；输出应为28行，每行是一种角度的5项预测。打开 `outputs/trial_v0_1/my_predictions.csv` 即可比较。同目录有 `examples/trial_v0_1/expected_output.csv` 可供核对。示例由模型训练所用房间构成，不能用于证明泛化准确性。

输出文件已存在时，换一个文件名，或在命令末尾加 `-Overwrite`。相对路径按执行命令时的当前目录解释；在其他位置运行也可以传入完整路径。所有自建输入、输出请保留在Project_2026内。

JSON输入方式相同（必须是对象数组，数字使用JSON数值）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\predict_trial.ps1 -InputPath .\examples\trial_v0_1\input.json -OutputPath .\outputs\trial_v0_1\my_predictions.json
```

## 2. 换成自己的房间与小时

复制 `examples/trial_v0_1/input_template.csv`，填入数据并另存为UTF-8 CSV。每行填12个数值字段，可以额外填Room_ID、Request_ID两个文本标识。不要直接传入包含真实输出或其他字段的36列原始数据库。

| 输入列名 | 含义与单位 | 约束 |
|---|---|---|
| Room_Area_m2 | 房间面积，m² | >0 |
| Glass_Area_m2 | 与模拟导出一致的玻璃面积，m² | ≥0，勿直接用另一种名义窗面积代替 |
| Orientation_X_model | 原模型坐标下朝向向量X | 与Y组成单位向量 |
| Orientation_Y_model | 原模型坐标下朝向向量Y | 不是地理方位角 |
| Solar_Obstruction_Ratio | 原数据定义的太阳遮挡比 | 0–1 |
| DOY_1based | 年内第几天 | 整数1–365，不支持闰年第366天 |
| Hour | 与原模拟相同的小时标签 | 整数7–18 |
| EPW_GHI_Wh_m2 | 该小时全球水平辐射量 | Wh/m²，≥0 |
| EPW_DNI_Wh_m2 | 该小时法向直射辐射量 | Wh/m²，≥0 |
| EPW_DHI_Wh_m2 | 该小时水平散射辐射量 | Wh/m²，≥0 |
| Solar_Altitude_mid_deg | 对应时刻太阳高度角 | 度，−90–90 |
| Solar_Azimuth_mid_deg | 对应时刻太阳方位角 | 度，0–360，沿用原数据约定 |

Room_ID不参与预测；只填Room_ID不能自动查出其BIM与天气。Request_ID用于关联自己的请求。缺失值、非有限值、真实输出列、自填PVSD_Angle_deg、未知列、重复列或非法范围将被拒绝。所有必需数值字段都要填；不要用0代表“不知道”。

模型内部仍是13项原始输入：上述12项＋自动枚举的PVSD_Angle_deg；DOY和太阳方位角转换成sin/cos后共15列。四个角度共享同一套模型参数。

天气输入应保持与原模拟相同的来源、时区和小时区间。当前EPW身份、Hour区间、太阳位置时间及模型北向仍待核实；软件只能验证格式/范围，不能替你验证这些物理对应关系。辐射列不要未经核对就混用瞬时W/m²和逐时Wh/m²。

## 3. 输出如何理解

每条输入返回4行，按0/30/60/90°排序。保留Input_Row（从1开始）、Room_ID、Request_ID、DOY、Hour、PVSD_Angle_deg，以及：

| 输出列 | 含义 | 单位 |
|---|---|---|
| E_mean_lux | 该小时房间传感器平均照度 | lux |
| E_P10_lux | 传感器照度第10百分位 | lux；不是最暗10%的平均值 |
| F_300_fraction | 照度≥300 lux的传感器空间比例 | 0–1；0.70表示70% |
| F_UDI_fraction | 照度在100–3000 lux的传感器空间比例 | 0–1；不是年度UDI |
| PV_AC_kWh | 房间案例对应PV系统该小时交流发电量 | kWh；不是kW |

负值裁剪到0，比例裁剪到[0,1]。这不保证5个独立输出之间的全部物理一致性。返回Model_Version和Model_SHA256以追踪精确模型，Usage_Status始终标明研究试用。

Coverage_Flags的含义：

- `PROVENANCE_UNCONFIRMED`：原始天气/时间/方位等来源仍未完全核实，所有记录都有。
- `DEVELOPMENT_ROOM`：标识属于7个拟合房间；这不等于独立测试通过。
- `NEW_OR_UNSPECIFIED_ROOM`：新房间或未填Room_ID。
- `UNSEEN_ROOM_FEATURE_COMBINATION`：5项房间特征组合不匹配任何开发房间。
- `OUTSIDE_TRAIN_RANGE:列名`：某输入超出训练数据该列最小/最大范围。
- `UNSEEN_MODEL_ORIENTATION`：朝向向量不在训练房间中。
- `ROOM_ID_GEOMETRY_MISMATCH`：填了开发房间ID，但几何特征与其训练记录不同。

这些是输入覆盖提示，不是置信区间或准确率。全部特征在范围内也不代表新房间预测可靠。标识可能填错，封存拦截也无法识别被匿名化或改名的测试房间；请继续遵守不输入三个TEST房间数据的约定。

## 4. Python统一接口

在项目根目录、项目conda环境中运行：

```python
import json
from pathlib import Path
from pvsd_trial import TrialPredictor

model = TrialPredictor()  # 启动时加载一次，后续重复使用
conditions = json.loads(Path('examples/trial_v0_1/input.json').read_text(encoding='utf-8'))
result = model.predict(conditions)  # 也接受pandas.DataFrame；每条输入返回4行
Path('outputs/trial_v0_1').mkdir(parents=True, exist_ok=True)
result.to_csv('outputs/trial_v0_1/python_predictions.csv', index=False)
```

接口无需读取原始CSV、BIM数据库或真实输出，也不联网。可让Grasshopper/Revit导出条件CSV，再调用PowerShell入口、读取结果CSV；本轮尚未开发Revit/Grasshopper插件或HTTP服务。选择角度时可人工比较四行结果，采光约束、偏好权重和控制规则需要单独定义/验证。

## 5. 文件、模型与复现

- `pvsd_trial.py`：独立Python/CLI推断模块。
- `scripts/predict_trial.ps1`：当前Windows环境的快捷入口。
- `models/trial_v0_1/`：模型、训练范围/参数和SHA-256清单；运行时必须保留全部三个文件。
- `configs/trial_v0_1.json`、`reports/trial_v0_1/parameter_freeze.json`：冻结规则与最终参数。
- `reports/trial_v0_1/verification.json`：接口验证记录，不是新房间准确率成绩。
- `reports/trial_v0_1/summary.md`：训练与软件检查结果。
- `examples/trial_v0_1/`：CSV/JSON示例、空模板与已核对输出。

5个目标使用不同回归器：平均照度为XGBoost/log1p；P10为RF相对值模型＋自己的RF均值模型；F300、FUDI、PV分别为固定参数HistGB。因此共6个估计器，调用方只看到统一5项输出。P10不使用调用者提供的真实平均照度，也不使用独立XGBoost的均值输出。

换电脑时，在克隆的项目根目录建立相同依赖环境（联网安装可用）：

```powershell
conda create --prefix .\.conda\envs\pvsd-ml-v1 python=3.11.18 pip -y
conda run --prefix .\.conda\envs\pvsd-ml-v1 python -m pip install -r .\configs\trial_v0_1_runtime_requirements.txt
```

这份精确版本清单用于推断；整个科研项目环境仍见environment.yml和既有精确环境记录。接口启动会核对主要依赖版本及模型文件SHA-256。只加载本项目可信模型：joblib使用pickle，不能加载未知来源的模型文件。若更换模型或参数，应创建新版本目录，勿覆盖本版。

如需重新训练以复现，使用 `scripts/train_trial_v0_1.py`；已有本版输出时脚本会拒绝覆盖。复现应在独立项目副本或新版本目录进行。不要为了运行预测而重新训练。
