# Tower A PVSD：机器学习方法与结果

基于建筑年度逐时模拟数据，输入房间BIM特征、时间、天气和候选百叶角度，分别预测五项采光及发电指标。本仓库用于公开方法与结果说明，不包含原始数据、训练权重、逐小时预测或开发日志。

## 从这里开始

- [方法说明](METHODS.md)：输入输出、特征、模型和验证方式。
- [可点击性能对比表](RESULTS.md)：三种验证的平均/最差MAE、综合候选建议；点击方法名直达代码。
- [历史MAE总表](results/history_mae.csv)：v1–v11原版及本次补充实验。
- [最新方案汇总](results/summary_metrics.csv)：258项指标，包括平均误差和最差房间。
- [逐房间结果](results/room_metrics.csv)：1,462条房间级指标，用于核对汇总结果。

当前训练代码覆盖v8–v11的17个五输出方案及v8的P10专项对照；v1–v7保留历史结果追溯，不保留中途开发脚本。2026-11-09的补充实验记为`v8+`至`v11+`，不改写原版当时的实验范围。

## 代码结构

```text
configs/study.json           固定房间划分、样条设置、源数据指纹
configs/tree_parameters.json 树模型固定参数
scripts/methods.py           特征、目标变换、分组调参、训练及预测
scripts/train.py             完整复现实验入口
scripts/check_results.py     无需数据即可核对公开结果
tests/test_methods.py        数值和数据划分检查
```

五个输出分别选参和拟合。Ridge允许共用矩阵分解，但各输出的回归系数和损失独立；没有把不同单位的误差混成一个“准确率”。

## 运行

环境与原实验一致：Python 3.11、scikit-learn 1.7.2、XGBoost 3.1.3。

```bash
conda env create -f environment.yml
conda activate pvsd-ml-v1
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

本次代码整理通过了输入预检、数值/分组测试、公开结果重算及与本地存档模型的推断一致性核对；没有启动新一轮全年训练。原实验的完整资料仍保留在本地，公开CSV属于当时已完成的实验结果。精简入口重新运行全部目标，可能有浮点舍入差异，不承诺字节级复现。
