# 跨房间特征与P10实验 v5

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run + validate
- Verification: saved predictions/model reloads verified; no independent TEST

本轮没有方案同时通过预设的三项开发筛选条件。

新增特征的XGBoost平均照度MAE：82.433 → 81.769 lux，下降0.81%，6/7房间改善，仍未达到预先约定的10%幅度。

Random Forest直接asinh P10将BM223 MAE从9.757降至1.479 lux（下降84.8%），但BM432从29.275增至44.361 lux；7房间平均MAE从9.748增至10.743 lux（恶化10.2%）。局部改善不能支持全局替换。

平均照度最低房间平均MAE为81.769 lux（B_XGBoost）；这是本轮开发结果的事后描述，不是锁定模型的独立测试分数。

P10照度最低房间平均MAE为9.748 lux（A_RandomForest）；这是本轮开发结果的事后描述，不是锁定模型的独立测试分数。

BM223的P10照度：本轮最低MAE 1.479 lux（C_RandomForest），真实均值0.616 lux。最优方法按本房间事后查看，不能据此给新房间选择模型。

BM522的平均照度：本轮最低MAE 108.370 lux（B_XGBoost），真实均值152.811 lux。最优方法按本房间事后查看，不能据此给新房间选择模型。

四组对照：A现有特征＋相对P10；B增强特征＋相对P10；C现有特征＋直接asinh P10；D增强特征＋直接asinh P10。平均照度都使用log1p，A/C完全共用平均照度预测，B/D也完全共用。

增强特征仅增加Glass_Area_m2/Room_Area_m2、Window_Count、Window_Area_Nominal_m2/Window_Count。原始13项输入编码为15列，增强后为16项输入、18列编码。平均每窗面积不是窗户高度或位置；本轮没有提取尚缺的房间进深等几何信息。

相对P10：预测P10/(平均照度+1 lux)，推断时乘以该候选自己的预测平均照度+1 lux；直接P10：拟合asinh(P10/1 lux)，用sinh还原并截断负值。1 lux尺度事先固定，未用留出房间标签估计。两种方式推断时都不使用真实平均照度。

外层每次6个房间训练、1个房间评分。内层3折，每次4个房间训练、2个房间验证；每个模型家族沿用v3的两个参数候选，以内层6个验证房间的平均原单位MAE选参数。全部选择固定后才统一计算外层分数。

预先约定：相对同一算法A组，房间平均MAE至少下降10%，至少5/7房间不恶化，最差房间MAE不增加。逐目标判断；这是开发筛选规则，不是工程验收标准。

只有7个开发房间，且已用于多轮方法设计。内层分组调参不能消除累计开发选择偏差。BM532、BM702、BMG26仍封存；本轮无新模拟、无角度决策优化、无全部7房间部署重训。
