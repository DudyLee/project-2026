# 性能对比与结果说明

更新：2026-11-09

本页汇总v8–v11的补测结果，“+”表示补充实验。已有开发结果保留来源标记，新增目标已完成训练与验证。

## 各输出的模型建议

以下建议综合考虑留一房间、留一朝向和测试结果，兼顾平均误差与最差房间误差。该组合尚未部署。表中每格为平均MAE / 最差房间MAE，全部方法的结果见后文。

|输出|建议方法|留一房间|留一朝向|三房间测试|
|---|---|---:|---:|---:|
|平均照度（lux）|[v11 CB](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L257-L259)|53.28 / 147.74|66.87 / 194.43|77.89 / 111.33|
|P10（lux）|[v8 T0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|9.40 / 29.28|9.45 / 29.31|18.68 / 43.25|
|F300（百分点）|[v11 Aq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|5.20 / 8.63|5.58 / 8.84|6.34 / 10.07|
|FUDI（百分点）|[v10 X0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|7.80 / 15.24|10.19 / 16.15|6.81 / 12.63|
|PV（kWh/小时）|[v8 T1](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|0.00896 / 0.01839|0.01099 / 0.02277|0.00310 / 0.00422|

- 平均照度建议使用CB，其留一房间与留一朝向结果较均衡，B0为备选。P10建议保留T0：X0虽然降低了测试房间BM702的误差，却明显增加了开发房间BM223的误差。
- F300建议使用Aq，主要考虑其最差房间误差；FUDI建议使用X0。两者均以v9 B为备选。PV建议使用T1，其平均和最差房间MAE在三类验证中均为最低。
- P10计算依赖配套的RF平均照度模型，替换为CB输出会改变原方法。上述建议已参考测试成绩，后续在同一测试集上的结果属于复评。

方法名跳转到固定提交中的定义分支；CA/CB直接跳到融合函数。同版本方法通过公共函数和配置实现，因此部分链接会定位同一分支。进一步查看：[特征](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L73-L94) · [目标变换与还原](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L134-L164) · [内层划分](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L166-L174) · [选参与训练](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L192-L236) · [v8各输出树参数](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/configs/tree_parameters.json)。

## 结果分析

- 五项输出的最佳方法不同，版本升级未带来一致改善。
- 测试中v8 T1平均照度MAE为59.53 lux、PV为0.00310 kWh/小时；v10 X0的P10为11.38 lux、FUDI为6.81个百分点。
- P10的相对预测方案在开发验证中更好，但在测试中的排名不同，选模需要考虑这一差异。
- v10 S0平均照度留一房间MAE为48.04 lux，留一朝向和测试MAE分别为115.46和86.29 lux，跨朝向预测仍不稳定。
- v11针对平均照度的改进未在P10和PV上取得相同收益。试用版0.1仍使用原模型。

## 数据文件与指标

|文件|范围|单位|
|---|---|---|
|[summary_metrics.csv](results/summary_metrics.csv)|258项版本/方法/目标/验证汇总，含最差房间|MAE为原始单位；比例为fraction|
|[room_metrics.csv](results/room_metrics.csv)|1,462条逐房间误差，可重算汇总|同上|
|[history_mae.csv](results/history_mae.csv)|v1–v11原版、试用版及补充实验历史总表|照度lux；F300/FUDI为百分点；PV为kWh|

`archived_experiment_*`是沿用开发结果，`supplement_new_fit`是新增开发结果，`supplement_all7_frozen`是本次全7拟合后的测试结果。历史表的空白表示当时未测；S1_direct是P10专项，其他输出不适用。历史表中不同验证方式不可混作一次测试。

以下每格为 **平均MAE / 最差房间MAE**，数值越低越好。平均值按房间等权计算，最差值取单房间MAE的最大值。F300和FUDI的单位为百分点，PV为每条小时记录的kWh误差。粗体表示同一验证、同一输出中最低的平均MAE，按未四舍五入的数值判断；模型建议还考虑最差房间和其他验证结果。

## 7开发房间：留一房间

|版本|方法|平均照度 lux|P10 lux|F300 百分点|FUDI 百分点|PV kWh/小时|
|---|---|---:|---:|---:|---:|---:|
|v8+|[T0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|82.43 / 231.64|**9.40** / 29.28|7.12 / 15.69|11.04 / 25.47|0.01261 / 0.02381|
|v8+|[T1](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|70.27 / 223.46|9.61 / 33.92|6.21 / 13.26|11.35 / 29.03|**0.00896** / 0.01839|
|v8+|[S0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|76.13 / 252.29|11.81 / 41.73|7.43 / 15.05|9.87 / 22.09|0.01869 / 0.03735|
|v8+|[S1](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|64.38 / 222.83|14.05 / 43.98|7.36 / 14.62|9.51 / 22.37|0.01759 / 0.03377|
|v8+|[S1_direct](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|—|12.34 / 53.44|—|—|—|
|v9+|[A](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L130-L130)|84.66 / 230.83|15.85 / 46.15|7.26 / 14.51|9.77 / 20.78|0.01332 / 0.03209|
|v9+|[B](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L130-L130)|83.04 / 234.72|15.63 / 44.06|5.88 / 10.78|7.95 / 16.59|0.01394 / 0.03163|
|v9+|[C](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L130-L130)|64.94 / 171.20|13.01 / 48.42|5.83 / 11.94|13.67 / 26.44|0.02030 / 0.04592|
|v10+|[X0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|70.46 / 219.99|13.99 / 40.37|6.14 / 11.16|**7.80** / 15.24|0.01481 / 0.03574|
|v10+|[Xq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|55.01 / 137.00|18.35 / 69.90|5.32 / 8.64|13.50 / 23.88|0.02090 / 0.04528|
|v10+|[S0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|**48.04** / 112.54|13.86 / 57.88|5.75 / 10.12|9.91 / 23.18|0.02007 / 0.03199|
|v10+|[Sq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|61.04 / 178.63|14.21 / 58.49|5.46 / 10.52|8.32 / 16.49|0.02164 / 0.04347|
|v11+|[A0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|66.59 / 194.74|14.90 / 57.88|6.09 / 13.84|11.57 / 28.48|0.02128 / 0.03733|
|v11+|[Aq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|59.95 / 178.63|13.53 / 51.64|**5.20** / 8.63|9.25 / 21.87|0.02223 / 0.04347|
|v11+|[B0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|50.93 / 124.31|14.79 / 57.57|6.00 / 11.17|9.64 / 22.83|0.02172 / 0.03697|
|v11+|[Bq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|61.72 / 185.53|14.89 / 51.10|5.33 / 9.75|9.31 / 21.93|0.02218 / 0.04064|
|v11+|[CA](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L257-L259)|61.49 / 184.57|12.05 / 54.75|5.53 / 10.97|9.50 / 25.15|0.01880 / 0.02882|
|v11+|[CB](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L257-L259)|53.28 / 147.74|13.50 / 54.33|5.50 / 9.56|8.84 / 22.12|0.01866 / 0.03152|

## 7开发房间：留一朝向

|版本|方法|平均照度 lux|P10 lux|F300 百分点|FUDI 百分点|PV kWh/小时|
|---|---|---:|---:|---:|---:|---:|
|v8+|[T0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|110.61 / 252.25|9.45 / 29.31|7.37 / 15.72|11.23 / 24.95|0.01542 / 0.02498|
|v8+|[T1](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|84.03 / 224.67|**9.08** / 30.22|7.12 / 17.73|11.49 / 28.71|**0.01099** / 0.02277|
|v8+|[S0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|110.95 / 230.25|10.88 / 43.35|7.58 / 14.25|10.97 / 20.98|0.02272 / 0.03873|
|v8+|[S1](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|91.49 / 198.62|10.73 / 43.76|7.14 / 13.80|10.76 / 21.00|0.02167 / 0.03820|
|v8+|[S1_direct](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|—|12.77 / 55.92|—|—|—|
|v9+|[A](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L130-L130)|90.54 / 232.97|14.74 / 46.15|6.43 / 12.75|9.58 / 20.78|0.01355 / 0.02860|
|v9+|[B](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L130-L130)|82.58 / 239.45|17.19 / 44.06|5.50 / 9.80|**9.41** / 17.06|0.01445 / 0.03103|
|v9+|[C](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L130-L130)|83.37 / 247.45|23.99 / 96.95|6.70 / 12.05|15.19 / 31.49|0.02251 / 0.05318|
|v10+|[X0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|75.07 / 225.78|14.74 / 40.37|6.25 / 11.29|10.19 / 16.15|0.01591 / 0.03702|
|v10+|[Xq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|78.63 / 210.32|21.94 / 92.69|6.82 / 11.48|15.23 / 28.10|0.02319 / 0.05681|
|v10+|[S0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|115.46 / 480.14|51.07 / 289.72|5.51 / 8.07|13.02 / 26.25|0.01979 / 0.03367|
|v10+|[Sq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|68.02 / 209.20|63.75 / 378.98|5.95 / 10.77|11.11 / 21.87|0.02388 / 0.04637|
|v11+|[A0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|70.50 / 198.25|15.14 / 64.09|5.82 / 10.39|13.04 / 25.29|0.01913 / 0.03662|
|v11+|[Aq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|69.22 / 206.36|19.54 / 67.20|5.58 / 8.84|12.66 / 25.15|0.02361 / 0.04637|
|v11+|[B0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|70.73 / 204.97|17.45 / 63.80|5.84 / 10.32|11.51 / 25.18|0.01919 / 0.03575|
|v11+|[Bq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|69.40 / 205.97|22.64 / 89.76|5.57 / 8.65|12.75 / 25.13|0.02458 / 0.05632|
|v11+|[CA](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L257-L259)|**66.59** / 193.83|16.99 / 64.94|5.45 / 8.18|12.56 / 23.35|0.01767 / 0.03892|
|v11+|[CB](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L257-L259)|66.87 / 194.43|19.78 / 69.62|**5.45** / 7.97|11.60 / 25.10|0.01753 / 0.04139|

## 3个已观察测试房间

|版本|方法|平均照度 lux|P10 lux|F300 百分点|FUDI 百分点|PV kWh/小时|
|---|---|---:|---:|---:|---:|---:|
|v8+|[T0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|111.07 / 236.97|18.68 / 43.25|6.53 / 15.43|12.03 / 27.04|0.00767 / 0.01434|
|v8+|[T1](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|**59.53** / 97.39|19.72 / 46.47|**5.21** / 11.32|11.28 / 25.78|**0.00310** / 0.00422|
|v8+|[S0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|122.19 / 176.16|19.61 / 31.00|10.28 / 15.15|12.24 / 18.95|0.01211 / 0.01596|
|v8+|[S1](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|84.04 / 125.88|18.62 / 28.06|8.80 / 11.62|11.47 / 17.19|0.01160 / 0.01603|
|v8+|[S1_direct](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L128-L129)|—|21.58 / 34.34|—|—|—|
|v9+|[A](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L130-L130)|71.09 / 147.44|12.71 / 24.36|5.69 / 11.41|7.80 / 12.18|0.00390 / 0.00583|
|v9+|[B](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L130-L130)|70.65 / 129.44|13.03 / 24.02|5.22 / 11.12|7.45 / 11.99|0.00360 / 0.00530|
|v9+|[C](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L130-L130)|62.20 / 103.62|13.16 / 24.95|5.67 / 11.37|7.02 / 11.62|0.00539 / 0.00727|
|v10+|[X0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|66.65 / 120.52|**11.38** / 22.76|5.64 / 12.09|**6.81** / 12.63|0.00356 / 0.00572|
|v10+|[Xq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|65.96 / 100.78|11.79 / 25.12|5.72 / 10.16|8.57 / 13.62|0.00951 / 0.01681|
|v10+|[S0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|86.29 / 134.61|24.00 / 42.85|6.46 / 10.89|10.05 / 14.87|0.01119 / 0.01509|
|v10+|[Sq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L131-L131)|92.27 / 151.43|24.82 / 48.13|6.06 / 9.23|9.45 / 15.16|0.02290 / 0.04323|
|v11+|[A0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|105.18 / 152.59|25.61 / 39.06|6.64 / 10.86|12.08 / 17.78|0.01119 / 0.01509|
|v11+|[Aq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|92.27 / 151.43|22.59 / 33.88|6.34 / 10.07|10.19 / 16.11|0.02777 / 0.04750|
|v11+|[B0](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|76.48 / 108.05|26.02 / 39.58|6.47 / 10.62|9.02 / 13.49|0.01128 / 0.01499|
|v11+|[Bq](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L132-L132)|102.23 / 144.64|23.19 / 34.89|6.29 / 10.14|10.20 / 17.18|0.02805 / 0.04608|
|v11+|[CA](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L257-L259)|83.09 / 113.30|24.07 / 36.39|6.35 / 10.37|9.52 / 14.48|0.01530 / 0.02427|
|v11+|[CB](https://github.com/DudyLee/project-2026/blob/d9760cf6b4e8e0fa689bdec6c53b6f6200365a34/scripts/methods.py#L257-L259)|77.89 / 111.33|24.56 / 37.16|6.28 / 10.31|8.78 / 13.75|0.01557 / 0.02401|

## 结果核对与局限

原实验已核对模型与源文件哈希、参数选择、测试前的模型固定记录及保存模型的预测结果，并从新增预测重新计算了1,098条房间指标。v11 B0/CB的原平均照度结果在数值精度内复现。仓库中的检查程序用于核对汇总表和逐房间表；从逐小时预测重新计算指标需要本地原始资料。

实验仅覆盖7个开发房间和3个测试房间，测试结果也已用于后续分析。这些成绩说明了模型在现有房间上的表现，对其他房间的适用性仍需进一步验证。
