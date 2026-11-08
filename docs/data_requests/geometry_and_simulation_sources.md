# Tower A：已有工程资料与几何字段核对清单

本清单对应本次批准的“误差诊断＋资料整理”。当前不要求新模拟；不需要解封BM532、BM702、BMG26。请优先提供7个TRAIN房间的已有文件，重点BM223、BM432、BM522。

## 当前已知与缺口

本地Project_2026/data/raw有两张CSV和作者README。未发现能确认属于这批Tower A模拟的原始Revit/Rhino/Honeybee几何、传感器坐标、逐传感器照度或EPW。工作区其他课程BIM与Group 10压缩包没有建立关联，未当作本项目来源。此结论是本地盘点，不代表Google Drive中一定没有；本轮未重新读取Drive。

## 优先取得的已有文件

1. **同一模拟版本的房间、窗户和遮挡几何**：Revit/IFC、Rhino/Grasshopper、HBJSON任一种可追溯导出。需要单位、坐标系、Room_ID及窗口对象ID映射；允许仅导出TRAIN房间及其影响范围内遮挡。
2. **原传感器网格与参数**：坐标、所属房间、间距、离地高度、近墙点剔除、房间外点检查、面积加权规则。不能仅提供传感器数量。
3. **已有逐传感器照度结果和导出脚本**：按existing_sensor_result_sample_requests.csv列出的24个工况抽查。提供sensor_id、x/y/z、lux及Room_ID/DOY/Hour/角度。核对均值、numpy线性P10及同一网格映射；如原始结果不存在，请标记缺失，不重新运行全年模拟。
4. **材料与单位来源**：原玻璃材质、可见光参数、README所述玻璃面积除以10的换算来源。当前没有证据断定换算错误，但需记录其依据。
5. **EPW、时间与角度约定**：EPW文件及哈希、是否40组共同使用、Hour7的时间区间、太阳位置取时点、时区/夏令时、模型北向、PVSD零度和旋转轴/板法线定义。

## 可直接填写的文件

- TRAIN_room_source_index.csv：已有房间编号、楼层、面积、窗户与传感器数量及窗口元素ID，便于定位。
- geometry_field_dictionary.csv：拟补字段、单位、定义、来源与新房间预测时可获得性。
- ../../data/templates/TRAIN_room_geometry_to_complete.csv：7个房间的待填模板。空白表示未知；状态为MISSING_NOT_FOR_TRAINING，不是可用于训练的新数据。
- ../../data/templates/window_geometry_to_complete.csv：逐窗字段模板。多窗房间保留逐窗信息，避免凭总面积猜高度或位置。
- ../../data/templates/simulation_provenance_to_complete.json：时间、北向、角度及材料来源，未知项保持null。

## 后续使用原则

先完成房间/窗户/网格映射和单位核对，再冻结少量几何特征。只有新房间预测时也能获得的字段才可入模。网格质检字段不自动变成训练特征，不用Sensor_Count作为房间ID替代品。

相对太阳角和入射角需要北向、太阳时间及PVSD定义一致后才能计算；房间在模型坐标下的几何长度提取则可先进行。几何特征提取本身不是新增全年模拟。

官方参考：Honeybee网格组件支持grid_size、dist_floor、wall_offset和remove_out；这些参数需要记录实际工程值，不能把软件默认值当成我们的设置。
https://docs.ladybug.tools/hb-radiance-primer/components/0_basicproperties/sensor_grid_from_rooms

本清单仅保存于项目，未向模拟作者发送消息。
