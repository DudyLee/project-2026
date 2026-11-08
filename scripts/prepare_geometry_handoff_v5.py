"""Prepare source inventory and empty, explicitly unverified handoff templates."""
import csv
import json
from pathlib import Path
import zipfile
import train_cross_room_v5 as T

ROOT=T.ROOT;OUT=ROOT/'reports/cross_room_diagnosis_v5';DOC=ROOT/'docs/data_requests';TPL=ROOT/'data/templates'
for p in [DOC,TPL]:p.mkdir(parents=True,exist_ok=True)
cfg=T.read(ROOT/'configs/diagnosis_v5.json');run=T.read(OUT/'run_manifest.json');rooms=run['train_rooms']
known={}
with (ROOT/cfg['source']).open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['Sample_Role']!='TRAIN':continue
        if r['Room_ID'] not in known:
            known[r['Room_ID']]={k:r[k] for k in ['Room_ID','Floor_0based','Room_Area_m2','Window_Count','Sensor_Count','Revit_Window_Element_IDs']}
assert sorted(known)==rooms
with (DOC/'TRAIN_room_source_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(next(iter(known.values()))));w.writeheader();w.writerows(known[r] for r in rooms)
fields=[
 ('Room_Depth_PrimaryFacade_m','m','主要外窗立面法线方向的房间地板几何投影跨度；多采光面房间另记主立面选取规则','房间边界＋窗所在立面','是，几何可取得时'),
 ('Room_Width_PrimaryFacade_m','m','主要外窗立面平行方向的房间地板几何投影跨度','房间边界＋窗所在立面','是'),
 ('Ceiling_Height_m','m','房间净高；非均匀净高需另附几何而非猜测单值','房间几何','是'),
 ('Workplane_Height_m','m','原照度传感器相对房间地板的实际高度','原网格/模拟参数','需统一预测网格定义'),
 ('Window_Head_AboveWorkplane_m','m','窗顶相对实际工作面的高度；多窗逐窗填写或注明汇总规则','窗几何＋网格高度','是，规则一致时'),
 ('Window_Sill_Height_m','m','窗台相对房间地板的高度；多窗保留逐窗数据','窗几何','是'),
 ('Glazing_Visible_Transmittance','fraction','可见光透射率；注明与原Radiance材质参数的对应方式，不直接假定相等','材料原始定义','需已有材料参数'),
 ('Grid_Size_m','m','原传感器网格尺寸；非均匀网格须提供坐标与权重','原GH/HB网格参数','网格质量核对项'),
 ('Grid_Wall_Offset_m','m','原网格近墙点剔除距离','原GH/HB网格参数','网格质量核对项'),
 ('Grid_Outside_Points_Removed','boolean','是否剔除房间之外的传感器点，保留原设置','原GH/HB网格参数','网格质量核对项'),
 ('Internal_Obstruction_Description','text','原模拟纳入的内部隔墙、家具或其他遮挡，不把未建模物体视为已纳入','原模拟几何','先用于模型范围核对')]
with (DOC/'geometry_field_dictionary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['field','unit','definition','required_source','available_at_new_room_inference']);w.writerows(fields)
roomcols=['Room_ID']+[r[0] for r in fields]+['Source_File','Source_Object_ID','Geometry_Units','Geometry_Version','Verified_By','Status']
with (TPL/'TRAIN_room_geometry_to_complete.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=roomcols);w.writeheader();w.writerows({'Room_ID':r,'Status':'MISSING_NOT_FOR_TRAINING'} for r in rooms)
with (TPL/'window_geometry_to_complete.csv').open('w',encoding='utf-8-sig',newline='') as f:
    csv.writer(f).writerow(['Room_ID','Window_Element_ID','Window_Width_m','Window_Height_m','Window_Sill_Height_m','Window_Head_Height_m','Glass_Area_m2','Center_X_model_m','Center_Y_model_m','Center_Z_model_m','Normal_X_model','Normal_Y_model','Normal_Z_model','Material_Identifier','Source_File','Verified_By'])
provenance={'status':'MISSING_NOT_VERIFIED','TRAIN_rooms_only':rooms,'EPW_filename':None,'EPW_sha256':None,'same_EPW_for_all_40_cases':None,
            'Hour7_interval_start':None,'Hour7_interval_end':None,'solar_position_timestamp_convention':None,
            'timezone_and_DST_convention':None,'model_north_rotation_deg':None,'rotation_sign_reference':None,
            'PVSD_angle_zero_definition':None,'PVSD_axis_and_panel_normals_source':None,
            'glass_area_divide_by_10_conversion_source':None,'room_to_grid_mapping_source':None,
            'P10_export_code_source':None,'sensor_area_weighting_rule':None,'source_simulation_versions':None}
T.dump(TPL/'simulation_provenance_to_complete.json',provenance)
cases=[{'Room_ID':r,'DOY_1based':day,'Hour':hour,'PVSD_Angle_deg':angle,'Source_Request':'Existing per-sensor illuminance only; do not run a new simulation'}
       for r in cfg['focal_rooms'] for day in [80,264] for hour in [9,12] for angle in [0,90]]
with (DOC/'existing_sensor_result_sample_requests.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
archive=ROOT.parent/'project/project_final(1).zip'
with zipfile.ZipFile(archive) as z:entries=z.namelist()
geometryext={'.rvt','.ifc','.3dm','.gh','.ghx','.hbjson','.hbpkl','.pts','.ill','.rad','.epw'}
local=[]
for p in ROOT.rglob('*'):
    if any(part in {'.conda','.cache'} for part in p.relative_to(ROOT).parts):continue
    if p.is_file() and p.suffix.lower() in geometryext:local.append(p.relative_to(ROOT).as_posix())
inventory={'scope':'Local Project_2026 source files and filename/metadata review of workspace project archive. No claim of exhaustive Google Drive inventory.',
  'raw_files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':T.sha(p)} for p in sorted((ROOT/'data/raw').glob('*')) if p.is_file()],
  'tower_A_geometry_grid_EPW_files_found_in_Project_2026':local,
  'workspace_project_archive_review':{'path':str(archive),'entries':entries,'interpretation':'Group 10 course project; no Tower A provenance or direct geometry/weather file established. Not extracted or used.'},
  'other_workspace_BIM_files':'Assignment/course models found by filename inventory; no verified Tower A match; not opened, copied or used.',
  'connected_Google_Drive_reader_available':False,'Google_Drive_rechecked':False,
  'matching_raw_sensor_results_available':False,'templates_contain_verified_new_geometry':False,
  'protected_TEST_rooms_excluded_from_handoff':True,'source_index_rows':len(known),'raw_sensor_sample_case_count':len(cases)}
T.dump(OUT/'source_inventory.json',inventory)
content='''# Tower A：已有工程资料与几何字段核对清单

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
'''
(DOC/'geometry_and_simulation_sources.md').write_text(content,encoding='utf-8')
T.dump(OUT/'handoff_manifest.json',{'script_sha256':T.sha(__file__),'files':{p.relative_to(ROOT).as_posix():T.sha(p) for folder in [DOC,TPL] for p in folder.glob('*') if p.is_file()},'raw_source_sha256':T.sha(ROOT/cfg['source'])})
print(f'Source checklist, {len(known)} TRAIN room references, field dictionary, three blank templates and {len(cases)} existing-case requests prepared; nothing sent externally.')
