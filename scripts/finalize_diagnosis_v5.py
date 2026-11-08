"""Record completed authorized diagnostic phase and package source-request materials."""
from pathlib import Path
import sys
import zipfile
from datetime import datetime,timezone
import train_cross_room_v5 as T

ROOT=T.ROOT;OUT=ROOT/'reports/cross_room_diagnosis_v5'
assert '--figures-reviewed' in sys.argv
run=T.read(OUT/'run_manifest.json');verify=T.read(OUT/'verification.json');build=T.read(OUT/'report_build_manifest.json');handoff=T.read(OUT/'handoff_manifest.json')
assert run['status']=='completed' and verify['status']=='passed'
assert all(T.sha(ROOT/p)==h for p,h in run['input_hashes'].items())
assert T.sha(ROOT/'scripts/diagnose_cross_room_v5.py')==run['script_sha256']
assert T.sha(ROOT/'scripts/verify_diagnosis_v5.py')==verify['script_sha256']
assert T.sha(ROOT/'scripts/report_diagnosis_v5.py')==build['script_sha256']
assert all(T.sha(OUT/p)==h for p,h in build['artifacts'].items())
assert all(T.sha(ROOT/p)==h for p,h in handoff['files'].items())
assert T.read(ROOT/'configs/proposals/P10_blend_v6_draft.json')['status']=='PROPOSED_NOT_APPROVED_NOT_EXECUTED'
presentation=T.read(OUT/'presentation_verification.json');presentation['figures']='passed: all three scientific plots visually inspected'
T.dump(OUT/'presentation_verification.json',presentation)
checks=[('Aggregation','All seven rooms retained with monthly/hourly/angle/weather strata; macro and room results kept distinct.'),
 ('Ecological inference','No all-476-room conclusion from seven-room summaries; angle ranking is not general prediction accuracy.'),
 ('Selection/Berkson','Representative rooms form a selected development sample; no population prevalence claim.'),
 ('Collider','No causal regression or target-dependent input feature selection; true-label substitutions expressly diagnostic.'),
 ('Base rate','Eligible angle-pair denominator and coverage reported; near-ties omitted only under declared gap thresholds.'),
 ('Regression to mean','No intervention or newly improved model claim from extreme-room diagnosis.'),
 ('Survivorship','All seven rooms and all nine saved methods analyzed; no difficult room removed.'),
 ('Look elsewhere','Predeclared weather bins and focal rooms; no significance testing or optimized cutoff search.'),
 ('Forking paths','No new parameter, weight or routing selection; v6 remains an unexecuted draft.'),
 ('Causation','Algebraic decomposition is an identity, not a physical causal estimate; weather-bin associations do not validate provenance.'),
 ('Reverse causality','Held-room true mean/ratio substitutions and pointwise best model use answers; never presented as deployable inference.')]
T.dump(OUT/'interpretation_checks.json',{'coverage':'11/11 checked','checks':[{'type':a,'assessment':b} for a,b in checks]})
exports=ROOT/'exports';exports.mkdir(exist_ok=True);package=exports/'geometry_source_handoff_v1.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for rel in handoff['files']:z.write(ROOT/rel,arcname=rel)
with zipfile.ZipFile(package) as z:assert set(z.namelist())==set(handoff['files'])
T.dump(OUT/'handoff_package.json',{'path':package.relative_to(ROOT).as_posix(),'sha256':T.sha(package),'file_count':len(handoff['files']),
 'contains':'TRAIN room source identifiers, field dictionary, blank templates and existing-case request list; no performance targets or TEST room data',
 'externally_sent':False})
approval={'status':'approved_diagnostic_and_source_preparation_completed','approval_user_message':'按照你的计划开始执行一下后续的实验和整理',
 'executed_scope':'Planned next phase 1 error diagnosis and phase 2 geometry/source preparation, as announced at start; no new model training.',
 'analysis_config':'configs/diagnosis_v5.json','analysis_config_sha256':T.sha(ROOT/'configs/diagnosis_v5.json'),
 'run':'reports/cross_room_diagnosis_v5/run_manifest.json','verification':'reports/cross_room_diagnosis_v5/verification.json',
 'new_model_fits':0,'TEST_predictions':0,'TEST_scores':0,'completed_utc':datetime.now(timezone.utc).isoformat(),
 'next_training_configuration':'configs/proposals/P10_blend_v6_draft.json; proposed only, not executed'}
T.dump(ROOT/'docs/approvals/diagnosis_v5_approval.json',approval)
status=T.read(ROOT/'docs/project_status.json')
status.update(updated_local_date='2026-11-07',latest_analysis='reports/cross_room_diagnosis_v5/report.html',
 latest_completed_analysis=run['analysis_id'],latest_approval='docs/approvals/diagnosis_v5_approval.json',
 pending_proposal='configs/proposals/P10_blend_v6_draft.json',pending_proposal_status='Diagnostic phase completed; optional P10 fusion draft not approved or executed',
 report='reports/cross_room_diagnosis_v5/report.html',configuration='configs/diagnosis_v5.json',verification='reports/cross_room_diagnosis_v5/verification.json',
 authorization_status='Approved diagnostic/source preparation phase completed. Further training configuration remains to be confirmed; TEST sealed.',
 model_status='Existing v5/v4 models retained; diagnosis identifies room-scale, environmental-response and two-stage error-cancellation issues. No replacements.',
 source_handoff='exports/geometry_source_handoff_v1.zip',TEST_predictions=0,TEST_scores=0,new_simulations=0,final_deployment_refit=False)
T.dump(ROOT/'docs/project_status.json',status)
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8');marker='## 第五轮之后：跨房间误差诊断与资料整理'
if marker not in text:
    facts=T.read(OUT/'report_data.json')['facts']
    addition='\n\n'+marker+'\n\n已按批准的下一步完成误差诊断与资料整理，新增模型拟合0。报告为`reports/cross_room_diagnosis_v5/report.html`，交互视图为同目录`cross-room-error-diagnosis.canvas.tsx`。\n\n'
    addition+='\n\n'.join(facts)+'\n\n'
    addition+='真实均值/比例替换仅用于解释误差，不是可部署模型或新验证分数。角度排序属于单指标诊断，不能代替综合推荐评估。所有7个TRAIN房间完整保留，三个TEST房间没有预测或评分。\n\n'
    addition+='脚本：`scripts/diagnose_cross_room_v5.py`计算诊断；`scripts/prepare_geometry_handoff_v5.py`生成资料清单与模板；`scripts/verify_diagnosis_v5.py`独立复核；`scripts/report_diagnosis_v5.py`构建报告。入口`run_diagnosis_v5.ps1`拒绝覆盖既有分析，`report_diagnosis_v5.ps1`可重建数值复核及报告。\n\n'
    addition+='资料包：`exports/geometry_source_handoff_v1.zip`，包含`docs/data_requests/`和`data/templates/`对应文件。空白几何字段标记为MISSING_NOT_FOR_TRAINING；没有伪造缺失几何、取得原始网格或向他人发消息。\n\n'
    addition+='下一轮可选P10融合草案为`configs/proposals/P10_blend_v6_draft.json`，状态PROPOSED_NOT_APPROVED_NOT_EXECUTED。优先取得已有几何/网格来源；该草案尚未选权重、未评分、未运行。\n'
    readme.write_text(text+addition,encoding='utf-8')
artifacts={p.relative_to(OUT).as_posix():T.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and 'logs' not in p.parts}
T.dump(OUT/'artifact_manifest.json',{'files':artifacts,'finalizer_sha256':T.sha(__file__),'completed_utc':datetime.now(timezone.utc).isoformat()})
print(f'Diagnosis finalized: {len(artifacts)} artifacts, 11/11 interpretation checks; source request package {package.name}; v6 not executed; TEST sealed.')
