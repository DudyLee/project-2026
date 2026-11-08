"""Close the approved v5 run after numerical, static and figure review."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import train_cross_room_v5 as T

ROOT=T.ROOT;OUT=ROOT/'reports/experiment_v5'
assert '--figures-reviewed' in sys.argv,'Inspect both figures before marking the artifact complete.'
run=T.read(OUT/'run_manifest.json');v=T.read(OUT/'verification.json');b=T.read(OUT/'report_build_manifest.json')
assert run['status']=='completed' and v['status']=='passed'
assert all(T.sha(ROOT/p)==h for p,h in run['input_hashes'].items())
assert T.sha(ROOT/'scripts/train_cross_room_v5.py')==run['script_sha256']
assert T.sha(ROOT/'scripts/evaluate_cross_room_v5.py')==v['evaluation_script_sha256']
assert T.sha(ROOT/'scripts/report_cross_room_v5.py')==b['report_script_sha256']
assert all(T.sha(OUT/p)==h for p,h in b['artifacts'].items())
presentation=T.read(OUT/'presentation_verification.json');presentation['figure_visual_inspection']='passed: both scientific figures visually inspected'
T.dump(OUT/'presentation_verification.json',presentation)
checks=[
 ('Simpson aggregation','Room-macro results and all seven paired room results shown; overall improvement is not assumed to hold in each room.'),
 ('Ecological inference','No claim of per-hour tolerance or coverage for all 476 rooms from room averages.'),
 ('Selection/Berkson','Seven representative development rooms are a selected sample; external population generalization remains unproven.'),
 ('Collider adjustment','No causal effect estimation; added features are input-derived, no target-derived covariates.'),
 ('Base rates','Regression task; no classification accuracy claim. Dark-room scales shown beside absolute error.'),
 ('Regression to mean','No before/after field intervention claim; paired frozen-baseline comparisons use identical held-room rows.'),
 ('Survivorship','All seven rooms and every predeclared profile/family/target are reported, including worse results.'),
 ('Look elsewhere','All candidates reported; best-room and best-method summaries labeled post-hoc development descriptions. No p-values/significance claims.'),
 ('Forking paths','Config frozen before training; two candidates per family; previous development history disclosed; TEST sealed.'),
 ('Causal attribution','Ablation evaluates predictive changes, not a proven physical cause of room illumination or a guaranteed feature benefit.'),
 ('Reverse causality','Inference uses X only; target transforms and ratio labels are fitting-stage outputs, not true targets supplied at prediction.')
]
T.dump(OUT/'interpretation_checks.json',{'coverage':'11/11 checked','checks':[{'type':a,'assessment':c} for a,c in checks],
 'scientific_inference_scope':'Descriptive development comparison, not independent confirmatory evidence or a calibrated uncertainty estimate.'})
approval=T.read(ROOT/'docs/approvals/cross_room_v5_approval.json')
approval.update(status='approved_and_completed',completed_utc=datetime.now(timezone.utc).isoformat(),
                config_sha256=T.sha(T.CONFIG),completed_run='reports/experiment_v5/run_manifest.json',verification='reports/experiment_v5/verification.json')
T.dump(ROOT/'docs/approvals/cross_room_v5_approval.json',approval)
status=T.read(ROOT/'docs/project_status.json')
status.update(updated_local_date='2026-11-07',latest_completed_experiment=run['experiment_id'],latest_analysis='reports/experiment_v5/report.html',
              pending_proposal=None,pending_proposal_status='Approved four-group cross-room v5 experiment completed; no further experiment inferred',
              latest_approval='docs/approvals/cross_room_v5_approval.json',report='reports/experiment_v5/report.html',configuration='configs/experiment_v5.json',
              verification='reports/experiment_v5/verification.json',model_status='V5 comparisons available per target; no automatic model replacement or deployment. V4 PV results retained.',
              authorization_status='Approved v5 work completed; further experiments and TEST unsealing require user confirmation.',
              TEST_predictions=0,TEST_scores=0,new_simulations=0,final_deployment_refit=False)
T.dump(ROOT/'docs/project_status.json',status)
readme=ROOT/'README.md';existing=readme.read_text(encoding='utf-8');marker='## 第五轮：跨房间特征与直接P10预测'
if marker not in existing:
    report=T.read(OUT/'report_data.json')
    addition='\n\n'+marker+'\n\n'
    addition+=f'用户批准后已完成{sum(run["fit_counts"].values())}次拟合（504次内层、{run["fit_counts"]["outer"]}次外层），28个外层模型包。报告入口为`reports/experiment_v5/report.html`，交互视图为同目录`Cross-room-features-P10.canvas.tsx`。\n\n'
    addition+='四组：A原13项特征＋相对P10；B增加3项房间特征＋相对P10；C原特征＋直接asinh P10；D增强特征＋直接asinh P10。平均照度均采用log1p，A/C与B/D分别共用相同平均照度输出。模型为Random Forest与XGBoost，每家族沿用v3的两个候选并在外层训练房间内部选择。HistGB作为冻结参考。\n\n'
    addition+='新增特征为玻璃面积/地板面积、窗户数量、名义窗面积/窗户数量；模型编码由15列增至18列。P10直接变换的尺度固定为1 lux。推断不使用留出房间真实照度。没有假设缺失的房间进深、窗户位置、北向或百叶几何。\n\n'
    addition+=report['headline']+'\n\n'+'\n\n'.join(report['findings'])+'\n\n'
    addition+='预设逐目标筛选：相对同算法A组，房间平均MAE下降至少10%、至少5/7房间不恶化、最差房间MAE不增加。所有组合均报告；最优分数属于开发结果，未自动替换旧模型。\n\n'
    addition+='配置`configs/experiment_v5.json`；协议`docs/experiment_v5_protocol.md`；批准记录`docs/approvals/cross_room_v5_approval.json`。训练入口`scripts/run_cross_room_v5.ps1`拒绝覆盖已开始的运行。已激活项目conda环境时，执行以下入口仅重算验证及报告，不训练新模型：\n\n```powershell\n.\\scripts\\report_cross_room_v5.ps1\n```\n\n'
    addition+='结果目录包含内外层预测、候选选择、逐房间误差、特征覆盖、配对比较、开发筛选条件和验证记录；模型保存于`models/experiment_v5/`。原始数据和冻结参考不变，3个TEST房间继续封存，无最终全7房间部署重训。图已检查，HTML静态链接/脚本已验证；未执行浏览器交互或原生Canvas编译验证。\n'
    readme.write_text(existing+addition,encoding='utf-8')
artifacts={p.relative_to(OUT).as_posix():T.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and 'logs' not in p.parts}
T.dump(OUT/'artifact_manifest.json',{'artifacts':artifacts,'finalizer_sha256':T.sha(__file__),'completed_utc':datetime.now(timezone.utc).isoformat()})
print(f'V5 finalized: {sum(run["fit_counts"].values())} fits, {len(artifacts)} report artifacts, 11/11 interpretation checks, TEST sealed.')
