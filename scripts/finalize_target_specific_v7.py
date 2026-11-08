"""Archive verified v7 evidence and update the project index without model deployment."""
from datetime import datetime, timezone
import sys
import train_target_specific_v7 as V

T=V.T;ROOT=V.ROOT;OUT=ROOT/'reports/experiment_v7'


def main():
    assert '--figures-reviewed' in sys.argv, 'Review both scientific figures first'
    cfg=T.read(V.CONFIG);run=T.read(OUT/'run_manifest.json');verify=T.read(OUT/'verification.json')
    build=T.read(OUT/'report_build_manifest.json');data=T.read(OUT/'report_data.json');decision=T.read(OUT/'candidate_decision.json')
    assert run['status']=='completed' and verify['status']=='passed'
    assert all(T.sha(ROOT/p)==h for p,h in run['input_hashes'].items())
    assert T.sha(ROOT/'scripts/train_target_specific_v7.py')==run['script_sha256']
    assert T.sha(ROOT/'scripts/verify_target_specific_v7.py')==verify['script_sha256']
    assert T.sha(ROOT/'scripts/report_target_specific_v7.py')==build['script_sha256']
    assert all(T.sha(OUT/p)==h for p,h in build['artifacts'].items())
    assert run['TEST_predictions']==run['TEST_scores']==verify['TEST_predictions']==verify['TEST_scores']==0
    presentation=T.read(OUT/'presentation_verification.json')
    presentation.update(figures='passed: both figures visually inspected',visual_reviewed_files=['figures/mean_and_worst_changes.png','figures/room_errors.png'])
    T.dump(OUT/'presentation_verification.json',presentation)
    checks=[
        ('Aggregation','All seven rooms reported; average, median, maximum and per-room MAE kept distinct. Targets with different units never averaged together.'),
        ('Ecological inference','No claim of accuracy for all 476 rooms or other buildings from this selected seven-room sample.'),
        ('Selection/Berkson','Candidate families and hypotheses informed by previous development results; this is not an independent confirmatory study.'),
        ('Collider','No target-dependent row filtering, room-name routing or adjustment using held-room outputs. Original 13 input columns retained.'),
        ('Base rate','All room-hour-angle rows remain. Descriptive tolerance rates and worst-room rates accompany MAE; tolerances are not engineering standards.'),
        ('Regression to mean','The worst-room identity may change. Compare maxima across the same complete room set, not just previously difficult rooms.'),
        ('Survivorship','Every planned output, selection policy and room is reported, including failed comparisons and tradeoffs.'),
        ('Look elsewhere','24 declared candidates, no search expansion after outer scores, no significance tests or unsupported confidence intervals.'),
        ('Forking paths','Protocol/config frozen before training; Mean_select is a declared control and is not promoted to the primary hypothesis after observing results.'),
        ('Causation','This compares predictive procedures on simulated labels; it does not establish causal reasons for geometry-related errors or field performance.'),
        ('Reverse causality','Outer targets used only for scoring; the relative P10 inference uses its own predicted mean, never held-room true mean.'),
    ]
    T.dump(OUT/'interpretation_checks.json',{'coverage':'11/11 checked','checks':[{'type':k,'assessment':v} for k,v in checks],
        'additional_limit':'Minimizing inner worst-room error does not guarantee lower outer worst-room error. Repeated model development remains a limitation.'})
    completed=datetime.now(timezone.utc).isoformat()
    approval={'status':'approved_target_specific_experiment_completed','approval_user_message':cfg['approval_user_message'],
        'configuration':'configs/experiment_v7.json','configuration_sha256':T.sha(V.CONFIG),
        'protocol':'docs/experiment_v7_protocol.md','protocol_sha256':T.sha(ROOT/'docs/experiment_v7_protocol.md'),
        'run':'reports/experiment_v7/run_manifest.json','verification':'reports/experiment_v7/verification.json',
        'fit_counts':run['fit_counts'],'additional_verification_refits':verify['bounded_reproducibility_estimator_refits'],
        'completed_utc':completed,'TEST_predictions':0,'TEST_scores':0,'new_simulations':0,'final_deployment_refit':False,
        'scope':'One declared target-specific grid and three policies, already authorized by user; no further permission requested and no undeclared expansion.'}
    T.dump(ROOT/'docs/approvals/target_specific_v7_approval.json',approval)
    passed=[t for t in V.TARGETS if decision['targets'][t]['passes_both_references']]
    status=T.read(ROOT/'docs/project_status.json')
    status.update(updated_local_date='2026-11-08',latest_completed_experiment=run['experiment_id'],
        latest_analysis='reports/experiment_v7/report.html',latest_completed_analysis='target_specific_worst_room_v7_evaluation',
        latest_approval='docs/approvals/target_specific_v7_approval.json',pending_proposal=None,
        pending_proposal_status='Authorized v7 plan and bounded grid completed; no pending execution.',
        report='reports/experiment_v7/report.html',configuration='configs/experiment_v7.json',verification='reports/experiment_v7/verification.json',
        authorization_status='User-authorized target-specific optimization completed and verified. TEST sealed.',
        model_status=f'V7 Worst_guarded passed both development references for {len(passed)}/5 targets. Candidates retained for further validation; no final deployment model or TEST evaluation.',
        v7_development_candidates=passed,TEST_predictions=0,TEST_scores=0,new_simulations=0,final_deployment_refit=False,
        next_priority='Review per-target tradeoffs, resolve existing geometry/weather provenance, then freeze a final evaluation plan; no additional simulations authorized.')
    T.dump(ROOT/'docs/project_status.json',status)
    registry=T.read(ROOT/'docs/model_evidence_registry.json')
    registry['updated_local_date']='2026-11-08';registry['latest_decision']=decision
    registry['v7_development_candidates']=[r for r in data['scoreboard'] if r['passes_both_references']]
    registry['v7_all_output_results']=data['scoreboard']
    registry['v7_result_source']='reports/experiment_v7/scoreboard.csv'
    registry['v7_result_source_sha256']=T.sha(OUT/'scoreboard.csv')
    if not any(x['label']=='v7' for x in registry['completed_work']):
        registry['completed_work'].append({'label':'v7','description':'逐目标损失与最差房间优先选择；完整报告平均/最差及逐房间表现','report':'reports/experiment_v7/report.html'})
    registry['current_comparison_reference_note']='Original comparison references remain listed for traceability. v7 development candidates are recorded separately; no final deployment model selected.'
    T.dump(ROOT/'docs/model_evidence_registry.json',registry)
    idx=ROOT/'docs/experiment_index.md';text=idx.read_text(encoding='utf-8')
    text=text.replace('更新：2026-11-07。最新完成v6；融合没有改善整体P10误差，原参考保留。',
        '更新：2026-11-08。最新完成v7逐目标优化。'+data['headline']+' 旧参考保留，新候选单独记录。')
    row='| v7 | 逐目标损失函数与最差房间优先选择 | [报告](../reports/experiment_v7/report.html) |'
    if row not in text:text=text.replace('\n## 当前比较参考','\n'+row+'\n\n## 当前比较参考',1)
    # Keep the overview table contiguous: the former last row already ends before the heading.
    text=text.replace('| [报告](../reports/experiment_v6/report.html) |\n\n'+row,
                      '| [报告](../reports/experiment_v6/report.html) |\n'+row)
    marker='## v7逐目标结果'
    if marker not in text:
        text+='\n\n'+marker+'\n\n'+data['headline']+'\n\n'+'\n\n'.join(data['facts'])+'\n\n'+data['gate']+'\n\n'
        text+='原参考数值和v7候选分开保存；未执行全7房间部署训练、TEST评价或新的角度决策验证。详细误差与每折所选模型见[v7报告](../reports/experiment_v7/report.html)。\n'
    idx.write_text(text,encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    text=text.replace('当前完成v6，P10融合未通过改进条件；测试房间继续封存。','当前完成v7逐目标优化；结果与候选状态见索引，测试房间继续封存。')
    marker='## 第七轮：逐目标优化与最差房间误差（2026-11-08完成）'
    if marker not in text:
        text+='\n\n'+marker+'\n\n'+data['headline']+'\n\n'+data['protocol']+'\n\n'+data['gate']+'\n\n'+'\n\n'.join(data['facts'])+'\n\n'
        text+=f"完成{run['fit_counts']['inner']}次内层、{run['fit_counts']['outer']}次外层拟合，7个外层模型包；验证另重复拟合7个回归器。全部五项目标原候选重现v3。独立复核1008条内层房间分数、168条候选汇总、105次选择、140组外层指标，重载核对1839600个预测值。\n\n"
        text+='报告`reports/experiment_v7/report.html`，交互视图`target-specific-worst-room.canvas.tsx`；配置`configs/experiment_v7.json`；方案`docs/experiment_v7_protocol.md`；模型`models/experiment_v7/`；批准记录`docs/approvals/target_specific_v7_approval.json`。\n\n'
        text+='训练入口`scripts/run_target_specific_v7.ps1`拒绝覆盖运行。复核/重建报告可用`scripts/report_target_specific_v7.ps1`，会执行7个回归器的有限重复拟合。图像审阅后用项目conda Python执行`scripts/finalize_target_specific_v7.py --figures-reviewed`更新归档。图已检查，HTML静态脚本/链接已通过；浏览器交互与原生Canvas编译未验证。\n\n'+data['limitations']+'\n'
    readme.write_text(text,encoding='utf-8')
    paths=['README.md','docs/project_status.json','docs/model_evidence_registry.json','docs/experiment_index.md','docs/approvals/target_specific_v7_approval.json']
    T.dump(OUT/'organization_manifest.json',{'completed_utc':completed,'files':{p:T.sha(ROOT/p) for p in paths},
        'scope':'Index and development candidate registry only. Earlier manifests document earlier file states; no final model deployment.'})
    artifacts={p.relative_to(OUT).as_posix():T.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and 'logs' not in p.parts}
    T.dump(OUT/'artifact_manifest.json',{'files':artifacts,'finalizer_sha256':T.sha(__file__),'completed_utc':completed,'interpretation_checks':'11/11 checked'})
    print(f'V7 finalized: {len(artifacts)} report artifacts, 11/11 interpretation checks, {len(passed)}/5 development candidates; TEST sealed.')


if __name__=='__main__':main()
