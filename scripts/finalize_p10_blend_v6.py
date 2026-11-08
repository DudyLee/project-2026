"""Finalize the completed bounded fusion experiment; no fitting or target-data reads."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import sys

import train_cross_room_v5 as T


ROOT = T.ROOT
OUT = ROOT / 'reports/experiment_v6'


def metric_row(relative, method, target=None):
    with (ROOT / relative).open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if row['method'] == method and (target is None or row['target'] == target)]
    assert len(rows) == 1, (relative, method, target)
    return rows[0]


def main():
    if '--figures-reviewed' not in sys.argv:
        raise SystemExit('Visually inspect both scientific figures before finalization.')
    config = T.read(ROOT / 'configs/experiment_v6.json')
    run = T.read(OUT / 'run_manifest.json')
    verify = T.read(OUT / 'verification.json')
    build = T.read(OUT / 'report_build_manifest.json')
    decision = T.read(OUT / 'candidate_decision.json')
    assert run['status'] == 'completed' and verify['status'] == 'passed'
    assert run['experiment_id'] == config['experiment_id']
    assert not decision['passes_development_gate']
    assert run['TEST_predictions'] == run['TEST_scores'] == 0
    assert verify['TEST_predictions'] == verify['TEST_scores'] == 0
    assert run['new_regression_estimator_fits'] == 0
    assert all(T.sha(ROOT / p) == h for p, h in run['input_hashes'].items())
    assert T.sha(ROOT / 'scripts/run_p10_blend_v6.py') == run['script_sha256']
    assert T.sha(ROOT / 'scripts/verify_p10_blend_v6.py') == verify['script_sha256']
    assert T.sha(ROOT / 'scripts/report_p10_blend_v6.py') == build['script_sha256']
    assert all(T.sha(OUT / p) == h for p, h in build['artifacts'].items())
    historical_draft = ROOT / config['source_proposal']
    assert T.read(historical_draft)['status'] == 'PROPOSED_NOT_APPROVED_NOT_EXECUTED'

    presentation = T.read(OUT / 'presentation_verification.json')
    presentation['figures'] = 'passed: both scientific plots visually inspected; logarithmic MAE axis and selected-weight markers labeled'
    presentation['visual_reviewed_files'] = ['figures/room_MAE.png', 'figures/inner_weight_selection.png']
    T.dump(OUT / 'presentation_verification.json', presentation)

    checks = [
        ('Aggregation', 'Equal-room macro MAE, median, worst room and all seven paired room outcomes are reported; no inference from pooled rows alone.'),
        ('Ecological inference', 'Seven selected development rooms do not establish accuracy across 476 rooms; no accuracy percentage inferred from R-squared.'),
        ('Selection/Berkson', 'Representative-room selection and previously examined development data limit external validity; TEST remains sealed.'),
        ('Collider', 'No outcome-dependent feature filtering or held-room target calibration; inputs are the original 13 predictors.'),
        ('Base rate', 'All seven rooms and 122640 outer rows remain; two improved, one tied, four worsened. Small low-light errors remain visible.'),
        ('Regression to mean', 'Improvement in an extreme room is not treated as evidence of overall or causal improvement.'),
        ('Survivorship', 'All weights and rooms are reported, including BM432 deterioration and the failed development gate.'),
        ('Look elsewhere', 'Five declared weights only; no significance tests, post-hoc finer grid or room-specific routing.'),
        ('Forking paths', 'Original-unit macro MAE and the prespecified gate determine the decision. Improved secondary asinh MAE does not override failure.'),
        ('Causation', 'Weight-selection transfer failure is observed; missing geometry remains a hypothesis, not a proven physical cause.'),
        ('Reverse causality', 'Held-room labels are used only for scoring. Weight selection uses inner OOF on the other six rooms; diagnostic true-label substitutions are not inference inputs.'),
    ]
    T.dump(OUT / 'interpretation_checks.json', {
        'coverage': '11/11 checked',
        'checks': [{'type': name, 'assessment': explanation} for name, explanation in checks],
        'selection_score_limit': 'The same inner OOF supports formulation-candidate and blend-weight selection; these inner scores are selection evidence, not independent performance estimates.',
    })

    completed_utc = datetime.now(timezone.utc).isoformat()
    approval = {
        'status': 'approved_bounded_P10_fusion_completed',
        'approval_user_message': config['approval_user_message'],
        'executed_scope': 'Previously prepared five-weight P10 fusion on saved v5 TRAIN-only nested predictions; verification and result organization.',
        'historical_draft': config['source_proposal'],
        'historical_draft_sha256': T.sha(historical_draft),
        'historical_status_note': 'The earlier draft preserves its status at creation. This later receipt and the execution config record approval and completion.',
        'configuration': 'configs/experiment_v6.json',
        'configuration_sha256': T.sha(ROOT / 'configs/experiment_v6.json'),
        'run': 'reports/experiment_v6/run_manifest.json',
        'verification': 'reports/experiment_v6/verification.json',
        'candidate_decision': 'reports/experiment_v6/candidate_decision.json',
        'completed_utc': completed_utc,
        'new_regression_estimator_fits': 0,
        'weight_selection_tasks': 7,
        'TEST_predictions': 0,
        'TEST_scores': 0,
        'new_simulations': 0,
        'further_scope': 'No finer weights, room routing, new experiments, deployment refit or TEST release inferred from this completion.',
    }
    T.dump(ROOT / 'docs/approvals/P10_blend_v6_approval.json', approval)

    status = T.read(ROOT / 'docs/project_status.json')
    status.update(
        updated_local_date='2026-11-07',
        latest_completed_experiment=run['experiment_id'],
        latest_analysis='reports/experiment_v6/report.html',
        latest_completed_analysis='nested_P10_blend_v6_evaluation',
        latest_approval='docs/approvals/P10_blend_v6_approval.json',
        pending_proposal=None,
        pending_proposal_status='The former v6 draft was subsequently approved and completed; no new training configuration is pending execution.',
        report='reports/experiment_v6/report.html',
        configuration='configs/experiment_v6.json',
        verification='reports/experiment_v6/verification.json',
        authorization_status='Authorized bounded v6 experiment completed; declared stopping rule applied. TEST sealed.',
        model_status='V6 blend failed the development gate. Retain A_RandomForest relative P10 comparison reference and all prior models; no deployment promotion.',
        stopped_branch='Finer blend weights or room-specific routing without new independent input information',
        next_priority='Obtain existing TRAIN geometry, sensor-grid and simulation provenance; no additional simulations.',
        source_handoff='exports/geometry_source_handoff_v1.zip',
        experiment_index='docs/experiment_index.md',
        evidence_registry='docs/model_evidence_registry.json',
        TEST_predictions=0, TEST_scores=0, new_simulations=0, final_deployment_refit=False,
    )
    T.dump(ROOT / 'docs/project_status.json', status)

    entries = [
        ('v1', 'experiment_v1', '8/13项特征与Ridge、HistGB基线'),
        ('v1诊断', 'diagnosis_v1', '误差归因、log1p与同房间时间验证；与跨房间分数分开解释'),
        ('v2', 'experiment_v2', '平均照度log1p与相对P10处理'),
        ('v3', 'experiment_v3', 'HistGB、Random Forest、XGBoost的按房间嵌套验证与角度评价'),
        ('v3决策诊断', 'decision_diagnosis_v3', '固定0°的优势、可改进空间和切换损失'),
        ('v4', 'experiment_v4', 'PV增量与内层选择切换门槛；收益局限于个别房间'),
        ('v5', 'experiment_v5', '房间特征增强与直接asinh P10；均未通过替换条件'),
        ('v5误差诊断', 'cross_room_diagnosis_v5', '两阶段误差、环境响应与现有几何资料核对'),
        ('v6', 'experiment_v6', '相对/直接P10融合；未通过条件，保留原参考'),
    ]
    registry = {
        'updated_local_date': '2026-11-07',
        'status': 'development_evidence_only_not_a_deployed_multioutput_model',
        'evaluation': 'Seven development rooms; outer leave-one-room-out. Repeated development use is not final independent testing.',
        'protected_test_rooms': config['protected_test_rooms'],
        'current_comparison_references': [],
        'completed_work': [],
        'latest_decision': decision,
        'limitations': [
            'Only seven independently varying development rooms; hourly rows are not independent room samples.',
            'A full-seven-room final estimator has not been fitted or evaluated on sealed TEST.',
            'Existing input provenance and geographic/temporal conventions remain unconfirmed.',
            'No demonstrated robust gain from dynamic angle recommendation across rooms.',
        ],
    }
    reference_specs = [
        ('E_mean_lux', 'A_XGBoost', 'reports/experiment_v5/summary_metrics.csv', 'log1p mean; retained comparison reference, not a newly selected final estimator', 'lux'),
        ('E_P10_lux', 'A_RandomForest', 'reports/experiment_v5/summary_metrics.csv', 'relative P10 paired with its own candidate mean model', 'lux'),
        ('F_300_fraction', 'Frozen_HistGB', 'reports/experiment_v3/summary_metrics.csv', 'unchanged frozen baseline; other v3 candidates remain available', 'fraction'),
        ('F_UDI_fraction', 'Frozen_HistGB', 'reports/experiment_v3/summary_metrics.csv', 'unchanged frozen baseline; other v3 candidates remain available', 'fraction'),
        ('PV_AC_kWh', 'Frozen_HistGB', 'reports/experiment_v3/summary_metrics.csv', 'unchanged absolute-performance baseline; angle decisions evaluated separately', 'kWh'),
    ]
    for target, method, source, note, unit in reference_specs:
        row = metric_row(source, method, target)
        registry['current_comparison_references'].append({
            'target': target, 'method': method, 'note': note, 'unit': unit,
            'room_count': int(row['room_count']),
            'macro_room_MAE': float(row['mae_mean']),
            'worst_room_MAE': float(row['mae_max']),
            'source': source, 'source_sha256': T.sha(ROOT / source),
            'promotion_status': 'comparison_reference_only',
        })
    for label, folder, description in entries:
        report = f'reports/{folder}/report.html'
        assert (ROOT / report).is_file()
        registry['completed_work'].append({'label': label, 'description': description, 'report': report})
    T.dump(ROOT / 'docs/model_evidence_registry.json', registry)

    index = [
        '# 实验与模型证据索引', '',
        '更新：2026-11-07。最新完成v6；融合没有改善整体P10误差，原参考保留。', '',
        '**这里记录开发证据，不是一套已经锁定并通过独立测试的最终模型。** 七个房间已被多轮用于开发；BM532、BM702、BMG26继续封存。', '',
        '| 阶段 | 内容 | 结果入口 |', '|---|---|---|',
    ]
    index += [f'| {label} | {description} | [报告](../reports/{folder}/report.html) |' for label, folder, description in entries]
    index += ['', '## 当前比较参考', '',
              '下表用于后续比较，不表示把各目标的事后最优算法拼成一个已验证的生产模型。F300、FUDI与PV列出冻结HistGB参考；v3的其他候选保留在对应报告。', '',
              '| 目标 | 参考方法 | 7房间平均MAE | 单位 |', '|---|---|---:|---|']
    index += [f"| {r['target']} | {r['method']} | {r['macro_room_MAE']:.6f} | {r['unit']} |" for r in registry['current_comparison_references']]
    index += ['', '比例误差以0–1表示：例如0.07118相当于7.118个百分点。MAE不是准确率。逐房间差异和最差误差须同时查看。', '',
        '## v6结论', '',
        'Random Forest的相对P10与直接asinh P10按五个候选权重融合。每个外层折只在其余六个房间的内层OOF中选权重，外层房间只评分。原始单位平均MAE从9.748升至10.729 lux，恶化10.1%；2个房间改善、1个持平、4个恶化。没有通过预设条件，不替换原参考。', '',
        '内层分数用于选模型和权重，不能作为独立效果证明；次要的asinh误差改善不能推翻主要MAE结论。按停止规则，暂停无新信息的更细权重或按房间路由搜索。', '',
        '## 下一步所需资料', '',
        '优先核对已有原始模型的房间进深、窗户位置、传感器网格，以及EPW/时间/北向约定。缺失几何是待验证解释，当前实验没有证明它必然能提高精度。只请求现有资料，不追加模拟。', '',
        '- [资料请求说明](data_requests/geometry_and_simulation_sources.md)',
        '- [资料包](../exports/geometry_source_handoff_v1.zip)',
        '- [机器可读证据索引](model_evidence_registry.json)',
        '- [当前状态](project_status.json)', '',
        '历史草案保留创建时状态。v6的后续批准与完成以[批准记录](approvals/P10_blend_v6_approval.json)、[执行配置](../configs/experiment_v6.json)及[验证记录](../reports/experiment_v6/verification.json)为准。', '',
    ]
    (ROOT / 'docs/experiment_index.md').write_text('\n'.join(index), encoding='utf-8')

    readme_path = ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    obsolete = '下一轮可选P10融合草案为`configs/proposals/P10_blend_v6_draft.json`，状态PROPOSED_NOT_APPROVED_NOT_EXECUTED。优先取得已有几何/网格来源；该草案尚未选权重、未评分、未运行。'
    historical = 'P10融合草案`configs/proposals/P10_blend_v6_draft.json`保留创建时的PROPOSED_NOT_APPROVED_NOT_EXECUTED历史状态。用户随后批准并已完成下述v6；当前状态以`docs/approvals/P10_blend_v6_approval.json`和`docs/project_status.json`为准。'
    readme = readme.replace(obsolete, historical)
    marker = '## 第六轮：内层选权重的P10融合（2026-11-07完成）'
    if marker not in readme:
        readme += '\n\n' + marker + '\n\n' + '\n\n'.join([
            '入口`reports/experiment_v6/report.html`；交互视图`reports/experiment_v6/P10-nested-blend.canvas.tsx`。统一实验索引见`docs/experiment_index.md`，机器可读参考记录见`docs/model_evidence_registry.json`。',
            '基于v5已冻结的Random Forest相对P10与直接asinh P10，按w×相对预测+(1−w)×直接预测融合。仅比较w=1、0.75、0.5、0.25、0；每个外层折在其余六个房间的3折分组内层OOF中选择原始单位平均MAE最低的权重，并列偏向较大w。七个权重先冻结，再加载外层答案评分。没有新回归器拟合，也没有按留出房间名称或真实标签路由。',
            'P10的7房间平均MAE：原相对模型9.747747 lux，直接asinh 10.743254 lux，内层选权重融合10.729370 lux。融合比原模型恶化10.07%，2房间改善、1房间持平、4房间恶化。BM223从9.757降至1.479，BM432从29.275升至44.361 lux。融合没有通过预设的≥10%平均改善、≥5/7房间不恶化、最差房间误差不增加三项条件。',
            '保留A_RandomForest相对P10作为比较参考。次要asinh误差较低不改变原始单位主指标的结论；按照停止规则，不在已看过的外层结果上继续细调权重或加入房间路由。其余四个预测目标未变，没有全7房间最终部署拟合。',
            '独立数值复核通过：210条内层房间分数、35条权重汇总、7次权重选择、14次底层候选选择、122640行融合预测、21组外层误差和开发条件；7份推断配方各重载31个样本，只接收13项输入，无Room_ID和真实目标。三个TEST房间没有预测或评分，无新增模拟。',
            '模型配方在`models/experiment_v6/`，引用并校验v5模型包。运行记录、预测、逐房间比较、图与报告在`reports/experiment_v6/`。两张图已目视检查，HTML静态链接和脚本检查通过；浏览器交互和原生Canvas编译未验证。',
            '复核及重建报告：在Project_2026内执行`& .\\scripts\\report_p10_blend_v6.ps1`，不会训练新回归器。该命令会重置图像审阅状态；重新目视审阅后，再用项目conda Python执行`scripts/finalize_p10_blend_v6.py --figures-reviewed`生成最终归档。选择入口`run_p10_blend_v6.ps1`拒绝覆盖本次已完成运行。',
        ]) + '\n'
    navigation = '最新状态与各轮结果：[实验索引](docs/experiment_index.md)。当前完成v6，P10融合未通过改进条件；测试房间继续封存。'
    if navigation not in readme:
        readme = readme.replace('# Tower A PVSD 项目\n', '# Tower A PVSD 项目\n\n' + navigation + '\n', 1)
    readme_path.write_text(readme, encoding='utf-8')

    documentation_paths = [
        'README.md', 'docs/project_status.json', 'docs/experiment_index.md',
        'docs/model_evidence_registry.json', 'docs/approvals/P10_blend_v6_approval.json',
    ]
    T.dump(OUT / 'organization_manifest.json', {
        'completed_utc': completed_utc,
        'files': {p: T.sha(ROOT / p) for p in documentation_paths},
        'model_recipes': {p.relative_to(ROOT).as_posix(): T.sha(p) for p in sorted((ROOT / 'models/experiment_v6').glob('*.json'))},
        'source_data_unchanged': True,
        'scope': 'Documentation and comparison references only; no new final model selection or fitting.',
    })
    artifacts = {p.relative_to(OUT).as_posix(): T.sha(p)
                 for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json' and 'logs' not in p.parts}
    T.dump(OUT / 'artifact_manifest.json', {
        'files': artifacts, 'finalizer_sha256': T.sha(Path(__file__)),
        'completed_utc': completed_utc, 'interpretation_checks': '11/11 checked',
    })
    print(f'V6 finalized: {len(artifacts)} report artifacts; {len(entries)} indexed stages; 11/11 interpretation checks. Baseline retained; TEST sealed.')


if __name__ == '__main__':
    main()
