"""Write numerical v8 report and update project indexes; no model fitting or plots."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
import directional_v8 as D
from train_directional_v8 import sha
from prepare_repository_snapshot import PATTERNS,TEXT

ROOT=D.ROOT
NAMES={'T0':'原特征＋现有树模型','T1':'方向交互＋树模型','S0':'原特征＋样条/Ridge','S1':'方向交互＋样条/Ridge','S1_direct':'方向交互＋样条/Ridge，直接P10'}
TARGET_NAMES={'E_mean_lux':'平均照度','E_P10_lux':'P10照度','F_UDI_fraction':'FUDI'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    cfg=D.read(D.CONFIG);out=ROOT/cfg['output'];run=D.read(out/'run_manifest.json');verification=D.read(out/'verification.json')
    assert run['status']=='completed_verified' and verification['status']=='passed'
    summary=pd.read_csv(out/'summary_metrics.csv');fold=pd.read_csv(out/'fold_metrics.csv');gates=pd.read_csv(out/'development_gate.csv')
    decision=D.read(out/'candidate_decision.json');preps=D.read(out/'preprocessing_audit.json')
    def row(protocol,method,target):return summary[(summary.protocol==protocol)&(summary.method==method)&(summary.target==target)].iloc[0]
    def err(v,target):return f'{v*(100 if target=="F_UDI_fraction" else 1):.3f}'
    rows=[]
    for target in D.TARGETS:
        for method in cfg['methods']:
            if target not in cfg['methods'][method]['targets']:continue
            a,b=row('LeaveOneRoomOut',method,target),row('LeaveOneOrientationOut',method,target)
            rows.append([TARGET_NAMES[target]+'（'+('百分点' if target=='F_UDI_fraction' else 'lux')+'）',method,err(a.macro_MAE,target),err(a.worst_MAE,target)+' / '+a.worst_room,err(b.macro_MAE,target),err(b.worst_MAE,target)+' / '+b.worst_room])
    report=['## Material Passport', '',
        '- 实验：directional_transfer_v8；批准日期2026-11-08。状态：训练完成，数值/模型/选参核验通过。',
        '- 仅7个开发房间122,640行；三个既有测试房间52,560行在数值转换前跳过。本轮测试训练、预测、评分均为0。',
        '- 523次估计器拟合：内层Ridge 378、外层Ridge 73、外层树72；训练用预处理拟合72次。核验另重建72个预处理器，不训练新的预测估计器。',
        f'- 训练耗时{run["elapsed_seconds"]:.1f}秒，核验{verification["seconds"]:.1f}秒；独立项目conda环境，4线程。无新增模拟或可视化。',
        '- 本轮计划受已观察BM702诊断启发；全部选参和候选比较只使用开发房间。结果属于反复使用开发数据的证据，不是新的独立最终测试。',
        '- 依据：[批准记录](../../docs/approvals/directional_v8_approval.json)、[配置](../../configs/experiment_v8.json)、[执行协议](../../docs/experiment_v8_protocol.md)。', '',
        '## 主要结论','',
        '**平均照度有明确开发集收益，但没有任何目标的候选同时通过全部预设替换条件。保留试用版0.1。**',
        'T1平均照度在普通留一房间的7个房间全部改善，平均MAE下降14.75%；S1平均MAE下降21.90%，但两者最差房间MAE仅下降3.53%和3.80%，未达到10%。FUDI的样条方案改善平均与最差误差，却仅2/7房间改善。P10新增方法没有稳定优于现有相对RF。', '',
        '## 两套验证的完整MAE','',
        '平均MAE=7个房间MAE等权平均；最差MAE=7个房间MAE的最大值。不是最大单小时误差，也不是准确率。FUDI按百分点显示；例如0.09513=9.513个百分点。', '',
        table(['目标 / 单位','方法','留一房间平均','留一房间最差 / 房间','留一朝向平均','留一朝向最差 / 房间'],rows),'',
        table(['方法','定义'],[[m,NAMES[m]] for m in NAMES]),'',
        'T0/T1：平均照度XGBoost depth3、log1p；P10为RF depth12的配套mean_log与P10/(mean+1)两分支；FUDI固定HistGB。S0/S1对6个连续输入展开样条后标准化并做Ridge，目标处理与树方案一致。S1_direct只将P10改为直接asinh回归。',
        'F0由13个原始输入确定性编码为15列。F1增加6个太阳方向/直射量与模型坐标方向的经验交互，成为21列；样条展开后分别39/45列。这些交互不是已核实的窗面入射辐射。', '',
        '## 逐目标解释','',
        '**平均照度。** S1把BM522的MAE从108.545降到30.543 lux，BM432仅从231.642降到222.832 lux，仍是主验证最差房间。T1在两种验证的平均MAE均低于基线，其留朝向平均84.034 lux也低于S1的91.485 lux；S1留朝向最差198.616 lux低于T1的224.674 lux。不同方法在平均和最差误差上存在取舍，不能只报最好的单一数字。', '',
        '**P10。** S1_direct在BM223从6.727降到0.636 lux，但在BM522从29.268升至53.441 lux；主验证平均比T0高31.35%，最差高82.55%。S1相对方案也未解决问题，例如LM703从0.140升至6.542 lux。新的方向表示不能单独解决低值房间、较亮房间之间的P10异质性。直接方案的误差改善不具有跨房间一致性。', '',
        '**FUDI。** S1主验证平均从11.036降到9.513个百分点，最差从25.471降到22.369个百分点。改善来自BM223（24.898→9.026）与BM432（25.471→13.072）；BM522则从11.646升至22.369，成为新的最差房间，另4个房间也变差。它不是稳定提升所有新房间的方案。FUDI单独训练，不能由平均照度改善推导它也会改善。', '',
        '## 每个开发房间的主验证MAE','']
    for target in D.TARGETS:
        methods=[m for m in cfg['methods'] if target in cfg['methods'][m]['targets']]
        g=fold[(fold.protocol=='LeaveOneRoomOut')&(fold.target==target)].pivot(index='room',columns='method',values='mae')
        report += [TARGET_NAMES[target]+'（'+('百分点' if target=='F_UDI_fraction' else 'lux')+'）：','',table(['房间']+methods,[[r]+[err(g.loc[r,m],target) for m in methods] for r in cfg['development_rooms']]),'']
    gate_rows=[]
    for g in gates.itertuples():
        gate_rows.append([g.method,TARGET_NAMES[g.target],f'{g.room_macro_MAE_change_pct:+.2f}%',f'{g.room_worst_MAE_change_pct:+.2f}%',str(g.room_nonworse_count)+'/7',f'{g.orientation_macro_MAE_change_pct:+.2f}%',f'{g.orientation_worst_MAE_change_pct:+.2f}%','通过' if g.passes_all else '未通过'])
    report += ['## 冻结规则的判断','',
        '候选逐目标须同时满足：留房间平均不恶化、最差至少下降10%、至少5/7房间不恶化；留朝向平均与最差均不恶化。负号表示MAE降低。', '',
        table(['方法','目标','房间平均变化','房间最差变化','不恶化房间','朝向平均变化','朝向最差变化','判断'],gate_rows),'',
        '0/10个候选-目标组合通过，三个目标均保留T0；没有事后放宽10%或5/7条件，也没有扩展参数网格。F300和PV未重训。所有折模型保留用于复核，未用全7房间部署重训或覆盖0.1模型。', '',
        '## 验证设计与处理细节','',
        '普通验证7折，每折训练6房间、留出1房间。留朝向分别留出+X的BM223、−Y的BM432/BM441/LM608、+Y的BM522/BM608/LM703；后两折各训练4房间、留出3房间。+X折复用同一个BM223折，所以只有9个唯一训练问题、10个逻辑外层折。两套结果不是独立重复实验，不能把它们当14个独立房间。',
        '内层按Room_ID做3折GroupKFold，仅比较lambda=0.01/0.1，alpha=lambda×当前拟合行数。每个目标按内层原单位房间等权MAE选择；P10两分支同lambda，独立平均照度目标可选择另一lambda。样条结点、标准化均在各自训练折内拟合。保存63次选择冻结记录后才形成外层评分。',
        '两套外层的每个房间都先独立计分，再按7房间等权汇总；不把只有1个房间的+X与各3个房间的±Y直接等权平均。T0使用与0.1一致的固定参数，但每折只在外层训练房间重新拟合。其P10平均9.398 lux是本轮匹配基线，不混用较早不同参数流程的9.748 lux。', '',
        '## 越界、方向覆盖与来源限制','']
    cliprows=[]
    for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
        for method in cfg['methods']:
            for target in cfg['methods'][method]['targets']:
                f=fold[(fold.protocol==protocol)&(fold.method==method)&(fold.target==target)]
                cliprows.append([protocol,method,TARGET_NAMES[target],f'{f.raw_out_of_bounds_fraction.mean()*100:.2f}%',err(f.raw_mae.mean(),target),err(f.mae.mean(),target)])
    report += [table(['验证','方法','目标','原始输出越界率','裁剪前平均MAE','主指标MAE'],cliprows),'',
        '相对P10的原始输出定义为ratio×exp(mean_log)；实际推断先分别截断负ratio、负配套均值。相对P10的分支裁剪率另见fold_metrics.csv中的component_clipping_fraction。不能将其原始输出越界率直接等同最终输出为0的比例。R²和RMSE完整保存在CSV；极低且方差很小的P10房间可能出现很负的R²，即使绝对MAE较小，故不把单一R²当准确率。',
        '留+X折的所有训练样本X=0，X和X×u/X×v恒为0；该折无法从数据识别这些方向系数。新交互让模型能表达更多联合变化，但没有创造缺失方向的独立样本。TRAIN仍无−X房间，因此本轮无法证明BM702已改善。',
        '原始EPW与模拟是否相同、逐时对齐、模型地理北旋转、百叶角度定义和传感器网格仍待源资料确认。留一朝向还同时减少训练房间数，分数变化包含覆盖与训练样本构成两方面，不能单独归因为朝向。', '',
        '## 核验与复现','',
        f'核验通过700条内层房间评分、63次lambda选择、523个拟合输入/标签哈希、72个训练折预处理统计、9个模型包和{verification["prediction_values_reproduced"]:,}项预测值，以及182条外层逐房间指标、汇总与替换判断。',
        'T0与历史同参数留房间输出的最大差：平均照度0、FUDI 0、P10约6.3×10⁻¹³ lux（并行RF浮点求和量级）。原始数据、冻结配置/脚本和试用版0.1模型哈希保持不变。详细检查见[verification.json](verification.json)。', '',
        '在项目conda环境中执行：`python scripts/train_directional_v8.py` → `python scripts/verify_directional_v8.py` → `python scripts/finalize_directional_v8.py`。训练器拒绝覆盖已存在run_manifest.json；重新实验必须先使用新的输出目录和配置，保留本轮证据。核验/汇总可在已有结果上运行。', '',
        '输出入口：[全部汇总](summary_metrics.csv)、[逐房间](fold_metrics.csv)、[逐房间变化](paired_room_comparison.csv)、[分小时/角度/DNI](stratified_metrics.csv)、[P10两分支](P10_components.csv)、[替换判断](development_gate.csv)、[运行记录](run_manifest.json)。', '',
        '## 后续方向（本轮未执行）','',
        '平均照度优先保留T1及其方向交互作为继续研究的候选，集中分析BM432在强直射和不同时段的剩余误差；不要用平均MAE下降掩盖最差房间尚未解决。P10优先核对亮房间与近零房间的比例分支和条件分布，而不是继续扩大同一asinh/Ridge网格。FUDI需研究BM223/BM432改善但BM522退步的房间差异；下一步模型结构选择仍应只用开发房间验证，不按房间ID做事后路由。',
        '这些是本轮结果支持的研究重点，不是已经确认有效的新算法，也不包含新的测试集复查。', '',
        '## 统计解释检查（11/11）','',
        table(['检查','结论/限制'],[
            ['Simpson悖论/汇总掩盖分组','已逐房间配对；FUDI平均改善但5/7房间变差，明确披露。'],
            ['生态谬误','不从年度或房间平均推断每个小时都准确；不将7房间推广为476房间已验证。'],
            ['Berkson/选择偏差','7个代表房间并非随机总体样本，Kmeans代表性不等于方向覆盖充分。'],
            ['碰撞偏差','不作因果调整结论；输入没有使用本行目标派生值，比例仅用于训练标签。'],
            ['基率忽略','这是回归；MAE不是准确率，近零P10房间与高值房间分别报告。'],
            ['回归均值','不以旧最差房间改善作为唯一成功依据，固定相同外层基线并同时报告新最差房间。'],
            ['幸存者偏差','全部预先指定单元、目标与折均完成，失败单元也完整保存。'],
            ['多重尝试/到处找显著','完整报告10个候选-目标比较，不作p值或显著性宣称。'],
            ['分析岔路/事后选择','冻结两lambda和门槛，不扩格；但既有多轮开发与观察测试反馈仍限制独立性。'],
            ['相关不等于因果','比较预测方法性能，不将误差变化视为真实建筑因果效应或确定的源数据问题。'],
            ['逆向因果/标签进入输入','推断仅用13原始输入，真实均值不进入P10推断；模型重载验证不提供目标或Room_ID。']]),'',
        '不计算把逐时记录当独立样本的置信区间或显著性检验；工程误差容限仅作描述。核验状态为VERIFIED（保存模型的推断与指标复现），不声称从头第二次训练或物理源数据已核实。','']
    (out/'summary.md').write_text('\n'.join(report),encoding='utf-8')
    status=D.read(ROOT/'docs/project_status.json')
    status.update({'latest_completed_experiment':cfg['experiment_id'],'latest_approval':'docs/approvals/directional_v8_approval.json',
        'report':'reports/experiment_v8/summary.md','configuration':'configs/experiment_v8.json','verification':'reports/experiment_v8/verification.json',
        'pending_proposal':None,'pending_proposal_status':'Approved v8 completed and verified; no subsequent experiment authorized by this record.',
        'authorization_status':'Latest request authorized bounded TRAIN-only v8; completed without TEST fitting/scoring or new simulation/visualization.',
        'next_priority':'Retain trial 0.1; directional mean candidates improve average but not worst-room gate, P10 and FUDI transfer remain unresolved.',
        'v8':{'status':'completed_verified','estimator_fits':523,'TEST_fit_rows':0,'TEST_predictions':0,'TEST_scores':0,
            'targets':decision['targets'],'full_seven_refit':False,'deployed_trial_unchanged':True}})
    D.dump(ROOT/'docs/project_status.json',status)
    registry=D.read(ROOT/'docs/model_evidence_registry.json')
    registry['status']='v8_verified_no_candidate_passes_all_gates_trial_0_1_retained'
    registry['BM702_post_holdout_diagnosis']['proposal_executed']=True
    registry['v8_directional_transfer']={'report':'reports/experiment_v8/summary.md','verification':'reports/experiment_v8/verification.json',
        'metrics_source':'reports/experiment_v8/summary_metrics.csv','metrics_sha256':sha(out/'summary_metrics.csv'),
        'summary_metrics':summary.to_dict(orient='records'),'decisions':decision['targets'],
        'TEST_use_in_this_run':False,'independent_final_test':False,'deployed_trial_unchanged':True}
    D.dump(ROOT/'docs/model_evidence_registry.json',registry)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    marker='<!-- v8-current -->'
    if marker not in text:
        text=text.replace('各轮开发结果：[实验索引](docs/experiment_index.md)。v7没有候选通过全部替换条件，试用版沿用保留方法。',
            '各轮开发结果：[实验索引](docs/experiment_index.md)。'+marker+'最新v8已完成方向交互、树/样条与直接P10对照：平均照度平均误差降低，但没有候选通过全部最差房间与稳定性条件，保留0.1模型。[v8数值报告](reports/experiment_v8/summary.md)；三个测试房间未参与本轮。')
        readme.write_text(text,encoding='utf-8')
    index=ROOT/'docs/experiment_index.md';text=index.read_text(encoding='utf-8')
    text=text.replace('更新：2026-11-08。最新完成v7逐目标优化。最差误差优先方案没有输出同时通过全部开发条件。 旧参考保留，新候选单独记录。',
        '更新：2026-11-08。最新完成v8方向交互与模型表示实验，523次拟合、两套分组验证通过核验。平均照度有平均误差收益，全部候选仍未同时通过最差房间和稳定性条件；固定0.1模型保留。')
    text=text.replace('**这里记录开发证据，不是一套已经锁定并通过独立测试的最终模型。** 七个房间已被多轮用于开发；BM532、BM702、BMG26继续封存。',
        '**这里记录开发证据，不能据此宣称新房间可靠。** 七个房间已被多轮用于开发；BM532、BM702、BMG26已在0.1固定模型评估中被观察，但继续排除在训练/选模之外。下文历史轮次“封存”是当时状态。')
    if '| v8 |' not in text:text=text.replace('| v7 | 逐目标损失函数与最差房间优先选择 | [报告](../reports/experiment_v7/report.html) |',
        '| v7 | 逐目标损失函数与最差房间优先选择 | [报告](../reports/experiment_v7/report.html) |\n| v8 | 方向交互、树/样条＋Ridge、P10直接对照；留房间/留朝向 | [数值报告](../reports/experiment_v8/summary.md) |')
    if marker not in text:text += '\n\n'+marker+'\n## v8方向交互与跨朝向验证\n\n仅7开发房间，完成523次估计器拟合。普通留房间：T1平均照度82.433→70.274 lux，7/7改善；S1为64.380 lux，但最差仍222.832 lux、仅改善3.80%。S1 FUDI平均11.036→9.513个百分点，仅2/7房间改善。P10直接对照平均12.344 lux，高于基线9.398 lux。0/10候选-目标组合通过全部冻结条件，模型0.1保持不变。\n\n[完整两套验证与最差房间](../reports/experiment_v8/summary.md) · [核验记录](../reports/experiment_v8/verification.json)。本轮没有TEST预测/评分、模拟或可视化。\n'
    index.write_text(text,encoding='utf-8')
    artifacts=[p for folder in [out,ROOT/cfg['model_output']] for p in folder.rglob('*') if p.is_file() and p.name not in ['artifact_manifest.json','repository_preflight.json']]
    artifacts += [D.CONFIG,ROOT/'docs/experiment_v8_protocol.md',ROOT/'docs/approvals/directional_v8_approval.json']
    artifacts += [ROOT/'scripts'/name for name in ['directional_v8.py','train_directional_v8.py','verify_directional_v8.py','finalize_directional_v8.py']]
    artifacts += [ROOT/'docs'/name for name in ['project_status.json','model_evidence_registry.json','experiment_index.md']]+[readme]
    records=[];findings=[]
    for p in sorted(set(artifacts)):
        relative=p.relative_to(ROOT).as_posix();size=p.stat().st_size
        assert size<100*2**20,relative
        records.append({'path':relative,'bytes':size,'sha256':sha(p)})
        if p.suffix.lower() in TEXT:
            content=p.read_text(encoding='utf-8-sig',errors='replace')
            for category,pattern in PATTERNS.items():
                for match in pattern.finditer(content):findings.append({'path':relative,'type':category,'line':content.count('\n',0,match.start())+1})
    D.dump(out/'repository_preflight.json',{'status':'passed' if not findings else 'review_required','files':len(records),'bytes':sum(r['bytes'] for r in records),'findings':findings,'scan_scope':'Changed v8 artifacts and project indexes; no credential values logged.'})
    assert not findings,findings
    D.dump(out/'artifact_manifest.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),'scope':'v8 outputs, code, configuration, approval and updated project indexes; excludes this manifest and preflight record','files':records})
    print(json.dumps({'report':'reports/experiment_v8/summary.md','files':len(records),'MiB':round(sum(r['bytes'] for r in records)/2**20,2),'candidates_passing':int(gates.passes_all.sum()),'status':'finalized'},ensure_ascii=False))


if __name__=='__main__':main()
