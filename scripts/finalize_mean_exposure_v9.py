"""Numerical v9 report, evidence indexes and artifact inventory; no training or plots."""
from datetime import datetime,timezone
import json
import pandas as pd
import mean_exposure_v9 as V
from train_mean_exposure_v9 import sha
from prepare_repository_snapshot import PATTERNS,TEXT

D=V.D;ROOT=V.ROOT
NAMES={'T0':'原特征＋XGB log1p（现有模型方法）','T1':'方向交互＋XGB log1p（v8）','S1':'方向交互＋样条Ridge log1p（v8）',
       'A':'方向交互＋XGB原单位L1','B':'增加q及辐射交互＋XGB原单位L1','C':'B特征＋E/q标签、q权重、推断乘回q'}


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    cfg=D.read(V.CONFIG);out=ROOT/cfg['output'];run=D.read(out/'run_manifest.json');verification=D.read(out/'verification.json')
    assert run['status']=='completed_verified' and verification['status']=='passed'
    summary=pd.read_csv(out/'summary_metrics.csv');fm=pd.read_csv(out/'fold_metrics.csv');gates=pd.read_csv(out/'development_gate.csv');pairs=pd.read_csv(out/'paired_room_comparison.csv');strata=pd.read_csv(out/'stratified_metrics.csv')
    decision=D.read(out/'candidate_decision.json');methods=cfg['references']+V.METHODS
    def row(protocol,method):return summary[(summary.protocol==protocol)&(summary.method==method)].iloc[0]
    def f(v):return f'{v:.3f}'
    passed=gates.loc[gates.passes_all,'method'].tolist();reached=gates.loc[gates.reaches_primary_targets,'method'].tolist()
    conclusion=f'达到平均MAE<50且BM432≤120的候选：{", ".join(reached) if reached else "无"}；通过全部稳定性条件：{", ".join(passed) if passed else "无"}。'
    report=['## Material Passport','',
        '- 实验：mean_illumination_exposure_v9；用户2026-11-08批准A/B/C三组方案。训练与核验均完成。',
        '- 仅E_mean_lux；7开发房间122,640行，三个测试房间52,560行在数值转换前排除。本轮TEST训练、调参、预测、评分均为0。',
        f'- 189次估计器拟合：162内层＋27外层，训练{run["elapsed_seconds"]:.1f}秒；核验无新增拟合。无模拟、可视化或全7部署重训，现有0.1模型保持原样。',
        '- 这是已反复使用开发房间的研究结果，方案受到此前误差诊断启发，不是新的独立最终测试。',
        '- [批准记录](../../docs/approvals/mean_exposure_v9_approval.json) · [冻结配置](../../configs/experiment_v9.json) · [原始方案](../../docs/next_experiment_BM432_mean.md)。原始方案保留提出时状态，本次授权以批准记录为准。','',
        '## 结果判断','',conclusion,'',
        '目标与稳定性条件分开报告；即使仅BM432改善，也不能省略其他房间退步。是否进入替换候选不等于已部署，本轮不覆盖模型。','',
        '## 两种验证的完整MAE（lux）','',
        '平均MAE是先逐房间计算后对7房间等权平均；最差是最大房间MAE，不是最大单小时误差。留朝向也按7房间等权，不按3个方向直接平均。','',
        table(['方法','留房间平均','最差 / 房间','BM432','留朝向平均','最差 / 房间','BM432'],[[m,f(row('LeaveOneRoomOut',m).macro_MAE),f(row('LeaveOneRoomOut',m).worst_MAE)+' / '+row('LeaveOneRoomOut',m).worst_room,
            f(row('LeaveOneRoomOut',m).BM432_MAE),f(row('LeaveOneOrientationOut',m).macro_MAE),f(row('LeaveOneOrientationOut',m).worst_MAE)+' / '+row('LeaveOneOrientationOut',m).worst_room,f(row('LeaveOneOrientationOut',m).BM432_MAE)] for m in methods]),'',
        table(['方法','定义'],[[m,NAMES[m]] for m in methods]),'']
    for method in V.METHODS:
        a=row('LeaveOneRoomOut',method);t=row('LeaveOneRoomOut','T1');s=row('LeaveOneRoomOut','S1');o=row('LeaveOneOrientationOut',method)
        report += [f'{method}：主验证平均{a.macro_MAE:.3f} lux，较T1变化{(a.macro_MAE/t.macro_MAE-1)*100:+.2f}%、较S1变化{(a.macro_MAE/s.macro_MAE-1)*100:+.2f}%；BM432为{a.BM432_MAE:.3f} lux，较S1变化{(a.BM432_MAE/s.BM432_MAE-1)*100:+.2f}%。留朝向平均{o.macro_MAE:.3f} lux。','']
    c=row('LeaveOneRoomOut','C');s=row('LeaveOneRoomOut','S1')
    lm=fm[(fm.protocol=='LeaveOneRoomOut')&(fm.room=='LM608')].set_index('method')
    report += [f'本轮的关键取舍：C把BM432从{s.BM432_MAE:.3f}降到{c.BM432_MAE:.3f} lux，但LM608从{lm.loc["S1","mae"]:.3f}升到{lm.loc["C","mae"]:.3f} lux，且C在该房间的平均偏差为{lm.loc["C","bias"]:+.3f} lux（高估）。C的平均MAE仍为{c.macro_MAE:.3f}，未优于S1的{s.macro_MAE:.3f}。', '',
        f'留朝向时C的BM432 MAE为{row("LeaveOneOrientationOut","C").BM432_MAE:.3f} lux，较S1的{row("LeaveOneOrientationOut","S1").BM432_MAE:.3f}更差。这表明归一化的收益依赖当前训练房间构成；不能把普通留房间中的BM432改善解释为稳定的未见方向迁移。', '',
        'A/B没有解决原有亮工况低估，C则在房间之间改变了高估/低估分布。结果不支持“取消log或加入单一亮度尺度就足够”的结论；也不能仅凭这些误差确定缺少哪一项几何。按停止规则保留原模型，不追加房间专用校准。','']
    bright_main=strata[(strata.protocol=='LeaveOneRoomOut')&(strata.room=='BM432')&(strata.dimension=='true_lux_bin')&(strata.value=='>1000')].set_index('method')
    report += [f'BM432的最高照度区间仍未解决：真值>1000 lux时，S1 MAE为{bright_main.loc["S1","mae"]:.3f}，C为{bright_main.loc["C","mae"]:.3f} lux，仅降低{(1-bright_main.loc["C","mae"]/bright_main.loc["S1","mae"])*100:.2f}%。不能用房间整体改善23%替代对高亮工况的评价。','']
    report += ['## 全部房间与退步检查','']
    for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
        g=fm[fm.protocol.eq(protocol)].pivot(index='room',columns='method',values='mae')
        report += [protocol+'，MAE单位lux：','',table(['房间']+methods,[[room]+[f(g.loc[room,m]) for m in methods] for room in cfg['development_rooms']]),'']
    deterioration=[]
    for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
        for method in V.METHODS:
            for reference in ['T1','S1']:
                p=pairs[(pairs.protocol==protocol)&(pairs.method==method)&(pairs.reference==reference)&(pairs.room!='BM432')]
                worse=p[p.worse]
                deterioration.append([protocol,method,reference,str(len(worse))+'/6','；'.join(f'{r.room} {r.change_lux:+.3f}' for r in worse.itertuples()) or '无'])
    report += ['除BM432以外的6个房间，正数表示MAE增加。T1是新增方向特征后的树对照，S1是v8平均分最佳对照；同时展示避免换基准掩盖退步。','',
        table(['验证','候选','比较参考','退步房间数','退步房间及MAE增加lux'],deterioration),'',
        '## 预先冻结的目标及稳定性条件','',
        '主目标：普通留房间平均MAE<50 lux、BM432≤120 lux。额外要求：普通留房间最差不超过v8 S1，至少5/7房间较T1不恶化；留朝向平均与最差均不超过T1。数值参照均来自相同外层划分的v8完整预测。','',
        table(['候选','平均<50','BM432≤120','最差保护','≥5/7不退步','朝向平均保护','朝向最差保护','全部通过'],[[r.method]+['是' if v else '否' for v in [r.macro_below_50,r.BM432_at_most_120,r.worst_nonworse_than_S1,r.five_rooms_nonworse_than_T1,r.orientation_macro_nonworse_than_T1,r.orientation_worst_nonworse_than_T1,r.passes_all]] for r in gates.itertuples()]),'',
        '没有通过就保留原方法；若有通过，仅记录为开发候选，仍不自动部署。未根据结果追加深度、模型、倍率或输入。','',
        '## BM432亮工况与偏差','']
    bright=[]
    for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
        for dimension,value in [('true_lux_bin','>1000'),('DNI_bin','>500'),('midday','10-14')]:
            g=strata[(strata.protocol==protocol)&(strata.room=='BM432')&(strata.dimension==dimension)&(strata.value==value)]
            for m in ['T1','S1']+V.METHODS:
                r=g[g.method.eq(m)].iloc[0]
                bright.append([protocol,dimension+' '+value,m,r.n,f(r.mae),f(r.bias),f'{r.within_50_lux*100:.1f}%'])
    report += [table(['验证','工况','方法','记录数','MAE lux','偏差lux','误差≤50比例'],bright),'',
        '偏差=预测−真值，负值为低估。真实照度分段只用于事后诊断，不用于预测时路由或筛除样本。完整小时、四角度与DNI分段保存在stratified_metrics.csv。','',
        '## 复现与来源核验','',
        'q=(玻璃面积/房间面积)×(年度受遮挡辐射/100)。F2=F1+[q,q×DHI,q×DNI×cos(太阳高度角)]。A使用21个编码特征；B/C使用24个，额外原始字段为既有BIM年度受遮挡辐射。q不是照度，也不是真实物理采光系数。',
        '已验证7开发房间年度辐射与BIM表一致、全年/角度内恒定、遮挡比例恒等式成立、q有限且为正。作者README将年度辐射定义为既有玻璃面积加权太阳辐射结果；其上游独立性依赖作者说明，本轮没有重新核实原始计算过程。没有用室内真实均值或房间专用倍率构造输入。',
        'C的标签为E/q，权重q/当前训练折平均q。核验确认其加权L1损失与原始lux绝对误差成比例；q均值只来自当前内层或外层拟合房间。预测时仅需要14个原始输入，不需要目标或Room_ID。A/B/C均恢复到lux后评分；负输出截0，无上限裁剪。',
        '每组只比较depth3/5，其余参数冻结；内层3折GroupKFold按房间等权MAE选深度，并列取3。9个唯一外层问题对应7折留房间＋3组留朝向，其中+X/BM223复用。留BM432时模型从未使用其标签；留−Y时BM432/BM441/LM608全部同时留出。',
        f'核验完成{verification["independent_inner_room_MAE_checks"]}条内层逐房间MAE、27次深度选择、189条输入/标签/权重哈希、9个模型包重载的{verification["saved_prediction_values_reproduced"]:,}项预测、84条外层指标及{verification["stratified_MAE_checks"]}条分段MAE；v8参考逐值一致，原始数据与0.1模型哈希不变。',
        '核验是保存模型推断和指标复现，不声称从头第二次训练或新的独立测试。EPW时间/北向与传感器网格来源仍待核对。', '',
        '在项目conda环境中依次执行`python scripts/train_mean_exposure_v9.py`、`python scripts/verify_mean_exposure_v9.py`、`python scripts/finalize_mean_exposure_v9.py`。训练器拒绝覆盖已有运行记录；新实验须使用新目录和配置。', '',
        '数值入口：[汇总](summary_metrics.csv)、[逐房间](fold_metrics.csv)、[退步对照](paired_room_comparison.csv)、[分段](stratified_metrics.csv)、[门槛判断](development_gate.csv)、[核验](verification.json)、[来源核对](input_source_audit.json)。', '',
        '## 统计解释检查（11/11）','',
        table(['检查','处理'],[
            ['Simpson/汇总掩盖分组','列出逐房间及两种参照的退步，不能用平均改善覆盖局部恶化。'],
            ['生态谬误','年度房间平均不证明每小时准确，7房间不能代表476房间均已验证。'],
            ['Berkson/选择偏差','代表房间来自既有筛选，方向和高照度覆盖有限。'],
            ['碰撞偏差','不将经验输入系数解释为因果作用，无目标派生输入。'],
            ['基率忽略','回归MAE不是准确率；低光与亮工况分别报告。'],
            ['回归均值','重点房间源自事后发现，必须同划分配对及全房间比较。'],
            ['幸存者偏差','全部三组、两种深度、外层折保留，失败也报告。'],
            ['多重尝试','不作显著性结论，候选范围冻结且不扩展。'],
            ['分析岔路','计划和规则在训练前冻结，但多轮使用开发结果仍影响独立性。'],
            ['相关与因果','预测改善不能证明物理机制或原数据错误。'],
            ['逆向因果/标签泄漏','推断不接受真值或Room_ID，q来自既有BIM输入；原始来源限制披露。']]),'',
        '停止规则：本轮只完成声明的189次拟合，不因未达目标追加搜索、BM432校准或TEST复查。','']
    (out/'summary.md').write_text('\n'.join(report),encoding='utf-8')
    status=D.read(ROOT/'docs/project_status.json')
    status.update({'latest_completed_experiment':cfg['experiment_id'],'latest_approval':'docs/approvals/mean_exposure_v9_approval.json',
        'report':'reports/experiment_v9/summary.md','configuration':'configs/experiment_v9.json','verification':'reports/experiment_v9/verification.json',
        'pending_proposal':None,'pending_proposal_status':'Approved A/B/C experiment completed and verified; no further expansion authorized by this record.',
        'authorization_status':'Bounded TRAIN-only mean exposure v9 completed; no TEST operations, simulation or visualization.',
        'v9':{'status':'completed_verified','estimator_fits':189,'targets_reached':reached,'passing_candidates':passed,'TEST_fit_rows':0,'TEST_predictions':0,'TEST_scores':0,'deployed_trial_unchanged':True},
        'next_priority':'Review v9 mean-illumination evidence and remaining room-specific errors; no automatic search expansion.'})
    D.dump(ROOT/'docs/project_status.json',status)
    registry=D.read(ROOT/'docs/model_evidence_registry.json');registry['status']='v9_verified_trial_0_1_unchanged'
    registry['v9_mean_exposure']={'report':'reports/experiment_v9/summary.md','verification':'reports/experiment_v9/verification.json','summary':summary.to_dict(orient='records'),
        'metric_source_sha256':sha(out/'summary_metrics.csv'),'decision':decision,'TEST_use':False,'independent_final_test':False}
    D.dump(ROOT/'docs/model_evidence_registry.json',registry)
    marker='<!-- v9-current -->'
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    text=text.replace('最新v8已完成','v8已完成')
    if marker not in text:
        text=text.replace('## 文件位置',marker+'\n最新v9：已完成平均照度A/B/C三组对照及留房间/留朝向验证。'+conclusion+'[完整数值与其他房间退步检查](reports/experiment_v9/summary.md)。本轮不改变0.1部署模型。\n\n## 文件位置')
        readme.write_text(text,encoding='utf-8')
    index=ROOT/'docs/experiment_index.md';text=index.read_text(encoding='utf-8')
    if marker not in text:
        text=text.replace('# 实验与模型证据索引','# 实验与模型证据索引\n\n'+marker+'\n最新完成v9平均照度实验，189次拟合并通过核验。'+conclusion+'[完整报告](../reports/experiment_v9/summary.md)。下文保留各轮当时结论。',1)
        text += '\n\n## v9平均照度与房间强度归一化\n\n'+conclusion+'\n\n[两种验证及逐房间变化](../reports/experiment_v9/summary.md) · [输入来源核对](../reports/experiment_v9/input_source_audit.json) · [核验](../reports/experiment_v9/verification.json)。仅7开发房间，其他输出与0.1模型不变。\n'
        index.write_text(text,encoding='utf-8')
    artifacts=[p for folder in [out,ROOT/cfg['model_output']] for p in folder.rglob('*') if p.is_file() and p.name not in ['artifact_manifest.json','repository_preflight.json']]
    artifacts += [V.CONFIG,ROOT/'docs/approvals/mean_exposure_v9_approval.json']+[ROOT/'scripts'/n for n in ['mean_exposure_v9.py','train_mean_exposure_v9.py','verify_mean_exposure_v9.py','finalize_mean_exposure_v9.py']]
    artifacts += [readme]+[ROOT/'docs'/n for n in ['project_status.json','model_evidence_registry.json','experiment_index.md']]
    records=[];findings=[]
    for p in sorted(set(artifacts)):
        assert p.stat().st_size<100*2**20
        rel=p.relative_to(ROOT).as_posix();records.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
        if p.suffix.lower() in TEXT:
            text=p.read_text(encoding='utf-8-sig',errors='replace')
            for category,pattern in PATTERNS.items():
                for m in pattern.finditer(text):findings.append({'path':rel,'type':category,'line':text.count('\n',0,m.start())+1})
    D.dump(out/'repository_preflight.json',{'status':'passed' if not findings else 'review_required','files':len(records),'findings':findings});assert not findings
    D.dump(out/'artifact_manifest.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),'scope':'v9 outputs/code/config/approval and current project indexes; excludes manifest and preflight','files':records})
    print(json.dumps({'status':'finalized','report':'reports/experiment_v9/summary.md','targets_reached':reached,'passing':passed,'MiB':sum(r['bytes'] for r in records)/2**20},ensure_ascii=False))


if __name__=='__main__':main()
