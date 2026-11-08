"""Publish numerical v10 evidence and project indexes without fitting or plots."""
from datetime import datetime,timezone
import json
import pandas as pd
import pair_balance_v10 as V
from train_pair_balance_v10 import sha
from prepare_repository_snapshot import PATTERNS,TEXT

D=V.D;ROOT=V.ROOT
NAMES={'v8_T1':'v8方向交互XGB log','v8_S1':'v8方向交互样条Ridge log','v9_C':'v9归一化XGB加权L1',
       'v10_X0':'F2＋XGB，log(E+1)','v10_Xq':'F2＋XGB，log(E/q+1)',
       'v10_S0':'F2＋样条Ridge，log(E+1)','v10_Sq':'F2＋样条Ridge，log(E/q+1)'}


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    cfg=D.read(V.CONFIG);out=ROOT/cfg['output'];run=D.read(out/'run_manifest.json');verification=D.read(out/'verification.json')
    assert run['status']=='completed_verified' and verification['status']=='passed'
    summary=pd.read_csv(out/'summary_metrics.csv');fm=pd.read_csv(out/'fold_metrics.csv');gates=pd.read_csv(out/'development_gate.csv');pairs=pd.read_csv(out/'paired_room_comparison.csv');strata=pd.read_csv(out/'stratified_metrics.csv')
    decision=D.read(out/'candidate_decision.json');methods=list(cfg['references'])+V.METHODS
    def row(protocol,method):return summary[(summary.protocol==protocol)&(summary.method==method)].iloc[0]
    def f(value):return f'{value:.3f}'
    reached=gates[gates.reaches_primary_targets].method.tolist();passed=gates[gates.passes_all].method.tolist();partial=gates[gates.partial_progress].method.tolist()
    conclusion=f'同时达到50/120/40的候选：{", ".join(reached) or "无"}；全部稳定性条件通过：{", ".join(passed) or "无"}。'
    contrasts=[]
    for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
        for a,b in [('v10_X0','v10_Xq'),('v10_S0','v10_Sq')]:
            x,y=row(protocol,a),row(protocol,b)
            contrasts.append({'protocol':protocol,'unnormalized':a,'normalized':b,'macro_MAE_change':float(y.macro_MAE-x.macro_MAE),
                'worst_MAE_change':float(y.worst_MAE-x.worst_MAE),'BM432_MAE_change':float(y.BM432_MAE-x.BM432_MAE),'LM608_MAE_change':float(y.LM608_MAE-x.LM608_MAE)})
    pd.DataFrame(contrasts).to_csv(out/'factorial_contrasts.csv',index=False)
    profiles=pd.read_csv(out/'development_profiles.csv').set_index('Room_ID');splits=D.read(out/'fold_splits.json');support=[]
    for job in splits['jobs']:
        for room in job['held_rooms']:
            for feature in ['q','Room_Area_m2','Annual_Incident_Shaded_kWh_m2']:
                lo=float(profiles.loc[job['fit_rooms'],feature].min());hi=float(profiles.loc[job['fit_rooms'],feature].max());value=float(profiles.loc[room,feature])
                support.append({'job_id':job['id'],'held_room':room,'feature':feature,'fit_min':lo,'fit_max':hi,'held_value':value,'outside_training_range':value<lo or value>hi})
    pd.DataFrame(support).to_csv(out/'static_feature_support.csv',index=False)
    report=['## Material Passport','',
        '- 实验：pair_balance_log_normalization_v10；用户2026-11-08批准四格方案。状态：训练与数值核验完成。',
        '- 仅7开发房间122,640行；TEST的52,560行在数值转换前排除，本轮无TEST训练、调参、预测或评分。',
        f'- 252次估计器拟合（216内层＋36外层）；训练预处理36次，核验另重建36个预处理器，无新增估计器拟合。训练耗时{run["elapsed_seconds"]:.1f}秒，核验{verification["seconds"]:.1f}秒。',
        '- 无新增模拟/可视化/全7部署重训。其他四个输出与0.1部署模型保持原样。',
        '- [批准记录](../../docs/approvals/pair_balance_v10_approval.json) · [配置](../../configs/experiment_v10.json) · [原始方案](../../docs/next_experiment_pair_balance_v10.md)。原始方案保留提出时状态，本次执行以批准记录为准。',
        '- 七房间已多轮用于开发，方案受既有误差诊断启发；这不是新的独立最终测试。','',
        '## 主要结论','',conclusion,'',
        '**v10_S0在留一房间验证首次同时达到平均MAE<50和BM432≤120，但LM608未达到40，留一朝向表现严重退步，不能替换现有模型。**',
        f'S0留房间平均{row("LeaveOneRoomOut","v10_S0").macro_MAE:.3f}、BM432 {row("LeaveOneRoomOut","v10_S0").BM432_MAE:.3f}、LM608 {row("LeaveOneRoomOut","v10_S0").LM608_MAE:.3f} lux；留朝向则为{row("LeaveOneOrientationOut","v10_S0").macro_MAE:.3f}、{row("LeaveOneOrientationOut","v10_S0").BM432_MAE:.3f}、{row("LeaveOneOrientationOut","v10_S0").LM608_MAE:.3f} lux。',
        'Sq在留朝向的平均误差更好，但普通留房间的BM432仍明显偏差。不能在知道某房间/验证协议表现后，用S0和Sq拼成事后最优模型。','',
        '## 两种验证的完整结果（lux）','',
        '每个房间先独立评分，再对7个房间等权平均；最差是最大房间MAE，不是最大单小时误差。三组方向不直接等权平均。','',
        table(['方法','留房间平均','BM432','LM608','留朝向平均','BM432','LM608'],[[m]+[f(row(p,m)[metric]) for p in ['LeaveOneRoomOut','LeaveOneOrientationOut'] for metric in ['macro_MAE','BM432_MAE','LM608_MAE']] for m in methods]),'',
        table(['方法','含义'],[[m,NAMES[m]] for m in methods]),'',
        '本轮表中所有方法在两种验证的最差房间均为BM432；最差MAE见BM432列，机器可读表保留worst_room/worst_MAE。v10_S0名称与v8历史S0不混用。','',
        '## 全部房间是否退步','']
    for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
        g=fm[fm.protocol.eq(protocol)].pivot(index='room',columns='method',values='mae')
        report += [protocol+'：','',table(['房间']+methods,[[r]+[f(g.loc[r,m]) for m in methods] for r in cfg['development_rooms']]),'']
    regressions=[]
    for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
        for method in V.METHODS:
            for reference in ['v8_S1','v9_C']:
                p=pairs[(pairs.protocol==protocol)&(pairs.method==method)&(pairs.reference==reference)&~pairs.room.isin(['BM432','LM608'])]
                worse=p[p.worse]
                regressions.append([protocol,method,reference,f'{len(worse)}/5','；'.join(f'{r.room} {r.change_lux:+.3f}' for r in worse.itertuples()) or '无'])
    report += ['排除两个重点房间后，另外5个房间的退步；正数为MAE增加。逐房间对照文件还包括T1参考。','',
        table(['验证','候选','参照','退步数','房间及增加lux'],regressions),'',
        'S0相对v8 S1，普通留房间改善BM223/BM432/BM441，退步BM522/BM608/LM608/LM703；相对v9 C则6/7房间改善，只有BM223退步。两种参照必须同时展示，不能只挑有利参考。','',
        '## 目标与冻结筛选条件','',
        '主目标同时满足平均MAE<50、BM432≤120、LM608≤40。另有预定义部分进展条件：平均低于v8 S1、BM432不高于v9 C、LM608不高于v8 S1。全部稳定性条件还包括最差不高于v9 C、至少5/7房间较v8 S1不恶化、留朝向平均/最差不高于v8 T1且LM608不高于v8 S1。', '',
        table(['候选','平均<50','BM432≤120','LM608≤40','较S1不恶化房间','预定义部分进展','全部通过'],[[r.method,'是' if r.macro_below_50 else '否','是' if r.BM432_at_most_120 else '否','是' if r.LM608_at_most_40 else '否',f'{r.rooms_nonworse_than_S1}/7','是' if r.partial_progress else '否','是' if r.passes_all else '否'] for r in gates.itertuples()]),'',
        f'预定义“部分进展”通过：{", ".join(partial) or "无"}。这不否认S0的两个单项目标有改善，但不能将两项达标写成三项达标，更不能把42.662视为≤40。精确数值与各保护条件见development_gate.csv。没有根据结果放宽门槛。','',
        '## 同输入上的归一化对照','',
        '下表是归一化版本减去非归一化版本的MAE，负值为改善。模型参数仍由各自内层选择，解释为完整训练流程对照，不是固定参数下的纯因果效应。','',
        table(['验证','比较','平均变化','BM432变化','LM608变化'],[[r['protocol'],r['normalized']+' − '+r['unnormalized'],f(r['macro_MAE_change']),f(r['BM432_MAE_change']),f(r['LM608_MAE_change'])] for r in contrasts]),'',
        '归一化收益依赖模型和验证条件：树模型中Xq改善普通留房间，但样条中Sq弱于S0；留朝向时Sq则明显优于S0。不能据此把q宣布为普遍有效或普遍无效。','',
        '## S0为何不能直接投入新方向使用','',
        '普通留BM432仍有同为−Y的BM441/LM608可训练；留−Y组时这三个房间一起留出，仅剩BM223和三个+Y房间。训练房间数由6变4，朝向与强度支持同时改变。',
        'BM432的q=1.0635；普通留房间训练q最大约0.6394，留−Y训练最大约0.5445。S0的Ridge lambda由普通留BM432折的0.1变为留−Y折的0.01，均是各自内层合法选择。这些变化与方向缺失、特征外推和log逆变换放大同时存在，不能未经对照把480 lux的误差归因为单一因素。',
        '已有保存输出显示留−Y时S0对BM432转为强烈高估；它在留房间上的改善不能证明未见方向可靠。没有用外层误差重新选择lambda、改成0.1或加入输出上限。','',
        '## BM432与LM608的关键分段','']
    detail=[]
    for protocol in ['LeaveOneRoomOut','LeaveOneOrientationOut']:
        for room in ['BM432','LM608']:
            for dimension,value in [('DNI_bin','>500'),('midday','10-14'),('true_lux_bin','>1000')]:
                g=strata[(strata.protocol==protocol)&(strata.room==room)&(strata.dimension==dimension)&(strata.value==value)]
                for method in ['v8_S1','v9_C','v10_S0','v10_Sq']:
                    h=g[g.method.eq(method)]
                    if not len(h):continue
                    r=h.iloc[0];detail.append([protocol,room,dimension+' '+value,method,int(r.n),f(r.mae),f(r.bias)])
    report += [table(['验证','房间','分段','方法','记录数','MAE','偏差'],detail),'',
        '偏差=预测−真值。没有>1000 lux记录的房间不伪造该分段指标；真实照度分段只是事后诊断，不用于输入、路由或剔除样本。全部小时、角度和DNI分段见stratified_metrics.csv。','',
        '## 训练与核验','',
        '四组均使用v9 F2：14原始输入、24编码/交互特征。q是玻璃/地板比×年度受遮挡辐射/100，仅作经验尺度；作者说明和七房间BIM一致性检查通过，原始辐射/天气时间/北向和网格来源仍未独立核实。',
        'XGB仅比较depth3/5，150棵树、学习率0.07、平方损失。样条六列展开后48列，标准化＋Ridge，仅lambda0.01/0.1，alpha=lambda×当前拟合行数。所有行等权，不使用旧C的q权重。归一化与未归一化均log1p，恢复lux后选参和评分。log损失不等于原lux MAE。',
        '所有内层样条结点与标准化只用当前拟合房间；不按随机小时切分。外层共有9个唯一训练问题、10个逻辑折，+X/BM223复用。36次选参冻结后才形成外层评分，不做全7训练再给这7房间打泛化分。',
        f'核验通过：400条内层逐房间分数、36次参数选择、252条输入/标签哈希、36个预处理器统计、9个模型包的{verification["checks"]["prediction_values"]:,}项预测、98条外层指标、{verification["checks"]["strata"]}条分段MAE和全部门槛判断。原参照逐值一致，原数据、冻结脚本/配置和0.1模型哈希不变。',
        '核验复现保存模型的推断与指标，没有重跑第二次训练；预处理重建另记账，不计作新的预测估计器拟合。', '',
        '复现入口：`python scripts/train_pair_balance_v10.py` → `python scripts/verify_pair_balance_v10.py` → `python scripts/finalize_pair_balance_v10.py`。使用项目conda环境。训练器拒绝覆盖已有运行记录，新实验须采用新输出目录/配置。', '',
        '[汇总](summary_metrics.csv) · [逐房间](fold_metrics.csv) · [退步对照](paired_room_comparison.csv) · [门槛](development_gate.csv) · [核验](verification.json) · [输入支持范围](static_feature_support.csv)。','',
        '## 统计解释检查（11/11）','',
        table(['检查','处理'],[
            ['Simpson/汇总掩盖分组','主验证平均48但朝向失败；全部房间及两套验证完整报告。'],
            ['生态谬误','房间平均不等于每小时准确，不将7房间成绩推及476房间。'],
            ['Berkson/选择偏差','既有代表房间与方向覆盖有限，压力验证训练数量也减少。'],
            ['碰撞偏差','不作因果调整推断，无真实输出派生输入。'],
            ['基率忽略','MAE不是准确率，亮度分段及误差容限比例分别报告。'],
            ['回归均值','重点房间来自多轮诊断，配对比较及其他房间保护同时使用。'],
            ['幸存者偏差','四格和全部折都报告，包括S0朝向失败。'],
            ['多重尝试','不作显著性检验，不将大量逐时样本当大量独立房间。'],
            ['分析岔路','预先冻结配置与停止规则，不因42.662接近40而改门槛。'],
            ['相关不等于因果','q外推、正则与错误变化只作解释线索，不确立根因。'],
            ['逆向因果/泄漏','特征和q不使用照度目标；推断不接受Room_ID或真值。']]),'',
        '停止规则已执行：252次拟合后不追加搜索、融合权重或TEST复查。现有模型保持不变。','']
    (out/'summary.md').write_text('\n'.join(report),encoding='utf-8')
    status=D.read(ROOT/'docs/project_status.json')
    status.update({'latest_completed_experiment':cfg['experiment_id'],'latest_approval':'docs/approvals/pair_balance_v10_approval.json','report':'reports/experiment_v10/summary.md',
        'configuration':'configs/experiment_v10.json','verification':'reports/experiment_v10/verification.json','pending_proposal':None,
        'pending_proposal_status':'Approved v10 completed; no further search authorized by this record.','authorization_status':'TRAIN-only four-cell experiment completed; no TEST, simulation or visualization.',
        'next_priority':'Review S0 room-validation progress and severe orientation failure; no automatic extra searches or deployment.',
        'v10':{'status':'completed_verified','estimator_fits':252,'targets_reached':reached,'partial_progress':partial,'passing_candidates':passed,'TEST_predictions':0,'TEST_scores':0,'TEST_fit_rows':0,'deployed_trial_unchanged':True}})
    status['BM432_LM608_diagnosis']['proposal_executed']=True;D.dump(ROOT/'docs/project_status.json',status)
    registry=D.read(ROOT/'docs/model_evidence_registry.json');registry['status']='v10_verified_no_full_pass_trial_0_1_retained'
    registry['BM432_LM608_post_v9_diagnosis']['proposal_executed']=True
    registry['v10_pair_balance']={'report':'reports/experiment_v10/summary.md','verification':'reports/experiment_v10/verification.json','summary':summary.to_dict(orient='records'),
        'metrics_sha256':sha(out/'summary_metrics.csv'),'decision':decision,'TEST_use':False,'independent_final_test':False};D.dump(ROOT/'docs/model_evidence_registry.json',registry)
    marker='<!-- v10-current -->';readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8').replace('最新v9：','v9：')
    if marker not in text:
        text=text.replace('## 文件位置',marker+'\n最新v10：S0留房间平均48.04、BM432 112.54、LM608 42.66 lux；未达全部目标且留朝向明显失败，保留0.1。[完整报告与退步检查](reports/experiment_v10/summary.md)。\n\n## 文件位置');readme.write_text(text,encoding='utf-8')
    index=ROOT/'docs/experiment_index.md';text=index.read_text(encoding='utf-8')
    if marker not in text:
        text=text.replace('# 实验与模型证据索引','# 实验与模型证据索引\n\n'+marker+'\n最新v10已完成252次拟合并核验。S0留房间达成平均<50和BM432≤120，但LM608及留朝向未达要求，全部候选未通过。[结果](../reports/experiment_v10/summary.md)。下文为历史轮次记录。',1)
        text+='\n\n## v10四格log目标/归一化实验\n\n'+conclusion+'S0普通留房间平均48.040、BM432 112.541、LM608 42.662；留朝向平均115.460、BM432 480.140、LM608 102.251。Sq留朝向平均68.018，但普通留房间尚未达标。无TEST使用或部署变更。\n';index.write_text(text,encoding='utf-8')
    artifacts=[p for folder in [out,ROOT/cfg['model_output']] for p in folder.rglob('*') if p.is_file() and p.name not in ['artifact_manifest.json','repository_preflight.json']]
    artifacts += [V.CONFIG,ROOT/'docs/approvals/pair_balance_v10_approval.json',readme]+[ROOT/'docs'/n for n in ['project_status.json','model_evidence_registry.json','experiment_index.md']]
    artifacts += [ROOT/'scripts'/n for n in ['pair_balance_v10.py','train_pair_balance_v10.py','verify_pair_balance_v10.py','finalize_pair_balance_v10.py']]
    records=[];findings=[]
    for p in sorted(set(artifacts)):
        assert p.stat().st_size<100*2**20;relative=p.relative_to(ROOT).as_posix();records.append({'path':relative,'bytes':p.stat().st_size,'sha256':sha(p)})
        if p.suffix.lower() in TEXT:
            text=p.read_text(encoding='utf-8-sig',errors='replace')
            for category,pattern in PATTERNS.items():
                for m in pattern.finditer(text):findings.append({'path':relative,'type':category,'line':text.count('\n',0,m.start())+1})
    D.dump(out/'repository_preflight.json',{'status':'passed' if not findings else 'review_required','files':len(records),'findings':findings});assert not findings
    D.dump(out/'artifact_manifest.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),'scope':'v10 outputs/code/config/approval and current project indexes; excludes manifest and preflight','files':records})
    print(json.dumps({'status':'finalized','report':'reports/experiment_v10/summary.md','reached_all_three':reached,'passing':passed,'MiB':sum(r['bytes'] for r in records)/2**20},ensure_ascii=False))


if __name__=='__main__':main()
