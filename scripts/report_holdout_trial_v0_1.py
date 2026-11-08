"""Numerical holdout report and state update; no plots, fitting or new predictions."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pvsd_trial as api
import prepare_repository_snapshot as audit

OUT = ROOT/"reports/holdout_trial_v0_1"
LABELS = {"E_mean_lux": ("平均照度", 1, "lux"), "E_P10_lux": ("P10照度", 1, "lux"),
          "F_300_fraction": ("F300", 100, "个百分点"), "F_UDI_fraction": ("FUDI", 100, "个百分点"),
          "PV_AC_kWh": ("逐时PV", 1, "kWh/小时")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fmt(value, target):
    return f"{value*LABELS[target][1]:.5f}" if target == "PV_AC_kWh" else f"{value*LABELS[target][1]:.2f}"


def main():
    verification = read(OUT/"verification.json")
    assert verification["status"] == "passed"
    run = read(OUT/"run_manifest.json")
    assert run["status"] == "completed_verified"
    frozen = read(OUT/"pretest_freeze.json")
    for relative, digest in frozen["hashes"].items():
        assert api.sha256(ROOT/relative) == digest, relative
    rooms = pd.read_csv(OUT/"room_metrics.csv")
    model = rooms[rooms.method == "Frozen_trial_v0_1"]
    summary = pd.read_csv(OUT/"summary_metrics.csv")
    model_summary = summary[summary.method == "Frozen_trial_v0_1"]
    baseline_summary = summary[summary.method == "TRAIN_angle_mean"]
    lines = ["# 固定模型0.1：三个留出房间首次测试", "", "## Material Passport", "",
             "- 状态：VERIFIED；2026-11-08。模型与原始数据哈希保持不变。",
             "- 依据用户“是不是可以看一下在测试方向的性能了”的请求，首次评估BM532、BM702、BMG26。此前封存约束仅为此次评分解除；本轮未训练或调参。",
             "- 固定模型：trial 0.1.0；7个开发房间122,640条记录拟合；预测并评分3个TEST房间52,560行、262,800个目标值。",
             "- 评价规则于读取TEST数值目标前冻结并提交（b156b69）；先保存输入产生的预测，再载入真实目标评分。所有行保留。",
             "- 模型SHA-256：`"+run["model_sha256"]+"`。", "",
             "## 结论", "",
             "模型具有一定跨房间预测能力，但表现很不均匀，尚不足以宣称对任意新房间稳定准确。BM532表现较好；BM702五项MAE均为测试房间中最大，平均照度、P10和FUDI的R²为负；BMG26处于低照度情况，部分小绝对误差不能被解释为全面可靠。PV是此次测试中相对较强的输出。", "",
             "## MAE：越小越好", "", "| 输出 | BM532 | BM702 | BMG26 | 3房间平均 | 单位 |", "|---|---:|---:|---:|---:|---|"]
    for t in api.TARGETS:
        r = model[model.target == t].set_index("room")
        s = model_summary[model_summary.target == t].iloc[0]
        lines.append(f"| {LABELS[t][0]} | {fmt(r.loc['BM532','mae'],t)} | {fmt(r.loc['BM702','mae'],t)} | {fmt(r.loc['BMG26','mae'],t)} | {fmt(s.macro_MAE,t)} | {LABELS[t][2]} |")
    lines += ["", "平均值对三个房间等权；每个房间均为17,520行，因此本次MAE的房间平均恰好等于合并MAE。没有将5个不同量纲的输出合成一个准确率。", "",
              "## R²：逐房间解释", "", "| 输出 | BM532 | BM702 | BMG26 |", "|---|---:|---:|---:|"]
    for t in api.TARGETS:
        r = model[model.target == t].set_index("room")
        values = ["未定义（真实值恒定）" if pd.isna(r.loc[room,"r2"]) else f"{r.loc[room,'r2']:.3f}" for room in ["BM532", "BM702", "BMG26"]]
        lines.append("| "+LABELS[t][0]+" | "+" | ".join(values)+" |")
    lines += ["", "负R²表示平方误差比用该测试房间真实均值作恒定预测更大；该真实均值仅用于指标定义，不能作为部署时可获得的输入。R²不是预测正确率。BMG26的P10模拟标签全部为0，其MAE约0.040 lux、容限内比例100%，但R²未定义；不能用这个容易的全零目标证明P10泛化可靠。", "",
              "## RMSE与偏差", "", "| 输出 | 平均房间RMSE | 最差MAE（均为BM702） | 平均有符号偏差 | 单位 |", "|---|---:|---:|---:|---|"]
    for t in api.TARGETS:
        s = model_summary[model_summary.target == t].iloc[0]
        lines.append(f"| {LABELS[t][0]} | {fmt(s.macro_RMSE,t)} | {fmt(s.worst_MAE,t)} | {fmt(s.macro_bias,t)} | {LABELS[t][2]} |")
    lines += ["", "这里RMSE是三个房间RMSE的平均值，合并RMSE另见pooled_metrics.csv。偏差定义为预测−真实，平均接近0仍可能是正负误差抵消。", "",
              "## 三个房间的区别", "",
              "- BM532：五项R²约0.870–0.985，F300/FUDI MAE约2.38/4.68个百分点，PV MAE约0.00333 kWh/小时。平均照度仍有68.92 lux误差，不能仅凭高R²视为达成任意工程容限。",
              "- BM702：平均照度MAE236.97 lux，P10 MAE43.25 lux，FUDI MAE27.04个百分点；P10和FUDI平均分别低估43.19 lux和25.74个百分点。其模型坐标朝向为(-1,0)，在7个训练房间中没有出现。方向覆盖不足是合理的待检验解释，不能从这一房间直接认定为唯一原因或声称补一个特征就能解决。",
              "- BMG26：平均照度真实均值约40.64 lux，MAE27.32 lux、R²0.324；F300/FUDI R²约0.334/0.295，P10真实值全部为0。绝对数值较小部分来自其低照度水平。",
              "- BM532和BMG26的遮挡比分别低于/高于训练范围，但表现不同。范围提示不是置信概率，不能仅用是否超范围判断误差。", "",
              "## 与简单参照和开发结果的关系", "",
              "参照仅使用TRAIN中各角度的目标均值，在读取TEST数值目标前冻结。模型五项输出的平均MAE均低于此简单参照，但BM702平均照度MAE236.97 lux仍高于该参照185.97 lux；整体均值会掩盖局部失败。", "",
              "| 输出 | 模型测试平均MAE | TRAIN按角度均值参照MAE | 单位 |", "|---|---:|---:|---|"]
    for t in api.TARGETS:
        m = model_summary[model_summary.target == t].iloc[0]
        b = baseline_summary[baseline_summary.target == t].iloc[0]
        lines.append(f"| {LABELS[t][0]} | {fmt(m.macro_MAE,t)} | {fmt(b.macro_MAE,t)} | {LABELS[t][2]} |")
    lines += ["", "开发留一房间参考的平均MAE为82.43 lux、9.40 lux、7.12个百分点、11.04个百分点、0.01261 kWh/小时（与本模型固定参数对应）。测试房间与开发房间不同、训练规模也从每折6房间变为7房间，因此不能把测试MAE的升降归因于模型改善。P10这里使用固定depth12候选的9.40 lux开发分数；不同于此前按外层分别选择参数的9.75 lux参考。", "",
              "## 容限内比例（描述性）", "", "| 输出 | 预先固定的绝对容限 | 3房间平均容限内比例 | 最低房间比例 |", "|---|---|---:|---:|"]
    for t in api.TARGETS:
        s = model_summary[model_summary.target == t].iloc[0]
        tol = frozen["config"]["tolerances_descriptive"][t]
        lines.append(f"| {LABELS[t][0]} | ±{fmt(tol,t)} {LABELS[t][2]} | {100*s.macro_within_tolerance:.1f}% | {100*s.min_within_tolerance:.1f}% |")
    lines += ["", "容限沿用先前探索场景，尚未定义为项目工程验收标准；这些比例不应称为模型的统一准确率。", "",
              "## 验证与研究边界", "",
              f"已独立复算{verification['independently_recomputed_metric_rows']}行评分；重载模型复算262,800个值，最大差异{max(verification['max_prediction_reload_difference'].values()):.3g}。未改模型文件、数据文件、特征或裁剪规则；0次拟合、0次追加模拟。",
              "只有3个独立房间组，逐时记录彼此相关；没有据此生成窄置信区间、p值或普遍建筑性能结论。测试输入的覆盖情况此前已有审查，代表房间选择流程仍待核实；这是同建筑同天气模拟的目标留出验证，不是实测或其他天气年的验证。",
              "EPW、Hour时间区间、模型北向、百叶定义与几何/传感器来源仍待核对。该测试也没有评价角度推荐策略或证明动态选角优于固定0°。", "",
              "## 后续状态", "",
              "这次结果保留为固定0.1.0模型的首次留出成绩。三个房间没有进入训练，但其结果现在已经被观察。若后续针对这些误差修改模型、输入或方法，同一批房间不再是新的未见最终测试集，必须披露测试反馈的使用。",
              "本轮不据TEST替换模型或调参。建议优先核对跨朝向输入和既有房间/窗户几何；新方案应先在开发房间做分组对照。若希望对改进后模型再次给出独立结论，需要另行规划独立验证来源；当前仍不追加模拟。", "",
              "## 文件", "", "- summary_metrics.csv：房间平均和最差值；room_metrics.csv：逐房间全部指标。",
              "- room_angle_metrics.csv、room_hour_metrics.csv：四角度与逐小时明细；pooled_metrics.csv：明确标注的合并指标。",
              "- predictions_before_targets.parquet、prediction_freeze.json：读取数值标签前保存的预测及哈希。",
              "- scored_predictions.parquet：全部真实值、模型值、TRAIN均值参照及关联键。",
              "- pretest_freeze.json、run_manifest.json、verification.json：冻结、执行和复核记录。",
              "- test_room_feature_coverage.csv、feature_range_coverage.csv：输入覆盖，非置信区间。",
              "- interpretation_checks.json：11/11统计解释检查，含明确的限制项。", ""]
    (OUT/"summary.md").write_text("\n".join(lines), encoding="utf-8")
    checks = [
        ("Simpson's paradox / aggregation", "CAUTION", "Five macro MAEs beat TRAIN-angle mean, but BM702 mean illuminance is worse; all room results reported."),
        ("Ecological fallacy", "CAUTION", "Three room groups only; hourly rows not treated as independent rooms or whole-building proof."),
        ("Berkson / selection bias", "CAUTION", "K-means representative-room selection and single-building/weather sampling limit transportability."),
        ("Collider bias", "NOT_APPLICABLE_TO_CLAIM", "Predictive scoring only; no adjusted causal coefficients or causal claims."),
        ("Base rate neglect", "CAUTION", "Zero-label rates and true means retained; BMG26 P10 all zero explicitly discussed."),
        ("Regression to the mean", "CHECKED", "No model change or improvement attribution between different development/test room sets."),
        ("Survivorship bias", "CHECKED", "All 52560 scheduled test rows included; no outlier or zero removal."),
        ("Look-elsewhere effect", "CHECKED", "All five targets, three rooms and four angles reported; no best-test candidate selected."),
        ("Garden of forking paths", "NOTE", "Many prior development rounds acknowledged; exact frozen model/rules committed before test labels; no post-test parameter changes."),
        ("Correlation versus causation", "CAUTION", "Unseen orientation is a plausible explanation, not proven sole cause."),
        ("Reverse causality / leakage", "CHECKED", "All model inputs feature-only; predicted P10 uses predicted companion mean; numeric TEST labels loaded after predictions saved.")]
    dump(OUT/"interpretation_checks.json", {"coverage": "11/11", "checks": [{"type": a, "assessment": b, "basis": c} for a,b,c in checks], "no_p_values_or_confidence_interval_claims": True})
    outcome = {"evaluation_id": "holdout_trial_v0_1", "status": "completed_verified", "model_version": api.VERSION,
               "model_sha256": run["model_sha256"], "test_rooms": run["test_rooms"], "TEST_prediction_rows": 52560, "TEST_scored_rows": 52560,
               "estimator_fits": 0, "still_excluded_from_training": True, "untouched_for_future_result_informed_models": False,
               "report": "reports/holdout_trial_v0_1/summary.md", "verification": "reports/holdout_trial_v0_1/verification.json",
               "conclusion": "Uneven cross-room performance; BM702 worst for all outputs. No broad new-room reliability or engineering acceptance claim."}
    dump(ROOT/"docs/holdout_status.json", outcome)
    p = ROOT/"docs/project_status.json"
    state = read(p)
    state.update({"latest_completed_experiment": "holdout_trial_v0_1", "latest_analysis": outcome["report"], "report": outcome["report"],
                  "configuration": "configs/holdout_trial_v0_1.json", "verification": outcome["verification"], "latest_approval": "docs/approvals/holdout_trial_v0_1_approval.json",
                  "pending_proposal_status": "User-requested frozen-model holdout evaluation complete; no further training started.",
                  "TEST_predictions": 52560, "TEST_scores": 52560, "TEST_target_values": 262800,
                  "test_status": "observed_once_for_frozen_trial_0.1.0; remains_excluded_from_training",
                  "model_status": outcome["conclusion"], "authorization_status": "Latest user request authorized this evaluation; no TEST training/tuning or further simulations.",
                  "latest_completed_analysis": "frozen_model_holdout_evaluation", "holdout_evaluation": outcome,
                  "next_priority": "Interpret frozen holdout evidence; resolve existing geometry/weather provenance. Any result-informed model changes must disclose reuse of now-observed test rooms."})
    dump(p, state)
    p = ROOT/"docs/model_evidence_registry.json"
    registry = read(p)
    registry["status"] = "fixed_trial_evaluated_on_three_previously_reserved_rooms_uneven_performance"
    registry["holdout_evaluation"] = outcome | {"summary_metrics": model_summary.to_dict("records"), "prior_feature_only_audit_disclosed": True}
    registry["trial_release"].update({"TEST_evaluated": True, "holdout_report": outcome["report"]})
    dump(p, registry)
    p = ROOT/"README.md"
    text = p.read_text(encoding="utf-8")
    text = text.replace("三个测试房间继续封存；此版本用于研究联调，尚未独立测试。", "2026-11-08已按用户请求完成三个留出房间的首次固定模型测试：[测试结果](reports/holdout_trial_v0_1/summary.md)。跨房间表现不均，BM702为突出弱项；三个房间仍未进入训练，但测试结果已被观察。", 1)
    if "当前测试状态以[留出评估记录]" not in text:
        text = text.replace("各轮开发结果：", "当前测试状态以[留出评估记录](docs/holdout_status.json)为准；下文历史轮次的“封存/未测试”描述是各轮执行时状态。\n\n各轮开发结果：", 1)
    p.write_text(text, encoding="utf-8")
    updates = {
        "docs/experiment_index.md": "\n\n## 固定0.1模型首次留出测试\n\n2026-11-08按用户最新请求评估BM532、BM702、BMG26，共52,560行；0次拟合。MAE平均为111.07 lux、18.68 lux、6.53个百分点、12.03个百分点、0.00767 kWh/小时。五项最差房间均为BM702。详见[数值报告](../reports/holdout_trial_v0_1/summary.md)。三个房间仍不参加训练，但已不是未观察的测试集。\n",
        "docs/modeling_priorities.md": "\n\n## 后续授权更新：首次留出评估\n\n用户随后请求“是不是可以看一下在测试方向的性能了”，据此对固定0.1模型完成一次TEST评估，未训练或调参。之前的封存要求仅对此次评分更新；当前状态见[留出评估记录](holdout_status.json)。后续若受此结果启发修改方法，不能再次把同一批房间称为未见最终测试集。性能优先、无新增可视化及不追加模拟的约束保持。\n",
        "docs/trial_v0_1_usage.md": "\n\n## 后续评价记录（2026-11-08）\n\n用户已授权并完成固定0.1模型的首次三房间留出评分，见[数值结果](../reports/holdout_trial_v0_1/summary.md)。本文前述封存/未独立测试是接口创建时状态。原模型和接口源码保持不变，旧Usage_Status文字及封存ID拦截属于固定版本元数据；当前科学评价状态以[holdout_status.json](holdout_status.json)为准。测试由专用授权脚本完成，未通过修改Room_ID绕开标识，也未重训模型。\n",
    }
    for relative, addition in updates.items():
        p = ROOT/relative
        text = p.read_text(encoding="utf-8")
        heading = addition.strip().split("\n")[0]
        if heading not in text:
            p.write_text(text+addition, encoding="utf-8")
    # Release-specific inventory; historical trial and initial repo manifests remain immutable.
    files = [ROOT/"configs/holdout_trial_v0_1.json", ROOT/"docs/holdout_trial_v0_1_protocol.md", ROOT/"docs/approvals/holdout_trial_v0_1_approval.json",
             ROOT/"scripts/evaluate_holdout_trial_v0_1.py", ROOT/"scripts/verify_holdout_trial_v0_1.py", Path(__file__), ROOT/"docs/holdout_status.json",
             ROOT/"docs/project_status.json", ROOT/"docs/model_evidence_registry.json", ROOT/"README.md"]
    files += [ROOT/p for p in updates]
    files += [p for p in OUT.iterdir() if p.is_file() and p.name not in ["artifact_manifest.json", "repository_preflight.json"]]
    inventory, findings = [], []
    for p in sorted(set(files)):
        relative = p.relative_to(ROOT).as_posix()
        assert p.stat().st_size < 100*2**20
        inventory.append({"path": relative, "bytes": p.stat().st_size, "sha256": api.sha256(p)})
        if p.suffix.lower() in audit.TEXT:
            text = p.read_text(encoding="utf-8-sig", errors="replace")
            for name, pattern in audit.PATTERNS.items():
                for match in pattern.finditer(text):
                    findings.append({"path": relative, "type": name, "line": text.count("\n", 0, match.start())+1})
    assert not findings, findings
    dump(OUT/"repository_preflight.json", {"status": "passed", "files": len(inventory), "credential_findings": findings})
    dump(OUT/"artifact_manifest.json", {"evaluation_id": "holdout_trial_v0_1", "files": inventory,
         "note": "Shared project docs describe current state and may change later; historical model and evaluation freeze hashes remain fixed."})
    print(f"Numerical report and holdout status updated. {len(inventory)} files inventoried; no plots or model modifications.")


if __name__ == "__main__":
    main()
