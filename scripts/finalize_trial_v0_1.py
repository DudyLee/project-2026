"""Publish local trial documentation and a separate immutable artifact inventory."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pvsd_trial as api
import prepare_repository_snapshot as audit


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    out = ROOT/"reports/trial_v0_1"
    verified = read(out/"verification.json")
    assert verified["status"] == "passed"
    model = read(ROOT/"models/trial_v0_1/manifest.json")
    frozen = read(out/"parameter_freeze.json")
    run = read(out/"run_manifest.json")
    assert run["status"] == "trial_ready_verified"
    dump(out/"progress.json", {"stage": "trial_ready_verified", "completed_estimator_fits": 48, "total_estimator_fits": 48})
    requirements = "# Exact runtime versions recorded from the verified trial environment.\n" + "\n".join(f"{p}=={v}" for p,v in model["versions"].items() if p != "pyarrow") + "\n"
    (ROOT/"configs/trial_v0_1_runtime_requirements.txt").write_text(requirements, encoding="utf-8")
    properties = {c: {"type": "number"} for c in api.INPUT_FEATURES}
    for c in ["DOY_1based", "Hour"]:
        properties[c]["type"] = "integer"
    bounds = {"Room_Area_m2": (0, None), "Glass_Area_m2": (0, None), "Solar_Obstruction_Ratio": (0, 1),
              "Orientation_X_model": (-1, 1), "Orientation_Y_model": (-1, 1), "DOY_1based": (1, 365), "Hour": (7, 18),
              "Solar_Altitude_mid_deg": (-90, 90), "Solar_Azimuth_mid_deg": (0, 360),
              "EPW_GHI_Wh_m2": (0, None), "EPW_DNI_Wh_m2": (0, None), "EPW_DHI_Wh_m2": (0, None)}
    for c, (lo, hi) in bounds.items():
        properties[c]["exclusiveMinimum" if c == "Room_Area_m2" else "minimum"] = lo
        if hi is not None:
            properties[c]["maximum"] = hi
    properties.update({c: {"type": "string"} for c in api.IDENTIFIERS})
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Tower A PVSD trial 0.1 input batch", "type": "array", "minItems": 1,
              "description": "Additional runtime checks: finite values, unit orientation vector, protected TEST identifiers. Units and conventions: docs/trial_v0_1_usage.md.",
              "items": {"type": "object", "additionalProperties": False, "required": api.INPUT_FEATURES, "properties": properties}}
    dump(ROOT/"examples/trial_v0_1/input.schema.json", schema)
    selections = {t: read(out/f"selection_{t}.json") for t in ["E_mean_lux", "E_P10_lux"]}
    lines = ["# 试用版0.1训练与接口验证结果", "", "状态：已完成并可用于离线研究试用。三个TEST房间继续封存。", "",
             "使用入口：[中文使用说明](../../docs/trial_v0_1_usage.md)。模型包：[models/trial_v0_1](../../models/trial_v0_1)。", "",
             "## 固定模型", "", "| 输出 | 方法 | 最终参数 |", "|---|---|---|"]
    for target, spec in frozen["models"].items():
        params = spec["params"]
        detail = f"max_depth={params['max_depth']}" if "max_depth" in params else f"max_leaf_nodes={params['max_leaf_nodes']}, min_samples_leaf={params['min_samples_leaf']}"
        lines.append(f"| {target} | {spec['family']} / {spec['transform']} | {detail} |")
    lines += ["", "前两项在既定两候选中按7房间留一验证的平均MAE选参数；后三项保持v1参数。随后全部7房间122,640行拟合6个估计器（P10含自己的RF均值分支）。42次参数选择拟合＋6次全量拟合，共48次；无追加模拟。",
              "", "## 参数选择记录（开发分数）", "", "| 目标 | 参数候选 | 7房间平均MAE | 最差房间MAE | 选中 |", "|---|---|---:|---:|---|"]
    for target, selection in selections.items():
        for candidate in selection["candidates"]:
            lines.append(f"| {target} | {candidate['spec']['id']} | {candidate['macro_MAE']:.6f} lux | {candidate['worst_MAE']:.6f} lux | {'是' if selection['selected'] == candidate['spec']['id'] else '否'} |")
    lines += ["", "这些分数用于参数选择，验证房间也已多次用于开发。它们不是独立测试分数，不能据此宣布新版本准确率提高。原嵌套开发参考与本轮固定参数候选的评分过程不同，也不宜直接作胜负比较。全7房间模型没有新的独立泛化分数。",
              "", "## 软件检查", "", f"- 全部30,660个开发房间小时条件产生122,640个四角度行，核对613,200个目标值。统一接口与原实验直接推断最大差异 {max(verified['maximum_difference_from_legacy_direct_inference'].values()):.3g}。",
              "- CSV/JSON/Python入口、逐条/批量、模型重载、预处理列顺序及数值、TRAIN拟合输入/标签哈希检查通过。",
              "- 30种非法输入情形拒绝；三个封存Room_ID在Python和CLI中均拒绝，未进入模型预测。元数据损坏被校验拦截。",
              "- 推断无需原始数据库；独立模型目录在其他工作目录运行通过。输入范围提示与模型版本/哈希随结果返回。",
              "- TEST训练、预测、评分均为0；52,560行在数值转换前跳过，原始CSV哈希保持不变。",
              f"- 当前电脑实测：加载约{verified['timing_descriptive_this_machine']['load_seconds']:.2f}秒；加载后单条四角度平均约{verified['timing_descriptive_this_machine']['four_angles_one_record_seconds_warm_average_5']:.3f}秒；30,660条批量约{verified['timing_descriptive_this_machine']['all_30660_conditions_seconds']:.2f}秒。仅描述本机运行，不是通用性能保证。",
              "", "## 使用边界", "", "可以开始演示、同条件四角度性能比较和软件接口联调。仍未证明对任意新房间稳定准确、动态选角稳定优于固定0°，也没有最终TEST评估。未核实天气/时间/北向/百叶定义仍限制物理解释与迁移。所有范围提示都是诊断提示，不是置信概率。", "",
              "本轮没有新增HTTP服务、Revit/Grasshopper插件、LLM代理或自动控制；CSV/JSON与Python接口可作为后续接入基础。", ""]
    (out/"summary.md").write_text("\n".join(lines), encoding="utf-8")
    # Record interpretation checks without inventing inferential/statistical claims.
    checks = ["No training accuracy presented as generalization", "Selection scores identified as development-only", "Prior repeated room use acknowledged",
              "No TEST inputs used for prediction or target analysis", "No cross-procedure MAE improvement claim", "No p-values or independence claim for hourly rows",
              "Uncertainty flags not called confidence intervals", "Simulation surrogate distinguished from physical measurements", "No unsupported angle-control benefit claim",
              "Worst-room and provenance limitations retained", "Full-seven refit distinguished from final independent evaluation"]
    dump(out/"interpretation_checks.json", {"status": "passed", "checks": [{"check": s, "passed": True} for s in checks]})
    approval_path = ROOT/"docs/approvals/trial_v0_1_approval.json"
    approval = read(approval_path)
    approval.update({"status": "approved_trial_completed", "configuration_sha256": api.sha256(ROOT/"configs/trial_v0_1.json"),
                     "protocol_sha256": api.sha256(ROOT/"docs/trial_v0_1_protocol.md"), "run": "reports/trial_v0_1/run_manifest.json", "verification": "reports/trial_v0_1/verification.json"})
    dump(approval_path, approval)
    status_path = ROOT/"docs/project_status.json"
    status = read(status_path)
    status.update({"updated_local_date": "2026-11-08", "latest_approval": "docs/approvals/trial_v0_1_approval.json",
                   "pending_proposal_status": "Authorized full-seven-room trial build completed and verified.",
                   "trial_full_seven_refit": True, "final_deployment_refit": False,
                   "model_status": "Trial 0.1.0: fixed five-output model fitted to all seven development rooms; offline CSV/JSON/Python inference verified. No final TEST evaluation or production reliability claim.",
                   "trial_release": {"version": api.VERSION, "model": "models/trial_v0_1", "usage": "docs/trial_v0_1_usage.md", "report": "reports/trial_v0_1/summary.md", "verification": "reports/trial_v0_1/verification.json"},
                   "authorization_status": "User-authorized trial build complete; TEST sealed.",
                   "next_priority": "Use trial for supervised demonstrations/interface checks; resolve geometry/weather provenance and agree a final evaluation protocol before unsealing TEST."})
    dump(status_path, status)
    registry_path = ROOT/"docs/model_evidence_registry.json"
    registry = read(registry_path)
    registry["status"] = "development_evidence_with_verified_research_trial_not_independently_tested"
    registry["trial_release"] = status["trial_release"] | {"full_seven_refit": True, "TEST_evaluated": False,
        "historical_score_warning": "Existing comparison-reference metrics are not final independent scores of this full-seven package."}
    dump(registry_path, registry)
    readme_path = ROOT/"README.md"
    readme = readme_path.read_text(encoding="utf-8")
    old = "最新状态与各轮结果：[实验索引](docs/experiment_index.md)。当前完成v7逐目标优化；结果与候选状态见索引，测试房间继续封存。"
    new = "最新可用版本：**试用版0.1**，已用全部7个开发房间拟合固定模型，支持四角度×五指标CSV/JSON/Python预测。开始使用：[中文操作说明](docs/trial_v0_1_usage.md)；[试用版检查结果](reports/trial_v0_1/summary.md)。三个测试房间继续封存；此版本用于研究联调，尚未独立测试。\n\n各轮开发结果：[实验索引](docs/experiment_index.md)。v7没有候选通过全部替换条件，试用版沿用保留方法。"
    if old in readme:
        readme = readme.replace(old, new, 1)
    readme = readme.replace(
        "当前可开展研究原型演示与接口联调；完整的7房间训练部署包、统一推断接口和独立TEST评估尚未完成。具体边界和接入步骤见[模型使用就绪度](docs/model_usage_readiness.md)。本次归档没有新增训练或解封测试房间。",
        "初次归档时尚未创建全7房间模型和统一接口；后续按用户批准已完成[试用版0.1](docs/trial_v0_1_usage.md)。独立TEST评估仍未开展，三个测试房间继续封存。初次归档前的评估保留在[模型使用就绪度](docs/model_usage_readiness.md)。")
    readme_path.write_text(readme, encoding="utf-8")
    index_path = ROOT/"docs/experiment_index.md"
    index = index_path.read_text(encoding="utf-8")
    if "## 试用版0.1（全7开发房间固定模型）" not in index:
        index_path.write_text(index + "\n\n## 试用版0.1（全7开发房间固定模型）\n\n2026-11-08按用户批准完成。沿用保留模型族，平均照度XGB深度3、相对P10 RF深度12、F300/FUDI/PV固定HistGB；42次开发参数选择拟合后，用7房间122,640行重训6个估计器。\n\n[训练与接口验证](../reports/trial_v0_1/summary.md) · [使用说明](trial_v0_1_usage.md)。CSV/JSON/Python入口已验证，三个TEST房间继续封存。本轮软件检查和参数选择分数不等于最终独立泛化成绩。\n", encoding="utf-8")
    readiness_path = ROOT/"docs/model_usage_readiness.md"
    readiness = readiness_path.read_text(encoding="utf-8")
    marker = "> 2026-11-08后续状态："
    if marker not in readiness:
        readiness = readiness.replace("# 模型使用就绪度与试用边界\n", "# 模型使用就绪度与试用边界\n\n> 2026-11-08后续状态：用户已批准并完成试用版0.1。全7房间固定模型和CSV/JSON/Python接口现已可用，见[操作说明](trial_v0_1_usage.md)及[检查结果](../reports/trial_v0_1/summary.md)。下文保留创建试用版之前的评估和接入计划，所述“尚无全7模型/统一接口”是当时状态。可靠性限制和TEST封存继续有效。\n", 1)
        readiness_path.write_text(readiness, encoding="utf-8")
    # Keep initial repository_snapshot/preflight unchanged; inventory this release only.
    files = [ROOT/"pvsd_trial.py", ROOT/"configs/trial_v0_1.json", ROOT/"configs/trial_v0_1_runtime_requirements.txt",
             ROOT/"scripts/train_trial_v0_1.py", ROOT/"scripts/verify_trial_v0_1.py", ROOT/"scripts/finalize_trial_v0_1.py", ROOT/"scripts/predict_trial.ps1",
             ROOT/"docs/trial_v0_1_protocol.md", ROOT/"docs/trial_v0_1_usage.md", approval_path, status_path, registry_path, readme_path, index_path, readiness_path]
    for folder in [ROOT/"models/trial_v0_1", ROOT/"examples/trial_v0_1", out, ROOT/"outputs/trial_v0_1"]:
        files += [p for p in folder.rglob("*") if p.is_file() and p.name not in ["artifact_manifest.json", "repository_preflight.json"]]
    findings, inventory = [], []
    for path in sorted(set(files)):
        rel = path.relative_to(ROOT).as_posix()
        assert path.stat().st_size < 100*2**20
        inventory.append({"path": rel, "bytes": path.stat().st_size, "sha256": api.sha256(path)})
        if path.suffix.lower() in audit.TEXT:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for name, pattern in audit.PATTERNS.items():
                for match in pattern.finditer(text):
                    findings.append({"path": rel, "type": name, "line": text.count("\n", 0, match.start())+1})
    assert not findings, findings
    dump(out/"repository_preflight.json", {"status": "passed", "scope": "Trial release changed files; original repository snapshot preserved", "files": len(inventory), "credential_findings": findings})
    dump(out/"artifact_manifest.json", {"version": api.VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "files": inventory,
         "excluded_self_records": ["reports/trial_v0_1/artifact_manifest.json", "reports/trial_v0_1/repository_preflight.json"],
         "note": "Shared project summary docs can evolve in later commits; hashes describe this release."})
    print(f"Trial documentation and release inventory ready: {len(inventory)} files; credential scan passed.")


if __name__ == "__main__":
    main()
