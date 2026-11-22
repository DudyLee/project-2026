"""Deterministic command extraction and selection of existing result cards."""
from __future__ import annotations

import re
import unicodedata

from .metric_requests import LIMIT, METRICS, mentioned_metrics, ranking_request, scoped_metrics
from .schema import Intent, ResultRef

RULE_FIELDS = {
    "power": r"pv(?: energy)?|power|solar(?: energy)?|发电|光伏",
    "daylight": r"(?<!useful\s)daylight|(?<!有效)采光",
    "udi": METRICS["F_UDI_fraction"], "f300": METRICS["F_300_fraction"], "p10": METRICS["E_P10_lux"], "mean": METRICS["E_mean_lux"],
}
TOPICS = {
    "p10": METRICS["E_P10_lux"],
    "udi": METRICS["F_UDI_fraction"],
    "weather": r"weather|climate|solar position|天气|气象",
    "hours": r"available (?:hours|dates)|hour|time range|after|before|时间范围|几点|有效日期",
    "scoring": r"scor(?:e|ing)|formul[ae]|weights?|preference rule|评分|权重|计分",
    "rooms": r"rooms?|room ids?|房间|房号|楼层",
    "coverage": r"unsimulated|validat(?:ed|ion)|training|未模拟|验证",
    "metrics": r"illuminance|300\s*lux|mean daylight|average light|PV energy|照度|光伏发电量",
}


def matches(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def rule_weights(text: str) -> dict:
    rule = {}
    weights = LIMIT.sub(" ", text)
    for name, pattern in RULE_FIELDS.items():
        weight = re.search(rf"(?<![a-z])(?:{pattern})\s*(?:weight\s*)?(?:to|at|=|:|为|设为|占)?\s*(-?\d{{1,3}}(?:\.\d+)?)\s*(?:%|percent\b)", weights)
        weight = weight or re.search(rf"(?<![\d.\-])(-?\d{{1,3}}(?:\.\d+)?)\s*(?:%|percent\b)\s*(?:for\s+|的)?(?:{pattern})(?![a-z])", weights)
        if weight:
            rule[name] = float(weight[1])
    if "daylight" in rule:
        daylight = rule.pop("daylight")
        if "power" not in rule:
            rule["power"] = 100 - daylight
        elif rule["power"] + daylight != 100:
            rule["power"] = -1
    weight_removals = re.sub(r"(?:remove|drop|clear|删除|取消)\s+[^,，.;；]*?(?:cutoffs?|限值)", "", text)
    for name in ("udi", "f300", "p10", "mean"):
        if matches(rf"(?:(?:remove|exclude|omit|drop|without|去掉|移除|不要|排除)\s*(?:the\s+)?(?:{RULE_FIELDS[name]})|take\s+(?:{RULE_FIELDS[name]})\s+out)", weight_removals):
            rule[name] = 0
    for name in ("p10", "mean"):
        metric = RULE_FIELDS[name]
        target = re.search(
            rf"(?:{metric})(?:\s+(?:illuminance|照度))?\s*(?:的|with\s+)?\s*(?:target|full credit|目标|满分)\s*(?:to|at|of|=|:|为)?\s*(\d+(?:\.\d+)?)\s*(?:lux|lx|勒克斯)?",
            text,
        ) or re.search(rf"(?:target|目标)\s*(?:for\s+)?(?:{metric})\s*(?:to|at|of|=|:|为)?\s*(\d+(?:\.\d+)?)", text)
        if target:
            rule[name + "_target"] = float(target[1])
    return rule


CUTOFF_EDIT = r"\brequire(?:s|d|ment)?\b|only (?:consider|allow)|(?:add|change|set|remove|clear).*?cutoffs?|要求|仅考虑|只考虑|(?:添加|修改|设置|删除|清除).*?限值"
RULE_EDIT = r"(?:edit|customi[sz]e|change|set|adjust).*?(?:scoring|weights?|preference rule)|调整.*权重|自定义|权重改"


def rule_cutoffs(text: str, ranking: dict | None) -> dict:
    if matches(r"\b(?:clear|remove) (?:the |all )?cutoffs?\b|清除(?:所有)?限值", text):
        return {"clear_cutoffs": True}
    removal = re.search(r"(?:remove|drop|clear|删除|取消)\s+(.*?)(?:cutoff|限值)", text)
    if removal:
        metrics = [metric for metric in mentioned_metrics(removal[1]) if metric != "PV_AC_kWh"]
        return {"remove_cutoffs": metrics} if metrics else {}
    filters = ranking.get("filters", []) if ranking else []
    return {"cutoffs": filters} if filters and all(item["metric"] != "PV_AC_kWh" for item in filters) else {}


def local_command(message: str, intent: Intent, has_scenario: bool) -> dict:
    text = unicodedata.normalize("NFKC", message).casefold()
    angles = [int(value) for value in re.findall(r"(?<!\d)(-?\d{1,3})\s*(?:°|度|deg(?:ree)?s?\b)", text)]
    if len(angles) < 2 and (angles or matches(r"\bangles?\b|角度", text)):
        pair = re.search(r"(?<![\d:])(\d{1,3})\s*(?:°|度)?\s*(?:vs\.?|and|与|和|、|to|compared (?:to|with))\s*(\d{1,3})", text)
        if pair:
            angles = [int(value) for value in pair.groups()]
    metrics = mentioned_metrics(text)
    informational = matches(r"what (?:is|are|does|do)|how (?:does|do|is|are)|where (?:does|is|do)|define|meaning|"
                            r"can (?:i|we) (?:run|predict)|which rooms|list (?:the )?rooms|什么|哪里|来自|定义|能否|哪些房间", text)
    topic = next((name for name, pattern in TOPICS.items() if matches(pattern, text)), "general")
    if matches(r"(?:show|restore|display|plot) (?:all|every) (?:five |predicted )?(?:metrics?|lines?|series)|显示所有指标|显示全部指标", text):
        metrics = list(METRICS)
    table = matches(r"(?:show|open|expand|hide|close|collapse).*?(?:all values|table)|全部数值|(?:展开|关闭|隐藏)表格", text)
    hiding = matches(r"\bhide\b|\bcollapse\b|\bclose\b|隐藏|关闭", text)
    visibility = "only" if matches(r"\bonly\b|只显示|仅显示", text) else ("hide" if hiding else "show")
    command = {"angles": list(dict.fromkeys(angles))[:4]}

    if matches(r"\bexport\b|\bdownload\b|\bcsv\b|\bjson\b|导出|下载", text):
        return {**command, "action": "export", "format": "csv" if "csv" in text else "json" if "json" in text else None}
    comparing = matches(r"\bcompare\b|\bvs\.?\b|versus|difference|how much.*(?:lose|gain)|对比|比较|相比|差多少", text)
    explaining = matches(r"\bwhy\b|\bexplain\b|为什么|为何|解释|怎么推荐", text)
    ranking = ranking_request(text) if not explaining else None
    if not explaining and not comparing and not informational:
        rule = rule_weights(text)
        edits_cutoffs = matches(CUTOFF_EDIT, text)
        ranks_angles = matches(r"\b(?:rank|sort|which angle)\b|show.*?angles|排序|哪个角度", text)
        if matches(RULE_EDIT, text) or (not ranks_angles and (rule or edits_cutoffs)):
            cutoff_patch = rule_cutoffs(text, ranking)
            if (edits_cutoffs or ranking) and not cutoff_patch:
                return {**command, "action": "rule", "rule_issue": "cutoff"}
            return {**command, "action": "rule", "rule": {**rule, **cutoff_patch}}
    if ranking is not None and (not comparing or ranking.get("filters")):
        return {**command, "action": "rank", **ranking}
    if informational and topic != "general" and not (angles and topic in {"p10", "udi", "metrics"}) and not (comparing and has_scenario):
        return {**command, "action": "help", "topic": topic}
    if comparing:
        if ranking and ranking.get("ranking_issue") == "filter":
            return {**command, "action": "rank", **ranking}
        return {**command, "action": "compare", "metrics": scoped_metrics(text)}
    if table or (metrics and matches(r"show|hide|plot|display|restore|显示|隐藏|绘制", text)):
        return {**command, "action": "chart", "metrics": metrics, "visibility": visibility,
                "table_open": not hiding if table else None}
    if angles and matches(r"inspect|highlight|pin|select|focus|查看|高亮|选择", text):
        return {**command, "action": "chart"}
    if (angles and (informational or matches(r"what.s|how much|多少", text))) or matches(
        r"\bwhy\b|\bexplain\b|which angle|为什么|为何|解释|怎么推荐|哪个角度", text
    ):
        return {**command, "action": "explain", "metrics": scoped_metrics(text)}
    if intent.preference and not has_scenario and not matches(r"\bpredict\b|预测", text):
        return {**command, "action": "rule", "preference": intent.preference}
    if has_scenario and matches(r"^(?:what|who|where|can you|how|几点|如何|谁|能不能)\b", text) and not matches(
        r"louv(?:er|re)|angle|daylight|pv|power|room|\bpredict\b|百叶|角度|预测|房间|光伏", text
    ):
        return command
    if has_scenario or matches(r"\bpredict\b|\bfind (?:a |the )?room|找房间|\bforecast\b|\brun (?:a )?(?:prediction|model)|what if|预测|试算", text):
        return {**command, "action": "predict"}
    return command


def referenced_results(message: str, results: list[ResultRef], active: str | None,
                       requested_id: str | None, rooms: list[str], days: list[int], hours: list[int]) -> list[ResultRef]:
    """Return an ordered set of matches, or an empty list for an unmatched reference."""
    by_id = {entry.id: entry for entry in results}
    if matches(r"\bfirst (?:result|prediction|card)\b|第一个预测|第一张结果", message):
        return results[:1]
    if matches(r"\b(?:previous|earlier) (?:one|result|prediction|card)\b|上一个结果|上一张结果", message):
        index = next((i for i, entry in enumerate(results) if entry.id == active), len(results))
        return results[index - 1:index] if index else []
    if rooms or days or hours:
        return [item for item in results if (not rooms or item.room_id in rooms)
                and (not days or item.day in days) and (not hours or item.hour in hours)]
    if requested_id in by_id:
        return [by_id[requested_id]]
    return [by_id[active]] if active in by_id else results[-1:]


PROJECT_HELP = {
    "general": (
        "Ask me to predict daylight and PV energy for a room and hour, explain a result, compare or rank angles, change scoring weights, adjust a chart, or export a result.",
        "你可以让我预测某个房间和时段的采光与发电量、解释结果、比较角度或场景、筛选和排序角度、调整评分权重、修改图表或导出结果。",
    ),
    "p10": (
        "P10 is the 10th percentile of illuminance across a room’s sampling points. If P10 is 40 lux, about 10% of points are at or below 40 lux. It helps you understand how dark the less well-lit parts of the room are.",
        "P10 是房间采样点照度的第 10 百分位数。例如 P10 为 40 lux，约 10% 的点照度为 40 lux 或更低。它反映房间较暗区域的采光情况。",
    ),
    "udi": (
        "Useful daylight is the percentage of sampling points in the model’s useful illuminance range for the selected hour. The 300 lux coverage metric counts points at or above 300 lux.",
        "有效照度比例是该时段照度落在模型有效范围内的采样点比例。300 lux 达标比例统计照度达到或超过 300 lux 的采样点。",
    ),
    "weather": (
        "The app looks up the original simulation weather by day and hour. The data include solar irradiance and position.",
        "应用按日期和小时查找原模拟天气，包括太阳辐照量和太阳位置。",
    ),
    "hours": (
        "Pick any date in a 365-day reference year and a whole hour from 07:00 through 18:00. Weather comes from the original simulation.",
        "支持一年中的第 1 至 365 天，以及 07:00 至 18:00 的整点。天气来自原模拟。",
    ),
    "metrics": (
        "Mean illuminance averages the room’s sampling points. The 300 lux coverage metric is the share at or above 300 lux. PV energy is the predicted AC electricity generated during the selected hour.",
        "平均照度是房间各采样点照度的平均值。300 lux 达标比例是照度达到或超过 300 lux 的采样点占比。光伏发电量是该小时的预测交流电量。",
    ),
    "scoring": (
        "The default daylight score weights useful daylight at 60% and 300 lux coverage at 40%. Use Edit preference to adjust weights, add lux targets, or set eligibility cutoffs. Preference scores compare eligible angles.",
        "默认采光分采用 60% 有效照度比例和 40% 300 lux 达标比例。可用「编辑偏好」调整权重，设置照度目标或限值。偏好分用于比较符合限值的角度。",
    ),
    "rooms": (
        "The catalog contains 476 Tower A rooms. Search by room ID or floor. If you give a floor, I can offer a shortlist and a room picker.",
        "目录包含 Tower A 的 476 个房间。可按房号或楼层搜索。提供楼层后，我可以给出候选房间和房间选择器。",
    ),
    "coverage": (
        "All catalog rooms can be queried. Rooms marked Unsimulated have no room-specific training or test simulation in the trial dataset. Treat their predictions as estimates.",
        "目录中的房间都可查询。标有「未模拟房间」表示试验数据集中没有该房间的训练或测试仿真记录，预测值需作为估计理解。",
    ),
}
