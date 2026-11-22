"""Convert language and catalog matches into field states and selectable controls."""
from __future__ import annotations

from datetime import date, timedelta
import re
import unicodedata

from .agents import Agents
from .catalog import Catalog
from .language import language_code, split_language_request
from .routing import PROJECT_HELP, local_command, matches, referenced_results
from .schema import Draft, Intent, ResultRef

DIGITS = "零一二三四五六七八九"
NUMERAL = r"[0-9]{1,2}|[零一二两三四五六七八九十]{1,3}"
ROOM_CODE = r"(?<![A-Za-z0-9])[A-Za-z]{1,5}[0-9]{2,5}[A-Za-z]?(?![A-Za-z0-9])"
HOUR_PART = r"凌晨|早上|上午|中午|下午|傍晚|晚上|早晨"
MONTHS = {name: index for index, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), 1)}
MONTH_NAME = "|".join(f"{name[:3]}(?:{name[3:]})?" if len(name) > 3 else name for name in MONTHS)
MONTH_NUMBER = {name[:3]: index for name, index in MONTHS.items()}
PREFS = {
    "power": ("发电优先", "发电表现", "发电量最大", "发电最多", "power first", "power priority"),
    "daylight": ("采光优先", "照度优先", "采光最好", "daylight priority", "daylight first"),
    "balanced": ("综合平衡", "兼顾", "平衡", "balanced", "balance"),
}


def number(text: str) -> int:
    text = text.replace("两", "二")
    if text.isdecimal():
        return int(text)
    if "十" in text:
        before, after = text.split("十", 1)
        return 10 * (DIGITS.index(before) if before else 1) + (DIGITS.index(after) if after else 0)
    return int("".join(str(DIGITS.index(digit)) for digit in text))


def doy(month: int, day: int) -> int:
    try:
        return date(2001, month, day).timetuple().tm_yday
    except ValueError:
        return 0


def local_intent(message: str, today: date) -> tuple[Intent, list[str]]:
    text = unicodedata.normalize("NFKC", message).strip()
    room_codes = list(dict.fromkeys(code.upper() for code in re.findall(ROOM_CODE, text) if code.upper() not in {"P10", "F300"}))
    floor_pattern = rf"({NUMERAL})\s*(?:楼|层|[Ff](?![A-Za-z0-9]))"
    floor_match = re.search(floor_pattern, text)
    if not floor_match:
        floor_match = re.search(r"(?i)\b(?:floor|level)\s+(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:floor|level)\b", text)
    try:
        floor = (number(next(group for group in floor_match.groups() if group is not None))
                 if floor_match else (0 if re.search(r"首层|地面层|ground floor", text, re.IGNORECASE) else None))
    except ValueError:
        floor = None

    date_patterns = [
        (r"(?<!\d)(?:\d{4}[-/])?(\d{1,2})[-/](\d{1,2})(?!\d)", lambda m: doy(int(m[1]), int(m[2]))),
        (rf"({NUMERAL})\s*月\s*({NUMERAL})\s*[日号]?", lambda m: doy(number(m[1]), number(m[2]))),
        (r"(?:日序\s*|第\s*|(?i:DOY|day)\s*)(\d{1,3})\s*(?:天)?", lambda m: int(m[1])),
        (rf"(?i)\b({MONTH_NAME})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?!\d)",
         lambda m: doy(MONTH_NUMBER[m[1][:3].lower()], int(m[2]))),
        (rf"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_NAME})\b",
         lambda m: doy(MONTH_NUMBER[m[2][:3].lower()], int(m[1]))),
        (r"今天|明天|后天|昨天|(?i:\b(?:today|tomorrow|yesterday)\b)",
         lambda m: relative_day(today, m[0].lower())),
    ]
    dates: list[int] = []
    for pattern, convert in date_patterns:
        for match in re.finditer(pattern, text):
            try:
                dates.append(convert(match))
            except ValueError:
                dates.append(0)

    hours: list[int] = []
    fractional = False
    hour_pattern = (
        rf"(?:(?P<part>{HOUR_PART})\s*)?"
        rf"(?P<hour>{NUMERAL})\s*"
        r"(?P<suffix>:\d{2}|时(?:[0-9零一二三四五六七八九十]+分)?|点(?:钟|半|[0-9零一二三四五六七八九十]+分)?|(?i:[ap]\.?m\.?))"
        r"(?:\s*(?P<meridiem>(?i:[ap]\.?m\.?)))?"
    )
    for match in re.finditer(hour_pattern, text):
        try:
            hour = number(match["hour"])
        except ValueError:
            continue
        suffix = match["suffix"].lower()
        if suffix == "点半" or "分" in suffix or (suffix.startswith(":") and suffix != ":00"):
            fractional = True
            continue
        part = match["part"]
        meridiem = (match["meridiem"] or suffix).lower()
        if part in {"下午", "傍晚", "晚上", "中午"} or meridiem.startswith("p"):
            hour += 12 if 1 <= hour <= 11 else 0
        if part == "凌晨" or meridiem.startswith("a"):
            hour = 0 if hour == 12 else hour
        hours.append(hour)
    bare_hour = re.fullmatch(rf"\s*({NUMERAL})\s*", text)
    before_hour = re.search(r"(?i)(?:\bat\s+|小时\s*)(\d{1,2})(?![\d.]|\s*(?:°|deg(?:rees?)?\b|度|lux\b|lx\b|wh\b|kwh\b|%|勒克斯|瓦时))", text)
    if not hours and before_hour and not fractional:
        hours.append(int(before_hour[1]))
    if bare_hour and not fractional:
        try:
            hours.append(number(bare_hour[1]))
        except ValueError:
            pass
    if not hours and re.search(r"(?i)\bnoon\b", text):
        hours.append(12)
    needs_hour = fractional or (not hours and bool(re.search(
        HOUR_PART + r"|\b(?:morning|afternoon|evening|night)\b", text, re.IGNORECASE)))
    lowered = text.casefold()
    preference = next((key for key, terms in PREFS.items() if any(term in lowered for term in terms)), None)
    explain = bool(re.search(r"为什么|为何|怎么推荐|解释|差多少|哪个角度|what.+(difference|recommend)|why|compare|explain", lowered))
    vague_day = not dates and bool(re.search(r"周末|周[一二三四五六日天]|本周|下周|月初|月中|月末|月底|上旬|中旬|下旬|\b(?:weekend|next week|this week)\b", text, re.IGNORECASE))
    return Intent(floor=floor, days=list(dict.fromkeys(dates))[:5], needs_day=vague_day,
                  hours=list(dict.fromkeys(hours))[:5], needs_hour=needs_hour,
                  preference=preference, explain=explain), room_codes


def relative_day(today: date, word: str) -> int:
    actual = today + timedelta(days={"今天": 0, "明天": 1, "后天": 2, "昨天": -1, "today": 0, "tomorrow": 1, "yesterday": -1}[word])
    return doy(actual.month, actual.day)


def value_issue(field: str, values: list[int], was_vague: bool = False) -> dict | None:
    low, high = (1, 365) if field == "day" else (7, 18)
    valid = list(dict.fromkeys(value for value in values if low <= value <= high))
    if len(values) > 1:
        return {"field": field, "reason": "multiple", "options": valid}
    if was_vague:
        return {"field": field, "reason": "ambiguous", "options": valid}
    if values and not valid:
        return {"field": field, "reason": "out_of_range", "options": []}
    if not values:
        return {"field": field, "reason": "missing", "options": []}
    return None


async def clarify(message: str, draft: Draft, catalog: Catalog, agents: Agents, today: date,
                  language: str = "en", results: list[ResultRef] | None = None, active: str | None = None) -> dict:
    requested, message = split_language_request(message)
    reply_language = requested or language_code(language) or "en"
    if not message:
        return await response(draft, [], "language", "local", reply_language, requested, agents)
    local, codes = local_intent(message, today)
    # An isolated number is a day only when replying to a missing date and an already set hour.
    if re.fullmatch(NUMERAL, message.strip()) and local.hours and draft.day is None and draft.hour is not None:
        local.days, local.hours = local.hours, []
    results = results or []
    has_local_scenario = bool(codes or local.floor is not None or local.days or local.hours
                              or local.needs_day or local.needs_hour)
    command = local_command(message, local, has_local_scenario)
    remote = await agents.interpret(message, draft, today.isoformat(), results, active)
    if requested is None and remote:
        requested = language_code(remote.reply_language)
        if requested:
            reply_language = requested
            if remote.language_only:
                return await response(draft, [], "language", "agent", reply_language, requested, agents)
    floor = local.floor if local.floor is not None else (remote.floor if remote else None)
    days = local.days or (remote.days if remote and not local.needs_day else [])
    hours = local.hours or (remote.hours if remote and not local.needs_hour else [])
    preference = local.preference or (remote.preference if remote else None)
    vague_day = local.needs_day or (not days and bool(remote and remote.needs_day))
    vague_hour = local.needs_hour or (not hours and bool(remote and remote.needs_hour))
    room_query = (remote.room_query if remote and remote.room_query
                  and remote.room_query.casefold() in message.casefold() else None)
    area_words = r"(?:最大|较大|大一点|largest|biggest)|(?:最小|较小|小一点|smallest)"
    if not room_query and not codes and re.search(r"会议室|办公室|朝[东西南北]|" + area_words, message, re.IGNORECASE):
        room_query = message
    requests_room = bool(codes or room_query or floor is not None)
    has_scenario = requests_room or bool(days or hours or vague_day or vague_hour)
    action = command.get("action")
    if action is None or (action == "predict" and remote and remote.action in {"explain", "compare", "rank"}):
        action = (remote.action or ("explain" if remote.explain else None)) if remote else action
    action = action or ("predict" if requests_room else "help")
    source = "agent" if remote is not None else "local"
    remote_matches = remote is not None and remote.action == action
    angles = command.get("angles") or (remote.angles if remote_matches else [])
    if action == "help":
        topic = command.get("topic") or (remote.topic if remote_matches else None) or "general"
        sentence = PROJECT_HELP[topic][int(reply_language == "zh")]
        return await response(draft, [], action, source, reply_language, requested, agents, sentence=sentence)

    plan = {"angles": [angle for angle in dict.fromkeys(angles) if angle in {0, 30, 60, 90}],
            "unsupported_angles": [angle for angle in angles if angle not in {0, 30, 60, 90}]}
    if action in {"chart", "compare", "explain"}:
        plan["metrics"] = command.get("metrics") or (remote.metrics if remote_matches else [])
    if action not in {"predict", "compare"}:
        refs = referenced_results(message, results, active, remote.result_id if remote_matches else None,
                                  codes, days, hours)
        plan.update(result_id=refs[0].id if len(refs) == 1 else None,
                    result_choices=[ref.id for ref in refs] if len(refs) > 1 else [],
                    reference_missed=bool(results and not refs))
        if action == "rule":
            remote_rule = remote.rule.model_dump(exclude_none=True) if remote_matches and remote.rule else {}
            patch = {**remote_rule, **command.get("rule", {})}
            issue = command.get("rule_issue") or (remote.rule_issue if remote_matches and command.get("action") != "rule" else None)
            explicit_editor = command.get("action") == "rule" and "rule" in command
            plan.update(rule=patch, preference=None if patch or explicit_editor else preference, rule_issue=issue)
        if action == "chart":
            plan.update(visibility=command.get("visibility") or (remote.visibility if remote_matches else None),
                        table_open=command.get("table_open") if command.get("table_open") is not None
                        else (remote.table_open if remote_matches else None))
        if action == "rank":
            if command.get("action") == "rank":
                plan.update(filters=command.get("filters", []), sort=command.get("sort"),
                            ranking_issue=command.get("ranking_issue"))
            elif remote_matches:
                plan.update(filters=[item.model_dump() for item in remote.filters],
                            sort=remote.sort.model_dump() if remote.sort else None, ranking_issue=remote.ranking_issue)
            if not any(plan.get(key) for key in ("filters", "sort", "ranking_issue")):
                plan["ranking_issue"] = "sort"
        if action == "export":
            plan["format"] = command.get("format") or (remote.format if remote_matches else None)
        return await response(draft, [], action, source, reply_language, requested, agents, **plan)

    if action == "compare":
        by_id = {ref.id: ref for ref in results}
        base = by_id.get(remote.result_id) if remote_matches else None
        base = base or by_id.get(active) or (results[-1] if results else None)
        second = by_id.get(remote.against_result_id) if remote_matches else None
        latest_pair = matches(r"(?:last|latest|most recent) two|最近两(?:次|条|张|个)", message)
        if latest_pair and results:
            base = results[-1]
        previous = latest_pair or matches(r"previous|earlier|prior|上一个|上一张|之前的", message)
        first = matches(r"first (?:result|one|prediction|card)|第一个|第一张", message)
        if previous and base:
            index = results.index(base)
            second = results[index - 1] if index else None
        elif first and results:
            second = results[0]
        if second == base:
            second = None
        # Use the active room if it was named, or require a unique first-room match.
        if len(codes) == 2 and not (previous or first) and not (base and base.room_id in codes):
            room_refs = referenced_results(message, results, active, None, [codes[0]], days, hours)
            if len(room_refs) == 1:
                base = room_refs[0]
        if len(codes) == 2 and base and base.room_id in codes:
            other = [ref for ref in results if ref.room_id in codes and ref.room_id != base.room_id
                     and (not days or ref.day in days) and (not hours or ref.hour in hours)]
            if len(other) > 1:
                return await response(draft, [], action, source, reply_language, requested, agents,
                                      result_id=base.id, against_choices=[ref.id for ref in other], **plan)
            if other:
                second = other[0]
        plan.update(result_id=base.id if base else None, against_result_id=second.id if second else None)
        if base and has_scenario and not second and not (previous or first):
            alternatives = [ref for ref in referenced_results(message, results, active, None, codes, days, hours)
                            if ref != base] if (codes or days or hours) and not (vague_day or vague_hour) else []
            if len(alternatives) > 1:
                return await response(draft, [], action, source, reply_language, requested, agents,
                                      against_choices=[ref.id for ref in alternatives], **plan)
            if len(alternatives) == 1 and len(codes) <= 1 and len(days) <= 1 and len(hours) <= 1:
                plan["against_result_id"] = alternatives[0].id
                return await response(draft, [], action, source, reply_language, requested, agents, **plan)
            plan["compare_scenario"] = True
            draft = Draft(room_id=base.room_id, day=base.day, hour=base.hour,
                          preference=draft.preference if base.id == active else None)
        else:
            plan.update(compare_cards=bool(second or previous or first),
                        compare_all=matches(r"all (?:four )?angles|these four|four angles|四个角度|所有角度", message))
            return await response(draft, [], action, source, reply_language, requested, agents, **plan)

    if action == "predict":
        plan["starts_new_prediction"] = matches(r"\bpredict\b|\bforecast\b|new prediction|预测|试算", message)
    result = draft.model_copy()
    issues = []

    if codes:
        matching_rooms = [catalog.rooms[code] for code in codes if code in catalog.rooms]
        if len(codes) == 1 and matching_rooms:
            result.room_id = matching_rooms[0].id
        else:
            result.room_id = None
            issues.append({"field": "room", "reason": "multiple" if len(codes) > 1 else "not_found",
                           "rooms": [room.card() for room in matching_rooms], "floor": floor})
    elif requests_room:
        pool = catalog.search(floor=floor)
        matched_query = catalog.search(room_query, floor) if room_query else []
        if len(matched_query) == 1:
            result.room_id = matched_query[0].id
        else:
            candidates = matched_query or pool
            size_hint = re.search(area_words, room_query or "", re.IGNORECASE)
            if size_hint:
                smaller = bool(re.search(r"小|small", size_hint[0], re.IGNORECASE))
                candidates = sorted(candidates, key=lambda room: room.area, reverse=not smaller)
            # Only candidate IDs and minimal catalog facts are sent to the specialist.
            if room_query and candidates:
                shortlist = candidates[:24]
                ids = await agents.match(room_query, [room.card() for room in shortlist])
                if ids:
                    candidates = [catalog.rooms[room_id] for room_id in ids]
            result.room_id = None
            issues.append({"field": "room", "reason": "multiple" if candidates else "not_found",
                           "rooms": [room.card() for room in candidates[:6]], "floor": floor})
    elif result.room_id not in catalog.rooms:
        result.room_id = None
        issues.append({"field": "room", "reason": "missing", "rooms": [], "floor": None})

    if preference:
        result.preference = preference
    for field, values, unresolved in (("day", days, vague_day), ("hour", hours, vague_hour)):
        if not values and not unresolved and getattr(result, field) is not None:
            continue
        issue = value_issue(field, values, unresolved)
        setattr(result, field, None if issue else values[0])
        if issue:
            issues.append(issue)

    return await response(result, issues, "clarify" if issues else "predict", source,
                          reply_language, requested, agents, **plan)


async def response(draft: Draft, issues: list[dict], action: str, interpretation: str,
                   language: str, requested: str | None, agents: Agents, sentence: str | None = None,
                   **plan) -> dict:
    reply = sentence
    if language not in {"en", "zh"}:
        source = sentence or {
            "language": "I’ll reply in your requested language.",
            "clarify": "Please confirm the details below.",
            "predict": "Here are the predictions for the four louver angles.",
        }.get(action)
        if source:
            reply = await agents.localize(source, language) or sentence
    return {
        "draft": draft.model_dump(), "issues": issues, "action": action,
        "interpretation": interpretation, "language": language,
        "requested_language": requested, "localized_reply": reply, **plan,
    }
