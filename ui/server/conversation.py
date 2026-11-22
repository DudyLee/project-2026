"""Convert language and catalog matches into field states and selectable controls."""
from __future__ import annotations

from datetime import date, timedelta
import re
import unicodedata

from .agents import Agents
from .catalog import Catalog
from .language import language_code, split_language_request
from .schema import Draft, Intent

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
    room_codes = list(dict.fromkeys(code.upper() for code in re.findall(ROOM_CODE, text)))
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
    before_hour = re.search(r"(?i)(?:\bat\s+|小时\s*)(\d{1,2})(?!\d)", text)
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
                  language: str = "en") -> dict:
    requested, message = split_language_request(message)
    reply_language = requested or language_code(language) or "en"
    if not message:
        return await response(draft, [], "language", "local", reply_language, requested, agents)
    local, codes = local_intent(message, today)
    # An isolated number is a day only when replying to a missing date and an already set hour.
    if re.fullmatch(NUMERAL, message.strip()) and local.hours and draft.day is None and draft.hour is not None:
        local.days, local.hours = local.hours, []
    remote = await agents.interpret(message, draft, today.isoformat())
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
    explanation = local.explain or bool(remote and remote.explain)
    room_query = (remote.room_query if remote and remote.room_query
                  and remote.room_query.casefold() in message.casefold() else None)
    area_words = r"(?:最大|较大|大一点|largest|biggest)|(?:最小|较小|小一点|smallest)"
    if not room_query and not codes and re.search(r"会议室|办公室|朝[东西南北]|" + area_words, message, re.IGNORECASE):
        room_query = message
    requests_room = bool(codes or room_query or floor is not None)
    result = draft.model_copy()
    issues = []

    if codes:
        matches = [catalog.rooms[code] for code in codes if code in catalog.rooms]
        if len(codes) == 1 and matches:
            result.room_id = matches[0].id
        else:
            result.room_id = None
            issues.append({"field": "room", "reason": "multiple" if len(codes) > 1 else "not_found",
                           "rooms": [room.card() for room in matches], "floor": floor})
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

    has_new_conditions = requests_room or bool(days or hours or vague_day or vague_hour or preference)
    action = "clarify" if issues else ("explain" if explanation and not has_new_conditions else "predict")
    return await response(result, issues, action, "agent" if remote is not None else "local",
                          reply_language, requested, agents)


async def response(draft: Draft, issues: list[dict], action: str, interpretation: str,
                   language: str, requested: str | None, agents: Agents) -> dict:
    text = None
    if language not in {"en", "zh"} and action != "explain":
        sentence = {
            "language": "I’ll reply in your requested language.",
            "clarify": "Please confirm the details below.",
            "predict": "Here are the predictions for the four louver angles.",
        }[action]
        text = await agents.localize(sentence, language)
    return {
        "draft": draft.model_dump(), "issues": issues, "action": action,
        "interpretation": interpretation, "language": language,
        "requested_language": requested, "localized_reply": text,
    }
