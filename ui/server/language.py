"""Resolve explicit chat-language requests independently of the interface locale."""
from __future__ import annotations

import re

LANGUAGE_NAMES = {
    "english": "en", "英语": "en", "英文": "en",
    "traditional chinese": "zh-Hant", "繁體中文": "zh-Hant", "繁体中文": "zh-Hant",
    "chinese": "zh", "mandarin": "zh", "simplified chinese": "zh",
    "中文": "zh", "汉语": "zh", "普通话": "zh", "简体中文": "zh",
    "french": "fr", "français": "fr", "法语": "fr", "法文": "fr",
    "spanish": "es", "español": "es", "西班牙语": "es",
    "japanese": "ja", "日本語": "ja", "日语": "ja", "日文": "ja",
    "korean": "ko", "한국어": "ko", "韩语": "ko", "韩文": "ko",
    "german": "de", "deutsch": "de", "德语": "de",
    "italian": "it", "italiano": "it", "意大利语": "it",
    "portuguese": "pt", "português": "pt", "葡萄牙语": "pt",
    "russian": "ru", "русский": "ru", "俄语": "ru",
    "arabic": "ar", "العربية": "ar", "阿拉伯语": "ar",
    "hindi": "hi", "印地语": "hi", "thai": "th", "泰语": "th",
    "vietnamese": "vi", "越南语": "vi", "indonesian": "id", "印度尼西亚语": "id",
}

NAMES = "|".join(re.escape(name) for name in sorted(LANGUAGE_NAMES, key=len, reverse=True))
LEAD = r"(?:(?:please|could you|can you|would you|can we|could we|may we|from now on|now)\s+)*"
TAIL = r"(?:\s+(?:please|from now on|for this chat))?"
VERB = r"(?:reply|respond|answer|speak|continue|chat|talk|communicate)"
REQUESTS = (
    re.compile(rf"\b{LEAD}(?:{VERB}(?:\s+(?:to me|with me))?\s+in|use)\s+({NAMES})(?![A-Za-z]){TAIL}",
               re.IGNORECASE),
    re.compile(rf"(?:以后|接下来|现在)?\s*(?:请|麻烦|可以|能不能)?\s*(?:用|改用|换成|切换到|说)\s*({NAMES})"
               r"(?:\s*(?:回复|回答|沟通|交流|说话|继续|对话|吧))*", re.IGNORECASE),
    re.compile(rf"^{LEAD}(?:请\s*)?({NAMES})\s*(?:please|thanks|回复|回答|交流)?\s*[.!?。！？，,]*$", re.IGNORECASE),
)


def language_code(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    name = LANGUAGE_NAMES.get(value.casefold())
    if name:
        return name
    if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,3}", value):
        return None
    code = value.split("-")[0].lower()
    if code == "en" or (code == "zh" and value.lower() in {"zh", "zh-cn", "zh-hans"}):
        return code
    return value


def split_language_request(message: str) -> tuple[str | None, str]:
    for pattern in REQUESTS:
        match = pattern.search(message)
        if match:
            remainder = (message[:match.start()] + " " + message[match.end():]).strip(" ,.!?，。！？；;\n")
            remainder = re.sub(r"^(?:and|then)\b[\s,]*", "", remainder, flags=re.IGNORECASE)
            remainder = re.sub(r"^(?:然后|并且|并|再)[，,\s]*", "", remainder)
            if re.fullmatch(r"(?:thanks|thank you|please|谢谢|多谢)", remainder, re.IGNORECASE):
                remainder = ""
            return LANGUAGE_NAMES[match[1].casefold()], remainder.strip()
    return None, message
