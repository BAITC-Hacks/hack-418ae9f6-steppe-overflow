"""Проверка фактов в объяснении LLM: каждое число в тексте должно следовать из результатов инструментов.

Числа считают инструменты; LLM может их только пересказать (в долях, процентах, округлённо).
Если в тексте есть число, которого нет в результатах, публикация отклоняется, и LLM
переписывает объяснение. Даты, время, годы и названия моделей (06Z, 0.25°, 9 км) не проверяются.
"""

from __future__ import annotations

import re
from typing import Iterable

# Числа, которые встречаются в названиях и единицах, а не в данных
_CONSTANTS = {0.0, 1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 24.0, 48.0, 80.0, 100.0, 120.0, 0.25}

_IGNORE = [
    r"\b\d{1,2}\.(?:0[1-9]|1[0-2])(?:\.\d{2,4})?\b",   # даты 13.02 / 13.02.2026
    r"\b\d{1,2}:\d{2}\b",                              # время 23:00
    r"\b20\d\d\b",                                     # годы
    r"\b\d{1,2}\s*[ZzЗ]\b", r"\b\d{1,2}\s*UTC\b",      # прогоны 06Z, 12 UTC
    r"\bv\d+\b", r"\bD\+\d\b", r"\bP\d{2}\b",          # версии, D+1, P10/P90
    r"\bUTC\s*[+−-]\s*\d+\b",                           # часовой пояс
    r"\b\d+(?:[.,]\d+)?\s*°",                           # 0.25°
    r"\b\d+\s*(?:км|м)\b(?!/)",                         # 9 км, 100 м (но не м/с)
]
_NUMBER = re.compile(r"(?<![\w])[-+−]?(\d+(?:[.,]\d+)?)\s*(%|п\.\s?п\.|процент\w*)?")


def numbers_in_text(text: str) -> list[tuple[float, bool, int]]:
    """Числа из текста: (значение, это проценты?, знаков после запятой)."""
    t = text
    for pattern in _IGNORE:
        t = re.sub(pattern, " ", t)
    out = []
    for m in _NUMBER.finditer(t):
        raw = m.group(1).replace(",", ".")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append((abs(float(raw)), m.group(2) is not None, decimals))
    return out


def _flatten(obj) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten(v)


def allowed_numbers(results: list[dict]) -> tuple[set[float], set[float]]:
    """(значения как есть, доли ×100 для процентов) из результатов инструментов."""
    raw, pct = set(_CONSTANTS), {0.0, 100.0}
    for r in results:
        for x in _flatten(r):
            raw.add(abs(x))
            if abs(x) <= 1.5:
                pct.add(abs(x) * 100)
    return raw, pct


def _matches(n: float, is_pct: bool, decimals: int, raw: set[float], pct: set[float]) -> bool:
    # Точность — как записано в тексте: «32 %» ↔ 0.315…0.325; «2,66» ↔ 2.655…2.665
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    candidates = pct if is_pct else raw | {a * 100 for a in raw if a <= 1.5 and decimals == 0}
    return any(abs(n - a) <= tol for a in candidates)


def check(text: str, results: list[dict]) -> dict:
    """{'passed': bool, 'numbers': кол-во проверенных чисел, 'unmatched': числа без источника}."""
    raw, pct = allowed_numbers(results)
    nums = numbers_in_text(text)
    bad = [n for n, is_pct, dec in nums if not _matches(n, is_pct, dec, raw, pct)]
    return {"passed": not bad, "numbers": len(nums), "unmatched": sorted(set(bad))}
