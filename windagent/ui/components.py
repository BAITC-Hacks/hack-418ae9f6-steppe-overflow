"""HTML-компоненты главной страницы: шапка, KPI-карточки, карточка «AI-прогноз».

Чистые функции (данные → HTML), без Streamlit — чтобы их можно было тестировать.
Все числа берутся из прогноза, журнала агента и прогнозов погоды; ничего не выдумывается.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

MONTHS_SHORT = ["", "янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
UNCERTAINTY = [(0.35, "Низкая", "#e3f3e8", "#0f6e36"), (0.5, "Средняя", "#fdf3dc", "#8a5a00"),
               (9.0, "Высокая", "#fbe4e4", "#a12b2b")]
SOURCE_NAMES = {"ifs": "ECMWF", "icon": "ICON", "gfs": "GFS", "ifs025": "ECMWF 0.25°"}

CSS = """
<style>
.sw-head { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap; }
.sw-head h1 { margin:0; padding:0; font-size:2.1rem; font-weight:800; color:#0c2f2b; }
.sw-sub { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:4px; color:#5b6b66; font-size:14px; }
.sw-tag { border:1px solid #dce8e1; border-radius:8px; padding:2px 8px; font-size:12.5px; color:#34423e; background:#fff; }
.sw-issue { text-align:right; font-size:13.5px; color:#5b6b66; }
.sw-issue b { color:#0c2f2b; }
.sw-ok { color:#0f6e36; font-weight:600; } .sw-ok::before { content:"●"; margin-right:6px; }
.sw-warn { color:#a12b2b; font-weight:600; } .sw-warn::before { content:"●"; margin-right:6px; }
.sw-kpis { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:14px; }
.sw-kpi { background:#fff; border:1px solid #dce8e1; border-radius:14px; padding:14px 16px; min-height:112px; }
.sw-kpi .l { color:#5b6b66; font-weight:600; font-size:13.5px; }
.sw-kpi .v { font-size:2.1rem; font-weight:800; color:#0c2f2b; line-height:1.25; margin-top:4px; }
.sw-kpi .v small { font-size:1rem; font-weight:700; margin-left:8px; }
.sw-kpi .s { color:#5b6b66; font-size:13px; margin-top:4px; }
.sw-pill { display:inline-block; border-radius:999px; padding:2px 10px; font-size:12.5px; font-weight:700; }
.sw-ai { display:grid; grid-template-columns:minmax(0, 1.7fr) minmax(0, 1fr); gap:24px; background:#eef6f1;
  border:1px solid #d3e7da; border-radius:16px; padding:18px 20px; }
.sw-ai .t { display:flex; align-items:center; gap:10px; font-weight:800; color:#0c2f2b; }
.sw-ai .t svg { width:22px; height:22px; }
.sw-ai .h { font-size:1.12rem; font-weight:800; color:#0c2f2b; margin:10px 0 4px; }
.sw-ai .b { color:#34423e; line-height:1.5; font-size:14.5px; }
.sw-ai .why { border-left:1px solid #cfe0d6; padding-left:20px; font-size:13.5px; color:#34423e; }
.sw-ai .why b { color:#0c2f2b; }
.sw-ai .why ul { margin:6px 0 0; padding-left:18px; } .sw-ai .why li { margin:3px 0; }
.sw-ai .rev { margin-top:8px; font-size:12.5px; color:#5b6b66; }
@media (max-width: 900px) {
  .sw-kpis { grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }
  .sw-ai { grid-template-columns:1fr; }
  .sw-ai .why { border-left:0; padding-left:0; border-top:1px solid #cfe0d6; padding-top:12px; }
  .sw-issue { text-align:left; }
}
@media (max-width: 640px) {
  .sw-head h1 { font-size:1.6rem; }
  .sw-kpi { padding:10px 12px; min-height:0; } .sw-kpi .v { font-size:1.55rem; }
}
</style>
"""

SPARKLE = ('<svg viewBox="0 0 24 24" fill="#12803f"><path d="M12 2l1.9 5.6L19.5 9.5l-5.6 1.9L12 17l-1.9-5.6'
           'L4.5 9.5l5.6-1.9L12 2zm7 11l.95 2.55L22.5 16.5l-2.55.95L19 20l-.95-2.55L15.5 16.5l2.55-.95L19 13z"/></svg>')


def fmt_short(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day:02d} {MONTHS_SHORT[ts.month]} {ts.year}, {ts:%H:%M}"


def uncertainty_level(width: float) -> tuple[str, str, str]:
    for limit, name, bg, fg in UNCERTAINTY:
        if width < limit:
            return name, bg, fg
    return UNCERTAINTY[-1][1:]


def header_html(title: str, issue_local: pd.Timestamp | None, data_ok: bool | None, subtitle: str,
                tag: str = "Мощность — в % номинала ВЭС (2 турбины)") -> str:
    right = ""
    if issue_local is not None:
        status = ('<span class="sw-ok">Только данные, опубликованные до выпуска</span>' if data_ok
                  else '<span class="sw-warn">Найден источник из будущего</span>' if data_ok is False else "")
        right = f'<div class="sw-issue">Выпуск прогноза: <b>{fmt_short(issue_local)}</b><br>{status}</div>'
    return (f'{CSS}<div class="sw-head"><div><h1>{html.escape(title)}</h1><div class="sw-sub">{html.escape(subtitle)}'
            f'<span class="sw-tag">{html.escape(tag)}</span></div></div>{right}</div>')


def kpi_html(f: pd.DataFrame, clim_delta: float | None, extra: tuple[str, str, str] | None = None) -> str:
    """Четыре карточки: средняя мощность, максимум, выработка, неопределённость (или своя 4-я)."""
    d1 = f[f["lead_day"] == 1]
    mean, peak_i = float(d1["p_farm"].mean()), d1["p_farm"].idxmax()
    peak, peak_t = float(d1.loc[peak_i, "p_farm"]), pd.Timestamp(d1.loc[peak_i, "target_time_local"])
    calm, high = int((d1["p_farm"] <= 0.05).sum()), int((d1["p_farm"] >= 0.8).sum())
    if clim_delta is None:
        delta = '<span class="s">&nbsp;</span>'
    else:
        up = clim_delta >= 0
        delta = (f'<span class="sw-pill" style="background:{"#e3f3e8" if up else "#fbe4e4"};'
                 f'color:{"#0f6e36" if up else "#a12b2b"}" title="Отклонение от климатической нормы для этого '
                 f'месяца и часов">{"↑" if up else "↓"} {abs(clim_delta) * 100:.0f} п.п. к норме</span>')
    regime = ("Штиль и работа на номинале не ожидаются" if not calm and not high else
              " · ".join(x for x in (f"штиль {calm} ч" if calm else "", f"на номинале {high} ч" if high else "") if x).capitalize())
    cards = [
        ("Средняя мощность завтра", f"{mean:.0%}", delta),
        ("Максимум завтра", f"{peak:.0%}<small>в {peak_t:%H:%M}</small>", f'<div class="s">{regime}</div>'),
        ("Выработка завтра", f"{d1['p_farm'].sum():.1f} ч", '<div class="s">в эквиваленте работы на номинале</div>'),
    ]
    if extra:
        cards.append(extra)
    elif "p_farm_q90" in f:
        width = float((f["p_farm_q90"] - f["p_farm_q10"]).mean())
        name, bg, fg = uncertainty_level(width)
        cards.append(("Неопределённость P10–P90",
                      f'{width * 100:.0f} п.п.<small><span class="sw-pill" style="background:{bg};color:{fg}">{name}</span></small>',
                      '<div class="s">средняя ширина интервала 80 %</div>'))
    inner = "".join(f'<div class="sw-kpi"><div class="l">{l}</div><div class="v">{v}</div>{s}</div>' for l, v, s in cards)
    return f'{CSS}<div class="sw-kpis">{inner}</div>'


def split_explanation(text: str) -> tuple[str, str]:
    """Первая фраза — заголовок карточки, остальное — пояснение."""
    parts = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])", text.strip(), maxsplit=1)
    return (parts[0], parts[1] if len(parts) > 1 else "")


def model_gaps(w: pd.DataFrame | None) -> list[tuple[str, float]]:
    """Среднее расхождение ECMWF с другими моделями по ветру на 100 м (м/с), по убыванию."""
    if w is None or "ifs__wind_speed_100m" not in w:
        return []
    out = []
    for src in ("icon", "gfs", "ifs025"):
        col = f"{src}__wind_speed_100m"
        if col in w:
            out.append((src, float((w[col] - w["ifs__wind_speed_100m"]).abs().mean())))
    return sorted(out, key=lambda x: -x[1])


def ai_card_html(explanation: str, confidence: str | None, mode: str, gaps: list[tuple[str, float]],
                 width: float | None, spread: float | None, revision: str | None = None) -> str:
    conf = confidence or "—"
    colors = {"высокая": ("#e3f3e8", "#0f6e36"), "средняя": ("#fdf3dc", "#8a5a00"), "низкая": ("#fbe4e4", "#a12b2b")}
    bg, fg = colors.get(conf, ("#eef2f0", "#34423e"))
    head, body = split_explanation(explanation)
    label = "AI-прогноз" if mode == "llm" else "Прогноз агента"
    reasons = [f"{SOURCE_NAMES['ifs']} ↔ {SOURCE_NAMES[s]}: {g:.1f} м/с" for s, g in gaps[:2]]
    if spread is not None:
        reasons.append(f"разброс четырёх моделей: {spread:.1f} м/с")
    if width is not None:
        reasons.append(f"диапазон мощности P10–P90: {width * 100:.0f} п.п.")
    why = ""
    if reasons:
        why = (f'<div class="why"><b>Почему уверенность {html.escape(conf)}?</b><div style="margin-top:4px">'
               f'Насколько расходятся прогнозы ветра на 100 м:</div><ul>'
               + "".join(f"<li>{html.escape(r)}</li>" for r in reasons) + "</ul></div>")
    rev = f'<div class="rev">{html.escape(revision)}</div>' if revision else ""
    return (f'{CSS}<div class="sw-ai"><div><div class="t">{SPARKLE}{label}'
            f'<span class="sw-pill" style="background:{bg};color:{fg}">{html.escape(conf.capitalize())} уверенность</span></div>'
            f'<div class="h">{html.escape(head)}</div><div class="b">{html.escape(body)}</div>{rev}</div>{why}</div>')


def weather_spread(w: pd.DataFrame | None) -> float | None:
    if w is None:
        return None
    cols = [c for c in w.columns if c.endswith("__wind_speed_100m") and not c.startswith("nb_")]
    return float(w[cols].std(axis=1).mean()) if cols else None


def previous_forecast(sub: pd.DataFrame, issue_date) -> pd.DataFrame | None:
    """Прогноз прошлого выпуска на те же часы (его D+2 = наше «завтра»)."""
    prev = sub[sub["issue_date"] == pd.Timestamp(issue_date) - pd.Timedelta(days=1)]
    return prev if len(prev) else None


def wind_for_hours(w: pd.DataFrame | None, f: pd.DataFrame) -> np.ndarray | None:
    if w is None or "ifs__wind_speed_100m" not in w:
        return None
    m = f[["target_time_local"]].merge(w[["target_time_local", "ifs__wind_speed_100m"]], on="target_time_local", how="left")
    return m["ifs__wind_speed_100m"].to_numpy()
