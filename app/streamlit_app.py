"""Веб-интерфейс Steppe Wind: прогноз выработки ВЭС на 24–48 часов.

Запуск: uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from windagent import forecast
from windagent.config import load_settings
from windagent.eval import metrics
from windagent.ui import charts, data, theme

st.set_page_config(page_title="Steppe Wind — прогноз ВЭС", page_icon="🌬️", layout="wide")

SETTINGS = load_settings()
OFFSET = SETTINGS["scada"]["utc_offset_hours"]
MONTHS = ["", "января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
          "сентября", "октября", "ноября", "декабря"]
PERIOD_NAMES = {"feb2025": "февраль 2025", "dec2025": "декабрь 2025", "jan2026": "январь 2026"}


def fmt_day(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {MONTHS[d.month]} {d.year}"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.0f}%"


# --- Кэш ---------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_submission() -> pd.DataFrame:
    return data.submission()


@st.cache_data(show_spinner=False)
def load_manifest(d: str) -> dict:
    return data.manifest(d)


@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame:
    return data.backtest_preds()


@st.cache_data(show_spinner="Собираю прогнозы погоды на момент выпуска…")
def load_weather(d: str) -> pd.DataFrame:
    return data.weather_for_issue(d)


@st.cache_resource(show_spinner="Загружаю модель…")
def load_model():
    return forecast.load_model()


# --- Общие элементы -----------------------------------------------------------------------


def issue_picker(key: str) -> pd.Timestamp:
    dates = data.test_issue_dates()
    default = st.session_state.get("issue_date", dates[9])
    idx = next((i for i, d in enumerate(dates) if d == default), 9)
    d = st.selectbox(
        "Дата выпуска прогноза",
        dates,
        index=idx,
        format_func=lambda x: f"{fmt_day(x)}, 14:00 → прогноз на {fmt_day(x + pd.Timedelta(days=1))} и "
                              f"{fmt_day(x + pd.Timedelta(days=2))}",
        key=key,
    )
    st.session_state["issue_date"] = d
    return pd.Timestamp(d)


def leakage_badge(m: dict) -> None:
    ok = m["checks"]["all_sources_published_before_as_of"]
    as_of_local = pd.Timestamp(m["as_of_utc"]) + pd.Timedelta(hours=OFFSET)
    if ok:
        st.success(
            f"✅ Все прогнозы погоды опубликованы до момента прогноза "
            f"({as_of_local:%d.%m.%Y %H:%M} местного времени). Данные из будущего не использованы.",
        )
    else:
        st.error("⚠️ Обнаружен источник, опубликованный позже момента прогноза.")


def stat_tiles(f: pd.DataFrame, extra: dict | None = None) -> None:
    s = data.summarize(f)
    cols = st.columns(4)
    cols[0].metric("Средняя мощность завтра", fmt_pct(s["mean_d1"]),
                   help="Средняя прогнозная мощность ВЭС на D+1 в долях номинала")
    cols[1].metric("Выработка завтра", f"{s['full_load_hours_d1']:.1f} ч",
                   help="В часах работы на номинальной мощности: сумма почасовой нормированной мощности за D+1")
    cols[2].metric(f"Пик завтра, в {s['peak_time']:%H:%M}", fmt_pct(s["peak_value"]))
    if extra:
        cols[3].metric(extra["label"], extra["value"], help=extra.get("help"))
    elif s["mean_width"] is not None:
        cols[3].metric("Неопределённость", f"±{s['mean_width'] / 2 * 100:.0f} п.п.",
                       help="Половина средней ширины интервала P10–P90")


# --- Страницы -----------------------------------------------------------------------------


def page_forecast():
    st.title("Прогноз выработки ВЭС")
    st.caption("Почасовой прогноз на 48 часов: сутки вперёд (D+1) и двое суток вперёд (D+2). "
               "Мощность — в долях номинала ВЭС (2 турбины).")

    mode = st.segmented_control(
        "Режим", ["Февраль 2026 — тестовый период", "История — прогноз против факта"],
        default="Февраль 2026 — тестовый период", label_visibility="collapsed",
    )

    if mode and mode.startswith("История"):
        page_history()
        return

    d = issue_picker("issue_forecast")
    sub = load_submission()
    f = sub[sub["issue_date"] == d].reset_index(drop=True)
    m = load_manifest(f"{d:%Y-%m-%d}")

    stat_tiles(f)
    show_t = st.toggle("Показать турбины по отдельности", value=False)
    st.plotly_chart(charts.forecast_chart(f, show_turbines=show_t), use_container_width=True)
    leakage_badge(m)

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🔄 Пересчитать выпуск сейчас", help="Заново: прогнозы погоды на момент выпуска → признаки → модель"):
            t0 = time.time()
            live, _ = forecast.forecast_issue(d, load_model())
            diff = float(np.abs(live["p_farm"].to_numpy() - f["p_farm"].to_numpy()).max())
            st.info(f"Пересчитано за {time.time() - t0:.1f} с. Расхождение с сохранённым прогнозом: {diff:.4f} "
                    f"({'совпадает' if diff < 1e-3 else 'отличается'}).")
    with c2:
        st.download_button("⬇️ Скачать прогноз этого выпуска (CSV)", f.to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_{d:%Y-%m-%d}.csv", mime="text/csv")

    with st.expander("Таблица: все 48 часов"):
        t = f[["target_time_local", "lead_day", "p_farm", "p_farm_q10", "p_farm_q90", "p_t1", "p_t2"]].rename(columns={
            "target_time_local": "час (UTC+6)", "lead_day": "сутки", "p_farm": "ВЭС", "p_farm_q10": "P10",
            "p_farm_q90": "P90", "p_t1": "турбина 1", "p_t2": "турбина 2"})
        st.dataframe(t, hide_index=True, use_container_width=True,
                     column_config={"час (UTC+6)": st.column_config.DatetimeColumn(format="DD.MM HH:mm")})

    st.download_button("⬇️ Скачать весь прогноз на февраль (28 выпусков, CSV)",
                       sub.to_csv(index=False).encode("utf-8"), file_name="forecast_feb2026.csv", mime="text/csv")


def page_history():
    bt = load_backtest()
    ens = bt[bt["model"] == "ensemble"]
    periods = list(dict.fromkeys(ens["period"]))
    c1, c2 = st.columns([1, 2])
    per = c1.selectbox("Контрольный период", periods, format_func=lambda p: PERIOD_NAMES.get(p, p))
    dates = sorted(ens.loc[ens["period"] == per, "issue_date"].unique())
    d = c2.selectbox("Дата выпуска", dates, index=len(dates) // 2, format_func=fmt_day)
    g = ens[(ens["period"] == per) & (ens["issue_date"] == d)].sort_values("target_time_utc")
    f = g[["target_time_local", "lead_day"]].copy()
    f["p_farm"] = g["p_farm_pred"].to_numpy()
    mae = metrics.mae(g["p_farm"], g["p_farm_pred"])
    st.caption("Модель обучена только на данных до начала периода; факт на момент прогноза был неизвестен.")
    stat_tiles(f, extra={"label": "Ошибка этого выпуска (MAE)", "value": f"{mae:.3f}",
                         "help": "Средняя абсолютная ошибка по 48 часам, в долях номинала"})
    st.plotly_chart(charts.forecast_chart(f, fact=g["p_farm"].reset_index(drop=True)), use_container_width=True)


def page_weather():
    st.title("Прогнозы погоды на момент выпуска")
    st.caption("Модель видит только те прогнозы, которые уже были опубликованы в момент выпуска — "
               "как в реальной работе диспетчера.")
    d = issue_picker("issue_weather")
    w = load_weather(f"{d:%Y-%m-%d}")
    st.plotly_chart(charts.weather_chart(w), use_container_width=True)
    st.caption("Чем сильнее расходятся модели, тем неувереннее прогноз — это учитывается в интервале P10–P90.")

    m = load_manifest(f"{d:%Y-%m-%d}")
    leakage_badge(m)
    as_of = pd.Timestamp(m["as_of_utc"])
    rows = {}
    for e in m["sources"]:
        if not e.get("model") or not e.get("max_published_at"):
            continue
        key = e["model"]
        pub = pd.Timestamp(e["max_published_at"])
        if key not in rows or pub > rows[key]["pub"]:
            rows[key] = {"pub": pub, "source": e["source"], "days": e.get("days_used")}
    names = {"ecmwf_ifs": "ECMWF IFS 9 км", "ecmwf_ifs025": "ECMWF IFS 0.25°", "gfs_seamless": "GFS", "icon_seamless": "ICON"}
    table = pd.DataFrame([
        {
            "модель": names.get(k, k),
            "API Open-Meteo": "Single Runs" if v["source"] == "single" else "Previous Runs",
            "самый свежий прогон опубликован (UTC+6)": v["pub"] + pd.Timedelta(hours=OFFSET),
            "запас до момента прогноза": f"{(as_of - v['pub']) / pd.Timedelta('1h'):.0f} ч",
            "проверка": "✅" if v["pub"] <= as_of else "❌",
        }
        for k, v in rows.items()
    ])
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"самый свежий прогон опубликован (UTC+6)": st.column_config.DatetimeColumn(format="DD.MM.YYYY HH:mm")})
    with st.expander("Таблица: ветер на 100 м по часам"):
        t = w[["target_time_local"] + [c for c in w.columns if c.endswith("__wind_speed_100m")]]
        st.dataframe(t.rename(columns=lambda c: c.replace("__wind_speed_100m", ", м/с").upper() if "__" in c else "час (UTC+6)"),
                     hide_index=True, use_container_width=True)


def page_quality():
    st.title("Качество прогноза")
    st.caption("Бэктест по тому же протоколу, что и тест: ежедневный выпуск в 14:00, прогноз на 48 ч, "
               "модель обучена только на прошлом.")
    bt = load_backtest()
    table = bt.groupby(["model", "period"]).apply(
        lambda g: metrics.mae(g["p_farm"], g["p_farm_pred"]), include_groups=False
    ).unstack()[list(dict.fromkeys(bt["period"]))]
    table = table.loc[[m for m in ("ensemble", "phys_ifs", "clim") if m in table.index]]
    table.columns = [PERIOD_NAMES.get(c, c) for c in table.columns]

    ens, ref, clim = (table.loc[m].mean() for m in ("ensemble", "phys_ifs", "clim"))
    c = st.columns(3)
    c[0].metric("MAE основной модели", f"{ens:.3f}", help="Средняя абсолютная ошибка в долях номинала, среднее по 3 периодам")
    c[1].metric("Лучше физической кривой", f"{(1 - ens / ref) * 100:.0f}%")
    c[2].metric("Лучше климатологии", f"{(1 - ens / clim) * 100:.0f}%")

    st.plotly_chart(charts.mae_by_model_chart(table), use_container_width=True)
    st.subheader("Ошибка по горизонту прогноза")
    st.plotly_chart(charts.error_by_horizon_chart(bt), use_container_width=True)
    with st.expander("Таблица MAE"):
        st.dataframe(table.rename(index=theme.MODEL_NAMES).round(3), use_container_width=True)
    st.info("**Почему не точнее?** Если подставить в модель фактический ветер на турбине, ошибка была бы всего "
            "0.03–0.04. Прогноз ветра на сутки вперёд ошибается в среднем на 2–2.5 м/с — именно это, "
            "а не модель, ограничивает точность.")


def page_about():
    st.title("Как это работает")
    st.markdown(f"""
**Задача.** Прогноз почасовой выработки ветроэлектростанции из двух турбин
(Алматинская область, 43.64° с. ш., 78.54° в. д.) на 24–48 часов вперёд.

**Цикл прогноза для даты выпуска D (14:00 по местному времени):**
1. **Погода.** Берутся прогнозы ECMWF, ICON и GFS из архива Open-Meteo — только опубликованные к моменту выпуска.
2. **Подготовка данных.** 48 целевых часов × ~90 признаков: ветер на разных высотах, направление, сдвиг ветра,
   плотность воздуха, согласие погодных моделей, календарь.
3. **Модель.** Ансамбль градиентного бустинга и физической кривой мощности турбин; интервал P10–P90 —
   квантильный бустинг с конформной калибровкой.
4. **Результат.** Прогноз по каждой турбине и по ВЭС + манифест: какие данные использованы и когда опубликованы.

**Честность ретроспективы.** Все данные проходят через `DataStore(as_of)`: он физически не отдаёт прогнозы погоды,
опубликованные после момента выпуска. Это проверяется автотестами для всех 28 выпусков февраля.

**Данные.** SCADA двух турбин с 11.03.2023 по 31.01.2026 (10-минутные записи); архив прогнозов погоды
[Open-Meteo](https://open-meteo.com/) с 15.03.2024.

Модель: `{load_model().version}`.
""")


pages = [
    st.Page(page_forecast, title="Прогноз", icon="📈", default=True),
    st.Page(page_weather, title="Погода", icon="🌬️", url_path="weather"),
    st.Page(page_quality, title="Качество", icon="🎯", url_path="quality"),
    st.Page(page_about, title="Как это работает", icon="ℹ️", url_path="about"),
]
with st.sidebar:
    st.markdown("### 🌬️ Steppe Wind")
    st.caption("Agentic AI прогноз выработки ВЭС · команда Steppe Overflow")
st.navigation(pages).run()
