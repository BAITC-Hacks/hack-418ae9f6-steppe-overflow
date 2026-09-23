"""Веб-интерфейс Steppe Wind: прогноз выработки ВЭС на 24–48 часов.

Запуск: uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from windagent import forecast
from windagent.config import ROOT, load_settings
from windagent.eval import metrics
from windagent.ui import agent_view as av
from windagent.ui import charts, data, theme

ASSETS = ROOT / "app" / "assets"
st.set_page_config(page_title="Steppe Wind — прогноз ВЭС", page_icon=str(ASSETS / "favicon.svg"), layout="wide")

SETTINGS = load_settings()
OFFSET = SETTINGS["scada"]["utc_offset_hours"]
MONTHS = ["", "января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
          "сентября", "октября", "ноября", "декабря"]
PERIOD_NAMES = {"feb2025": "февраль 2025", "dec2025": "декабрь 2025", "jan2026": "январь 2026"}
CONFIDENCE_STYLE = {"высокая": ("#e3f3e8", "#0f6e36"), "средняя": ("#fdf3dc", "#8a5a00"), "низкая": ("#fbe4e4", "#a12b2b")}


def fmt_day(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {MONTHS[d.month]} {d.year}"


def fmt_issue(d) -> str:
    """«9 февраля → на 10–11 февраля» (коротко, чтобы помещалось на телефоне)."""
    d = pd.Timestamp(d)
    d1, d2 = d + pd.Timedelta(days=1), d + pd.Timedelta(days=2)
    target = (f"{d1.day}–{d2.day} {MONTHS[d2.month]}" if d1.month == d2.month
              else f"{d1.day} {MONTHS[d1.month]} – {d2.day} {MONTHS[d2.month]}")
    return f"{d.day} {MONTHS[d.month]} → на {target}"


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


def load_agent_run(d) -> dict | None:
    """Свежий живой запуск этой сессии, иначе сохранённый журнал из репозитория."""
    if st.session_state.get(f"live_run_{pd.Timestamp(d):%Y-%m-%d}"):
        run = av.load_run(av.run_dir(d, live=True))
        if run:
            return run
    return av.load_run(av.run_dir(d))


# --- Оформление ----------------------------------------------------------------------------

CSS = """
<style>
/* Меньше пустого места над заголовком */
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; }
h1 { padding-top: 0.2rem; }

/* Карточки ключевых показателей */
[data-testid="stMetric"] {
  background: #fff; border: 1px solid #dce8e1; border-radius: 14px; padding: 14px 16px 12px;
}
[data-testid="stMetricLabel"] p { color: #5b6b66; font-weight: 600; }
[data-testid="stMetricValue"] { font-weight: 700; }

/* Блок «Главное на завтра» */
.st-key-brief { background: linear-gradient(135deg, #f4f9f6 0%, #eef4f9 100%); border: 1px solid #dce8e1;
  border-radius: 16px; padding: 16px 20px 10px; }
.sw-chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: 12.5px; font-weight: 700; margin-right: 6px; }

/* Навигация по датам: кнопки ← → по краям выбора */
.st-key-datenav button { min-height: 42px; }

/* Плавающая кнопка чата: круглая, внизу справа, на всех страницах, с символом Steppe Wind */
.st-key-chat_fab { position: fixed; right: 28px; bottom: 28px; z-index: 1000; width: auto !important; }
.st-key-chat_fab button {
  width: 60px; height: 60px; border-radius: 50%; padding: 0; border: none;
  background: #12803f url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA4OC41MiA0MS4yNSI+PHBhdGggZmlsbD0iI2ZmZmZmZiIgZD0iTTgwLjcyLDIyLjA3Yy0yLjQ3LDAtNC40NSwxLjEzLTYuMDYsMi43OGwtNi45Niw3LjE0LTYuNDcsNi42OWMtMS4yOCwxLjMyLTIuOTUsMi41Mi00Ljg0LDIuNTNsLTQuODkuMDRjLTEuNy4wMS0yLjg1LTEuMzQtMi44NS0zdi0xNS4wM2MwLS45OS0uODctMS4xLTEuNTEtMS4xNi0zLjEzLS4zLTUuNDksMS4wMS03LjU3LDMuMTdsLTEzLjI3LDEzLjc2Yy0xLjI5LDEuMzMtMi45MSwyLjI1LTQuNzgsMi4yNGwtNC41Mi0uMDJjLTEuNDIsMC0yLjctMS4zMi0yLjctMi43OGwtLjAyLTE1LjQyYzAtLjU3LS42MS0uOTYtMS4xMi0uOTZILjk3Qy41MSwyMi4wNywwLDIxLjU2LDAsMjEuMWwuMDktMTIuOTZDLjEyLDMuNjEsNC4xOC4wNiw4LjU4LjA0TDE2LjIsMGMuMzksMCwuOTEuNDYuOTEuOTFsLjA1LDE3LjI3YzAsLjc3LDEuMDEsMS4yNiwxLjUzLDEuMzYuNjguMTMsMS42Ni0uMDUsMi4yMS0uNjRsMTMuMDMtMTMuODJDMzkuNi0uOTEsNDMuMzkuMTIsNTAuNSwwYy41NCwwLDEuMDUuNSwxLjA1LDEuMDZsLjAyLDE2LjljMCwuNjkuNjcsMS4zNCwxLjE0LDEuNTIuNTYuMjEsMS43Ni4yMywyLjIzLS4yNmw2LjAzLTYuMTcsNy40NC04LjAxYzIuNTYtMi43NSw1Ljg3LTQuOTMsOS43MS00Ljk2TDg3LjMxLDBjLjI5LDAsLjc0LjEyLjkuMjlzLjI5LjY5LjI5Ljk1bC4wMiw3Ljg0LS4wMywxMi4wMmMwLC40MS0uMzcuOTQtLjguOTRoLTYuOTZaIi8+PC9zdmc+") center / 32px no-repeat;
  box-shadow: 0 8px 24px rgba(12, 71, 65, 0.28); transition: transform .15s ease, box-shadow .15s ease;
}
.st-key-chat_fab button:hover { background-color: #0f6e36; transform: translateY(-2px);
  box-shadow: 0 12px 28px rgba(12, 71, 65, 0.34); }
.st-key-chat_fab button > div { visibility: hidden; }  /* иконка и стрелка popover — вместо них символ */
/* Подсказка слева от кнопки: показывается при наведении и один раз после загрузки */
.st-key-chat_fab::before, .st-key-chat_fab::after {
  content: "Спросите AI-помощника"; position: absolute; right: 72px; top: 50%; transform: translateY(-50%);
  white-space: nowrap; background: #0c4741; color: #fff; padding: 8px 12px; border-radius: 10px;
  font: 600 13px Manrope, Roboto, sans-serif; box-shadow: 0 6px 18px rgba(12, 71, 65, .22);
  pointer-events: none; opacity: 0;
}
.st-key-chat_fab::before { transition: opacity .15s ease; }
.st-key-chat_fab:hover::before { opacity: 1; }
.st-key-chat_fab::after { animation: sw-hint 6s ease 1.2s 1 both; }
@keyframes sw-hint { 0% { opacity: 0; } 8% { opacity: 1; } 85% { opacity: 1; } 100% { opacity: 0; } }
[data-testid="stPopoverBody"]:has(.st-key-chat_panel) { width: min(420px, calc(100vw - 32px)); }

/* Телефон */
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 6rem; }
  h1 { font-size: 1.7rem !important; line-height: 1.2 !important; }
  h2, h3 { font-size: 1.2rem !important; }
  .st-key-kpi [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .6rem !important; }
  .st-key-kpi [data-testid="stColumn"] { min-width: calc(50% - .3rem) !important; flex: 1 1 calc(50% - .3rem) !important; }
  [data-testid="stMetric"] { padding: 10px 12px; }
  [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
  .st-key-brief { padding: 12px 14px 6px; }
  .st-key-chat_fab { right: 16px; bottom: 16px; }
  .st-key-chat_fab button { width: 52px; height: 52px; background-size: 28px; }
  .st-key-chat_fab::before, .st-key-chat_fab::after { right: 62px; font-size: 12px; }
}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


GITHUB_URL = "https://github.com/BAITC-Hacks/hack-418ae9f6-steppe-overflow"


def status_block() -> None:
    """Статус системы в боковой панели: модель, данные, режим агента, сборка, ссылка на код."""
    model = load_model()
    llm = av.llm_configured()
    import os

    agent = (f'<span style="color:#0f6e36">● LLM включён</span> · {os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")}'
             if llm else '<span style="color:#8a5a00">● режим правил</span> (без ключа OpenAI)')
    rev_file = ROOT / "REVISION"
    rev = rev_file.read_text().strip() if rev_file.exists() else "локальная"
    trained = pd.Timestamp(model.meta["trained_until_utc"]) + pd.Timedelta(hours=OFFSET)
    rows = [
        ("Модель", f"<code>{model.version}</code>"),
        ("Данные SCADA", f"до {trained:%d.%m.%Y}"),
        ("Архив погоды", "15.03.2024 – 01.03.2026"),
        ("AI-агент", agent),
        ("Сборка", f"<code>{rev}</code>"),
    ]
    body = "".join(f'<div style="display:flex;justify-content:space-between;gap:8px;padding:3px 0">'
                   f'<span style="color:#5b6b66">{k}</span><span style="text-align:right">{v}</span></div>' for k, v in rows)
    st.html(
        f'<div style="background:#fff;border:1px solid #d5e0ea;border-radius:12px;padding:10px 12px;'
        f'font:12.5px Manrope,Roboto,sans-serif;color:#0c2f2b">'
        f'<div style="font-weight:700;margin-bottom:4px">Статус системы</div>{body}'
        f'<a href="{GITHUB_URL}" target="_blank" style="display:inline-block;margin-top:6px;color:#12803f;'
        f'font-weight:700;text-decoration:none">Код на GitHub ↗</a></div>'
    )


# PWA: манифест и service worker отдаёт Caddy (deploy/Caddyfile); здесь — подключение на странице
# и кнопка «Установить приложение», которая появляется, когда браузер разрешает установку.
PWA_HTML = """
<button class="sw-install" style="display:none;align-items:center;gap:8px;width:100%;margin-top:6px;
  padding:9px 12px;border-radius:10px;border:1px solid #c9d8cf;background:#fff;color:#0c4741;
  font:600 14px Manrope,Roboto,sans-serif;cursor:pointer">📲 Установить приложение</button>
<script>
(function () {
  if (["localhost", "127.0.0.1"].includes(location.hostname)) return;
  const w = window, doc = document;
  const show = () => doc.querySelectorAll(".sw-install").forEach((b) => {
    b.style.display = "flex";
    b.onclick = async () => { if (!w.__swPrompt) return; w.__swPrompt.prompt(); await w.__swPrompt.userChoice;
      w.__swPrompt = null; doc.querySelectorAll(".sw-install").forEach((x) => (x.style.display = "none")); };
  });
  if (!w.__swPwa) {
    w.__swPwa = true;
    const add = (tag, attrs) => { const el = doc.createElement(tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v)); doc.head.appendChild(el); };
    add("link", { rel: "manifest", href: "/manifest.webmanifest" });
    add("meta", { name: "theme-color", content: "#0c4741" });
    add("link", { rel: "apple-touch-icon", href: "/pwa/apple-touch-icon.png" });
    add("meta", { name: "apple-mobile-web-app-capable", content: "yes" });
    add("meta", { name: "mobile-web-app-capable", content: "yes" });
    add("meta", { name: "apple-mobile-web-app-title", content: "Steppe Wind" });
    if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");
    w.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); w.__swPrompt = e; show(); });
  }
  if (w.__swPrompt) show();
})();
</script>
"""


def plot(fig) -> None:
    """Графики в фирменном стиле: своя тема вместо стандартной Streamlit, без панели инструментов."""
    st.plotly_chart(fig, use_container_width=True, theme=None, config=theme.PLOTLY_CONFIG)


def chip(text: str, bg: str, fg: str) -> str:
    return f'<span class="sw-chip" style="background:{bg};color:{fg}">{text}</span>'


# --- Выбор даты выпуска (общий для всех страниц) ---------------------------------------------------


def _shift_issue(key: str, delta: int) -> None:
    dates = data.test_issue_dates()
    cur = pd.Timestamp(st.session_state.get(key, st.session_state.get("issue_date", dates[9])))
    i = min(max(dates.index(cur) + delta, 0), len(dates) - 1)
    st.session_state[key] = dates[i]
    st.session_state["issue_date"] = dates[i]


def issue_picker(key: str) -> pd.Timestamp:
    dates = data.test_issue_dates()
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get("issue_date", dates[9])
    cur = pd.Timestamp(st.session_state[key])
    with st.container(horizontal=True, vertical_alignment="bottom", key="datenav", gap="small"):
        st.button("", icon=":material/chevron_left:", key=f"{key}_prev", help="Предыдущий выпуск",
                  on_click=_shift_issue, args=(key, -1), disabled=cur == dates[0])
        d = st.selectbox("Выпуск прогноза (в 14:00 местного времени)", dates, key=key, format_func=fmt_issue,
                         width="stretch")
        st.button("", icon=":material/chevron_right:", key=f"{key}_next", help="Следующий выпуск",
                  on_click=_shift_issue, args=(key, 1), disabled=cur == dates[-1])
    st.session_state["issue_date"] = d
    return pd.Timestamp(d)


def leakage_note(m: dict) -> str:
    as_of_local = pd.Timestamp(m["as_of_utc"]) + pd.Timedelta(hours=OFFSET)
    if m["checks"]["all_sources_published_before_as_of"]:
        return chip(f"✓ только данные, опубликованные до {as_of_local:%d.%m %H:%M}", "#e3f3e8", "#0f6e36")
    return chip("⚠ найден источник из будущего", "#fbe4e4", "#a12b2b")


def kpi_tiles(f: pd.DataFrame, vs_clim: float | None = None, extra: dict | None = None) -> None:
    s = data.summarize(f)
    with st.container(key="kpi"):
        cols = st.columns(4)
        cols[0].metric("Мощность завтра", fmt_pct(s["mean_d1"]),
                       delta=None if vs_clim is None else f"{vs_clim * 100:+.0f} п.п. к норме",
                       help="Средняя прогнозная мощность ВЭС на завтра, % номинала. Сравнение — с климатической "
                            "нормой для этого месяца и часа")
        cols[1].metric("Выработка завтра", f"{s['full_load_hours_d1']:.1f} ч",
                       help="В часах работы на полной мощности: сумма почасовой мощности за завтра")
        cols[2].metric(f"Пик завтра · {s['peak_time']:%H:%M}", fmt_pct(s["peak_value"]))
        if extra:
            cols[3].metric(extra["label"], extra["value"], help=extra.get("help"))
        elif s["mean_width"] is not None:
            cols[3].metric("Интервал P10–P90", f"{s['mean_width'] * 100:.0f} п.п.",
                           help="Средняя ширина интервала: с вероятностью 80% факт окажется внутри него")


def agent_brief(d: pd.Timestamp, m: dict) -> float | None:
    """Блок «Главное на завтра» — объяснение агента. Возвращает отклонение от нормы для плитки."""
    run = load_agent_run(d)
    with st.container(key="brief"):
        if run is None:
            st.markdown(leakage_note(m), unsafe_allow_html=True)
            return None
        b = av.run_brief(run)
        conf = b["confidence"] or "—"
        bg, fg = CONFIDENCE_STYLE.get(conf, ("#eef2f0", "#34423e"))
        mode = "LLM-агент" if b["mode"] == "llm" else "агент (правила)"
        st.markdown(
            f'<div style="margin-bottom:6px">{chip("🤖 " + mode, "#ebf1f7", "#0c4741")}'
            f'{chip("уверенность: " + conf, bg, fg)}{leakage_note(m)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Главное на завтра.** {b['explanation']}")
        if b["revision"]:
            r = b["revision"]
            change = f", в среднем на {r['mean_abs_change'] * 100:.0f} п.п." if r["mean_abs_change"] is not None else ""
            c1, c2 = st.columns([4, 1], vertical_alignment="center")
            c1.caption(f"В {r['as_of_local'][-5:]} вышел новый прогон ECMWF — агент выпустил ревизию "
                       f"v{r['version']}{change}. В сдачу идёт версия на 14:00.")
            c2.page_link(PAGE_AGENT, label="Подробнее", icon=":material/arrow_forward:")
        return b["vs_climatology"]


# --- Страницы -----------------------------------------------------------------------------


def page_forecast():
    st.title("Прогноз выработки ВЭС")
    st.caption("Почасовой прогноз на 48 часов: завтра и послезавтра. Мощность — в % номинала ВЭС (2 турбины).")

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

    vs_clim = agent_brief(d, m)
    kpi_tiles(f, vs_clim)
    show_t = st.toggle("Показать турбины по отдельности", value=False)
    plot(charts.forecast_chart(f, show_turbines=show_t))

    with st.container(horizontal=True, key="toolbar", gap="small"):
        recalc = st.button("Пересчитать выпуск", icon=":material/refresh:",
                           help="Заново: прогнозы погоды на момент выпуска → признаки → модель")
        st.download_button("CSV выпуска", f.to_csv(index=False).encode("utf-8"), icon=":material/download:",
                           file_name=f"forecast_{d:%Y-%m-%d}.csv", mime="text/csv")
        st.download_button("CSV всего февраля", sub.to_csv(index=False).encode("utf-8"), icon=":material/download:",
                           file_name="forecast_feb2026.csv", mime="text/csv", help="28 выпусков × 48 часов")
    if recalc:
        t0 = time.time()
        live, _ = forecast.forecast_issue(d, load_model())
        diff = float(np.abs(live["p_farm"].to_numpy() - f["p_farm"].to_numpy()).max())
        st.info(f"Пересчитано за {time.time() - t0:.1f} с. Максимальное расхождение с сохранённым прогнозом: "
                f"{diff * 100:.2f} п.п. ({'совпадает' if diff < 1e-3 else 'отличается'}).")

    with st.expander("Таблица: все 48 часов"):
        t = f[["target_time_local", "lead_day", "p_farm", "p_farm_q10", "p_farm_q90", "p_t1", "p_t2"]].rename(columns={
            "target_time_local": "час (UTC+6)", "lead_day": "сутки", "p_farm": "ВЭС", "p_farm_q10": "P10",
            "p_farm_q90": "P90", "p_t1": "турбина 1", "p_t2": "турбина 2"})
        st.dataframe(t, hide_index=True, use_container_width=True,
                     column_config={"час (UTC+6)": st.column_config.DatetimeColumn(format="DD.MM HH:mm")})


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
    kpi_tiles(f, extra={"label": "Ошибка этого выпуска", "value": f"{mae * 100:.0f} п.п.",
                        "help": f"Средняя абсолютная ошибка по 48 часам (MAE = {mae:.3f})"})
    plot(charts.forecast_chart(f, fact=g["p_farm"].reset_index(drop=True)))


def page_weather():
    st.title("Погода на момент выпуска")
    st.caption("Модель видит только те прогнозы погоды, которые уже были опубликованы в момент выпуска — "
               "как в реальной работе диспетчера.")
    d = issue_picker("issue_weather")
    w = load_weather(f"{d:%Y-%m-%d}")
    show_all = st.toggle("Показать все погодные модели", value=False,
                         help="ECMWF, GFS, ECMWF 0.25° и ICON по отдельности")
    plot(charts.weather_chart(w, show_all=show_all))
    spread = float(w[[c for c in w.columns if c.endswith("__wind_speed_100m")]].std(axis=1).mean())
    st.caption(f"Голубой коридор — разброс четырёх моделей (в среднем ±{spread:.1f} м/с). Чем он шире, "
               f"тем неувереннее прогноз — это учитывается в интервале P10–P90.")

    m = load_manifest(f"{d:%Y-%m-%d}")
    st.markdown(leakage_note(m), unsafe_allow_html=True)
    as_of = pd.Timestamp(m["as_of_utc"])
    rows = {}
    for e in m["sources"]:
        if not e.get("model") or not e.get("max_published_at"):
            continue
        key = e["model"]
        pub = pd.Timestamp(e["max_published_at"])
        if key not in rows or pub > rows[key]["pub"]:
            rows[key] = {"pub": pub, "source": e["source"]}
    names = {"ecmwf_ifs": "ECMWF IFS 9 км", "ecmwf_ifs025": "ECMWF IFS 0.25°", "gfs_seamless": "GFS", "icon_seamless": "ICON"}
    table = pd.DataFrame([
        {
            "модель": names.get(k, k),
            "API Open-Meteo": "Single Runs" if v["source"] == "single" else "Previous Runs",
            "свежайший прогон опубликован (UTC+6)": v["pub"] + pd.Timedelta(hours=OFFSET),
            "запас до момента прогноза": f"{(as_of - v['pub']) / pd.Timedelta('1h'):.0f} ч",
            "проверка": "✓" if v["pub"] <= as_of else "✗",
        }
        for k, v in rows.items()
    ])
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"свежайший прогон опубликован (UTC+6)": st.column_config.DatetimeColumn(format="DD.MM.YYYY HH:mm")})
    with st.expander("Таблица: ветер на 100 м по часам"):
        t = w[["target_time_local"] + [c for c in w.columns if c.endswith("__wind_speed_100m")]]
        st.dataframe(t.rename(columns=lambda c: c.replace("__wind_speed_100m", ", м/с").upper() if "__" in c else "час (UTC+6)"),
                     hide_index=True, use_container_width=True)


def page_quality():
    st.title("Качество прогноза")
    st.caption("Бэктест по тому же протоколу, что и тест: ежедневный выпуск в 14:00, прогноз на 48 ч, "
               "модель обучена только на прошлом. Три контрольных периода, ~4 300 часов.")
    bt = load_backtest()
    table = bt.groupby(["model", "period"]).apply(
        lambda g: metrics.mae(g["p_farm"], g["p_farm_pred"]), include_groups=False
    ).unstack()[list(dict.fromkeys(bt["period"]))]
    table = table.loc[[m for m in ("ensemble", "phys_ifs", "clim") if m in table.index]]
    table.columns = [PERIOD_NAMES.get(c, c) for c in table.columns]

    ens, ref, clim = (table.loc[m].mean() for m in ("ensemble", "phys_ifs", "clim"))
    with st.container(key="kpi"):
        c = st.columns(3)
        c[0].metric("Средняя ошибка (MAE)", f"{ens * 100:.1f} п.п.", help=f"MAE = {ens:.3f} доли номинала")
        c[1].metric("Лучше физической кривой", f"{(1 - ens / ref) * 100:.0f}%")
        c[2].metric("Лучше климатологии", f"{(1 - ens / clim) * 100:.0f}%")
    st.markdown(
        f"**Как читать.** В среднем прогноз ошибается на **{ens * 100:.0f}% номинальной мощности** в час. "
        f"Простой способ «прогноз ветра ECMWF → кривая мощности турбин» ошибается на {ref * 100:.0f}%, "
        f"а «средняя выработка для этого месяца и часа» — на {clim * 100:.0f}%."
    )

    plot(charts.mae_by_model_chart(table))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Прогноз против факта")
        plot(charts.forecast_vs_fact_chart(bt))
        st.caption("Каждая точка — один час. Чем ближе к диагонали, тем точнее. Тёмная линия — средний прогноз "
                   "для каждого уровня факта: при слабом ветре модель немного завышает, на номинале — занижает.")
    with c2:
        st.subheader("Распределение ошибок")
        plot(charts.error_hist_chart(bt))
        st.caption("Ошибка = прогноз − факт. Большие ошибки — это в основном сдвиг по времени начала или "
                   "окончания ветра: погодная модель угадала событие, но не час.")
    st.subheader("Ошибка по горизонту прогноза")
    plot(charts.error_by_horizon_chart(bt))
    with st.expander("Таблица MAE"):
        st.dataframe(table.rename(index=theme.MODEL_NAMES).round(3), use_container_width=True)
    st.info("**Почему не точнее?** Если подставить в модель фактический ветер на турбине, ошибка была бы всего "
            "3–4% номинала. Прогноз ветра на сутки вперёд ошибается в среднем на 2–2.5 м/с — именно это, "
            "а не модель, ограничивает точность.")


def client_id() -> str:
    """Адрес посетителя (за Caddy — из X-Forwarded-For) для дневных лимитов."""
    try:
        fwd = st.context.headers.get("X-Forwarded-For", "")
        return fwd.split(",")[0].strip() or (st.context.ip_address or "local")
    except Exception:
        return "local"


# --- Чат (плавающий виджет) ------------------------------------------------------------------

SUGGESTIONS = [
    "Когда завтра пик выработки?",
    "Насколько уверенный прогноз и почему?",
    "Будет ли штиль в ближайшие двое суток?",
]


@st.fragment
def chat_panel() -> None:
    with st.container(key="chat_panel"):
        dates = data.test_issue_dates()
        d = pd.Timestamp(st.session_state.get("issue_date", dates[9]))
        st.markdown("**AI-помощник**")
        st.caption(f"Выпуск {fmt_day(d)}, 14:00 — прогноз на {fmt_day(d + pd.Timedelta(days=1))} "
                   f"и {fmt_day(d + pd.Timedelta(days=2))}. Дату можно сменить на любой странице.")
        run = load_agent_run(d)
        if run is None:
            st.warning("Для этой даты нет журнала агента.")
            return
        if not av.llm_configured():
            st.info("Чат заработает после подключения ключа OpenAI на сервере. "
                    "Пока объяснение прогноза — на странице «AI-агент».")
            return
        key = f"chat_{d:%Y-%m-%d}"
        history = st.session_state.setdefault(key, [])
        q = None
        if history:
            with st.container(height=300, border=False, autoscroll=True):
                for m in history:
                    avatar = ":material/person:" if m["role"] == "user" else str(ASSETS / "symbol.svg")
                    st.chat_message(m["role"], avatar=avatar).write(m["content"])
        else:
            st.caption("Например:")
            for i, sug in enumerate(SUGGESTIONS):
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    q = sug
        typed = st.chat_input("Ваш вопрос о прогнозе…", max_chars=400, key="chat_input_fab")
        q = typed or q
        if q:
            allowed, msg = av.take_quota("chat", client_id())
            if not allowed:
                answer = f"Лимит чата: {msg}."
            else:
                try:
                    with st.spinner("Думаю…"):
                        answer = av.chat_answer(q, run, history)
                except Exception as e:  # сеть или API — не роняем страницу
                    answer = f"Не удалось получить ответ ({type(e).__name__})."
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
            st.rerun(scope="fragment")


def chat_widget() -> None:
    with st.container(key="chat_fab"):
        with st.popover("", icon=":material/chat:"):
            chat_panel()


# --- AI-агент --------------------------------------------------------------------------------


def render_run(run: dict) -> None:
    last = run["versions"][-1]
    st.subheader("Объяснение для диспетчера")
    st.info(last["explanation"])
    if len(run["versions"]) > 1:
        with st.expander(f"Объяснение версии v1 (на момент выпуска, {run['versions'][0]['as_of_local']})"):
            st.write(run["versions"][0]["explanation"])
    st.caption(f"Итог агента: {run['summary']}")

    st.subheader("Версии прогноза")
    plot(charts.versions_chart(run["versions"]))
    st.caption("v1 построена строго по данным на 14:00 — это прогноз для сдачи. Ревизия выпускается, только если "
               "новый прогон ECMWF заметно меняет прогноз. Журналы посчитаны на сервере (x86), файл сдачи — на ARM: "
               "в отдельных часах возможны расхождения меньше 1 п.п. из-за округления на разных процессорах.")

    with st.expander(f"Все шаги агента подробно ({len(run['steps'])})"):
        for step in run["steps"]:
            icon, title = av.STEP_TITLES.get(step["tool"], ("•", step["tool"]))
            as_of_local = pd.Timestamp(step["as_of_utc"]) + pd.Timedelta(hours=OFFSET)
            st.markdown(f"{icon} **{step['step']}. {title}** · данные на {as_of_local:%d.%m %H:%M} — "
                        f"{av.step_summary(step)}")
            if step.get("reason") and step["tool"] != "decision":
                st.caption(f"Почему: {step['reason']}")
            if step.get("result") is not None:
                with st.popover("Данные шага"):
                    st.json(step["result"], expanded=True)


def page_agent():
    st.title("AI-агент")
    st.caption("Агент сам проходит цикл: погода → проверка данных → модель → анализ → публикация → "
               "пересчёт при выходе нового прогона. Числа считают инструменты, агент принимает решения и объясняет.")
    d = issue_picker("issue_agent")
    key = f"live_run_{d:%Y-%m-%d}"

    llm = av.llm_configured()
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        mode = st.segmented_control("Режим", ["LLM (OpenAI)", "Правила"], default="LLM (OpenAI)" if llm else "Правила",
                                    label_visibility="collapsed", disabled=not llm)
        start = st.button("Запустить агента сейчас", type="primary", icon=":material/play_arrow:")
        st.caption("Повторяет весь цикл заново на архивных данных этой даты: правила ~3 с, LLM ~30 с.")

    scheme = st.empty()
    if start:
        use_llm = llm and (mode or "").startswith("LLM")
        allowed, msg = (av.take_quota("agent_llm", client_id()) if use_llm else (True, ""))
        if not allowed:
            st.warning(f"LLM-режим недоступен: {msg}. Запускаю по правилам.")
            use_llm = False
        from windagent.agent.orchestrator import run_agent

        with st.spinner("Агент работает…"):
            r = run_agent(d, load_model(), mode="llm" if use_llm else "rules", out_dir=av.run_dir(d, live=True),
                          on_step=lambda steps: scheme.html(av.stepper_html(av.pipeline_stages(steps), running=True)))
        st.session_state[key] = True
        st.toast(f"Готово: {r['versions']} верс. за {r['steps']} шагов ({av.MODE_NAMES.get(r['mode'], r['mode'])})",
                 icon=":material/check_circle:")

    run = load_agent_run(d)
    if run is None:
        st.warning("Для этой даты нет журнала агента — нажмите «Запустить агента сейчас».")
        return
    scheme.html(av.stepper_html(av.pipeline_stages(run["steps"])))
    source = "только что выполненный запуск" if st.session_state.get(key) else "сохранённый запуск из репозитория"
    st.caption(f"Режим: {av.MODE_NAMES.get(run['mode'], run['mode'])} · версий: {len(run['versions'])} · "
               f"шагов: {len(run['steps'])} · {source}")
    render_run(run)
    st.caption("Вопросы о прогнозе можно задать в чате — круглая кнопка внизу справа.")


# --- Свои данные -------------------------------------------------------------------------------


def page_upload():
    st.title("Проверка на ваших данных")
    st.caption("Загрузите фактическую выработку — сравним её с нашими прогнозами. Файлы обрабатываются "
               "в памяти и нигде не сохраняются.")
    with st.expander("Какие файлы подходят", expanded=False):
        st.markdown(
            "- **Формат датасета организаторов** — CSV с колонками: ID, время, скорость ветра, нормализованная "
            "мощность, температура (10-минутные записи). Можно загрузить сразу два файла (турбины 1 и 2) — "
            "посчитаем среднее по ВЭС.\n"
            "- **Простой CSV** — колонка времени и колонка мощности (доля номинала 0–1 или проценты 0–100); "
            "любая частота, приводится к часу.\n\n"
            "Сравнение возможно для периодов наших прогнозов: **февраль 2026** (тестовый прогноз с интервалом "
            "P10–P90) и контрольные периоды бэктеста — февраль 2025, декабрь 2025, январь 2026."
        )
    c1, c2 = st.columns([3, 1])
    files = c1.file_uploader("CSV-файлы (до 2 шт.)", type=["csv", "txt"], accept_multiple_files=True)
    tz = c2.selectbox("Часовой пояс меток", [6, 5, 0], format_func=lambda h: f"UTC+{h}" if h else "UTC",
                      help="В датасете организаторов — UTC+6")
    use_sample = c2.button("Попробовать на примере", icon=":material/science:",
                           help="Январь 2026 из датасета (турбина 1)")

    payloads = []
    if use_sample:
        payloads = [("пример: январь 2026, турбина 1", data_sample())]
        st.session_state["upload_sample"] = True
    elif files:
        payloads = [(f.name, f.getvalue()) for f in files[:2]]
        st.session_state["upload_sample"] = False
    elif st.session_state.get("upload_sample"):
        payloads = [("пример: январь 2026, турбина 1", data_sample())]
    if not payloads:
        return

    from windagent.ui import upload

    try:
        parsed = [upload.parse_upload(raw, utc_offset_hours=tz) for _, raw in payloads]
        fact = upload.combine([p for p, _ in parsed])
        res = upload.evaluate(fact, load_catalog())
    except upload.UploadError as e:
        st.error(f"Не удалось обработать файл: {e}")
        return
    for (name, _), (_, kind) in zip(payloads, parsed):
        st.caption(f"📄 {name} — {kind}")
    p0, p1 = res["period"]
    st.success(f"Сопоставлено {res['hours']} ч ({p0:%d.%m.%Y} – {p1:%d.%m.%Y}) с источником: {', '.join(res['sources'])}.")

    with st.container(key="kpi"):
        cols = st.columns(4)
        for i, lead in enumerate((1, 2)):
            sc = res["by_lead"].get(lead)
            if sc:
                cols[i].metric(f"Ошибка, {'на сутки вперёд' if lead == 1 else 'на двое суток'}",
                               f"{sc['MAE'] * 100:.1f} п.п.",
                               help=f"MAE {sc['MAE']:.3f}, RMSE {sc['RMSE']:.3f}, смещение {sc['bias']:+.3f}; {sc['n']} ч")
        sc1 = res["by_lead"].get(1, {})
        if "coverage_p10_p90" in sc1:
            cols[2].metric("Попадание в P10–P90", f"{sc1['coverage_p10_p90']:.0%}", help="Номинал интервала — 80%")
        if sc1:
            cols[3].metric("Смещение", f"{sc1['bias'] * 100:+.1f} п.п.",
                           help="Положительное — прогноз на сутки вперёд в среднем выше факта")

    plot(charts.evaluation_chart(res["merged"]))
    if len(res["daily_mae"]) > 1:
        st.subheader("Ошибка по дням")
        plot(charts.daily_mae_chart(res["daily_mae"]))
    m = res["merged"]
    with st.expander("Таблица сопоставления"):
        st.dataframe(m[["target_time_local", "lead_day", "p_farm", "fact", "source"]].rename(columns={
            "target_time_local": "час (UTC+6)", "lead_day": "сутки", "p_farm": "прогноз", "fact": "факт",
            "source": "источник"}), hide_index=True, use_container_width=True)
    st.download_button("Скачать сопоставление (CSV)", m.to_csv(index=False).encode("utf-8"),
                       icon=":material/download:", file_name="forecast_vs_fact.csv", mime="text/csv")


@st.cache_data(show_spinner=False)
def data_sample() -> bytes:
    from windagent.ui import upload

    return upload.sample_file()


@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    from windagent.ui import upload

    return upload.forecast_catalog()


# --- О системе -----------------------------------------------------------------------------------


def page_about():
    st.title("О системе")
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
5. **Агент.** Проверяет погоду и результат, пишет объяснение, публикует версию v1, а после выхода нового прогона
   ECMWF пересчитывает прогноз и публикует ревизию, если изменение существенное (экран «AI-агент»).

**Честность ретроспективы.** Все данные проходят через `DataStore(as_of)`: он физически не отдаёт прогнозы погоды,
опубликованные после момента выпуска. Это проверяется автотестами для всех 28 выпусков февраля.

**Данные.** SCADA двух турбин с 11.03.2023 по 31.01.2026 (10-минутные записи); архив прогнозов погоды
[Open-Meteo](https://open-meteo.com/) с 15.03.2024.

Модель: `{load_model().version}`.
""")


PAGE_AGENT = st.Page(page_agent, title="AI-агент", icon=":material/smart_toy:", url_path="agent")
pages = [
    st.Page(page_forecast, title="Прогноз", icon=":material/show_chart:", default=True),
    PAGE_AGENT,
    st.Page(page_weather, title="Погода", icon=":material/air:", url_path="weather"),
    st.Page(page_quality, title="Качество", icon=":material/insights:", url_path="quality"),
    st.Page(page_upload, title="Свои данные", icon=":material/upload_file:", url_path="upload"),
    st.Page(page_about, title="О системе", icon=":material/info:", url_path="about"),
]
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "symbol.svg"), size="large")
inject_css()
chat_widget()
with st.sidebar:
    st.caption("Agentic AI прогноз выработки ВЭС · Steppe Overflow")
    status_block()
    st.html(PWA_HTML, unsafe_allow_javascript=True)
st.navigation(pages).run()
