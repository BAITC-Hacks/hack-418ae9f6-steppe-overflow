"""Инструменты агента. Каждый инструмент — детерминированная функция над состоянием выпуска.

Все числа считают инструменты; агент (правила или LLM) только вызывает их,
принимает решения и формулирует объяснение. Результат каждого инструмента —
компактный JSON-совместимый словарь.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from windagent import protocol
from windagent.config import load_settings, resolve
from windagent.data import scada
from windagent.data.store import DataStore
from windagent.features import build
from windagent.forecast import OUTPUT_COLUMNS, ProductionModel

SOURCES = {"ifs": "ecmwf_ifs", "ifs025": "ecmwf_ifs025", "gfs": "gfs_seamless", "icon": "icon_seamless"}
SOURCE_NAMES = {"ifs": "ECMWF IFS 9 км", "ifs025": "ECMWF IFS 0.25°", "gfs": "GFS", "icon": "ICON"}
AGENT_DIR = "artifacts/agent"

# Пороги решений (одинаковые для правил и для подсказок LLM)
SUSPICIOUS_COVERAGE = 0.9        # прогноз покрывает меньше 90 % часов
SUSPICIOUS_DEVIATION_MS = 4.0    # среднее отклонение от медианы моделей, м/с
REVISION_MEAN_DIFF = 0.03        # ревизия, если средний сдвиг прогноза больше 3 п.п.
REVISION_MAX_DIFF = 0.15         # …или хотя бы один час сдвинулся больше чем на 15 п.п.


def _ts(x) -> str | None:
    return None if x is None or pd.isna(x) else str(pd.Timestamp(x))


def _local(x, offset: int) -> str:
    return f"{pd.Timestamp(x) + pd.Timedelta(hours=offset):%d.%m %H:%M}"


@dataclass
class AgentState:
    issue_date: pd.Timestamp
    model: ProductionModel
    settings: dict = field(default_factory=load_settings)
    out_dir: Path | None = None
    as_of: pd.Timestamp | None = None
    store: DataStore | None = None
    X: pd.DataFrame | None = None
    forecast: pd.DataFrame | None = None
    excluded: list[str] = field(default_factory=list)
    versions: list[dict] = field(default_factory=list)
    last_analysis: dict | None = None

    def __post_init__(self):
        self.issue_date = pd.Timestamp(self.issue_date).normalize()
        self.as_of = self.as_of or protocol.issue_time_utc(self.issue_date, self.settings)
        self.out_dir = self.out_dir or resolve(AGENT_DIR) / f"{self.issue_date:%Y-%m-%d}"

    @property
    def offset(self) -> int:
        return self.settings["scada"]["utc_offset_hours"]

    @property
    def targets(self) -> pd.DatetimeIndex:
        return protocol.target_hours_utc(self.issue_date, self.settings)


# --- Инструменты ----------------------------------------------------------------------------


def check_available_runs(state: AgentState) -> dict:
    """Какие прогоны погоды уже опубликованы на текущий момент (as_of)."""
    w = state.settings["weather"]
    store = DataStore(state.as_of, settings=state.settings)
    out = []
    for model in w["single_runs"]:
        run = store.latest_run(model)
        pub = store.published_at(model, run) if run is not None else None
        out.append({"source": "ifs", "model": model, "latest_run_utc": _ts(run), "published_utc": _ts(pub),
                    "age_h": None if run is None else round((state.as_of - run) / pd.Timedelta("1h"), 1)})
    for model, cfg in w["previous_runs"].items():
        lag = pd.Timedelta(hours=cfg["publish_lag_h"])
        run = (state.as_of - lag).floor("6h")
        src = next(k for k, v in SOURCES.items() if v == model)
        out.append({"source": src, "model": model, "latest_run_utc": _ts(run), "published_utc": _ts(run + lag),
                     "age_h": round((state.as_of - run) / pd.Timedelta("1h"), 1)})
    return {"as_of_utc": _ts(state.as_of), "as_of_local": _local(state.as_of, state.offset), "runs": out}


def fetch_weather(state: AgentState) -> dict:
    """Собирает прогнозы всех погодных моделей на 48 целевых часов (только опубликованные к as_of)."""
    state.store = DataStore(state.as_of, settings=state.settings)
    state.X = build.issue_frame(state.issue_date, store=state.store, settings=state.settings)
    state.forecast = None
    cov = {}
    for pre in SOURCES:
        col = f"{pre}__wind_speed_100m"
        cov[pre] = round(float(state.X[col].notna().mean()), 3) if col in state.X else 0.0
    ifs_runs = pd.to_datetime(state.X.get("ifs__run_time")).dropna().unique()
    return {
        "hours": int(len(state.X)),
        "coverage_ws100": cov,
        "ifs_runs_used_utc": [str(r) for r in sorted(ifs_runs)],
        "all_published_before_as_of": all(
            e.get("max_published_at") is None or pd.Timestamp(e["max_published_at"]) <= state.as_of
            for e in state.store.manifest
        ),
    }


def validate_weather(state: AgentState) -> dict:
    """Проверка качества прогнозов погоды: покрытие, физичность, согласие моделей."""
    X = _need(state.X, "fetch_weather")
    ws = pd.DataFrame({pre: X.get(f"{pre}__wind_speed_100m") for pre in SOURCES}).astype(float)
    median = ws.median(axis=1)
    report, suspicious = {}, []
    for pre in SOURCES:
        s = ws[pre]
        cov = float(s.notna().mean())
        bad_range = int(((s < 0) | (s > 40)).sum())
        dev = float((s - median).abs().mean()) if s.notna().any() else float("nan")
        issues = []
        if cov < SUSPICIOUS_COVERAGE:
            issues.append(f"покрытие {cov:.0%}")
        if bad_range:
            issues.append(f"{bad_range} нефизичных значений")
        if dev > SUSPICIOUS_DEVIATION_MS:
            issues.append(f"сильно отличается от остальных моделей ({dev:.1f} м/с)")
        report[pre] = {"coverage": round(cov, 3), "mean_ws100": round(float(s.mean()), 2),
                       "deviation_from_median_ms": round(dev, 2), "issues": issues}
        if issues:
            suspicious.append(pre)
    spread = float(ws.std(axis=1).mean())
    return {
        "sources": report,
        "suspicious_sources": suspicious,
        "ensemble_spread_ms": round(spread, 2),
        "agreement": "хорошее" if spread < 1.5 else "умеренное" if spread < 3 else "слабое",
        "recommendation": (f"исключить из расчёта: {', '.join(suspicious)}" if suspicious
                           else "все источники пригодны"),
    }


def run_forecast(state: AgentState, exclude_sources: list[str] | None = None) -> dict:
    """Запуск модели. exclude_sources — погодные модели, которые нужно игнорировать."""
    X = _need(state.X, "fetch_weather").copy()
    exclude = [s for s in (exclude_sources or []) if s in SOURCES]
    for pre in exclude:
        cols = [c for c in X.columns if c.startswith(f"{pre}__") and not c.endswith(("run_time", "day"))]
        X[cols] = np.nan
    pred = state.model.predict(X)
    f = X[build.META].copy()
    for c in ("p_t1", "p_t2", "p_farm", "p_farm_q10", "p_farm_q90"):
        f[c] = pred[c].to_numpy().round(4)
    f["ifs_run_time"] = X.get("ifs__run_time")
    f["model_version"] = state.model.version
    state.forecast = f[OUTPUT_COLUMNS]
    state.excluded = exclude
    d1, d2 = f[f["lead_day"] == 1], f[f["lead_day"] == 2]
    return {
        "excluded_sources": exclude,
        "mean_p_farm_d1": round(float(d1["p_farm"].mean()), 3),
        "mean_p_farm_d2": round(float(d2["p_farm"].mean()), 3),
        "equivalent_full_load_hours_d1": round(float(d1["p_farm"].sum()), 1),
        "mean_interval_width": round(float((f["p_farm_q90"] - f["p_farm_q10"]).mean()), 3),
    }


def analyze_forecast(state: AgentState) -> dict:
    """Анализ результата: правдоподобие, согласованность с погодой, сравнение с климатом и прошлым выпуском."""
    f = _need(state.forecast, "run_forecast")
    X = state.X
    flags = []
    p = f["p_farm"]
    if p.isna().any() or (p < 0).any() or (p > 1).any():
        flags.append("значения вне диапазона 0–1 или пропуски")
    jump = float(p.diff().abs().max())
    if jump > 0.6:
        flags.append(f"резкий скачок мощности за час: {jump:.2f}")

    ens = X[[f"{s}__wind_speed_100m" for s in SOURCES if f"{s}__wind_speed_100m" in X]].median(axis=1)
    inconsistent = int(((ens > 13) & (p < 0.3)).sum() + ((ens < 2.5) & (p > 0.3)).sum())
    if inconsistent:
        flags.append(f"{inconsistent} ч, где мощность не согласуется с прогнозом ветра")

    # Климат: средняя мощность этого месяца и часа по истории до момента выпуска
    h = scada.load_hourly(state.settings)
    h = h[h.index + pd.Timedelta("1h") <= state.as_of]
    loc = h.index + pd.Timedelta(hours=state.offset)
    clim = h["p_farm"].groupby([loc.month, loc.hour]).mean()
    tl = pd.DatetimeIndex(f["target_time_local"])
    clim_vals = clim.reindex(pd.MultiIndex.from_arrays([tl.month, tl.hour])).to_numpy()
    clim_mean = float(np.nanmean(clim_vals))

    d1 = f[f["lead_day"] == 1]
    peak = d1.loc[d1["p_farm"].idxmax()]
    width = float((f["p_farm_q90"] - f["p_farm_q10"]).mean())
    spread = float(X[[f"{s}__wind_speed_100m" for s in SOURCES]].std(axis=1).mean())
    confidence = "высокая" if width < 0.35 and spread < 1.5 else "низкая" if width > 0.55 or spread > 3 else "средняя"

    prev = _previous_forecast(state)
    change = None
    if prev is not None:
        m = f.merge(prev[["target_time_utc", "p_farm"]], on="target_time_utc", suffixes=("", "_prev"))
        if len(m):
            change = {"hours_compared": int(len(m)),
                      "mean_abs_change": round(float((m["p_farm"] - m["p_farm_prev"]).abs().mean()), 3),
                      "mean_shift": round(float((m["p_farm"] - m["p_farm_prev"]).mean()), 3)}

    calm = d1.loc[d1["p_farm"] <= 0.05, "target_time_local"]
    result = {
        "flags": flags,
        "confidence": confidence,
        "mean_p_farm_d1": round(float(d1["p_farm"].mean()), 3),
        "peak_d1": {"value": round(float(peak["p_farm"]), 3), "time_local": f"{pd.Timestamp(peak['target_time_local']):%d.%m %H:%M}"},
        "calm_hours_d1": int(len(calm)),
        "high_hours_d1": int((d1["p_farm"] >= 0.8).sum()),
        "climatology_mean": round(clim_mean, 3),
        "vs_climatology": round(float(p.mean()) - clim_mean, 3),
        "mean_interval_width": round(width, 3),
        "ensemble_spread_ms": round(spread, 2),
        "change_vs_previous_issue": change,
        "mean_wind_ifs_d1_ms": round(float(X.loc[X["lead_day"] == 1, "ifs__wind_speed_100m"].mean()), 1),
    }
    state.last_analysis = result
    return result


def publish_forecast(state: AgentState, explanation: str) -> dict:
    """Публикует текущий прогноз как новую версию: CSV + манифест + объяснение."""
    f = _need(state.forecast, "run_forecast")
    version = len(state.versions) + 1
    state.out_dir.mkdir(parents=True, exist_ok=True)
    path = state.out_dir / f"forecast_v{version}.csv"
    f.to_csv(path, index=False)
    record = {
        "version": version,
        "as_of_utc": _ts(state.as_of),
        "as_of_local": _local(state.as_of, state.offset),
        "excluded_sources": state.excluded,
        "explanation": explanation,
        "file": str(path.relative_to(resolve(".")) if path.is_relative_to(resolve(".")) else path),
        "sources": state.store.manifest if state.store else [],
        "forecast": f,
    }
    state.versions.append(record)
    return {"version": version, "file": record["file"], "as_of_local": record["as_of_local"]}


def advance_to_next_update(state: AgentState, max_hours: int = 12) -> dict:
    """Переводит «часы» вперёд до публикации следующего прогона ECMWF (но не позже начала прогнозного окна).

    Нужен, чтобы пересчитать прогноз при обновлении входных данных.
    """
    model = next(iter(state.settings["weather"]["single_runs"]))
    lag = pd.Timedelta(hours=state.settings["weather"]["single_runs"][model]["publish_lag_h"])
    limit = min(state.as_of + pd.Timedelta(hours=max_hours), state.targets[0])
    store_future = DataStore(limit, settings=state.settings)
    newer = [r for r in store_future.available_runs(model) if r + lag > state.as_of]
    if not newer:
        return {"new_run_available": False, "as_of_utc": _ts(state.as_of),
                "reason": f"до {_local(limit, state.offset)} новых прогонов нет"}
    run = newer[0]
    old = state.as_of
    state.as_of = run + lag
    state.X, state.forecast, state.store = None, None, None
    return {"new_run_available": True, "model": model, "run_utc": _ts(run),
            "previous_as_of_local": _local(old, state.offset), "new_as_of_local": _local(state.as_of, state.offset),
            "new_as_of_utc": _ts(state.as_of)}


def compare_with_published(state: AgentState) -> dict:
    """Насколько текущий прогноз отличается от последней опубликованной версии."""
    f = _need(state.forecast, "run_forecast")
    if not state.versions:
        return {"has_published": False}
    last = state.versions[-1]["forecast"]
    d = (f["p_farm"].to_numpy() - last["p_farm"].to_numpy())
    mean_abs, max_abs = float(np.abs(d).mean()), float(np.abs(d).max())
    significant = mean_abs > REVISION_MEAN_DIFF or max_abs > REVISION_MAX_DIFF
    i = int(np.abs(d).argmax())
    return {
        "has_published": True,
        "compared_with_version": state.versions[-1]["version"],
        "mean_abs_change": round(mean_abs, 3),
        "max_abs_change": round(max_abs, 3),
        "max_change_at_local": f"{pd.Timestamp(f['target_time_local'].iloc[i]):%d.%m %H:%M}",
        "mean_shift": round(float(d.mean()), 3),
        "significant": significant,
        "recommendation": "опубликовать ревизию" if significant else "изменения незначительны — ревизия не нужна",
    }


# --- Служебное ------------------------------------------------------------------------------


class ToolError(RuntimeError):
    pass


def _need(x, tool: str):
    if x is None:
        raise ToolError(f"сначала нужно вызвать {tool}")
    return x


def _previous_forecast(state: AgentState) -> pd.DataFrame | None:
    """Прогноз предыдущего выпуска (для сравнения перекрывающихся часов)."""
    prev_date = state.issue_date - pd.Timedelta(days=1)
    agent_dir = resolve(AGENT_DIR) / f"{prev_date:%Y-%m-%d}"
    files = sorted(agent_dir.glob("forecast_v*.csv"))
    if files:
        return pd.read_csv(files[0], parse_dates=["target_time_utc"])
    sub = resolve("artifacts/submission/forecast_feb2026.csv")
    if sub.exists():
        s = pd.read_csv(sub, parse_dates=["issue_date", "target_time_utc"])
        s = s[s["issue_date"] == prev_date]
        return s if len(s) else None
    return None


# Реестр для оркестраторов: имя → (функция, описание, JSON-схема аргументов)
TOOLS = {
    "check_available_runs": (check_available_runs,
                             "Показывает, какие прогоны погодных моделей уже опубликованы на текущий момент.", {}),
    "fetch_weather": (fetch_weather,
                      "Собирает прогнозы погоды всех моделей на 48 целевых часов (только опубликованные к текущему моменту).", {}),
    "validate_weather": (validate_weather,
                         "Проверяет качество прогнозов погоды: покрытие, физичность значений, согласие между моделями.", {}),
    "run_forecast": (run_forecast,
                     "Запускает модель прогноза мощности. Можно исключить подозрительные погодные модели.",
                     {"exclude_sources": {"type": "array", "items": {"type": "string", "enum": list(SOURCES)},
                                          "description": "погодные модели, которые не использовать (обычно пусто)"}}),
    "analyze_forecast": (analyze_forecast,
                         "Анализирует прогноз: правдоподобие, согласованность с ветром, сравнение с климатом и прошлым выпуском, уверенность.", {}),
    "publish_forecast": (publish_forecast,
                         "Публикует текущий прогноз как новую версию с объяснением для диспетчера.",
                         {"explanation": {"type": "string", "description": "объяснение прогноза на русском, 3–6 предложений, только факты из результатов инструментов"}}),
    "advance_to_next_update": (advance_to_next_update,
                               "Переводит время вперёд до публикации следующего прогона ECMWF, чтобы пересчитать прогноз на новых данных.", {}),
    "compare_with_published": (compare_with_published,
                               "Сравнивает текущий прогноз с последней опубликованной версией и советует, нужна ли ревизия.", {}),
}


def call_tool(state: AgentState, name: str, args: dict | None = None) -> dict:
    fn = TOOLS[name][0]
    return fn(state, **(args or {}))


def save_run(state: AgentState, steps: list[dict], mode: str, summary: str) -> Path:
    """Журнал запуска агента: шаги, решения, версии прогноза."""
    state.out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "issue_date": f"{state.issue_date:%Y-%m-%d}",
        "mode": mode,
        "model_version": state.model.version,
        "created_unix": int(time.time()),
        "summary": summary,
        "steps": steps,
        "versions": [{k: v for k, v in ver.items() if k != "forecast"} for ver in state.versions],
    }
    path = state.out_dir / "run.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return path
