"""Загрузка данных для веб-интерфейса (без зависимостей от Streamlit)."""

from __future__ import annotations

import json

import pandas as pd

from windagent import protocol
from windagent.config import load_settings, resolve
from windagent.data import scada
from windagent.data.store import DataStore
from windagent.features import build

SUB = "artifacts/submission"


def test_issue_dates() -> list[pd.Timestamp]:
    f = load_settings()["forecast"]
    return protocol.issue_dates(f["test_first_issue"], f["test_last_issue"])


def submission() -> pd.DataFrame:
    return pd.read_csv(
        resolve(SUB) / "forecast_feb2026.csv",
        parse_dates=["issue_date", "issue_time_utc", "target_time_utc", "target_time_local", "ifs_run_time"],
    )


def manifest(issue_date) -> dict:
    path = resolve(SUB) / "manifests" / f"{pd.Timestamp(issue_date):%Y-%m-%d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def backtest_preds() -> pd.DataFrame:
    return pd.read_parquet(resolve("artifacts/reports/backtest_preds.parquet"))


def backtest_issue_dates() -> dict[str, list[pd.Timestamp]]:
    p = backtest_preds()
    return {per: sorted(g["issue_date"].unique()) for per, g in p.groupby("period", sort=False)}


def weather_for_issue(issue_date) -> pd.DataFrame:
    """Прогнозы ветра на 100 м всех погодных моделей, доступные на момент выпуска."""
    s = load_settings()
    store = DataStore(protocol.issue_time_utc(issue_date, s), settings=s)
    X = build.issue_frame(issue_date, store=store, settings=s)
    cols = ["target_time_local", "lead_day"] + [
        c for c in X.columns
        if c.endswith("__wind_speed_100m") or c in ("ifs__lag_ws100_std", "ifs__temperature_2m", "ifs__wind_direction_100m")
    ]
    return X[cols]


def scada_hourly() -> pd.DataFrame:
    h = scada.load_hourly()
    h = h.copy()
    h["ts_local"] = h.index + pd.Timedelta(hours=load_settings()["scada"]["utc_offset_hours"])
    return h


def summarize(f: pd.DataFrame) -> dict:
    """Ключевые числа прогноза на 48 ч для плиток."""
    d1 = f[f["lead_day"] == 1]
    peak = d1.loc[d1["p_farm"].idxmax()]
    return {
        "mean_d1": float(d1["p_farm"].mean()),
        "full_load_hours_d1": float(d1["p_farm"].sum()),  # сумма нормированной мощности = часы при номинале
        "peak_value": float(peak["p_farm"]),
        "peak_time": pd.Timestamp(peak["target_time_local"]),
        "mean_width": float((f["p_farm_q90"] - f["p_farm_q10"]).mean()) if "p_farm_q90" in f else None,
        "high_hours": int((d1["p_farm"] >= 0.8).sum()),
        "calm_hours": int((d1["p_farm"] <= 0.05).sum()),
    }
