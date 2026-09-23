"""Сборка таблицы «выпуск × целевой час» из погодных прогнозов, доступных на момент выпуска.

Одна и та же функция используется и для обучения (по историческим выпускам),
и для прогноза — поэтому распределение признаков в обучении и в тесте одинаковое.
Колонки погоды: <префикс модели>__<переменная>, например ifs__wind_speed_100m.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from windagent import protocol
from windagent.config import load_settings
from windagent.data.store import DataStore

META = ["issue_date", "issue_time_utc", "target_time_utc", "target_time_local", "horizon_h", "lead_day"]
TRUTH = ["p_t1", "p_t2", "p_farm", "p_clean_t1", "p_clean_t2", "p_farm_clean"]
LAGGED_RUNS = 4


def issue_frame(issue_date, store: DataStore | None = None, settings: dict | None = None) -> pd.DataFrame:
    """48 строк (целевые часы) с сырыми прогнозами всех моделей погоды, доступными на момент выпуска."""
    s = settings or load_settings()
    store = store or DataStore(protocol.issue_time_utc(issue_date, s), settings=s)
    tf = protocol.target_frame(issue_date, s)
    t = pd.DatetimeIndex(tf["target_time_utc"])
    out = tf.copy()
    out.insert(0, "issue_date", pd.Timestamp(issue_date).normalize())
    out.insert(1, "issue_time_utc", store.as_of)
    prefix = s["nwp_prefix"]
    w = s["weather"]

    for model in w["single_runs"]:
        pre = prefix[model]
        try:
            f = store.single_run(model, t)
        except LookupError:
            f = pd.DataFrame(index=t)
        for c in f.columns:
            if c == "published_at":
                continue
            out[f"{pre}__{c}"] = f[c].to_numpy()
        # Возраст прогона в момент выпуска: чем свежее, тем точнее
        out[f"{pre}__run_age_h"] = (
            ((store.as_of - pd.to_datetime(f["run_time"])) / pd.Timedelta("1h")).to_numpy()
            if "run_time" in f else np.nan
        )
        # Лаговый ансамбль: последние прогоны модели (за ~сутки). Разброс между ними —
        # оценка неопределённости прогноза ветра.
        lagged = store.single_runs_lagged(model, t, n_runs=LAGGED_RUNS)
        if lagged:
            ws = np.column_stack([r["wind_speed_100m"].to_numpy(dtype=float) for r in lagged])
            with warnings.catch_warnings():  # часы, где ни один прогон не дотягивается, → NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                out[f"{pre}__lag_ws100_mean"] = np.nanmean(ws, axis=1)
                out[f"{pre}__lag_ws100_std"] = np.nanstd(ws, axis=1)

    for model in w["previous_runs"]:
        pre = prefix[model]
        f = store.previous_runs(model, t)
        for c in f.columns:
            if c == "published_at":
                continue
            out[f"{pre}__{c}"] = f[c].to_numpy()

    store.check_manifest()
    return out.reset_index(drop=True)


def build_dataset(issue_dates: Iterable, settings: dict | None = None) -> pd.DataFrame:
    """Признаки для набора выпусков (без истинных значений)."""
    s = settings or load_settings()
    frames = [issue_frame(d, settings=s) for d in issue_dates]
    return pd.concat(frames, ignore_index=True)


def attach_truth(df: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Добавляет факт SCADA по целевому часу. Используется только для обучения и оценки."""
    cols = [c for c in TRUTH if c in hourly.columns]
    truth = hourly[cols].reindex(pd.DatetimeIndex(df["target_time_utc"]))
    out = df.copy()
    for c in cols:
        out[c] = truth[c].to_numpy()
    return out


def history_issue_dates(settings: dict | None = None) -> list[pd.Timestamp]:
    """Все даты выпусков, для которых есть архив прогнозов и факт SCADA."""
    s = settings or load_settings()
    first = pd.Timestamp(s["weather"]["archive_start"]) + pd.Timedelta(days=1)
    last = pd.Timestamp(s["forecast"]["test_first_issue"]) - pd.Timedelta(days=2)
    return protocol.issue_dates(first, last)
