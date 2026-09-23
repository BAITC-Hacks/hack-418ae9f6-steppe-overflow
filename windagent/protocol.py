"""Протокол выпусков прогноза.

Выпуск в дату D (локальное время данных, UTC+6) делается в D issue_hour_local:00
и покрывает 48 часов: D+1 00:00 … D+2 23:00 локального времени.
lead_day = 1 для D+1 и 2 для D+2.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from windagent.config import load_settings


def _cfg(settings: dict | None) -> tuple[int, int, int]:
    s = settings or load_settings()
    f = s["forecast"]
    return s["scada"]["utc_offset_hours"], f["issue_hour_local"], f["horizon_days"]


def issue_time_utc(issue_date: date | str, settings: dict | None = None) -> pd.Timestamp:
    offset, hour, _ = _cfg(settings)
    return pd.Timestamp(issue_date).normalize() + pd.Timedelta(hours=hour - offset)


def target_hours_utc(issue_date: date | str, settings: dict | None = None) -> pd.DatetimeIndex:
    """Начала целевых часов (UTC) для выпуска в дату D."""
    offset, _, days = _cfg(settings)
    first_local = pd.Timestamp(issue_date).normalize() + pd.Timedelta(days=1)
    local = pd.date_range(first_local, periods=24 * days, freq="h")
    return local - pd.Timedelta(hours=offset)


def target_frame(issue_date: date | str, settings: dict | None = None) -> pd.DataFrame:
    """Таблица целевых часов: время UTC и локальное, горизонт от выпуска, сутки упреждения."""
    offset, _, _ = _cfg(settings)
    t = target_hours_utc(issue_date, settings)
    issue = issue_time_utc(issue_date, settings)
    local = t + pd.Timedelta(hours=offset)
    return pd.DataFrame(
        {
            "target_time_utc": t,
            "target_time_local": local,
            "horizon_h": ((t - issue) / pd.Timedelta("1h")).astype(int),
            "lead_day": (local.normalize() - pd.Timestamp(issue_date).normalize()).days,
        }
    )


def issue_dates(start: date | str, end: date | str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="D"))
