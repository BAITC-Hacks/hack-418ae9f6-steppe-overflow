"""Низкоуровневый клиент Open-Meteo (Single Runs, Previous Runs, ERA5).

Все ответы запрашиваются и возвращаются в UTC, скорость ветра — в м/с.
Время в DataFrame — наивные datetime в UTC.
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime

import httpx
import pandas as pd

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoError(RuntimeError):
    pass


class OpenMeteoClient:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        *,
        timeout: float = 60.0,
        retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        max_rps: float = 3.0,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.retries = retries
        self._http = httpx.Client(timeout=timeout, transport=transport)
        # Бесплатный Open-Meteo: 600 запросов/мин, 5000/час; запрос с >10 переменными
        # считается дороже. Ограничиваем частоту общим для всех потоков «турникетом».
        self._min_interval = 1.0 / max_rps if max_rps else 0.0
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- HTTP ----------------------------------------------------------------

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            time.sleep(wait)

    def _get(self, url: str, params: dict) -> dict | None:
        """GET с повторами. None — если API сообщает, что данных нет (а не сбой)."""
        base = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }
        delay = 2.0
        last_err: Exception | None = None
        for _ in range(self.retries):
            self._throttle()
            try:
                r = self._http.get(url, params={**base, **params})
            except httpx.TransportError as e:
                last_err = e
            else:
                if r.status_code == 200:
                    return r.json()
                body = _json_or_none(r)
                reason = (body or {}).get("reason", "")
                if r.status_code == 400 and "not available" in reason:
                    return None
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise OpenMeteoError(f"{r.status_code}: {reason or r.text[:200]}")
                last_err = OpenMeteoError(f"{r.status_code}: {reason}")
                if r.status_code == 429:
                    # Минутный лимит сбрасывается через минуту, часовой — через час
                    time.sleep(3600 if "Hourly" in reason else 65)
                    continue
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise OpenMeteoError(f"не удалось получить {url}: {last_err}")

    # --- Запросы -------------------------------------------------------------

    def single_run(
        self, model: str, run_time: datetime, variables: list[str], forecast_hours: int
    ) -> pd.DataFrame | None:
        """Один прогон модели. None — если прогона нет в архиве."""
        data = self._get(
            SINGLE_RUNS_URL,
            {
                "models": model,
                "run": pd.Timestamp(run_time).strftime("%Y-%m-%dT%H:%M"),
                "hourly": ",".join(variables),
                "forecast_hours": forecast_hours,
            },
        )
        if data is None:
            return None
        df = _hourly_frame(data)
        if df[variables].isna().all().all():
            return None  # API вернул сетку без значений — прогона фактически нет
        df.insert(0, "run_time", pd.Timestamp(run_time))
        return df.rename(columns={"time": "valid_time"})

    def previous_runs(
        self, model: str, start: date, end: date, variables: list[str], days: list[int]
    ) -> pd.DataFrame:
        """Прогнозы, выпущенные N суток назад, в «длинном» формате: valid_time, day, переменные."""
        cols = [f"{v}_previous_day{d}" for d in days for v in variables]
        data = self._get(
            PREVIOUS_RUNS_URL,
            {
                "models": model,
                "start_date": str(start),
                "end_date": str(end),
                "hourly": ",".join(cols),
            },
        )
        wide = _hourly_frame(data)
        parts = []
        for d in days:
            part = wide[["time"] + [f"{v}_previous_day{d}" for v in variables]].copy()
            part.columns = ["valid_time"] + variables
            part.insert(1, "day", d)
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    def era5(self, start: date, end: date, variables: list[str]) -> pd.DataFrame:
        data = self._get(
            ARCHIVE_URL,
            {"start_date": str(start), "end_date": str(end), "hourly": ",".join(variables)},
        )
        return _hourly_frame(data).rename(columns={"time": "valid_time"})


def _hourly_frame(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    num = df.columns.drop("time")
    df[num] = df[num].astype("float32")
    return df


def _json_or_none(r: httpx.Response) -> dict | None:
    try:
        return r.json()
    except ValueError:
        return None
