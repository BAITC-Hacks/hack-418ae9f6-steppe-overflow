"""DataStore(as_of) — единственная точка доступа к данным для модели и агента.

Гарантия: всё, что отдаёт DataStore, было доступно в момент as_of.
  * Прогон погодной модели доступен, если run_time + publish_lag ≤ as_of.
  * Час SCADA доступен, если он закончился: ts + 1 ч ≤ as_of.
Любая попытка получить более поздние данные → LeakageError.
Каждое обращение пишется в manifest — он сохраняется вместе с прогнозом.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from windagent.config import load_settings
from windagent.data import weather


class LeakageError(RuntimeError):
    """Попытка использовать данные, недоступные на момент as_of."""


def _floor6(t: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(t).floor("6h")


class DataStore:
    def __init__(
        self,
        as_of: str | pd.Timestamp,
        *,
        settings: dict | None = None,
        weather_dir: Path | None = None,
        scada_hourly: pd.DataFrame | None = None,
    ):
        self.as_of = pd.Timestamp(as_of)
        self.settings = settings or load_settings()
        self.weather_dir = weather_dir or weather.cache_dir(self.settings)
        self._scada = scada_hourly
        self.manifest: list[dict] = []

    # --- Прогоны с точным временем старта (Single Runs) --------------------------

    def _lag(self, model: str) -> pd.Timedelta:
        w = self.settings["weather"]
        cfg = w["single_runs"].get(model) or w["previous_runs"][model]
        return pd.Timedelta(hours=cfg["publish_lag_h"])

    def published_at(self, model: str, run_time: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(run_time) + self._lag(model)

    def available_runs(self, model: str) -> list[pd.Timestamp]:
        """Прогоны из кэша, опубликованные не позже as_of (по возрастанию)."""
        df = weather.load_single(model, self.weather_dir)
        if not len(df):
            return []
        runs = pd.DatetimeIndex(df["run_time"].unique()).sort_values()
        return list(runs[runs + self._lag(model) <= self.as_of])

    def latest_run(self, model: str) -> pd.Timestamp | None:
        runs = self.available_runs(model)
        return runs[-1] if runs else None

    def single_run(
        self, model: str, targets: pd.DatetimeIndex, run_time: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Прогноз на целевые часы. По умолчанию — самый свежий доступный прогон.

        Если свежий прогон покрывает не все часы (прогона нет в архиве, короткий
        горизонт), пропуски заполняются из более старых доступных прогонов.
        Возвращает индекс targets, переменные, run_time, published_at, lead_h (по строкам).
        """
        if run_time is not None:
            return self._one_run(model, targets, pd.Timestamp(run_time))
        runs = self.available_runs(model)
        if not runs:
            raise LookupError(f"{model}: нет ни одного прогона, доступного на {self.as_of}")
        out = None
        for r in reversed(runs[-self.MAX_FALLBACK_RUNS:]):
            f = self._one_run(model, targets, r, log=False)
            f.loc[f.drop(columns=["run_time", "published_at", "lead_h"]).isna().all(axis=1),
                  ["run_time", "published_at", "lead_h"]] = [pd.NaT, pd.NaT, np.nan]
            out = f if out is None else out.combine_first(f)
            if out.drop(columns=["run_time", "published_at", "lead_h"]).notna().any(axis=1).all():
                break
        self._log("single", model, targets=targets, frame=out,
                  run_time=pd.to_datetime(out["run_time"]).max())
        return out

    # Сколько прошлых прогонов можно использовать для дозаполнения (4 = последние сутки)
    MAX_FALLBACK_RUNS = 4

    def _one_run(self, model: str, targets: pd.DatetimeIndex, run_time: pd.Timestamp, log: bool = True) -> pd.DataFrame:
        pub = self.published_at(model, run_time)
        if pub > self.as_of:
            raise LeakageError(
                f"{model} прогон {run_time} опубликован {pub}, а сейчас (as_of) {self.as_of}"
            )
        df = weather.load_single(model, self.weather_dir)
        run = df[df["run_time"] == run_time].set_index("valid_time")
        out = run.drop(columns="run_time").reindex(targets)
        out["run_time"] = run_time
        out["published_at"] = pub
        out["lead_h"] = (targets - run_time) / pd.Timedelta("1h")
        if log:
            self._log("single", model, run_time=run_time, published_at=pub, targets=targets, frame=out)
        return out

    def single_runs_lagged(self, model: str, targets: pd.DatetimeIndex, n_runs: int) -> list[pd.DataFrame]:
        """Несколько последних доступных прогонов (лаговый ансамбль): от свежего к старому."""
        runs = self.available_runs(model)[-n_runs:][::-1]
        return [self._one_run(model, targets, r) for r in runs]

    # --- Previous Runs (previous_dayN) ------------------------------------------

    def previous_runs(self, model: str, targets: pd.DatetimeIndex) -> pd.DataFrame:
        """Для каждого часа берёт previous_dayN с минимальным N, прогон которого уже опубликован.

        previous_dayN для часа V взят из прогона run ≈ floor6(V) − 24N ч (проверено
        сравнением с Single Runs ECMWF). Доступен, если run + lag ≤ as_of.
        """
        days = sorted(self.settings["weather"]["previous_days"])
        lag = self._lag(model)
        df = weather.load_previous(model, self.weather_dir)
        by_day = {d: g.drop(columns="day").set_index("valid_time") for d, g in df.groupby("day")}

        out = pd.DataFrame(index=targets)
        out["day"] = np.nan
        out["run_time"] = pd.NaT
        base = _floor6(targets)
        for d in days:
            run = base - pd.Timedelta(hours=24 * d)
            ok = (run + lag <= self.as_of) & out["day"].isna().to_numpy()
            if d not in by_day or not ok.any():
                continue
            vals = by_day[d].reindex(targets[ok])
            has = vals.notna().any(axis=1).to_numpy()
            idx = targets[ok][has]
            for c in vals.columns:
                out.loc[idx, c] = vals.loc[idx, c].to_numpy()
            out.loc[idx, "day"] = d
            out.loc[idx, "run_time"] = run[ok][has]
        out["published_at"] = pd.to_datetime(out["run_time"]) + lag
        out["lead_h"] = (targets - pd.to_datetime(out["run_time"])) / pd.Timedelta("1h")
        self._log("previous", model, targets=targets, frame=out)
        return out

    # --- SCADA ----------------------------------------------------------------

    def scada_hourly(self) -> pd.DataFrame:
        """Почасовые SCADA-данные, завершившиеся к as_of."""
        if self._scada is None:
            from windagent.data import scada

            self._scada = scada.load_hourly(self.settings)
        h = self._scada
        out = h[h.index + pd.Timedelta("1h") <= self.as_of]
        self.manifest.append(
            {"source": "scada", "last_hour": str(out.index.max()) if len(out) else None}
        )
        return out

    # --- Контроль ----------------------------------------------------------------

    def _log(self, source, model, *, targets, frame, run_time=None, published_at=None):
        pub = pd.to_datetime(frame["published_at"]).dropna()
        if len(pub) and pub.max() > self.as_of:  # защита «в глубину»
            raise LeakageError(f"{source}/{model}: опубликовано {pub.max()} > as_of {self.as_of}")
        entry = {
            "source": source,
            "model": model,
            "targets": f"{targets.min()} … {targets.max()}",
            "coverage": float(frame.drop(columns=["run_time", "published_at", "lead_h", "day"], errors="ignore")
                              .notna().any(axis=1).mean()),
            "max_published_at": str(pub.max()) if len(pub) else None,
        }
        if run_time is not None:
            entry["run_time"] = str(run_time)
        if "day" in frame:
            entry["days_used"] = {int(k): int(v) for k, v in frame["day"].value_counts().items()}
        self.manifest.append(entry)

    def check_manifest(self) -> None:
        """Проверка, что ни один использованный источник не опубликован после as_of."""
        for e in self.manifest:
            mp = e.get("max_published_at")
            if mp and pd.Timestamp(mp) > self.as_of:
                raise LeakageError(f"утечка в манифесте: {e}")
