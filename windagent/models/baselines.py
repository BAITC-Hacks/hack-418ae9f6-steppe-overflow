"""Базовые модели прогноза — точки отсчёта для оценки основной модели.

Все модели реализуют интерфейс Forecaster:
  fit(train, ctx)  — train: строки «выпуск × час» с признаками и фактом, только до ctx.cutoff;
  predict(X)       — DataFrame с колонками p_t1, p_t2, p_farm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from windagent.data import scada
from windagent.models.power_curve import PowerCurve

TARGETS = ("p_t1", "p_t2", "p_farm")
TURBINE_CAP = {"p_t1": 0.99, "p_t2": 0.97, "p_farm": 0.98}


@dataclass
class FitContext:
    cutoff: pd.Timestamp          # момент выпуска первого прогноза периода (UTC)
    hourly: pd.DataFrame          # почасовой SCADA, только часы, закончившиеся до cutoff
    settings: dict


class Forecaster:
    name = "base"

    def fit(self, train: pd.DataFrame, ctx: FitContext) -> "Forecaster":
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


def _wind(X: pd.DataFrame, source: str, fallback: tuple[str, ...] = ("ifs025", "icon", "gfs")) -> np.ndarray:
    """Ветер 100 м из заданной модели; пропуски заполняются следующей моделью по списку."""
    ws = X.get(f"{source}__wind_speed_100m", pd.Series(np.nan, index=X.index)).astype(float)
    for fb in fallback:
        if fb != source and f"{fb}__wind_speed_100m" in X:
            ws = ws.fillna(X[f"{fb}__wind_speed_100m"].astype(float))
    return ws.to_numpy()


class Climatology(Forecaster):
    """B0: средняя мощность по (месяц, час суток местного времени) за всю историю до cutoff."""

    name = "clim"

    def fit(self, train, ctx):
        h = ctx.hourly
        local = h.index + pd.Timedelta(hours=ctx.settings["scada"]["utc_offset_hours"])
        key = pd.MultiIndex.from_arrays([local.month, local.hour], names=["month", "hour"])
        self.table_ = h[list(TARGETS)].groupby(key).mean()
        self.overall_ = h[list(TARGETS)].mean()
        return self

    def predict(self, X):
        loc = pd.DatetimeIndex(X["target_time_local"])
        key = pd.MultiIndex.from_arrays([loc.month, loc.hour], names=["month", "hour"])
        out = self.table_.reindex(key).fillna(self.overall_)
        out.index = X.index
        return out


class PhysicalCurve(Forecaster):
    """B1: прогноз ветра на 100 м → кривая мощности турбины, построенная по анемометру гондолы.

    Чистая «физика» без подстройки под ошибки прогноза погоды.
    """

    def __init__(self, source: str = "ifs"):
        self.source = source
        self.name = f"phys_{source}"

    def fit(self, train, ctx):
        self.curves_ = {}
        for tb in scada.TURBINES:
            d = scada.load_turbine_10min(tb, ctx.settings)
            d = d[(d["ts_utc"] + pd.Timedelta("10min") <= ctx.cutoff) & d["ok"]]
            self.curves_[f"p_{tb}"] = PowerCurve(p_max=TURBINE_CAP[f"p_{tb}"], bin_width=0.5).fit(d["ws"], d["p"])
        return self

    def predict(self, X):
        ws = _wind(X, self.source)
        out = pd.DataFrame({k: c.predict(ws) for k, c in self.curves_.items()}, index=X.index)
        out["p_farm"] = out[["p_t1", "p_t2"]].mean(axis=1)
        return out


class NwpCurve(Forecaster):
    """B2: кривая «прогноз ветра модели → фактическая мощность», обученная на исторических выпусках.

    Учитывает систематическую ошибку прогноза ветра на площадке (простейший MOS).
    """

    def __init__(self, source: str = "ifs"):
        self.source = source
        self.name = f"curve_{source}"

    def fit(self, train, ctx):
        ws = _wind(train, self.source)
        self.curves_ = {}
        for t in TARGETS:
            y = train[t.replace("p_", "p_clean_") if t != "p_farm" else "p_farm_clean"].to_numpy()
            self.curves_[t] = PowerCurve(p_max=TURBINE_CAP[t]).fit(ws, y)
        return self

    def predict(self, X):
        ws = _wind(X, self.source)
        return pd.DataFrame({t: c.predict(ws) for t, c in self.curves_.items()}, index=X.index)


def default_baselines() -> list[Forecaster]:
    return [
        Climatology(),
        PhysicalCurve("ifs"),
        NwpCurve("ifs"),
        NwpCurve("ifs025"),
        NwpCurve("icon"),
        NwpCurve("gfs"),
    ]
