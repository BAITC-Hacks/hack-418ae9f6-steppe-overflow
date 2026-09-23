"""Производные признаки из сырых прогнозов погоды (детерминированно, без обучения).

Работает на таблице «выпуск × час»: сглаживания и сдвиги считаются внутри
одного выпуска, поэтому соседние выпуски не «подсматривают» друг в друга.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

SOURCES = ("ifs", "ifs025", "icon", "gfs")
R_DRY_AIR = 287.05


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].astype(float) if name in df else pd.Series(np.nan, index=df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["issue_date", "target_time_utc"]).copy()
    g = out.groupby("issue_date", sort=False)

    for src in SOURCES:
        ws100 = _col(out, f"{src}__wind_speed_100m")
        ws10 = _col(out, f"{src}__wind_speed_10m")
        d = np.deg2rad(_col(out, f"{src}__wind_direction_100m"))
        out[f"{src}__dir_sin"] = np.sin(d)
        out[f"{src}__dir_cos"] = np.cos(d)
        # Компоненты ветра: направление с учётом силы
        out[f"{src}__u100"] = -ws100 * np.sin(d)
        out[f"{src}__v100"] = -ws100 * np.cos(d)
        # Показатель степенного профиля ветра (стратификация атмосферы)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{src}__shear"] = np.log(ws100.clip(lower=0.1) / ws10.clip(lower=0.1)) / np.log(10)
        out[f"{src}__ws100_cube"] = ws100.clip(upper=15) ** 3

    # Основная модель — ECMWF IFS: сглаживание и сдвиги во времени (фазовые ошибки фронтов)
    ws = "ifs__wind_speed_100m"
    out["ifs__ws100_roll3"] = g[ws].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    out["ifs__ws100_roll7"] = g[ws].transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())
    out["ifs__ws100_prev"] = g[ws].shift(1)
    out["ifs__ws100_next"] = g[ws].shift(-1)
    out["ifs__ws100_trend"] = out["ifs__ws100_next"] - out["ifs__ws100_prev"]
    out["ifs__ws100_day_mean"] = g[ws].transform("mean")

    # Плотность воздуха: мощность ∝ ρ·v³
    t_k = _col(out, "ifs__temperature_2m") + 273.15
    out["ifs__rho"] = _col(out, "ifs__surface_pressure") * 100 / (R_DRY_AIR * t_k)

    # Мультимодельный ансамбль: согласие моделей = уверенность
    ens = np.column_stack([_col(out, f"{s}__wind_speed_100m") for s in SOURCES])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["ens__ws100_mean"] = np.nanmean(ens, axis=1)
        out["ens__ws100_std"] = np.nanstd(ens, axis=1)
        out["ens__ws100_min"] = np.nanmin(ens, axis=1)
        out["ens__ws100_max"] = np.nanmax(ens, axis=1)

    # Календарь (местное время)
    loc = pd.DatetimeIndex(out["target_time_local"])
    out["cal__hour_sin"] = np.sin(2 * np.pi * loc.hour / 24)
    out["cal__hour_cos"] = np.cos(2 * np.pi * loc.hour / 24)
    out["cal__doy_sin"] = np.sin(2 * np.pi * loc.dayofyear / 365.25)
    out["cal__doy_cos"] = np.cos(2 * np.pi * loc.dayofyear / 365.25)
    return out.loc[df.index]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Все признаки модели: погодные (<src>__*), ансамблевые, календарные и горизонт."""
    skip = {"run_time", "published_at"}
    cols = [c for c in df.columns if "__" in c and c.split("__", 1)[1] not in skip]
    return cols + ["horizon_h", "lead_day"]
