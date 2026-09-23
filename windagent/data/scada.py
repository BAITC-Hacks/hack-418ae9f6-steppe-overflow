"""SCADA-данные турбин: загрузка, контроль качества, почасовая агрегация.

Исходные CSV — 10-минутные записи с метками в UTC+6. Внутри проекта всё
хранится в UTC (наивные datetime), час обозначается своим началом:
запись 14:00…14:50 по UTC+6 → час 08:00 UTC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from windagent.config import load_settings, resolve

TURBINES = ("t1", "t2")

# Порядок колонок в исходных файлах (имена на русском, поэтому берём по позиции)
_RAW_COLUMNS = ["id", "ts_local", "ws", "p", "t"]


def read_raw(path: str | Path, utc_offset_hours: int) -> pd.DataFrame:
    """Читает один CSV турбины. Возвращает ts_local, ts_utc, ws, p, t."""
    df = pd.read_csv(path, header=0, names=_RAW_COLUMNS)
    df["ts_local"] = pd.to_datetime(df["ts_local"], format="%Y-%m-%d %H:%M:%S")
    df["ts_utc"] = df["ts_local"] - pd.Timedelta(hours=utc_offset_hours)
    df = df.drop(columns="id").drop_duplicates("ts_utc").sort_values("ts_utc")
    return df[["ts_local", "ts_utc", "ws", "p", "t"]].reset_index(drop=True)


def add_qc_flags(df: pd.DataFrame, qc: dict) -> pd.DataFrame:
    """Помечает 10-минутные записи, непригодные для обучения кривой мощности.

    downtime    — ветер есть, а мощность на минимуме (простой / авария / ТО);
    curtailment — плато одинаковой мощности ниже номинала при меняющемся ветре
                  (внешнее ограничение мощности).
    Записи не удаляются: оценка прогноза идёт по «сырому» факту.
    """
    out = df.copy()
    out["downtime"] = (out["ws"] >= qc["downtime_ws_min"]) & (out["p"] <= qc["downtime_p_max"])

    # Серии подряд идущих одинаковых значений мощности без разрывов во времени
    new_run = (out["p"] != out["p"].shift()) | (out["ts_utc"].diff() != pd.Timedelta("10min"))
    run_id = new_run.cumsum()
    grp = out.groupby(run_id)
    run_len = grp["p"].transform("size")
    run_ws_mean = grp["ws"].transform("mean")
    run_ws_std = grp["ws"].transform("std").fillna(0.0)
    out["curtailment"] = (
        (run_len >= qc["curtail_min_run"])
        & (run_ws_mean >= qc["curtail_ws_min"])
        & (out["p"] >= qc["curtail_p_min"])
        & (out["p"] <= qc["curtail_p_max"])
        & (run_ws_std > 0.3)
    )
    out["ok"] = ~(out["downtime"] | out["curtailment"])
    return out


def to_hourly(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Агрегация 10-мин → час (индекс — начало часа в UTC, сетка без пропусков).

    p        — среднее по всем записям часа (то, с чем сравнивается прогноз);
    p_clean  — среднее только по записям без QC-флагов (для обучения);
    valid    — в часе не меньше min_obs записей.
    """
    x = df.set_index("ts_utc")
    hour = x.index.floor("h")
    g = x.groupby(hour)
    hourly = pd.DataFrame(
        {
            "ws": g["ws"].mean(),
            "p": g["p"].mean(),
            "t": g["t"].mean(),
            "n_obs": g["p"].size(),
            "n_ok": g["ok"].sum(),
            "p_clean": x["p"].where(x["ok"]).groupby(hour).mean(),
        }
    )
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full)
    hourly.index.name = "ts_utc"
    hourly[["n_obs", "n_ok"]] = hourly[["n_obs", "n_ok"]].fillna(0).astype(int)
    hourly["valid"] = hourly["n_obs"] >= min_obs
    # Час чистый, если почти все записи без флагов
    hourly["clean"] = hourly["valid"] & (hourly["n_ok"] >= min_obs)
    for col in ("ws", "p", "t", "p_clean"):
        hourly.loc[~hourly["valid"], col] = np.nan
    hourly.loc[~hourly["clean"], "p_clean"] = np.nan
    return hourly


def load_turbine_10min(turbine: str, settings: dict | None = None) -> pd.DataFrame:
    s = settings or load_settings()
    sc = s["scada"]
    raw = read_raw(resolve(sc["files"][turbine]), sc["utc_offset_hours"])
    return add_qc_flags(raw, sc["qc"])


def load_hourly(settings: dict | None = None) -> pd.DataFrame:
    """Почасовой датасет ВЭС: колонки {ws,p,t,p_clean,n_obs,valid,clean}_{t1,t2} и p_farm.

    p_farm — среднее валидных турбин за час (если одна турбина без данных,
    берётся вторая); p_farm_clean — то же по чистым часам.
    """
    s = settings or load_settings()
    min_obs = s["scada"]["min_obs_per_hour"]
    parts = []
    for tb in TURBINES:
        h = to_hourly(load_turbine_10min(tb, s), min_obs)
        parts.append(h.add_suffix(f"_{tb}"))
    wide = pd.concat(parts, axis=1)
    wide.index.name = "ts_utc"
    wide["p_farm"] = wide[[f"p_{tb}" for tb in TURBINES]].mean(axis=1, skipna=True)
    wide["p_farm_clean"] = wide[[f"p_clean_{tb}" for tb in TURBINES]].mean(axis=1, skipna=True)
    wide["ws_farm"] = wide[[f"ws_{tb}" for tb in TURBINES]].mean(axis=1, skipna=True)
    wide["t_farm"] = wide[[f"t_{tb}" for tb in TURBINES]].mean(axis=1, skipna=True)
    return wide


def summary(settings: dict | None = None) -> str:
    """Короткая текстовая сводка по данным (для CLI и отчётов)."""
    s = settings or load_settings()
    lines = []
    for tb in TURBINES:
        d = load_turbine_10min(tb, s)
        expected = int((d["ts_utc"].max() - d["ts_utc"].min()) / pd.Timedelta("10min")) + 1
        lines.append(
            f"{tb}: {len(d):,} записей, {d['ts_local'].min()} … {d['ts_local'].max()} (UTC+{s['scada']['utc_offset_hours']}), "
            f"пропущено {1 - len(d) / expected:.1%}, простои {d['downtime'].mean():.1%}, "
            f"ограничения {d['curtailment'].mean():.1%}, средняя мощность {d['p'].mean():.3f}"
        )
    h = load_hourly(s)
    lines.append(
        f"ВЭС (почасово): {len(h):,} часов, с данными {h['p_farm'].notna().mean():.1%}, "
        f"средняя мощность {h['p_farm'].mean():.3f}"
    )
    return "\n".join(lines)
