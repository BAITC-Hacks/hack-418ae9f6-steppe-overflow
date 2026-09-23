"""Локальный кэш архивных прогнозов погоды и его наполнение из Open-Meteo.

Файлы в cache/weather/ (коммитятся в репозиторий, чтобы бэктест
воспроизводился без сети):
  single_<model>.parquet           run_time, valid_time, переменные
  single_<model>_missing.json      прогоны, которых нет в архиве API
  previous_<model>.parquet         valid_time, day, переменные
  era5.parquet                     valid_time, переменные (реанализ, только для анализа)
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

from windagent.config import load_settings, resolve
from windagent.data.openmeteo import OpenMeteoClient, OpenMeteoError


def cache_dir(settings: dict | None = None) -> Path:
    s = settings or load_settings()
    return resolve(s["weather"]["cache_dir"])


def single_variables(model: str, settings: dict | None = None) -> list[str]:
    w = (settings or load_settings())["weather"]
    return w["variables"] + w["single_runs"][model].get("extra_variables", [])


def make_client(settings: dict | None = None) -> OpenMeteoClient:
    s = settings or load_settings()
    return OpenMeteoClient(s["site"]["latitude"], s["site"]["longitude"])


# --- Чтение кэша ----------------------------------------------------------------


def _read_parquet(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


@lru_cache(maxsize=32)
def _cached_read(path: str, mtime: float) -> pd.DataFrame | None:
    return _read_parquet(Path(path))


def _read(path: Path) -> pd.DataFrame | None:
    # Кэш в памяти инвалидируется при изменении файла
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return _cached_read(str(path), mtime)


def load_single(model: str, directory: Path | None = None) -> pd.DataFrame:
    df = _read((directory or cache_dir()) / f"single_{model}.parquet")
    if df is None:
        return pd.DataFrame(columns=["run_time", "valid_time"])
    return df


def load_neighbor(point: str, directory: Path | None = None) -> pd.DataFrame:
    df = _read((directory or cache_dir()) / f"single_ecmwf_ifs_nb_{point}.parquet")
    if df is None:
        return pd.DataFrame(columns=["run_time", "valid_time"])
    return df


def download_neighbors(start: date, end: date, *, settings: dict | None = None, directory: Path | None = None,
                       workers: int = 3, log: Callable[[str], None] = print) -> list[dict]:
    """Прогоны ECMWF в соседних точках сетки (config: neighbors)."""
    s = settings or load_settings()
    nb = s["neighbors"]
    out = []
    for name, pt in nb["points"].items():
        with OpenMeteoClient(pt["latitude"], pt["longitude"]) as client:
            out.append(download_single_runs(
                nb["model"], start, end, client=client, settings=s, directory=directory, workers=workers, log=log,
                variables=nb["variables"], run_hours=nb["run_hours"], stem=f"single_ecmwf_ifs_nb_{name}",
            ))
    return out


def load_previous(model: str, directory: Path | None = None) -> pd.DataFrame:
    df = _read((directory or cache_dir()) / f"previous_{model}.parquet")
    if df is None:
        return pd.DataFrame(columns=["valid_time", "day"])
    return df


def load_era5(directory: Path | None = None) -> pd.DataFrame | None:
    return _read((directory or cache_dir()) / "era5.parquet")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


# --- Загрузка ------------------------------------------------------------------


def planned_runs(model: str, start: date, end: date, settings: dict | None = None,
                 run_hours: list[int] | None = None) -> list[pd.Timestamp]:
    hours = run_hours or (settings or load_settings())["weather"]["single_runs"][model]["run_hours"]
    days = pd.date_range(start, end, freq="D")
    return [d + pd.Timedelta(hours=h) for d in days for h in hours]


def download_single_runs(
    model: str,
    start: date,
    end: date,
    *,
    client: OpenMeteoClient | None = None,
    settings: dict | None = None,
    directory: Path | None = None,
    workers: int = 4,
    save_every: int = 200,
    log: Callable[[str], None] = print,
    variables: list[str] | None = None,
    run_hours: list[int] | None = None,
    stem: str | None = None,
) -> dict:
    """Докачивает в кэш прогоны модели за период (уже скачанные пропускаются).

    stem/variables/run_hours позволяют качать ту же модель в другой точке (соседние точки сетки).
    """
    s = settings or load_settings()
    d = directory or cache_dir(s)
    stem = stem or f"single_{model}"
    path = d / f"{stem}.parquet"
    miss_path = d / f"{stem}_missing.json"
    cfg = s["weather"]["single_runs"][model]
    variables = variables or single_variables(model, s)

    have = _read_parquet(path)
    done = set(pd.to_datetime(have["run_time"].unique())) if have is not None else set()
    missing = set(pd.to_datetime(json.loads(miss_path.read_text()))) if miss_path.exists() else set()
    todo = [r for r in planned_runs(model, start, end, s, run_hours) if r not in done and r not in missing]
    log(f"{model}: в кэше {len(done)} прогонов, нет в архиве {len(missing)}, к загрузке {len(todo)}")

    own_client = client is None
    client = client or make_client(s)
    frames = [have] if have is not None else []
    new_count = 0
    failed: list[pd.Timestamp] = []

    def flush():
        if frames:
            all_ = pd.concat(frames, ignore_index=True)
            all_ = all_.drop_duplicates(["run_time", "valid_time"]).sort_values(["run_time", "valid_time"])
            frames[:] = [all_]
            _write(all_, path)
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        miss_path.write_text(json.dumps(sorted(str(m) for m in missing), indent=0))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(client.single_run, model, r, variables, cfg["forecast_hours"]): r for r in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                try:
                    df = fut.result()
                except OpenMeteoError as e:
                    # Сбой одного прогона не прерывает загрузку; при повторном запуске докачается
                    failed.append(r)
                    log(f"  ошибка {r}: {e}")
                    continue
                if df is None:
                    missing.add(r)
                else:
                    frames.append(df)
                    new_count += 1
                if i % save_every == 0:
                    flush()
                    log(f"  {i}/{len(todo)}")
    finally:
        flush()  # сохраняем скачанное даже при прерывании
        if own_client:
            client.close()
    return {
        "model": model,
        "downloaded": new_count,
        "not_in_archive": len(missing),
        "failed": len(failed),
        "total": len(done) + new_count,
    }


def download_previous_runs(
    model: str,
    start: date,
    end: date,
    *,
    client: OpenMeteoClient | None = None,
    settings: dict | None = None,
    directory: Path | None = None,
    chunk_days: int = 92,
    log: Callable[[str], None] = print,
) -> dict:
    """Скачивает previous_dayN за период кусками; данные за период в кэше заменяются."""
    s = settings or load_settings()
    d = directory or cache_dir(s)
    path = d / f"previous_{model}.parquet"
    w = s["weather"]
    own_client = client is None
    client = client or make_client(s)
    parts = []
    try:
        cur = pd.Timestamp(start)
        while cur <= pd.Timestamp(end):
            stop = min(cur + pd.Timedelta(days=chunk_days - 1), pd.Timestamp(end))
            parts.append(client.previous_runs(model, cur.date(), stop.date(), w["variables"], w["previous_days"]))
            log(f"{model}: {cur.date()} … {stop.date()}")
            cur = stop + pd.Timedelta(days=1)
    finally:
        if own_client:
            client.close()
    new = pd.concat(parts, ignore_index=True)
    have = _read_parquet(path)
    if have is not None:
        new = pd.concat([have, new]).drop_duplicates(["valid_time", "day"], keep="last")
    new = new.sort_values(["day", "valid_time"]).reset_index(drop=True)
    _write(new, path)
    filled = new.dropna(subset=["wind_speed_100m"])
    return {"model": model, "rows": len(new), "rows_with_100m": len(filled)}


def download_era5(
    start: date,
    end: date,
    *,
    client: OpenMeteoClient | None = None,
    settings: dict | None = None,
    directory: Path | None = None,
) -> dict:
    s = settings or load_settings()
    d = directory or cache_dir(s)
    own_client = client is None
    client = client or make_client(s)
    try:
        df = client.era5(start, end, s["weather"]["variables"])
    finally:
        if own_client:
            client.close()
    _write(df, d / "era5.parquet")
    return {"rows": len(df)}


def status(settings: dict | None = None) -> str:
    """Текстовая сводка по содержимому кэша."""
    s = settings or load_settings()
    d = cache_dir(s)
    lines = [f"Кэш погоды: {d}"]
    for m in s["weather"]["single_runs"]:
        df = load_single(m, d)
        if len(df):
            lines.append(
                f"  single  {m}: {df['run_time'].nunique()} прогонов, "
                f"{df['run_time'].min()} … {df['run_time'].max()}"
            )
        else:
            lines.append(f"  single  {m}: пусто")
    for m in s["weather"]["previous_runs"]:
        df = load_previous(m, d)
        if len(df):
            ok = df.dropna(subset=["wind_speed_100m"])
            by_day = ok.groupby("day")["valid_time"].min().dt.date.to_dict() if len(ok) else {}
            lines.append(
                f"  previous {m}: {len(df)} строк; ветер 100 м доступен с (по day): {by_day}"
            )
        else:
            lines.append(f"  previous {m}: пусто")
    e = load_era5(d)
    lines.append(f"  era5: {len(e) if e is not None else 0} часов")
    return "\n".join(lines)
