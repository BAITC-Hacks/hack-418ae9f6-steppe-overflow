"""Проверка прогноза на данных пользователя: разбор загруженного CSV и сравнение с прогнозами.

Поддерживаемые форматы:
  * формат датасета организаторов: ID, время, скорость ветра, нормализованная мощность, температура
    (10-минутные записи, одна турбина на файл; два файла → среднее = ВЭС);
  * простой CSV: колонка времени + колонка нормализованной мощности (0–1 или 0–100 %),
    любая частота (приводится к часу).
Файлы обрабатываются в памяти и нигде не сохраняются.
"""

from __future__ import annotations

import io
import warnings

import pandas as pd

from windagent.config import load_settings, resolve
from windagent.data import scada
from windagent.eval import metrics

MAX_ROWS = 600_000


class UploadError(ValueError):
    pass


def _read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UploadError("не удалось прочитать файл: неизвестная кодировка")
    try:
        df = pd.read_csv(io.StringIO(text), sep=_detect_sep(text), nrows=MAX_ROWS)
    except Exception as e:  # noqa: BLE001 — показываем пользователю понятную причину
        raise UploadError(f"файл не похож на CSV: {e}") from e
    if df.shape[1] < 2 or len(df) < 2:
        raise UploadError("в файле должно быть хотя бы 2 колонки и 2 строки")
    return df


def _detect_sep(text: str) -> str:
    """Разделитель, который даёт одинаковое число колонок во всех первых строках (приоритет ; и табуляции:
    в русских файлах запятая бывает и в заголовках, и в дробных числах)."""
    lines = [ln for ln in text.splitlines()[:20] if ln.strip()]
    for sep in (";", "\t", ","):
        counts = {ln.count(sep) for ln in lines}
        if len(counts) == 1 and counts.pop() > 0:
            return sep
    return ","


def _is_time(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return bool(pd.to_datetime(s.head(20), errors="coerce").notna().all())


def _to_number(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def parse_upload(raw: bytes, utc_offset_hours: int | None = None) -> tuple[pd.Series, str]:
    """CSV → почасовая нормализованная мощность (индекс — начало часа в UTC) и описание формата."""
    offset = load_settings()["scada"]["utc_offset_hours"] if utc_offset_hours is None else utc_offset_hours
    df = _read_csv(raw)

    # Формат организаторов: 5 колонок, вторая — время
    if df.shape[1] == 5 and _is_time(df.iloc[:, 1]):
        d = pd.DataFrame({
            "ts_local": pd.to_datetime(df.iloc[:, 1], errors="coerce"),
            "ws": _to_number(df.iloc[:, 2]),
            "p": _to_number(df.iloc[:, 3]),
            "t": _to_number(df.iloc[:, 4]),
        }).dropna(subset=["ts_local", "p"])
        d["ts_utc"] = d["ts_local"] - pd.Timedelta(hours=offset)
        d = d.drop_duplicates("ts_utc").sort_values("ts_utc")
        qc = scada.add_qc_flags(d, load_settings()["scada"]["qc"])
        hourly = scada.to_hourly(qc, min_obs=load_settings()["scada"]["min_obs_per_hour"])
        p = hourly["p"].dropna()
        kind = "формат датасета организаторов (10-минутные записи)"
    else:
        time_col = next((c for c in df.columns if _is_time(df[c])), None)
        if time_col is None:
            raise UploadError("не найдена колонка со временем")
        nums = {c: _to_number(df[c]) for c in df.columns if c != time_col}
        named = [c for c in nums if any(k in str(c).lower() for k in ("мощн", "power", "p_farm", "выработ"))]
        cand = named or [c for c, s in nums.items() if s.notna().mean() > 0.9 and s.min() >= 0]
        if not cand:
            raise UploadError("не найдена колонка с мощностью")
        col = cand[0]
        p = nums[col]
        if p.max() > 1.05:
            if p.max() <= 100.5:
                p = p / 100  # проценты номинала
            else:
                raise UploadError("мощность должна быть нормализованной: доля номинала (0–1) или проценты (0–100)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ts = pd.to_datetime(df[time_col], errors="coerce") - pd.Timedelta(hours=offset)
        s = pd.Series(p.to_numpy(), index=ts).dropna()
        s = s[~s.index.isna()].sort_index()
        p = s.groupby(s.index.floor("h")).mean()
        kind = f"простой CSV (время: «{time_col}», мощность: «{col}»)"
    if not len(p):
        raise UploadError("после разбора не осталось ни одного часа с данными")
    p = p.clip(0, 1)
    p.index.name = "target_time_utc"
    return p.rename("fact"), kind


def combine(series: list[pd.Series]) -> pd.Series:
    """Несколько файлов (турбины) → средняя мощность ВЭС по часам."""
    if len(series) == 1:
        return series[0]
    return pd.concat(series, axis=1).mean(axis=1).rename("fact")


def forecast_catalog() -> pd.DataFrame:
    """Все наши прогнозы: тестовый период (с интервалом) и бэктест основной модели."""
    sub = pd.read_csv(resolve("artifacts/submission/forecast_feb2026.csv"),
                      parse_dates=["issue_date", "target_time_utc", "target_time_local"])
    sub = sub[["issue_date", "target_time_utc", "target_time_local", "lead_day", "p_farm", "p_farm_q10", "p_farm_q90"]]
    sub["source"] = "прогноз на февраль 2026"
    bt = pd.read_parquet(resolve("artifacts/reports/backtest_preds.parquet"))
    bt = bt[bt["model"] == "ensemble"].rename(columns={"p_farm_pred": "p_farm_fc"})
    bt = bt[["issue_date", "target_time_utc", "target_time_local", "lead_day", "p_farm_fc"]].rename(columns={"p_farm_fc": "p_farm"})
    bt["source"] = "бэктест основной модели"
    cat = pd.concat([sub, bt], ignore_index=True)
    return cat.drop_duplicates(["target_time_utc", "lead_day"], keep="first")


def evaluate(fact: pd.Series, catalog: pd.DataFrame | None = None) -> dict:
    """Сопоставляет факт с нашими прогнозами. Возвращает таблицу и метрики по суткам упреждения."""
    cat = forecast_catalog() if catalog is None else catalog
    m = cat.merge(fact.reset_index(), on="target_time_utc", how="inner")
    if m.empty:
        raise UploadError("загруженные данные не пересекаются с периодами наших прогнозов "
                          "(февраль 2026, февраль 2025, декабрь 2025, январь 2026)")
    out = {"merged": m.sort_values(["lead_day", "target_time_utc"]), "by_lead": {}, "sources": sorted(m["source"].unique())}
    for lead in (1, 2):
        g = m[m["lead_day"] == lead]
        if len(g):
            s = metrics.score(g["fact"], g["p_farm"])
            if g["p_farm_q10"].notna().any():
                s["coverage_p10_p90"] = metrics.coverage(g["fact"], g["p_farm_q10"], g["p_farm_q90"])
            out["by_lead"][lead] = s
    d1 = m[m["lead_day"] == 1].sort_values("target_time_utc")
    local = d1["target_time_local"].dt.normalize()
    out["daily_mae"] = (d1["p_farm"] - d1["fact"]).abs().groupby(local).mean()
    out["period"] = (m["target_time_local"].min(), m["target_time_local"].max())
    out["hours"] = int(m["target_time_utc"].nunique())
    return out


def sample_file(start: str = "2026-01-01", end: str = "2026-01-31") -> bytes:
    """Пример для кнопки «попробовать»: фрагмент исходного датасета (турбина 1) в формате организаторов."""
    path = resolve(load_settings()["scada"]["files"]["t1"])
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.iloc[:, 1])
    part = df[(ts >= start) & (ts < pd.Timestamp(end) + pd.Timedelta(days=1))]
    return part.to_csv(index=False).encode("utf-8")
