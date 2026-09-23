import numpy as np
import pandas as pd
import pytest

from windagent.config import load_settings
from windagent.data import scada

QC = load_settings()["scada"]["qc"]


def _frame(ts_local, ws, p, t=0.0, offset=6):
    ts_local = pd.to_datetime(ts_local)
    return pd.DataFrame(
        {
            "ts_local": ts_local,
            "ts_utc": ts_local - pd.Timedelta(hours=offset),
            "ws": ws,
            "p": p,
            "t": t,
        }
    )


def test_read_raw_parses_time_without_leading_zero_and_shifts_to_utc(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text(
        "ID,Время,WS,P,T\n"
        "1,2023-03-11 0:00:00,6.7,0.39,15.3\n"
        "2,2023-03-11 0:10:00,6.8,0.40,15.2\n",
        encoding="utf-8",
    )
    df = scada.read_raw(csv, utc_offset_hours=6)
    assert list(df.columns) == ["ts_local", "ts_utc", "ws", "p", "t"]
    assert df["ts_local"].iloc[0] == pd.Timestamp("2023-03-11 00:00")
    assert df["ts_utc"].iloc[0] == pd.Timestamp("2023-03-10 18:00")


def test_downtime_flag():
    df = _frame(
        pd.date_range("2025-01-01", periods=3, freq="10min"),
        ws=[8.0, 8.0, 2.0],
        p=[0.01, 0.50, 0.01],
    )
    out = scada.add_qc_flags(df, QC)
    # Штиль с минимальной мощностью — это не простой
    assert out["downtime"].tolist() == [True, False, False]


def test_curtailment_flag_on_plateau_with_changing_wind():
    n = 8
    df = _frame(
        pd.date_range("2025-01-01", periods=n, freq="10min"),
        ws=np.linspace(9, 13, n),
        p=[0.61] * n,
    )
    out = scada.add_qc_flags(df, QC)
    assert out["curtailment"].all()
    assert not out["ok"].any()


def test_rated_plateau_is_not_curtailment():
    n = 12
    df = _frame(
        pd.date_range("2025-01-01", periods=n, freq="10min"),
        ws=np.linspace(12, 16, n),
        p=[0.99] * n,
    )
    out = scada.add_qc_flags(df, QC)
    assert not out["curtailment"].any()


def test_to_hourly_aggregation_and_validity():
    # Час 1: 6 записей; час 2: 3 записи (невалиден); час 3: пропущен полностью; час 4: 6 записей
    ts = list(pd.date_range("2025-01-01 06:00", periods=6, freq="10min"))
    ts += list(pd.date_range("2025-01-01 07:00", periods=3, freq="10min"))
    ts += list(pd.date_range("2025-01-01 09:00", periods=6, freq="10min"))
    p = [0.2] * 6 + [0.5] * 3 + [0.8] * 6
    df = scada.add_qc_flags(_frame(ts, ws=6.0, p=p), QC)
    h = scada.to_hourly(df, min_obs=4)

    assert h.index[0] == pd.Timestamp("2025-01-01 00:00")  # 06:00 UTC+6 → 00:00 UTC
    assert len(h) == 4  # сетка без пропусков
    assert h["p"].iloc[0] == pytest.approx(0.2)
    assert h["valid"].tolist() == [True, False, False, True]
    assert np.isnan(h["p"].iloc[1]) and np.isnan(h["p"].iloc[2])
    assert h["n_obs"].tolist() == [6, 3, 0, 6]


# --- Тесты на реальных данных -------------------------------------------------


@pytest.fixture(scope="module")
def hourly():
    return scada.load_hourly()


@pytest.mark.parametrize("tb", scada.TURBINES)
def test_real_10min_data_integrity(tb):
    d = scada.load_turbine_10min(tb)
    assert d["ts_utc"].is_unique
    assert d["ts_utc"].is_monotonic_increasing
    assert d[["ws", "p", "t"]].notna().all().all()
    assert d["p"].between(0, 1).all()
    assert d["ts_local"].min() == pd.Timestamp("2023-03-11 00:00")
    assert d["ts_local"].max() == pd.Timestamp("2026-01-31 23:50")
    # Доля помеченных записей разумная: флаги не «съедают» данные
    assert d["ok"].mean() > 0.95


def test_real_hourly_dataset(hourly):
    assert hourly.index.is_unique and hourly.index.is_monotonic_increasing
    assert hourly.index.to_series().diff().dropna().eq(pd.Timedelta("1h")).all()
    # Последний час: 31.01.2026 23:00 UTC+6 = 17:00 UTC; февраля в данных нет
    assert hourly.index.max() == pd.Timestamp("2026-01-31 17:00")
    assert hourly["p_farm"].dropna().between(0, 1).all()
    assert hourly["p_farm"].notna().mean() > 0.97
    # ВЭС = среднее двух турбин, если обе есть
    both = hourly.dropna(subset=["p_t1", "p_t2"]).iloc[:100]
    assert np.allclose(both["p_farm"], (both["p_t1"] + both["p_t2"]) / 2)
    # Большая дыра турбины 1 летом 2024 закрыта данными турбины 2
    gap = hourly.loc["2024-06-01":"2024-06-30"]
    assert gap["p_t1"].isna().mean() > 0.9
    assert gap["p_farm"].notna().mean() > 0.9
