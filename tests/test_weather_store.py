"""Тесты слоя погоды и DataStore. Работают без сети: HTTP подменяется, кэш синтетический."""

import json

import httpx
import numpy as np
import pandas as pd
import pytest

from windagent import protocol
from windagent.config import load_settings
from windagent.data import weather
from windagent.data.openmeteo import OpenMeteoClient, OpenMeteoError
from windagent.data.store import DataStore, LeakageError

SETTINGS = load_settings()
VARS = weather.single_variables("ecmwf_ifs", SETTINGS)


# --- Синтетический кэш --------------------------------------------------------


def _single_cache(directory, first="2026-02-08", last="2026-02-11"):
    """Прогоны ECMWF каждые 6 ч; значение ветра = номер часа старта прогона (для проверки выбора)."""
    rows = []
    for run in pd.date_range(first, last, freq="6h"):
        valid = pd.date_range(run, periods=72, freq="h")
        df = pd.DataFrame({"run_time": run, "valid_time": valid})
        for v in VARS:
            df[v] = np.float32(run.hour)
        rows.append(df)
    weather._write(pd.concat(rows, ignore_index=True), directory / "single_ecmwf_ifs.parquet")


def _previous_cache(directory, model="gfs_seamless"):
    """previous_dayN: значение ветра = N, чтобы видеть, какой день выбран."""
    valid = pd.date_range("2026-02-01", "2026-02-20", freq="h")
    parts = []
    for d in SETTINGS["weather"]["previous_days"]:
        df = pd.DataFrame({"valid_time": valid, "day": d})
        for v in SETTINGS["weather"]["variables"]:
            df[v] = np.float32(d)
        parts.append(df)
    weather._write(pd.concat(parts, ignore_index=True), directory / f"previous_{model}.parquet")


@pytest.fixture
def cache(tmp_path):
    _single_cache(tmp_path)
    _previous_cache(tmp_path)
    return tmp_path


# --- Протокол выпусков ----------------------------------------------------------


def test_protocol_issue_and_targets():
    issue = protocol.issue_time_utc("2026-01-31")
    assert issue == pd.Timestamp("2026-01-31 08:00")  # 14:00 UTC+6
    tf = protocol.target_frame("2026-01-31")
    assert len(tf) == 48
    assert tf["target_time_local"].iloc[0] == pd.Timestamp("2026-02-01 00:00")
    assert tf["target_time_local"].iloc[-1] == pd.Timestamp("2026-02-02 23:00")
    assert tf["target_time_utc"].iloc[0] == pd.Timestamp("2026-01-31 18:00")
    assert tf["lead_day"].tolist() == [1] * 24 + [2] * 24
    assert tf["horizon_h"].iloc[0] == 10 and tf["horizon_h"].iloc[-1] == 57
    assert len(protocol.issue_dates("2026-01-31", "2026-02-27")) == 28


# --- Single Runs -----------------------------------------------------------------


def test_latest_run_respects_publish_lag(cache):
    # Прогон 00Z публикуется в 07:00 (лаг 7 ч)
    assert DataStore("2026-02-10 08:00", weather_dir=cache).latest_run("ecmwf_ifs") == pd.Timestamp("2026-02-10 00:00")
    assert DataStore("2026-02-10 06:59", weather_dir=cache).latest_run("ecmwf_ifs") == pd.Timestamp("2026-02-09 18:00")
    assert DataStore("2026-02-10 07:00", weather_dir=cache).latest_run("ecmwf_ifs") == pd.Timestamp("2026-02-10 00:00")


def test_single_run_returns_latest_available_values(cache):
    st = DataStore("2026-02-10 08:00", weather_dir=cache)
    t = protocol.target_hours_utc("2026-02-10")
    f = st.single_run("ecmwf_ifs", t)
    assert (f["run_time"] == pd.Timestamp("2026-02-10 00:00")).all()
    assert (f["published_at"] <= st.as_of).all()
    assert f["wind_speed_100m"].notna().all()
    assert f["lead_h"].min() == 18 and f["lead_h"].max() == 65


def test_requesting_future_run_raises_leakage(cache):
    st = DataStore("2026-02-10 08:00", weather_dir=cache)
    t = protocol.target_hours_utc("2026-02-10")
    with pytest.raises(LeakageError):
        st.single_run("ecmwf_ifs", t, run_time="2026-02-10 06:00")


def test_lagged_ensemble_is_newest_first_and_all_available(cache):
    st = DataStore("2026-02-10 08:00", weather_dir=cache)
    runs = st.single_runs_lagged("ecmwf_ifs", protocol.target_hours_utc("2026-02-10"), n_runs=3)
    assert [r["run_time"].iloc[0] for r in runs] == [
        pd.Timestamp("2026-02-10 00:00"),
        pd.Timestamp("2026-02-09 18:00"),
        pd.Timestamp("2026-02-09 12:00"),
    ]
    st.check_manifest()


# --- Previous Runs -------------------------------------------------------------------


def test_previous_runs_choose_freshest_allowed_day(cache):
    st = DataStore("2026-02-10 08:00", weather_dir=cache)
    t = protocol.target_hours_utc("2026-02-10")
    p = st.previous_runs("gfs_seamless", t)
    lag = pd.Timedelta(hours=SETTINGS["weather"]["previous_runs"]["gfs_seamless"]["publish_lag_h"])

    assert p["day"].notna().all()
    # Ни один использованный прогон не опубликован позже as_of
    assert (pd.to_datetime(p["run_time"]) + lag <= st.as_of).all()
    # Значение ветра = номер дня → убеждаемся, что взяты данные именно выбранного дня
    assert (p["wind_speed_100m"] == p["day"]).all()
    # Выбран минимальный допустимый день: с day-1 прогон был бы ещё не опубликован
    for ts, row in p.iterrows():
        d = int(row["day"])
        if d > 1:
            earlier_run = pd.Timestamp(ts).floor("6h") - pd.Timedelta(hours=24 * (d - 1))
            assert earlier_run + lag > st.as_of


def test_previous_runs_first_target_uses_day1(cache):
    st = DataStore("2026-02-10 08:00", weather_dir=cache)
    p = st.previous_runs("gfs_seamless", protocol.target_hours_utc("2026-02-10"))
    # Час 18:00 UTC 10.02 → run = 18:00 09.02 (+5 ч = 23:00 09.02) — уже опубликован
    assert p["day"].iloc[0] == 1
    assert p["run_time"].iloc[0] == pd.Timestamp("2026-02-09 18:00")


def test_previous_runs_nothing_allowed_gives_nan(cache):
    # as_of слишком ранний: даже day3 ещё не вышел для этих часов
    st = DataStore("2026-02-05 00:00", weather_dir=cache)
    p = st.previous_runs("gfs_seamless", pd.date_range("2026-02-15", periods=6, freq="h"))
    assert p["day"].isna().all()


# --- SCADA ---------------------------------------------------------------------------


def test_scada_only_completed_hours():
    idx = pd.date_range("2026-01-31 00:00", periods=24, freq="h")
    h = pd.DataFrame({"p_farm": np.linspace(0, 1, 24)}, index=idx)
    st = DataStore("2026-01-31 08:00", scada_hourly=h)
    out = st.scada_hourly()
    assert out.index.max() == pd.Timestamp("2026-01-31 07:00")  # час 07:00–08:00 закончился в 08:00
    assert (out.index + pd.Timedelta("1h") <= st.as_of).all()


# --- HTTP-клиент (без сети) ---------------------------------------------------------


def _client(handler):
    return OpenMeteoClient(43.6, 78.5, transport=httpx.MockTransport(handler), retries=3, max_rps=0)


def test_client_parses_single_run():
    def handler(request):
        assert request.url.params["run"] == "2026-02-10T00:00"
        assert request.url.params["wind_speed_unit"] == "ms"
        return httpx.Response(200, json={"hourly": {"time": ["2026-02-10T00:00", "2026-02-10T01:00"],
                                                    "wind_speed_100m": [5.0, 6.5]}})

    df = _client(handler).single_run("ecmwf_ifs", pd.Timestamp("2026-02-10"), ["wind_speed_100m"], 2)
    assert list(df.columns) == ["run_time", "valid_time", "wind_speed_100m"]
    assert df["wind_speed_100m"].tolist() == [5.0, 6.5]


def test_client_not_available_and_all_null_are_none():
    def not_avail(request):
        return httpx.Response(400, json={"error": True, "reason": "The requested model run is not available."})

    def all_null(request):
        return httpx.Response(200, json={"hourly": {"time": ["2026-02-10T00:00"], "wind_speed_100m": [None]}})

    assert _client(not_avail).single_run("x", pd.Timestamp("2026-02-10"), ["wind_speed_100m"], 1) is None
    assert _client(all_null).single_run("x", pd.Timestamp("2026-02-10"), ["wind_speed_100m"], 1) is None


def test_client_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr("windagent.data.openmeteo.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"reason": "Too many requests"})
        return httpx.Response(200, json={"hourly": {"time": ["2026-02-10T00:00"], "wind_speed_100m": [1.0]}})

    df = _client(handler).single_run("x", pd.Timestamp("2026-02-10"), ["wind_speed_100m"], 1)
    assert calls["n"] == 3 and len(df) == 1


def test_client_raises_on_bad_request():
    def handler(request):
        return httpx.Response(400, json={"error": True, "reason": "Cannot initialize variable"})

    with pytest.raises(OpenMeteoError):
        _client(handler).single_run("x", pd.Timestamp("2026-02-10"), ["bad"], 1)


def test_download_skips_cached_and_remembers_missing(tmp_path):
    requested = []

    def handler(request):
        run = request.url.params["run"]
        requested.append(run)
        if run.endswith("06:00"):
            return httpx.Response(400, json={"error": True, "reason": "The requested model run is not available."})
        hours = pd.date_range(run, periods=3, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
        return httpx.Response(200, json={"hourly": {"time": hours, **{v: [1.0] * 3 for v in VARS}}})

    client = _client(handler)
    r1 = weather.download_single_runs("ecmwf_ifs", "2026-02-10", "2026-02-10", client=client,
                                      directory=tmp_path, log=lambda m: None)
    assert r1 == {"model": "ecmwf_ifs", "downloaded": 3, "not_in_archive": 1, "failed": 0, "total": 3}
    assert json.loads((tmp_path / "single_ecmwf_ifs_missing.json").read_text()) == ["2026-02-10 06:00:00"]

    requested.clear()
    r2 = weather.download_single_runs("ecmwf_ifs", "2026-02-10", "2026-02-10", client=client,
                                      directory=tmp_path, log=lambda m: None)
    assert requested == [] and r2["downloaded"] == 0


# --- Живой API (запуск: pytest -m network) ------------------------------------------


@pytest.mark.network
def test_live_single_run_available():
    with weather.make_client() as c:
        df = c.single_run("ecmwf_ifs", pd.Timestamp("2026-02-10 00:00"), ["wind_speed_100m"], 48)
    assert df is not None and df["wind_speed_100m"].notna().all()


# --- Реальный кэш: протокол тестового периода ----------------------------------------


def _real_cache_ready():
    return len(weather.load_single("ecmwf_ifs")) > 0 and len(weather.load_previous("gfs_seamless")) > 0


@pytest.mark.skipif(not _real_cache_ready(), reason="кэш погоды не скачан")
def test_real_cache_no_leakage_for_all_test_issues():
    f = SETTINGS["forecast"]
    for d in protocol.issue_dates(f["test_first_issue"], f["test_last_issue"]):
        st = DataStore(protocol.issue_time_utc(d))
        t = protocol.target_hours_utc(d)
        single = st.single_run("ecmwf_ifs", t)
        assert single["wind_speed_100m"].notna().all(), d
        assert single["run_time"].iloc[0] == d  # к 14:00 местного доступен прогон 00Z того же дня
        for m in SETTINGS["weather"]["previous_runs"]:
            prev = st.previous_runs(m, t)
            assert prev["wind_speed_100m"].notna().all(), (d, m)
        st.check_manifest()


def test_single_run_fills_gaps_from_older_runs(tmp_path):
    # Прогона 00Z 10.02 нет в архиве → часы покрываются прогоном 18Z 09.02
    _single_cache(tmp_path)
    df = weather.load_single("ecmwf_ifs", tmp_path)
    weather._write(df[df["run_time"] != pd.Timestamp("2026-02-10 00:00")], tmp_path / "single_ecmwf_ifs.parquet")
    st = DataStore("2026-02-10 08:00", weather_dir=tmp_path)
    t = protocol.target_hours_utc("2026-02-10")
    f = st.single_run("ecmwf_ifs", t)
    assert f["wind_speed_100m"].notna().all()
    assert (f["run_time"] == pd.Timestamp("2026-02-09 18:00")).all()
    assert (f["wind_speed_100m"] == 18).all()


def test_single_run_mixes_runs_when_horizon_is_short(tmp_path):
    # Свежий прогон с коротким горизонтом: хвост берётся из более старого прогона
    _single_cache(tmp_path)
    df = weather.load_single("ecmwf_ifs", tmp_path)
    fresh = pd.Timestamp("2026-02-10 00:00")
    df = df[~((df["run_time"] == fresh) & (df["valid_time"] > pd.Timestamp("2026-02-11 00:00")))]
    weather._write(df, tmp_path / "single_ecmwf_ifs.parquet")
    st = DataStore("2026-02-10 08:00", weather_dir=tmp_path)
    f = st.single_run("ecmwf_ifs", protocol.target_hours_utc("2026-02-10"))
    assert f["wind_speed_100m"].notna().all()
    assert (f.loc[:"2026-02-11 00:00", "run_time"] == fresh).all()
    assert (pd.to_datetime(f.loc["2026-02-11 01:00":, "run_time"]) < fresh).all()
    assert (pd.to_datetime(f["published_at"]) <= st.as_of).all()
