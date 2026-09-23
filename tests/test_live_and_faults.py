"""Живой режим (без сети: архивный кэш вместо свежего), демо-сценарии сбоев, соседние точки ECMWF."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from windagent import forecast, live
from windagent.agent import tools as T
from windagent.agent.orchestrator import run_agent
from windagent.config import load_settings, resolve
from windagent.data import weather
from windagent.data.store import DataStore

pytestmark = pytest.mark.skipif(not resolve(forecast.MODEL_PATH).exists(), reason="нет обученной модели")
SETTINGS = load_settings()


@pytest.fixture(scope="module")
def model():
    return forecast.load_model()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("WINDAGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cache = tmp_path / "live_cache"
    cache.mkdir()
    for f in resolve("cache/weather").glob("*.parquet"):
        shutil.copy(f, cache / f.name)
    return tmp_path


def test_live_run_publishes_and_compares_with_previous(model, runtime):
    first = live.run_live(model, mode="rules", now=pd.Timestamp("2026-02-10 08:00"), refresh=False, log=lambda m: None)
    ptr = live.latest_pointer()
    assert ptr["issue_date"] == "2026-02-10" and ptr["ecmwf_run"] == "2026-02-10 00:00:00"
    assert (runtime / "live" / "2026-02-10" / "1400" / "run.json").exists()
    assert first["versions"] == 1
    # Через 6 часов вышел новый прогон — агент сравнивает с предыдущим живым прогнозом
    live.run_live(model, mode="rules", now=pd.Timestamp("2026-02-10 14:00"), refresh=False, log=lambda m: None)
    run = json.loads((runtime / "live" / "2026-02-10" / "2000" / "run.json").read_text(encoding="utf-8"))
    analysis = next(s["result"] for s in run["steps"] if s["tool"] == "analyze_forecast")
    assert analysis["change_vs_previous_issue"]["hours_compared"] == 48
    assert len(live.history("2026-02-10")) == 2
    # Ничего из будущего: все источники опубликованы до момента прогноза
    for s in run["steps"]:
        if s["tool"] == "fetch_weather":
            assert s["result"]["all_published_before_as_of"]


@pytest.mark.parametrize("fault, excluded", [("icon_corrupt", ["icon"]), ("gfs_gaps", ["gfs"])])
def test_fault_scenarios_make_agent_exclude_source(model, tmp_path, fault, excluded):
    r = run_agent("2026-02-10", model, mode="rules", recheck=False, out_dir=tmp_path, fault=fault)
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["fault"] == fault and run["versions"][0]["excluded_sources"] == excluded
    assert "Исключены источники" in r["explanation"]


def test_late_ecmwf_is_detected_and_lowers_confidence(model, tmp_path):
    st = T.AgentState(issue_date="2026-02-10", model=model, out_dir=tmp_path, fault="ecmwf_late")
    T.fetch_weather(st)
    v = T.validate_weather(st)
    assert v["ecmwf_stale"] and v["ecmwf_run_age_h"] == 14
    T.run_forecast(st)
    a = T.analyze_forecast(st)
    assert any("устарел" in f for f in a["flags"])


def test_neighbor_respects_publication_time(tmp_path):
    rows = []
    for run in pd.date_range("2026-02-09", "2026-02-10", freq="D"):
        valid = pd.date_range(run, periods=72, freq="h")
        rows.append(pd.DataFrame({"run_time": run, "valid_time": valid,
                                  "wind_speed_100m": np.float32(run.day), "wind_direction_100m": np.float32(270)}))
    weather._write(pd.concat(rows), tmp_path / "single_ecmwf_ifs_nb_w.parquet")
    t = pd.date_range("2026-02-10 18:00", periods=6, freq="h")
    early = DataStore("2026-02-10 06:00", weather_dir=tmp_path).neighbor("w", t)   # 00Z ещё не вышел
    late = DataStore("2026-02-10 08:00", weather_dir=tmp_path).neighbor("w", t)
    assert (early["wind_speed_100m"] == 9).all() and (late["wind_speed_100m"] == 10).all()
    assert (pd.to_datetime(late["published_at"]) <= pd.Timestamp("2026-02-10 08:00")).all()
