"""Проверки итогового прогноза на февраль 2026 и его воспроизводимости."""

import json

import numpy as np
import pandas as pd
import pytest

from windagent import forecast, protocol
from windagent.config import load_settings, resolve
from windagent.data.store import LeakageError
from windagent.models.baselines import TARGETS, Forecaster

SETTINGS = load_settings()
SUB = resolve(forecast.SUBMISSION_DIR)
FULL = SUB / "forecast_feb2026.csv"
DAY_AHEAD = SUB / "forecast_feb2026_dayahead.csv"


class _Const(Forecaster):
    name = "const"

    def predict(self, X):
        return pd.DataFrame({t: 0.5 for t in TARGETS}, index=X.index)


class _Interval:
    def predict(self, X):
        return pd.DataFrame({"p_farm_q10": 0.6, "p_farm_q90": 1.2}, index=X.index)


def test_interval_always_contains_point_and_respects_cap():
    m = forecast.ProductionModel(point=_Const(), interval=_Interval(), meta={})
    m.point.predict = _Const().predict  # EnsembleForecaster не нужен для проверки логики
    out = m.predict(pd.DataFrame(index=range(3)))
    assert (out["p_farm_q10"] <= out["p_farm"]).all()  # 0.6 > 0.5 → опущен до прогноза
    assert (out["p_farm_q90"] == forecast.P_MAX).all()  # 1.2 → ограничен физическим максимумом


def test_model_trained_after_issue_is_rejected():
    m = forecast.ProductionModel(point=_Const(), interval=_Interval(),
                                 meta={"trained_until_utc": "2026-02-15 08:00:00", "version": "x"})
    with pytest.raises(LeakageError):
        forecast.forecast_issue("2026-02-10", m, SETTINGS)


# --- Файлы сабмита -----------------------------------------------------------------------

needs_submission = pytest.mark.skipif(not FULL.exists(), reason="прогноз не построен: windagent submission")


@needs_submission
def test_submission_shape_and_values():
    f = pd.read_csv(FULL, parse_dates=["issue_date", "target_time_utc", "target_time_local"])
    assert len(f) == 28 * 48 and f["issue_date"].nunique() == 28
    assert f.groupby("issue_date").size().eq(48).all()
    p = f[["p_t1", "p_t2", "p_farm", "p_farm_q10", "p_farm_q90"]]
    assert p.notna().all().all()
    assert p.ge(0).all().all() and p.le(1).all().all()
    assert (f["p_farm_q10"] <= f["p_farm"]).all() and (f["p_farm"] <= f["p_farm_q90"]).all()
    assert f["p_farm_q90"].max() <= forecast.P_MAX
    assert f["model_version"].nunique() == 1


@needs_submission
def test_day_ahead_covers_every_hour_of_february():
    d = pd.read_csv(DAY_AHEAD, parse_dates=["target_time_local"])
    expected = pd.date_range("2026-02-01 00:00", "2026-02-28 23:00", freq="h")
    assert d["target_time_local"].is_unique
    assert pd.DatetimeIndex(d["target_time_local"]).equals(expected)
    assert (d["lead_day"] == 1).all()


@needs_submission
def test_all_manifests_have_no_leakage():
    files = sorted((SUB / "manifests").glob("*.json"))
    assert len(files) == 28
    for path in files:
        m = json.loads(path.read_text(encoding="utf-8"))
        as_of = pd.Timestamp(m["as_of_utc"])
        assert as_of == protocol.issue_time_utc(m["issue_date"], SETTINGS)
        assert m["checks"]["all_sources_published_before_as_of"] is True
        assert pd.Timestamp(m["model_trained_until_utc"]) <= as_of
        for src in m["sources"]:
            if src.get("max_published_at"):
                assert pd.Timestamp(src["max_published_at"]) <= as_of, (path.name, src)


@needs_submission
def test_forecast_is_reproducible_from_committed_model():
    model = forecast.load_model()
    f, _ = forecast.forecast_issue("2026-02-10", model, SETTINGS)
    saved = pd.read_csv(FULL, parse_dates=["issue_date"])
    saved = saved[saved["issue_date"] == "2026-02-10"].reset_index(drop=True)
    assert np.allclose(f["p_farm"].to_numpy(), saved["p_farm"].to_numpy(), atol=1e-4)
    assert np.allclose(f["p_farm_q90"].to_numpy(), saved["p_farm_q90"].to_numpy(), atol=1e-4)
