import numpy as np
import pandas as pd
import pytest

from windagent import protocol
from windagent.config import load_settings
from windagent.eval import backtest, metrics
from windagent.features import build
from windagent.models.baselines import Climatology, FitContext, NwpCurve
from windagent.models.power_curve import PowerCurve

SETTINGS = load_settings()


# --- Метрики --------------------------------------------------------------------------


def test_metrics_known_values():
    t = [0.0, 0.5, 1.0, np.nan]
    p = [0.1, 0.3, 1.0, 0.7]
    s = metrics.score(t, p)
    assert s["n"] == 3
    assert s["MAE"] == pytest.approx(0.1)
    assert s["RMSE"] == pytest.approx(np.sqrt((0.01 + 0.04) / 3))
    assert s["bias"] == pytest.approx((0.1 - 0.2) / 3)
    assert s["nMAE_%"] == pytest.approx(10.0)
    assert metrics.pinball([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert metrics.pinball([0.0], [1.0], 0.9) == pytest.approx(0.1)
    assert metrics.coverage([0.5, 0.9], [0.4, 0.4], [0.6, 0.6]) == 0.5


def test_skill_vs_reference():
    table = pd.DataFrame({"model": ["clim", "m"], "MAE": [0.2, 0.15]})
    out = metrics.add_skill(table, "clim")
    assert out.set_index("model").loc["m", "skill_vs_clim"] == pytest.approx(0.25)


# --- Кривая мощности ------------------------------------------------------------------


def _logistic(ws):
    return 0.01 + 0.98 / (1 + np.exp(-(ws - 8) * 0.9))


def test_power_curve_recovers_shape_and_is_monotone():
    rng = np.random.default_rng(0)
    ws = rng.uniform(0, 20, 5000)
    p = np.clip(_logistic(ws) + rng.normal(0, 0.03, ws.size), 0, 1)
    pc = PowerCurve(bin_width=0.5).fit(ws, p)
    grid = np.linspace(0, 20, 81)
    pred = pc.predict(grid)
    assert np.all(np.diff(pred) >= -1e-12)
    assert pred.min() >= 0.01 and pred.max() <= 0.99
    assert np.abs(pred - _logistic(grid)).mean() < 0.03
    assert np.isnan(pc.predict([np.nan]))[0]


# --- Бэктест на синтетике -----------------------------------------------------------------


def _synthetic_dataset(first="2025-01-01", last="2025-03-10"):
    """Выпуски с признаком ветра и фактом, который зависит от ветра по логистической кривой."""
    rng = np.random.default_rng(1)
    parts = []
    for d in protocol.issue_dates(first, last):
        tf = protocol.target_frame(d, SETTINGS)
        tf.insert(0, "issue_date", d)
        tf.insert(1, "issue_time_utc", protocol.issue_time_utc(d, SETTINGS))
        parts.append(tf)
    ds = pd.concat(parts, ignore_index=True)
    # Ветер — функция целевого часа (один и тот же для двух выпусков, покрывающих час)
    hours = (ds["target_time_utc"] - pd.Timestamp("2025-01-01")) / pd.Timedelta("1h")
    ws = 8 + 5 * np.sin(hours / 7.0)
    ds["ifs__wind_speed_100m"] = ws + rng.normal(0, 0.5, len(ds))
    for c in ("p_t1", "p_t2", "p_farm", "p_clean_t1", "p_clean_t2", "p_farm_clean"):
        ds[c] = np.clip(_logistic(ws), 0, 1)
    return ds


def _hourly_from(ds):
    h = ds.drop_duplicates("target_time_utc").set_index("target_time_utc")[["p_t1", "p_t2", "p_farm"]]
    return h.sort_index()


def test_split_uses_only_known_truth():
    ds = _synthetic_dataset()
    train, test, cutoff = backtest.split(ds, "2025-02-01", "2025-02-10", SETTINGS)
    assert cutoff == protocol.issue_time_utc("2025-02-01", SETTINGS)
    assert (train["target_time_utc"] + pd.Timedelta("1h") <= cutoff).all()
    assert (train["issue_date"] < pd.Timestamp("2025-02-01")).all()
    assert test["issue_date"].nunique() == 10 and len(test) == 10 * 48


def test_run_backtest_end_to_end_curve_beats_climatology():
    ds = _synthetic_dataset()
    preds = backtest.run_backtest(
        [Climatology(), NwpCurve("ifs")], ds, "2025-02-01", "2025-02-28", SETTINGS, hourly=_hourly_from(ds)
    )
    assert set(preds["model"]) == {"clim", "curve_ifs"}
    assert preds["p_farm_pred"].between(0, 1).all()
    table = backtest.report(preds)
    overall = table[table["lead_day"] == "all"].set_index("model")
    assert overall.loc["curve_ifs", "MAE"] < 0.5 * overall.loc["clim", "MAE"]
    assert set(table["lead_day"]) == {"all", "1", "2"}


def test_climatology_ignores_data_after_cutoff():
    idx = pd.date_range("2025-01-01", "2025-03-01", freq="h")
    h = pd.DataFrame({"p_t1": 0.2, "p_t2": 0.2, "p_farm": 0.2}, index=idx)
    cutoff = pd.Timestamp("2025-02-01")
    h.loc[h.index >= cutoff, :] = 0.9  # «будущее» — не должно влиять
    ctx = FitContext(cutoff=cutoff, hourly=h[h.index + pd.Timedelta("1h") <= cutoff], settings=SETTINGS)
    m = Climatology().fit(None, ctx)
    X = protocol.target_frame("2025-02-10", SETTINGS)
    assert np.allclose(m.predict(X)["p_farm"], 0.2)


# --- Сборка признаков на реальном кэше -----------------------------------------------------


def _cache_ready():
    from windagent.data import weather

    return len(weather.load_previous("gfs_seamless")) > 0


@pytest.mark.skipif(not _cache_ready(), reason="кэш погоды не скачан")
def test_issue_frame_real_cache():
    f = build.issue_frame("2026-02-10", settings=SETTINGS)
    assert len(f) == 48
    assert f["issue_time_utc"].iloc[0] == pd.Timestamp("2026-02-10 08:00")
    # К 14:00 местного (08:00 UTC) доступен прогон ECMWF 00Z: ему 8 часов
    assert (f["ifs__run_age_h"] == 8).all()
    assert f["ifs__lead_h"].min() == 18 and f["ifs__lead_h"].max() == 65
    for pre in ("ifs025", "gfs", "icon"):
        assert f[f"{pre}__wind_speed_100m"].notna().all()
        assert f[f"{pre}__day"].isin([1, 2, 3]).all()
