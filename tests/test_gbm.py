import numpy as np
import pandas as pd

from windagent.config import load_settings
from windagent.eval import backtest
from windagent.features.engineer import add_features, feature_columns
from windagent.models.baselines import TARGETS, Forecaster
from windagent.models.gbm import EnsembleForecaster, GbmForecaster

from tests.test_baselines_backtest import _hourly_from, _synthetic_dataset

SETTINGS = load_settings()


def test_features_stay_within_issue_and_keep_row_order():
    ds = _synthetic_dataset("2025-01-01", "2025-01-05")
    F = add_features(ds)
    assert F.index.equals(ds.index)
    # Сдвиг «предыдущий час» не переносит значение из соседнего выпуска
    first_rows = F.groupby("issue_date").head(1)
    assert first_rows["ifs__ws100_prev"].isna().all()
    cols = feature_columns(F)
    assert "horizon_h" in cols and "cal__hour_sin" in cols
    assert not any(c in cols for c in ("p_farm", "p_t1", "target_time_utc"))


class _Const(Forecaster):
    def __init__(self, v, name):
        self.v, self.name = v, name

    def predict(self, X):
        return pd.DataFrame({t: self.v for t in TARGETS}, index=X.index)


def test_ensemble_is_mean_of_members():
    X = pd.DataFrame(index=range(3))
    e = EnsembleForecaster(members=[_Const(0.2, "a"), _Const(0.4, "b"), _Const(0.9, "c")])
    assert np.allclose(e.predict(X)["p_farm"], 0.5)


def test_gbm_end_to_end_on_synthetic(monkeypatch):
    ds = _synthetic_dataset()
    # Признак без информации (все NaN) не должен ломать обучение
    ds["gfs__wind_gusts_10m"] = np.nan
    # Физическая кривая в GBM строится по SCADA; на синтетике подменяем её простой кривой
    from windagent.models import gbm

    class FakeCurve(Forecaster):
        name = "phys_ifs"

        def fit(self, train, ctx):
            return self

        def predict(self, X):
            ws = X["ifs__wind_speed_100m"].astype(float)
            p = (1 / (1 + np.exp(-(ws - 8) * 0.9))).clip(0.01, 0.99)
            return pd.DataFrame({"p_t1": p, "p_t2": p, "p_farm": p}, index=X.index)

    monkeypatch.setattr(gbm, "PhysicalCurve", lambda src="ifs": FakeCurve())
    model = GbmForecaster(max_iter=50)
    preds = backtest.run_backtest([model], ds, "2025-02-01", "2025-02-28", SETTINGS, hourly=_hourly_from(ds))
    assert "gfs__wind_gusts_10m" not in model.columns_
    assert preds["p_farm_pred"].between(0, 1).all()
    table = backtest.report(preds.assign(model="gbm"), reference="gbm")
    assert table.loc[table["lead_day"] == "all", "MAE"].iloc[0] < 0.1
