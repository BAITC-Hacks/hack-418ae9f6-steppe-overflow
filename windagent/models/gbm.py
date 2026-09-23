"""Основная модель: градиентный бустинг над признаками прогнозов погоды.

К признакам погоды добавляются прогнозы «физической» кривой мощности
(кривая турбин по анемометру, применённая к ветру каждой погодной модели) —
это даёт бустингу хорошую отправную точку и делает его устойчивее.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from windagent.features.engineer import SOURCES, add_features, feature_columns
from windagent.models.baselines import TARGETS, FitContext, Forecaster, PhysicalCurve

DEFAULT_PARAMS = dict(
    learning_rate=0.03,
    max_iter=600,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    max_features=0.8,
    random_state=42,
)


class GbmForecaster(Forecaster):
    def __init__(
        self,
        loss: str = "absolute_error",
        train_on: str = "raw",
        name: str | None = None,
        targets: tuple[str, ...] = TARGETS,
        **params,
    ):
        """loss: absolute_error (оптимум для MAE) или squared_error (для RMSE).
        train_on: raw — учиться на фактической мощности (с простоями), clean — без простоев."""
        self.loss = loss
        self.train_on = train_on
        self.targets = targets
        self.params = {**DEFAULT_PARAMS, **params}
        short = {"absolute_error": "l1", "squared_error": "l2", "quantile": f"q{int(params.get('quantile', 0.5) * 100)}"}
        self.name = name or f"gbm_{short[loss]}_{train_on}"

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        F = add_features(X)
        for src in SOURCES:
            ws = F[f"{src}__wind_speed_100m"] if f"{src}__wind_speed_100m" in F else np.nan
            pc = self.curve_.predict(pd.DataFrame({"ifs__wind_speed_100m": ws}, index=F.index))
            F[f"pc__{src}"] = pc["p_farm"].to_numpy()
        return F

    def _target(self, train: pd.DataFrame, t: str) -> pd.Series:
        if self.train_on == "clean":
            return train["p_farm_clean"] if t == "p_farm" else train[t.replace("p_", "p_clean_")]
        return train[t]

    def fit(self, train: pd.DataFrame, ctx: FitContext) -> "GbmForecaster":
        # Физическая кривая только по SCADA до cutoff; ветер подставляется из разных моделей
        self.curve_ = PhysicalCurve("ifs").fit(train, ctx)
        F = self._prepare(train)
        # Признаки без информации в обучающей выборке (все пропуски или одно значение) не нужны
        self.columns_ = [c for c in feature_columns(F) if F[c].nunique(dropna=True) > 1]
        self.models_ = {}
        for t in self.targets:
            y = self._target(train, t)
            ok = y.notna().to_numpy()
            m = HistGradientBoostingRegressor(loss=self.loss, **self.params)
            m.fit(F.loc[ok, self.columns_], y[ok])
            self.models_[t] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        F = self._prepare(X).reindex(columns=self.columns_)
        return pd.DataFrame({t: m.predict(F) for t, m in self.models_.items()}, index=X.index).clip(0.0, 1.0)


class EnsembleForecaster(Forecaster):
    """Итоговая модель E: среднее бустинга с MAE-потерей, бустинга с MSE-потерей и физической кривой.

    Компоненты ошибаются по-разному (медиана / среднее / физика), поэтому равновзвешенное
    среднее устойчиво лучше каждого из них на всех контрольных периодах.
    """

    name = "ensemble"

    def __init__(self, members: list[Forecaster] | None = None, targets: tuple[str, ...] = TARGETS):
        self.members = members or [
            GbmForecaster("absolute_error", targets=targets),
            GbmForecaster("squared_error", targets=targets),
            PhysicalCurve("ifs"),
        ]
        self.targets = targets

    def fit(self, train: pd.DataFrame, ctx: FitContext) -> "EnsembleForecaster":
        for m in self.members:
            m.fit(train, ctx)
        return self

    def member_predictions(self, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {m.name: m.predict(X)[list(self.targets)] for m in self.members}

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = list(self.member_predictions(X).values())
        return (sum(preds) / len(preds)).clip(0.0, 1.0)
