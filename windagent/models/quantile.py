"""Интервал неопределённости P10–P90 для мощности ВЭС: квантильный бустинг + конформная калибровка.

Квантильный бустинг на обучающей выборке даёт слишком узкие интервалы (переобучение),
поэтому интервал калибруется по схеме CQR (Conformalized Quantile Regression):
  1. последние cal_days перед cutoff — калибровочное окно; модели квантилей учатся до него;
  2. на окне считается, насколько факт выходит за интервал: s = max(q_lo − y, y − q_hi);
  3. поправка = квантиль s уровня coverage; для прогноза используются те же модели,
     что калибровались (строгая схема CQR: гарантия покрытия сохраняется).
Всё — только на данных до cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from windagent import protocol
from windagent.models.baselines import FitContext
from windagent.models.gbm import GbmForecaster


class ConformalInterval:
    def __init__(self, target: str = "p_farm", lo: float = 0.1, hi: float = 0.9, coverage: float = 0.8,
                 cal_days: int = 60, bins: tuple[float, ...] = ()):
        """bins — границы уровня прогноза для раздельной калибровки (Mondrian CQR), напр. (0.2, 0.6):
        ошибка при штиле, среднем ветре и на номинале разная, и поправка своя для каждого уровня."""
        self.target = target
        self.lo, self.hi, self.coverage, self.cal_days = lo, hi, coverage, cal_days
        self.bins = tuple(bins)

    def _models(self):
        t = (self.target,)
        return (
            GbmForecaster("quantile", quantile=self.lo, targets=t),
            GbmForecaster("quantile", quantile=self.hi, targets=t),
        )

    def fit(self, train: pd.DataFrame, ctx: FitContext) -> "ConformalInterval":
        cal_start = ctx.cutoff - pd.Timedelta(days=self.cal_days)
        first_cal_issue = (cal_start + pd.Timedelta(hours=ctx.settings["scada"]["utc_offset_hours"])).normalize()
        cal_cutoff = protocol.issue_time_utc(first_cal_issue, ctx.settings)
        inner = train[train["target_time_utc"] + pd.Timedelta("1h") <= cal_cutoff]
        cal = train[train["issue_date"] >= first_cal_issue]
        inner_ctx = FitContext(
            cutoff=cal_cutoff,
            hourly=ctx.hourly[ctx.hourly.index + pd.Timedelta("1h") <= cal_cutoff],
            settings=ctx.settings,
        )
        m_lo, m_hi = [m.fit(inner, inner_ctx) for m in self._models()]
        y = cal[self.target].to_numpy(dtype=float)
        q_lo = m_lo.predict(cal)[self.target].to_numpy()
        q_hi = m_hi.predict(cal)[self.target].to_numpy()
        ok = ~np.isnan(y)
        scores = np.maximum(q_lo[ok] - y[ok], y[ok] - q_hi[ok])
        self.margin_ = self._quantile(scores)
        self.n_cal_ = len(scores)
        mid = ((q_lo + q_hi) / 2)[ok]
        group = np.digitize(mid, self.bins)
        self.margins_ = {int(g): self._quantile(scores[group == g]) if (group == g).sum() >= 50 else self.margin_
                         for g in range(len(self.bins) + 1)}
        self.m_lo_, self.m_hi_ = m_lo, m_hi
        return self

    def _quantile(self, scores: np.ndarray) -> float:
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        return float(np.quantile(scores, level, method="higher"))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        q_lo = self.m_lo_.predict(X)[self.target]
        q_hi = self.m_hi_.predict(X)[self.target]
        margins = getattr(self, "margins_", None)
        if margins and self.bins:
            group = np.digitize(((q_lo + q_hi) / 2).to_numpy(), self.bins)
            m = pd.Series([margins.get(int(g), self.margin_) for g in group], index=X.index)
        else:
            m = self.margin_
        lo = q_lo - m
        hi = q_hi + m
        return pd.DataFrame({f"{self.target}_q10": lo.clip(0, 1), f"{self.target}_q90": hi.clip(0, 1)}, index=X.index)
