"""Эмпирическая кривая мощности: монотонная зависимость мощности от скорости ветра."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class PowerCurve:
    """Монотонно неубывающая кривая p(ws), построенная изотонической регрессией.

    Изотоническая регрессия на сырых точках даёт условное среднее мощности при
    данной скорости — если ws взят из прогноза погоды, кривая автоматически
    «сглаживается» с учётом ошибки прогноза ветра.
    """

    def __init__(self, p_min: float = 0.01, p_max: float = 0.99, bin_width: float | None = None):
        self.p_min = p_min
        self.p_max = p_max
        self.bin_width = bin_width
        self._iso: IsotonicRegression | None = None

    def fit(self, ws, p) -> "PowerCurve":
        ws = np.asarray(ws, dtype=float)
        p = np.asarray(p, dtype=float)
        ok = ~(np.isnan(ws) | np.isnan(p))
        ws, p = ws[ok], p[ok]
        if len(ws) < 10:
            raise ValueError("слишком мало точек для кривой мощности")
        weights = None
        if self.bin_width:
            # Медианы по корзинам — устойчиво к выбросам и быстро на 10-мин данных
            df = pd.DataFrame({"b": np.floor(ws / self.bin_width), "p": p})
            g = df.groupby("b")["p"]
            ws = (g.median().index.to_numpy() + 0.5) * self.bin_width
            weights = g.size().to_numpy()
            p = g.median().to_numpy()
        self._iso = IsotonicRegression(y_min=self.p_min, y_max=self.p_max, increasing=True, out_of_bounds="clip")
        self._iso.fit(ws, p, sample_weight=weights)
        return self

    def predict(self, ws) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("кривая не обучена")
        ws = np.asarray(ws, dtype=float)
        out = np.full(ws.shape, np.nan)
        ok = ~np.isnan(ws)
        if ok.any():
            out[ok] = self._iso.predict(ws[ok])
        return out

    def table(self, ws_grid=None) -> pd.DataFrame:
        grid = np.arange(0, 25.5, 0.5) if ws_grid is None else np.asarray(ws_grid)
        return pd.DataFrame({"ws": grid, "p": self.predict(grid)})
