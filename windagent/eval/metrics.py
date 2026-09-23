"""Метрики качества прогноза нормализованной мощности (номинал = 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = ~(np.isnan(t) | np.isnan(p))
    return t[ok], p[ok]


def mae(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    return float(np.mean(np.abs(p - t))) if len(t) else np.nan


def rmse(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2))) if len(t) else np.nan


def bias(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    return float(np.mean(p - t)) if len(t) else np.nan


def pinball(y_true, y_pred, q: float) -> float:
    t, p = _pair(y_true, y_pred)
    d = t - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d))) if len(t) else np.nan


def coverage(y_true, lo, hi) -> float:
    t = np.asarray(y_true, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    ok = ~(np.isnan(t) | np.isnan(lo) | np.isnan(hi))
    return float(np.mean((t[ok] >= lo[ok]) & (t[ok] <= hi[ok]))) if ok.any() else np.nan


def score(y_true, y_pred) -> dict:
    """MAE, RMSE, смещение. Мощность нормализована, поэтому nMAE (%) = MAE × 100."""
    t, _ = _pair(y_true, y_pred)
    return {
        "n": int(len(t)),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "nMAE_%": mae(y_true, y_pred) * 100,
    }


def score_table(preds: pd.DataFrame, truth_col: str = "p_farm", by: list[str] | None = None) -> pd.DataFrame:
    """Таблица метрик по моделям (колонка model) и, при желании, по группам (lead_day и т.п.)."""
    keys = ["model"] + (by or [])
    rows = []
    for k, g in preds.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k)), **score(g[truth_col], g["pred"])})
    return pd.DataFrame(rows)


def add_skill(table: pd.DataFrame, reference: str, metric: str = "MAE", by: list[str] | None = None) -> pd.DataFrame:
    """Skill = 1 − metric / metric(reference): доля улучшения относительно эталона."""
    out = table.copy()
    keys = by or []
    ref = out[out["model"] == reference].set_index(keys)[metric] if keys else out.loc[out["model"] == reference, metric].iloc[0]
    if keys:
        out[f"skill_vs_{reference}"] = 1 - out[metric] / out.set_index(keys).index.map(ref).to_numpy()
    else:
        out[f"skill_vs_{reference}"] = 1 - out[metric] / ref
    return out
