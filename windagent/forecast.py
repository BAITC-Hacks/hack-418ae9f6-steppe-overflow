"""Итоговый прогноз: обучение модели, прогноз одного выпуска с манифестом, прогон тестового периода.

Модель для теста обучается один раз — на всех фактах, известных к моменту первого
тестового выпуска (31.01.2026 14:00 UTC+6). SCADA за февраль нет, поэтому на
последующих выпусках модель не переобучается; меняются только прогнозы погоды.
"""

from __future__ import annotations

import gzip
import json
import pickle
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from windagent import protocol
from windagent.config import load_settings, resolve
from windagent.data import scada
from windagent.data.store import DataStore, LeakageError
from windagent.eval import backtest
from windagent.features import build
from windagent.models.baselines import FitContext
from windagent.models.gbm import EnsembleForecaster
from windagent.models.quantile import ConformalInterval

MODEL_PATH = "artifacts/models/production.pkl.gz"
P_MAX = 0.99  # физический максимум нормированной мощности (номинал турбины 1)
SUBMISSION_DIR = "artifacts/submission"
OUTPUT_COLUMNS = [
    "issue_date", "issue_time_utc", "target_time_utc", "target_time_local", "horizon_h", "lead_day",
    "p_t1", "p_t2", "p_farm", "p_farm_q10", "p_farm_q90", "ifs_run_time", "model_version",
]


@dataclass
class ProductionModel:
    point: EnsembleForecaster = field(default_factory=EnsembleForecaster)
    interval: ConformalInterval = field(default_factory=ConformalInterval)
    meta: dict = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.meta.get("version", "untrained")

    def fit(self, train: pd.DataFrame, ctx: FitContext) -> "ProductionModel":
        self.point.fit(train, ctx)
        self.interval.fit(train, ctx)
        self.meta = {
            "version": f"ensemble-v1-{ctx.cutoff:%Y%m%d}",
            "trained_until_utc": str(ctx.cutoff),
            "train_rows": int(len(train)),
            "train_issues": f"{train['issue_date'].min():%Y-%m-%d} … {train['issue_date'].max():%Y-%m-%d}",
            "interval_margin": self.interval.margin_,
        }
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.point.predict(X)
        q = self.interval.predict(X)
        out = p.copy()
        # Интервал всегда содержит точечный прогноз
        out["p_farm_q10"] = np.minimum(q["p_farm_q10"], p["p_farm"])
        out["p_farm_q90"] = np.maximum(q["p_farm_q90"].clip(upper=P_MAX), p["p_farm"])
        return out


def train_production(settings: dict | None = None, first_issue=None, log=print) -> ProductionModel:
    """Обучает модель на всех фактах, известных к первому тестовому выпуску, и сохраняет её."""
    s = settings or load_settings()
    first_issue = pd.Timestamp(first_issue or s["forecast"]["test_first_issue"])
    ds = backtest.load_dataset(s)
    train, _, cutoff = backtest.split(ds, first_issue, first_issue, s)
    assert (train["target_time_utc"] + pd.Timedelta("1h") <= cutoff).all()
    hourly = scada.load_hourly(s)
    ctx = FitContext(cutoff=cutoff, hourly=hourly[hourly.index + pd.Timedelta("1h") <= cutoff], settings=s)
    log(f"Обучение на {len(train)} строках (выпуски {train['issue_date'].min():%Y-%m-%d} … "
        f"{train['issue_date'].max():%Y-%m-%d}), факт до {cutoff} UTC")
    model = ProductionModel().fit(train, ctx)
    save_model(model)
    log(f"Модель {model.version} сохранена: {resolve(MODEL_PATH)}")
    return model


def save_model(model: ProductionModel, path: str | Path = MODEL_PATH) -> None:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wb", compresslevel=6) as f:
        pickle.dump(model, f)


def load_model(path: str | Path = MODEL_PATH) -> ProductionModel:
    p = resolve(path)
    return _load_model_cached(str(p), p.stat().st_mtime if p.exists() else 0.0)


@lru_cache(maxsize=2)
def _load_model_cached(path: str, mtime: float) -> ProductionModel:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"нет обученной модели {p}; запустите: windagent train")
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def forecast_issue(issue_date, model: ProductionModel, settings: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Прогноз одного выпуска: 48 часов + манифест использованных данных."""
    s = settings or load_settings()
    as_of = protocol.issue_time_utc(issue_date, s)
    cutoff = pd.Timestamp(model.meta["trained_until_utc"])
    if cutoff > as_of:
        raise LeakageError(f"модель обучена на фактах до {cutoff}, а выпуск делается в {as_of}")
    store = DataStore(as_of, settings=s)
    X = build.issue_frame(issue_date, store=store, settings=s)
    pred = model.predict(X)

    out = X[build.META].copy()
    for c in ("p_t1", "p_t2", "p_farm", "p_farm_q10", "p_farm_q90"):
        out[c] = pred[c].to_numpy().round(4)
    out["ifs_run_time"] = X.get("ifs__run_time")
    out["model_version"] = model.version
    store.check_manifest()
    manifest = {
        "issue_date": f"{pd.Timestamp(issue_date):%Y-%m-%d}",
        "as_of_utc": str(as_of),
        "model_version": model.version,
        "model_trained_until_utc": model.meta["trained_until_utc"],
        "sources": store.manifest,
        "checks": {
            "all_sources_published_before_as_of": all(
                e.get("max_published_at") is None or pd.Timestamp(e["max_published_at"]) <= as_of
                for e in store.manifest
            ),
            "rows": int(len(out)),
            "nan_in_forecast": bool(out[["p_t1", "p_t2", "p_farm"]].isna().any().any()),
        },
    }
    return out[OUTPUT_COLUMNS], manifest


def run_period(first_issue, last_issue, model: ProductionModel, settings: dict | None = None,
               out_dir: str | Path = SUBMISSION_DIR, name: str = "forecast_feb2026", log=print) -> dict:
    """Все выпуски периода → CSV со всеми выпусками, CSV «сутки вперёд», манифесты."""
    s = settings or load_settings()
    out_dir = resolve(out_dir)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    frames = []
    for d in protocol.issue_dates(first_issue, last_issue):
        f, m = forecast_issue(d, model, s)
        frames.append(f)
        (out_dir / "manifests" / f"{m['issue_date']}.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out_dir / f"{name}.csv", index=False)
    # «Сутки вперёд»: каждому часу — прогноз, выпущенный накануне (lead_day = 1)
    day_ahead = full[full["lead_day"] == 1].sort_values("target_time_utc")
    day_ahead.to_csv(out_dir / f"{name}_dayahead.csv", index=False)
    summary = {
        "issues": int(full["issue_date"].nunique()),
        "rows": int(len(full)),
        "day_ahead_rows": int(len(day_ahead)),
        "first_target_local": str(day_ahead["target_time_local"].min()),
        "last_target_local": str(day_ahead["target_time_local"].max()),
        "mean_p_farm_day_ahead": round(float(day_ahead["p_farm"].mean()), 4),
        "model_version": model.version,
    }
    log(f"Выпусков: {summary['issues']}, строк: {summary['rows']} → {out_dir}")
    return summary
