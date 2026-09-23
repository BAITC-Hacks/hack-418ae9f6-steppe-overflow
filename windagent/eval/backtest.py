"""Бэктест по протоколу теста: ежедневные выпуски, 48 ч вперёд, обучение только на прошлом."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from windagent import protocol
from windagent.config import load_settings, resolve
from windagent.data import scada
from windagent.eval import metrics
from windagent.features import build
from windagent.models.baselines import TARGETS, FitContext, Forecaster

DATASET_PATH = "artifacts/features/dataset.parquet"


def dataset_path(settings: dict | None = None) -> Path:
    return resolve(DATASET_PATH)


def build_full_dataset(settings: dict | None = None, log=print) -> pd.DataFrame:
    """Признаки по всем историческим выпускам + тестовому периоду, с фактом там, где он есть."""
    s = settings or load_settings()
    f = s["forecast"]
    dates = build.history_issue_dates(s) + protocol.issue_dates(f["test_first_issue"], f["test_last_issue"])
    parts = []
    for i, d in enumerate(dates, 1):
        parts.append(build.issue_frame(d, settings=s))
        if i % 100 == 0:
            log(f"  выпусков: {i}/{len(dates)}")
    ds = build.attach_truth(pd.concat(parts, ignore_index=True), scada.load_hourly(s))
    path = dataset_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(path, index=False, compression="zstd")
    log(f"Датасет: {len(ds)} строк, {ds['issue_date'].nunique()} выпусков → {path}")
    return ds


def load_dataset(settings: dict | None = None) -> pd.DataFrame:
    path = dataset_path(settings)
    if not path.exists():
        raise FileNotFoundError(f"нет {path}; соберите: windagent features build")
    return pd.read_parquet(path)


def split(dataset: pd.DataFrame, first_issue, last_issue, settings: dict | None = None):
    """Обучающая часть — только часы, факт по которым известен к моменту первого выпуска."""
    cutoff = protocol.issue_time_utc(first_issue, settings)
    first, last = pd.Timestamp(first_issue), pd.Timestamp(last_issue)
    train = dataset[
        (dataset["issue_date"] < first)
        & (dataset["target_time_utc"] + pd.Timedelta("1h") <= cutoff)
    ]
    test = dataset[(dataset["issue_date"] >= first) & (dataset["issue_date"] <= last)]
    return train, test, cutoff


def run_backtest(
    models: list[Forecaster],
    dataset: pd.DataFrame,
    first_issue,
    last_issue,
    settings: dict | None = None,
    hourly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Обучает каждую модель на прошлом и прогнозирует выпуски периода.

    Возвращает «длинную» таблицу: мета выпуска, model, <target>_pred и факт.
    """
    s = settings or load_settings()
    train, test, cutoff = split(dataset, first_issue, last_issue, s)
    assert (train["target_time_utc"] + pd.Timedelta("1h") <= cutoff).all(), "факт из будущего в обучении"
    hourly = scada.load_hourly(s) if hourly is None else hourly
    ctx = FitContext(cutoff=cutoff, hourly=hourly[hourly.index + pd.Timedelta("1h") <= cutoff], settings=s)

    truth_cols = [c for c in TARGETS if c in test.columns]
    out = []
    for m in models:
        m.fit(train, ctx)
        pred = m.predict(test).clip(0, 1)
        frame = test[build.META + truth_cols].copy()
        frame.insert(0, "model", m.name)
        for t in TARGETS:
            frame[f"{t}_pred"] = pred[t].to_numpy() if t in pred else float("nan")
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def report(preds: pd.DataFrame, target: str = "p_farm", reference: str = "clim") -> pd.DataFrame:
    """Метрики по моделям: всего и по суткам упреждения (1 = D+1, 2 = D+2)."""
    p = preds.rename(columns={f"{target}_pred": "pred"})
    total = metrics.score_table(p, truth_col=target)
    total["lead_day"] = "all"
    by_day = metrics.score_table(p, truth_col=target, by=["lead_day"])
    by_day["lead_day"] = by_day["lead_day"].astype(str)
    table = pd.concat([total, by_day], ignore_index=True)
    table = metrics.add_skill(table, reference, by=["lead_day"])
    return table.sort_values(["lead_day", "MAE"]).reset_index(drop=True)


MODEL_TITLES = {
    "ensemble": "**Основная модель: ансамбль (бустинг MAE + бустинг MSE + физическая кривая)**",
    "phys_ifs": "ECMWF IFS → кривая мощности по анемометру турбин",
    "curve_ifs": "ECMWF IFS → кривая, обученная на прогнозах",
    "curve_ifs025": "ECMWF 0.25° (Previous Runs) → обученная кривая",
    "curve_icon": "ICON (Previous Runs) → обученная кривая",
    "curve_gfs": "GFS (Previous Runs) → обученная кривая",
    "clim": "Климатология (среднее по месяцу и часу)",
}


def markdown_summary(preds: pd.DataFrame, title: str, target: str = "p_farm") -> str:
    """Таблицы MAE и RMSE по моделям и периодам (колонка period) в Markdown."""
    reports = {per: report(g, target) for per, g in preds.groupby("period", sort=False)}
    lines = [
        f"# {title}",
        "",
        "Прогноз нормированной мощности ВЭС (0–1). Протокол: выпуск ежедневно в 14:00 (UTC+6), "
        "прогноз на D+1 и D+2; обучение только на данных до первого выпуска периода.",
    ]
    for metric in ("MAE", "RMSE"):
        tab = pd.DataFrame(
            {per: r[r["lead_day"] == "all"].set_index("model")[metric] for per, r in reports.items()}
        )
        tab["среднее"] = tab.mean(axis=1)
        tab = tab.sort_values("среднее")
        lines += ["", f"## {metric}", "", "| Модель | " + " | ".join(tab.columns) + " |",
                  "|---|" + "---|" * len(tab.columns)]
        for m, r in tab.iterrows():
            lines.append(f"| {MODEL_TITLES.get(m, m)} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    return "\n".join(lines) + "\n"
