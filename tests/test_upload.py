"""Загрузка пользовательских данных: разбор форматов и сопоставление с прогнозами."""

import numpy as np
import pandas as pd
import pytest

from windagent.config import resolve
from windagent.ui import upload

needs_artifacts = pytest.mark.skipif(not resolve("artifacts/reports/backtest_preds.parquet").exists(),
                                     reason="нет артефактов прогноза")


def test_organizers_format_is_aggregated_to_hours():
    raw = ("ID,Статистическое время,Средняя скорость ветра(m/s),Нормализованная активная мощность,T\n"
           + "".join(f"{i},2026-01-10 {h}:{m:02d}:00,8.0,0.{5 if h == 0 else 7},1.0\n"
                     for i, (h, m) in enumerate((h, m) for h in (0, 1) for m in range(0, 60, 10))))
    p, kind = upload.parse_upload(raw.encode("utf-8"), utc_offset_hours=6)
    assert "организаторов" in kind
    assert list(p.index) == [pd.Timestamp("2026-01-09 18:00"), pd.Timestamp("2026-01-09 19:00")]
    assert p.tolist() == pytest.approx([0.5, 0.7])


def test_simple_csv_with_percent_and_semicolon():
    raw = "время;мощность, %\n2026-02-01 00:00;50\n2026-02-01 00:30;70\n2026-02-01 01:00;20\n".encode("cp1251")
    p, kind = upload.parse_upload(raw, utc_offset_hours=0)
    assert "простой CSV" in kind
    assert p.loc["2026-02-01 00:00"] == pytest.approx(0.6)  # среднее за час, проценты → доля
    assert p.loc["2026-02-01 01:00"] == pytest.approx(0.2)


@pytest.mark.parametrize("raw, msg", [
    (b"a\n1\n", "2 колонки"),
    (b"x,y\nfoo,1\nbar,2\n", "временем"),
    (b"t,p\n2026-02-01 00:00,500\n2026-02-01 01:00,700\n", "нормализованной"),
])
def test_bad_files_give_clear_errors(raw, msg):
    with pytest.raises(upload.UploadError, match=msg):
        upload.parse_upload(raw, utc_offset_hours=0)


def test_combine_two_turbines_is_mean():
    idx = pd.date_range("2026-02-01", periods=3, freq="h")
    a, b = pd.Series([0.2, 0.4, np.nan], index=idx), pd.Series([0.4, 0.6, 0.8], index=idx)
    assert upload.combine([a, b]).tolist() == pytest.approx([0.3, 0.5, 0.8])


@needs_artifacts
def test_sample_matches_backtest_metrics():
    p, _ = upload.parse_upload(upload.sample_file())
    res = upload.evaluate(p)
    assert res["hours"] == 744 and res["sources"] == ["бэктест основной модели"]
    assert 0.1 < res["by_lead"][1]["MAE"] < 0.2
    # 31 января покрыто только прогнозом на D+2 (последний выпуск бэктеста — 29.01)
    assert len(res["daily_mae"]) == 30


@needs_artifacts
def test_no_overlap_is_reported():
    s = pd.Series([0.5], index=pd.DatetimeIndex(["2030-01-01"], name="target_time_utc"), name="fact")
    with pytest.raises(upload.UploadError, match="не пересекаются"):
        upload.evaluate(s)
