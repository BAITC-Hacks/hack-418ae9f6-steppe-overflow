"""Smoke-тесты веб-интерфейса: данные, графики и запуск приложения без ошибок."""

import pandas as pd
import pytest

from windagent.config import resolve
from windagent.ui import charts, data

needs_artifacts = pytest.mark.skipif(
    not (resolve("artifacts/submission/forecast_feb2026.csv").exists()
         and resolve("artifacts/reports/backtest_preds.parquet").exists()),
    reason="нет артефактов прогноза",
)


def _issue():
    f = data.submission()
    return f[f["issue_date"] == f["issue_date"].min()].reset_index(drop=True)


@needs_artifacts
def test_summary_with_and_without_interval():
    f = _issue()
    s = data.summarize(f)
    assert 0 <= s["mean_d1"] <= 1 and s["mean_width"] > 0
    assert s["full_load_hours_d1"] == pytest.approx(f.loc[f["lead_day"] == 1, "p_farm"].sum())
    # История: интервала нет — плитки не должны падать
    s2 = data.summarize(f[["target_time_local", "lead_day", "p_farm"]])
    assert s2["mean_width"] is None


@needs_artifacts
def test_charts_build():
    f = _issue()
    fig = charts.forecast_chart(f, fact=pd.Series(0.5, index=f.index), show_turbines=True)
    names = [t.name for t in fig.data]
    assert "Прогноз ВЭС" in names and "Факт SCADA" in names and "Интервал P10–P90" in names
    w = data.weather_for_issue(f["issue_date"].iloc[0])
    assert len(charts.weather_chart(w).data) == 3  # коридор (2 границы) + ECMWF
    assert len(charts.weather_chart(w, show_all=True).data) == 6
    assert len(charts.forecast_vs_fact_chart(data.backtest_preds()).data) == 3
    bt = data.backtest_preds()
    assert len(charts.error_by_horizon_chart(bt).data) == 3


@needs_artifacts
def test_app_runs_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(resolve("app/streamlit_app.py")), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("Прогноз выработки ВЭС" in t.value for t in at.title)
    # Переключение в режим «История — прогноз против факта»
    at.segmented_control[0].set_value("История — прогноз против факта").run()
    assert not at.exception, at.exception


@needs_artifacts
def test_climatology_delta_is_for_tomorrow_only():
    f = _issue()
    delta = data.climatology_delta_d1(f, f["issue_time_utc"].iloc[0])
    d1 = f[f["lead_day"] == 1]["p_farm"].mean()
    assert delta is not None and -1 < delta < 1
    assert 0 <= d1 - delta <= 1  # норма — это доля номинала
