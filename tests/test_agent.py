"""Тесты агента: инструменты, оркестратор на правилах, LLM-цикл на подставном клиенте, запасной режим."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from windagent import forecast, protocol
from windagent.agent import tools as T
from windagent.agent.orchestrator import run_agent, tool_schemas
from windagent.config import resolve

pytestmark = pytest.mark.skipif(not resolve(forecast.MODEL_PATH).exists(), reason="нет обученной модели")
DATE = "2026-02-10"


@pytest.fixture(scope="module")
def model():
    return forecast.load_model()


def test_rules_agent_full_cycle(model, tmp_path):
    r = run_agent(DATE, model, mode="rules", out_dir=tmp_path)
    assert r["mode"] == "rules" and r["versions"] >= 1
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    tools_called = [s["tool"] for s in run["steps"]]
    for t in ("check_available_runs", "fetch_weather", "validate_weather", "run_forecast",
              "analyze_forecast", "publish_forecast", "advance_to_next_update"):
        assert t in tools_called
    assert all(s["ok"] for s in run["steps"])
    # Версия 1 сделана строго на момент выпуска и совпадает с файлом сдачи
    v1 = pd.read_csv(tmp_path / "forecast_v1.csv", parse_dates=["issue_date"])
    sub = pd.read_csv(resolve("artifacts/submission/forecast_feb2026.csv"), parse_dates=["issue_date"])
    sub = sub[sub["issue_date"] == DATE].reset_index(drop=True)
    assert np.allclose(v1["p_farm"], sub["p_farm"], atol=1e-4)
    assert run["versions"][0]["as_of_utc"] == str(protocol.issue_time_utc(DATE))


def test_advance_uses_only_runs_published_before_new_clock(model, tmp_path):
    st = T.AgentState(issue_date=DATE, model=model, out_dir=tmp_path)
    old = st.as_of
    u = T.advance_to_next_update(st)
    assert u["new_run_available"]
    run, new = pd.Timestamp(u["run_utc"]), pd.Timestamp(u["new_as_of_utc"])
    assert old < new <= st.targets[0]
    T.fetch_weather(st)
    assert pd.to_datetime(st.X["ifs__run_time"]).max() == run
    st.store.check_manifest()


def test_validate_flags_corrupted_source_and_rules_exclude_it(model, tmp_path):
    st = T.AgentState(issue_date=DATE, model=model, out_dir=tmp_path)
    T.fetch_weather(st)
    st.X["gfs__wind_speed_100m"] = 45.0  # нефизичный прогноз
    v = T.validate_weather(st)
    assert v["suspicious_sources"] == ["gfs"]
    r = T.run_forecast(st, exclude_sources=v["suspicious_sources"])
    assert r["excluded_sources"] == ["gfs"]
    assert st.forecast["p_farm"].between(0, 1).all()


def test_tools_require_order(model, tmp_path):
    st = T.AgentState(issue_date=DATE, model=model, out_dir=tmp_path)
    with pytest.raises(T.ToolError):
        T.run_forecast(st)


def test_tool_schemas_are_strict_objects():
    for s in tool_schemas():
        p = s["parameters"]
        assert s["strict"] and p["type"] == "object" and p["additionalProperties"] is False
        assert set(p["required"]) == set(p["properties"])


class FakeLLM:
    """Подставной клиент OpenAI: отдаёт заранее заданную последовательность вызовов инструментов."""

    def __init__(self, script, fail=False):
        self.script, self.fail, self.calls = list(script), fail, []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        if self.fail:
            raise ConnectionError("нет сети")
        self.calls.append(kw)
        step = self.script.pop(0)
        if isinstance(step, str):
            return SimpleNamespace(output=[SimpleNamespace(type="message")], output_text=step)
        name, args = step
        call = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id=f"c{len(self.calls)}")
        return SimpleNamespace(output=[call], output_text="")


def test_llm_agent_executes_tool_calls(model, tmp_path):
    script = [("check_available_runs", {}), ("fetch_weather", {}), ("validate_weather", {}),
              ("run_forecast", {"exclude_sources": []}), ("analyze_forecast", {}),
              ("publish_forecast", {"explanation": "Тестовое объяснение."}),
              "Опубликована версия v1."]
    fake = FakeLLM(script)
    r = run_agent(DATE, model, mode="llm", recheck=False, client=fake, out_dir=tmp_path)
    assert r["mode"] == "llm" and r["versions"] == 1
    assert r["explanation"] == "Тестовое объяснение." and r["summary"] == "Опубликована версия v1."
    # Результат каждого инструмента возвращается модели
    last_input = fake.calls[-1]["input"]
    outputs = [i for i in last_input if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert len(outputs) == 6


def test_llm_that_never_publishes_is_completed_by_rules(model, tmp_path):
    fake = FakeLLM(["Готово, ничего делать не буду."])
    r = run_agent(DATE, model, mode="llm", recheck=False, client=fake, out_dir=tmp_path)
    assert r["versions"] == 1


def test_llm_failure_falls_back_to_rules(model, tmp_path):
    r = run_agent(DATE, model, mode="llm", recheck=False, client=FakeLLM([], fail=True), out_dir=tmp_path)
    assert r["mode"] == "rules (fallback)" and r["versions"] == 1


def test_fact_check_rejects_invented_numbers_and_llm_retries(model, tmp_path):
    script = [("check_available_runs", {}), ("fetch_weather", {}), ("validate_weather", {}),
              ("run_forecast", {"exclude_sources": []}), ("analyze_forecast", {}),
              ("publish_forecast", {"explanation": "Завтра ветер до 23,7 м/с."}),  # выдумано
              ("publish_forecast", {"explanation": "Прогноз опубликован, подробности в таблице."}),
              "Готово."]
    fake = FakeLLM(script)
    r = run_agent(DATE, model, mode="llm", recheck=False, client=fake, out_dir=tmp_path)
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    publishes = [s for s in run["steps"] if s["tool"] == "publish_forecast"]
    assert publishes[0]["ok"] is False and "проверка фактов" in publishes[0]["result"]["error"]
    assert publishes[1]["ok"] is True and r["versions"] == 1
    assert run["versions"][0]["fact_check"]["passed"] is True


def test_fact_check_numbers():
    from windagent.agent import factcheck as fc

    results = [{"mean_p_farm_d1": 0.321, "peak_d1": {"value": 0.71}, "ensemble_spread_ms": 2.66}]
    ok = fc.check("Завтра 32 % номинала, пик 71 % в 10.02 23:00, разброс 2,66 м/с (прогон 06Z, IFS 0.25°).", results)
    assert ok["passed"] and ok["numbers"] == 3
    bad = fc.check("Завтра 55 % номинала.", results)
    assert not bad["passed"] and bad["unmatched"] == [55.0]
