"""Экран агента: журнал шагов, контекст чата, лимиты для публичного сайта."""

import json
from types import SimpleNamespace

import pytest

from windagent.config import resolve
from windagent.ui import agent_view as av

RUN_DIR = resolve("artifacts/agent/2026-02-10")
needs_run = pytest.mark.skipif(not (RUN_DIR / "run.json").exists(), reason="нет журнала агента")


@needs_run
def test_saved_run_loads_with_forecasts():
    run = av.load_run(RUN_DIR)
    assert run["versions"] and all(v["forecast"] is not None and len(v["forecast"]) == 48 for v in run["versions"])


@needs_run
def test_every_step_has_title_and_summary():
    run = av.load_run(RUN_DIR)
    for step in run["steps"]:
        assert step["tool"] in av.STEP_TITLES
        assert av.step_summary(step), step["tool"]


@needs_run
def test_chat_context_contains_facts():
    run = av.load_run(RUN_DIR)
    ctx = av.chat_context(run)
    assert run["versions"][-1]["explanation"] in ctx
    assert ctx.count("[") >= 48  # почасовой прогноз с интервалом


@needs_run
def test_chat_answer_uses_context_and_history():
    run = av.load_run(RUN_DIR)
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(output_text="Пик завтра около 23:00.")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    history = [{"role": "user", "content": "привет"}, {"role": "assistant", "content": "здравствуйте"}]
    ans = av.chat_answer("Когда пик?", run, history, client=client)
    assert ans == "Пик завтра около 23:00."
    assert "ДАННЫЕ" in seen["instructions"] and run["issue_date"] in seen["instructions"]
    assert seen["input"][-1] == {"role": "user", "content": "Когда пик?"} and len(seen["input"]) == 3


def test_quota_per_ip_and_total(tmp_path, monkeypatch):
    monkeypatch.setitem(av.LIMITS, "chat", {"per_ip": 2, "total": 3})
    q = tmp_path / "quota.json"
    assert av.take_quota("chat", "1.1.1.1", q)[0]
    assert av.take_quota("chat", "1.1.1.1", q)[0]
    assert not av.take_quota("chat", "1.1.1.1", q)[0]  # лимит адреса
    assert av.take_quota("chat", "2.2.2.2", q)[0]
    ok, msg = av.take_quota("chat", "3.3.3.3", q)
    assert not ok and "дневной лимит сайта" in msg  # общий лимит
    assert json.loads(q.read_text())["chat"]["total"] == 3


@needs_run
def test_brief_and_pipeline_from_saved_run():
    run = av.load_run(RUN_DIR)
    b = av.run_brief(run)
    assert b["explanation"] == run["versions"][0]["explanation"]  # на главной — версия на момент выпуска
    assert b["confidence"] in ("высокая", "средняя", "низкая")
    stages = av.pipeline_stages(run["steps"])
    assert stages[0]["title"] == "Погода" and stages[0]["phase"] == 1
    assert any(s["title"] == "Публикация v1" for s in stages)
    if len(run["versions"]) > 1:
        assert b["revision"]["version"] == 2 and any(s["phase"] == 2 for s in stages)
    html = av.stepper_html(stages)
    assert html.count('class="sw-node') == len(stages) and "sw-row" in html


def test_stepper_marks_running_and_errors():
    steps = [
        {"tool": "fetch_weather", "ok": True, "as_of_utc": "2026-02-10 08:00:00",
         "result": {"coverage_ws100": {"ifs": 1.0, "gfs": 1.0}}},
        {"tool": "run_forecast", "ok": False, "as_of_utc": "2026-02-10 08:00:00", "result": {"error": "x"}},
    ]
    html = av.stepper_html(av.pipeline_stages(steps), running=True)
    assert "sw-node ok" in html and "sw-node err" in html
