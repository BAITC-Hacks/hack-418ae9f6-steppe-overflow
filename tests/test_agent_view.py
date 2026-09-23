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
