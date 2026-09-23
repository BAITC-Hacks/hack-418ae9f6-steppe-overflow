"""Оркестраторы агента: на правилах (работает всегда) и на LLM OpenAI (tool calling через Responses API).

Оба выполняют один цикл:
  погода → проверка → модель → анализ → публикация → (новый прогон?) → пересчёт → сравнение → ревизия.
LLM выбирает порядок вызовов и решения сама; при любой ошибке LLM агент
автоматически доводит выпуск до конца по правилам.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

import pandas as pd

from windagent.agent import tools as T
from windagent.config import load_env

MAX_LLM_STEPS = 24

SYSTEM_PROMPT = """Ты — AI-агент прогноза выработки ветроэлектростанции (2 турбины, Алматинская область).
Твоя задача — выпустить почасовой прогноз мощности на 48 часов (D+1 и D+2) и сопроводить его объяснением
для диспетчера. Все числа получай только из инструментов, ничего не выдумывай и не вычисляй сам:
публикация проверяет, что каждое число в объяснении есть в результатах инструментов.

Порядок работы:
1. check_available_runs — какие прогнозы погоды доступны сейчас.
2. fetch_weather, затем validate_weather. Если validate_weather рекомендует исключить источник —
   исключи его в run_forecast (exclude_sources). ECMWF IFS («ifs») — главный источник; исключай его
   только при покрытии ниже 90 %.
3. run_forecast, затем analyze_forecast. Если в анализе есть флаги — подумай, связаны ли они с
   погодными данными; при необходимости перезапусти run_forecast без подозрительного источника.
4. publish_forecast с объяснением (3–6 предложений, по-русски): средняя мощность и пик завтра,
   периоды штиля или работы на номинале, согласие погодных моделей и уверенность, отличие от
   прошлого выпуска и от климата, какие данные использованы.
5. Если разрешён пересчёт: advance_to_next_update. Если вышел новый прогон — снова fetch_weather,
   validate_weather, run_forecast, analyze_forecast, затем compare_with_published. Публикуй
   ревизию (publish_forecast) только если compare_with_published советует; в объяснении ревизии
   скажи, что изменилось и почему.
6. В конце дай короткий итог (2–3 предложения): что опубликовано и с какой уверенностью.

С климатической нормой сравнивай завтрашний день: climatology_mean_d1 и vs_climatology_d1.
Оформление текстов: мощность — в процентах номинала, округлённо («32 %», а не «0,32»); изменения —
в процентных пунктах («на 7 п.п.»); время — местное, в формате «13.02 23:00»; скорость ветра — в м/с.
Не используй обозначения D+1/D+2 и названия полей JSON — пиши «завтра», «послезавтра»; модели
погоды называй ECMWF, ICON, GFS. Пиши для диспетчера: коротко, по делу, без технических терминов."""


# --- Объяснение по шаблону (режим правил и запасной вариант) -------------------------------------


def template_explanation(state: T.AgentState, validation: dict, analysis: dict, revision: dict | None = None) -> str:
    d1 = state.issue_date + pd.Timedelta(days=1)
    a = analysis
    parts = [
        f"Прогноз на {d1:%d.%m}: средняя мощность ВЭС {a['mean_p_farm_d1']:.0%} номинала "
        f"(ветер ECMWF на 100 м в среднем {a['mean_wind_ifs_d1_ms']} м/с), пик {a['peak_d1']['value']:.0%} "
        f"около {a['peak_d1']['time_local'][-5:]}."
    ]
    if a["high_hours_d1"]:
        parts.append(f"Работа близко к номиналу (≥80 %) ожидается {a['high_hours_d1']} ч.")
    if a["calm_hours_d1"]:
        parts.append(f"Штиль (≤5 %) — {a['calm_hours_d1']} ч.")
    parts.append(
        f"Согласие погодных моделей {validation['agreement']} (разброс {validation['ensemble_spread_ms']} м/с), "
        f"уверенность прогноза {a['confidence']}."
    )
    diff = a.get("vs_climatology_d1", a["vs_climatology"])
    if abs(diff) >= 0.05:
        parts.append(f"Это {'выше' if diff > 0 else 'ниже'} климатической нормы для завтра "
                     f"({a.get('climatology_mean_d1', a['climatology_mean']):.0%}) на {abs(diff) * 100:.0f} п.п.")
    if a["change_vs_previous_issue"]:
        c = a["change_vs_previous_issue"]
        parts.append(f"По сравнению с прошлым выпуском прогноз на пересекающиеся часы сдвинулся в среднем на "
                     f"{c['mean_shift']:+.0%}.")
    if state.excluded:
        parts.append(f"Исключены источники: {', '.join(T.SOURCE_NAMES[s] for s in state.excluded)} — "
                     "проверка качества данных показала проблемы.")
    if validation.get("ecmwf_stale"):
        parts.append(f"Свежий прогон ECMWF ещё не опубликован, использован прогон возрастом "
                     f"{validation['ecmwf_run_age_h']:.0f} ч — уверенность понижена.")
    if revision:
        parts.insert(0, f"Ревизия после выхода нового прогона ECMWF ({revision['new_as_of_local']}): прогноз "
                        f"изменился в среднем на {revision['mean_abs_change']:.0%}, максимум на "
                        f"{revision['max_abs_change']:.0%} ({revision['max_change_at_local']}).")
    return " ".join(parts)


# --- Журнал шагов -----------------------------------------------------------------------------


class Recorder:
    def __init__(self, state: T.AgentState, log: Callable[[str], None] | None = None,
                 on_step: Callable[[list[dict]], None] | None = None):
        self.state = state
        self.steps: list[dict] = []
        self.log = log or (lambda m: None)
        self.on_step = on_step or (lambda steps: None)

    def call(self, name: str, args: dict | None = None, reason: str = "") -> dict:
        t0 = time.time()
        try:
            result = T.call_tool(self.state, name, args)
            ok = True
            if name != "publish_forecast":
                self.state.tool_results.append(result)
        except T.ToolError as e:
            result, ok = {"error": str(e)}, False
        self.steps.append({
            "step": len(self.steps) + 1, "tool": name, "args": args or {}, "reason": reason, "ok": ok,
            "as_of_utc": str(self.state.as_of), "seconds": round(time.time() - t0, 2), "result": result,
        })
        shown = json.dumps(args, ensure_ascii=False) if args else ""
        shown = shown if len(shown) <= 60 else shown[:57] + "…"
        self.log(f"  [{len(self.steps):>2}] {name}{'(' + shown + ')' if shown else ''}{' — ' + reason if reason else ''}")
        self.on_step(self.steps)
        return result

    def note(self, text: str) -> None:
        self.steps.append({"step": len(self.steps) + 1, "tool": "decision", "reason": text, "ok": True,
                           "as_of_utc": str(self.state.as_of)})
        self.log(f"  [{len(self.steps):>2}] решение: {text}")
        self.on_step(self.steps)


# --- Оркестратор на правилах --------------------------------------------------------------------


def run_rules(state: T.AgentState, recheck: bool = True, rec: Recorder | None = None) -> tuple[list[dict], str]:
    rec = rec or Recorder(state)
    rec.call("check_available_runs", reason="какие прогнозы погоды уже опубликованы")
    _cycle(rec, first=True)
    if recheck:
        upd = rec.call("advance_to_next_update", reason="проверяем, не вышел ли новый прогон погоды")
        if upd.get("new_run_available"):
            _cycle(rec, first=False, update=upd)
        else:
            rec.note("новых прогонов до начала прогнозного окна нет — пересчёт не нужен")
    v = state.versions[-1]
    summary = (f"Опубликовано версий: {len(state.versions)}. Итоговая версия v{v['version']} "
               f"(данные на {v['as_of_local']}), уверенность {state.last_analysis['confidence']}.")
    return rec.steps, summary


def _cycle(rec: Recorder, first: bool, update: dict | None = None) -> None:
    rec.call("fetch_weather", reason="собираем прогнозы погоды на 48 ч")
    val = rec.call("validate_weather", reason="проверяем качество и согласие моделей")
    exclude = [s for s in val["suspicious_sources"] if s != "ifs" or val["sources"]["ifs"]["coverage"] < T.SUSPICIOUS_COVERAGE]
    if exclude:
        why = "; ".join(f"{T.SOURCE_NAMES[s]}: {', '.join(val['sources'][s]['issues'])}" for s in exclude)
        rec.note(f"исключаю подозрительные источники — {why}")
    if val.get("ecmwf_stale"):
        rec.note(f"свежий прогон ECMWF ещё не опубликован — работаю на прогоне возрастом {val['ecmwf_run_age_h']:.0f} ч, "
                 "уверенность будет понижена")
    rec.call("run_forecast", {"exclude_sources": exclude}, reason="запуск модели")
    ana = rec.call("analyze_forecast", reason="проверка правдоподобия результата")
    if ana["flags"]:
        rec.note("флаги анализа: " + "; ".join(ana["flags"]))
    if first:
        rec.call("publish_forecast", {"explanation": template_explanation(rec.state, val, ana)},
                 reason="публикуем первую версию прогноза")
        return
    cmp = rec.call("compare_with_published", reason="сравниваем с опубликованной версией")
    if cmp["significant"]:
        rec.call("publish_forecast", {"explanation": template_explanation(rec.state, val, ana, {**cmp, **update})},
                 reason="изменение существенное — публикуем ревизию")
    else:
        rec.note(f"изменение незначительное (в среднем {cmp['mean_abs_change']:.1%}) — остаётся версия "
                 f"v{rec.state.versions[-1]['version']}")


# --- Оркестратор на LLM (OpenAI Responses API) --------------------------------------------------


def tool_schemas() -> list[dict]:
    schemas = []
    for name, (_, desc, props) in T.TOOLS.items():
        schemas.append({
            "type": "function",
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props),
                           "additionalProperties": False},
            "strict": True,
        })
    return schemas


def run_llm(state: T.AgentState, recheck: bool = True, client=None, model: str | None = None,
            log: Callable[[str], None] | None = None, on_step=None) -> tuple[list[dict], str]:
    """Агентный цикл: модель OpenAI вызывает инструменты, пока не выпустит прогноз и не даст итог."""
    load_env()
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    model = model or os.environ.get("OPENAI_MODEL") or "gpt-5.6-terra"
    state.fact_check = True
    rec = Recorder(state, log, on_step)
    task = (f"Выпусти прогноз для даты выпуска {state.issue_date:%Y-%m-%d} "
            f"(момент прогноза {T._local(state.as_of, state.offset)} местного времени). "
            f"Пересчёт при появлении новых данных: {'разрешён' if recheck else 'не нужен'}.")
    items: list = [{"role": "user", "content": task}]
    final_text = ""
    for _ in range(MAX_LLM_STEPS):
        resp = client.responses.create(model=model, instructions=SYSTEM_PROMPT, input=items,
                                       tools=tool_schemas(), parallel_tool_calls=False)
        calls = [o for o in resp.output if getattr(o, "type", None) == "function_call"]
        items += resp.output
        if not calls:
            final_text = resp.output_text or ""
            break
        for c in calls:
            args = json.loads(c.arguments or "{}")
            if c.name not in T.TOOLS:
                result = {"error": f"неизвестный инструмент {c.name}"}
            else:
                result = rec.call(c.name, args, reason="решение LLM")
            items.append({"type": "function_call_output", "call_id": c.call_id,
                          "output": json.dumps(result, ensure_ascii=False, default=str)})
    if not state.versions:
        # LLM не опубликовала прогноз — доводим выпуск по правилам
        rec.note("LLM не опубликовала прогноз — завершаю выпуск по правилам")
        steps, summary = run_rules(state, recheck=False, rec=rec)
        return steps, summary
    return rec.steps, final_text.strip() or "Прогноз опубликован."


# --- Точка входа ---------------------------------------------------------------------------------


def llm_available() -> bool:
    load_env()
    return bool(os.environ.get("OPENAI_API_KEY"))


def run_agent(issue_date, model, mode: str = "auto", recheck: bool = True, settings: dict | None = None,
              log: Callable[[str], None] | None = None, client=None, out_dir=None, on_step=None,
              as_of=None, weather_dir=None, prev_forecast=None, fault: str | None = None) -> dict:
    """Полный запуск агента на одну дату выпуска. mode: auto | rules | llm.

    as_of/weather_dir/prev_forecast — для живого режима; fault — демо-сценарий сбоя данных.
    """
    extra = {k: v for k, v in (("settings", settings), ("out_dir", out_dir), ("as_of", as_of),
                                ("weather_dir", weather_dir), ("prev_forecast", prev_forecast),
                                ("fault", fault)) if v is not None}
    if as_of is not None:
        extra["as_of"] = pd.Timestamp(as_of)
    state = T.AgentState(issue_date=issue_date, model=model, **extra)
    use_llm = mode == "llm" or (mode == "auto" and (client is not None or llm_available()))
    used = "rules"
    if use_llm:
        try:
            steps, summary = run_llm(state, recheck=recheck, client=client, log=log, on_step=on_step)
            used = "llm"
        except Exception as e:  # сеть, ключ, лимиты — прогноз всё равно должен выйти
            (log or print)(f"  LLM недоступна ({type(e).__name__}: {e}); продолжаю по правилам")
            state = T.AgentState(issue_date=issue_date, model=model, **extra)
            rec = Recorder(state, log, on_step)
            rec.note(f"LLM недоступна ({type(e).__name__}) — работаю по правилам")
            steps, summary = run_rules(state, recheck=recheck, rec=rec)
            used = "rules (fallback)"
    else:
        steps, summary = run_rules(state, recheck=recheck, rec=Recorder(state, log, on_step))
    path = T.save_run(state, steps, used, summary)
    return {"issue_date": f"{state.issue_date:%Y-%m-%d}", "mode": used, "versions": len(state.versions),
            "summary": summary, "explanation": state.versions[-1]["explanation"], "run_file": str(path),
            "steps": len(steps)}
