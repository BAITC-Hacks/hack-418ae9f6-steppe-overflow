"""Живой режим: агент строит прогноз на настоящее «завтра» по свежим прогнозам погоды.

Цикл планировщика (windagent live --watch):
  1. обновить живой кэш погоды из Open-Meteo (последние прогоны ECMWF, Previous Runs GFS/ICON/ECMWF 0.25);
  2. если вышел новый прогон ECMWF (или наступили новые сутки) — запустить агента: момент прогноза = сейчас;
  3. агент сравнивает новый прогноз с предыдущим живым и объясняет, что изменилось;
  4. результат — в общий каталог (WINDAGENT_RUNTIME_DIR/live), его читает сайт.

Модель та же, что для теста (обучена на фактах до 31.01.2026); SCADA для «сейчас» не нужна —
прогноз строится только по прогнозам погоды.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from windagent.config import load_env, load_settings
from windagent.data import weather
from windagent.data.store import DataStore

MODEL = "ecmwf_ifs"


def runtime_dir() -> Path:
    import tempfile

    d = Path(os.environ.get("WINDAGENT_RUNTIME_DIR", Path(tempfile.gettempdir()) / "steppewind"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def live_dir() -> Path:
    d = runtime_dir() / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def live_cache() -> Path:
    d = runtime_dir() / "live_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")


def local_date(ts: pd.Timestamp, settings: dict | None = None) -> pd.Timestamp:
    s = settings or load_settings()
    return (ts + pd.Timedelta(hours=s["scada"]["utc_offset_hours"])).normalize()


def refresh_weather(now: pd.Timestamp | None = None, settings: dict | None = None,
                    log: Callable[[str], None] = print) -> dict:
    """Докачивает в живой кэш свежие прогоны. Список «нет в архиве» для свежих суток не запоминается:
    прогон, которого ещё нет, может выйти через час."""
    s = settings or load_settings()
    now = now or now_utc()
    d = live_cache()
    start, end = (now - pd.Timedelta(days=2)).date(), now.date()
    for m in list(s["weather"]["single_runs"]):
        (d / f"single_{m}_missing.json").unlink(missing_ok=True)
        weather.download_single_runs(m, start, end, settings=s, directory=d, workers=2, log=lambda x: None)
    for m in s["weather"]["previous_runs"]:
        weather.download_previous_runs(m, (now - pd.Timedelta(days=3)).date(), (now + pd.Timedelta(days=3)).date(),
                                       settings=s, directory=d, log=lambda x: None)
    if "neighbors" in s:
        for name in s["neighbors"]["points"]:
            (d / f"single_ecmwf_ifs_nb_{name}_missing.json").unlink(missing_ok=True)
        weather.download_neighbors(start, end, settings=s, directory=d, workers=2, log=lambda x: None)
    latest = DataStore(now, settings=s, weather_dir=d).latest_run(MODEL)
    log(f"живой кэш обновлён; свежайший опубликованный прогон ECMWF: {latest}")
    return {"latest_run": None if latest is None else str(latest)}


def latest_pointer() -> dict | None:
    p = live_dir() / "latest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def history(issue_date=None) -> list[dict]:
    """Все живые запуски (новые сверху), по желанию — только для одной даты выпуска."""
    out = []
    for f in sorted(live_dir().glob("*/*/run.json"), reverse=True):
        r = json.loads(f.read_text(encoding="utf-8"))
        if issue_date is None or r["issue_date"] == f"{pd.Timestamp(issue_date):%Y-%m-%d}":
            r["dir"] = str(f.parent)
            out.append(r)
    return out


def _last_forecast() -> pd.DataFrame | None:
    ptr = latest_pointer()
    if not ptr:
        return None
    files = sorted(Path(ptr["dir"]).glob("forecast_v*.csv"))
    return pd.read_csv(files[-1], parse_dates=["target_time_utc"]) if files else None


def llm_runs_today() -> int:
    today = local_date(now_utc())
    return sum(1 for r in history(today) if r.get("mode") == "llm")


def run_live(model, mode: str = "auto", now: pd.Timestamp | None = None, settings: dict | None = None,
             refresh: bool = True, log: Callable[[str], None] = print, on_step=None, max_llm_per_day: int = 8) -> dict:
    """Один живой выпуск: свежая погода → агент (момент прогноза = сейчас) → публикация в live/."""
    from windagent.agent.orchestrator import run_agent

    s = settings or load_settings()
    now = now or now_utc()
    if refresh:
        refresh_weather(now, s, log)
    load_env()
    if mode == "auto":
        mode = "llm" if os.environ.get("OPENAI_API_KEY") and llm_runs_today() < max_llm_per_day else "rules"
    issue = local_date(now, s)
    out = live_dir() / f"{issue:%Y-%m-%d}" / f"{now + pd.Timedelta(hours=s['scada']['utc_offset_hours']):%H%M}"
    latest = DataStore(now, settings=s, weather_dir=live_cache()).latest_run(MODEL)
    r = run_agent(issue, model, mode=mode, recheck=False, settings=s, out_dir=out, as_of=now,
                  weather_dir=live_cache(), prev_forecast=_last_forecast(), log=log, on_step=on_step)
    pointer = {"issue_date": f"{issue:%Y-%m-%d}", "as_of_utc": str(now), "dir": str(out), "mode": r["mode"],
               "ecmwf_run": None if latest is None else str(latest), "updated_unix": int(time.time())}
    (live_dir() / "latest.json").write_text(json.dumps(pointer, ensure_ascii=False, indent=1), encoding="utf-8")
    return {**r, **pointer}


def watch(model, interval_s: int = 900, log: Callable[[str], None] = print) -> None:
    """Планировщик: пересчитывает живой прогноз, когда выходит новый прогон ECMWF или начинаются новые сутки."""
    s = load_settings()
    while True:
        try:
            now = now_utc()
            info = refresh_weather(now, s, log)
            ptr = latest_pointer() or {}
            issue = f"{local_date(now, s):%Y-%m-%d}"
            if info["latest_run"] and (info["latest_run"] != ptr.get("ecmwf_run") or issue != ptr.get("issue_date")):
                log(f"новые входные данные (прогон {info['latest_run']}, выпуск {issue}) — запускаю агента")
                r = run_live(model, now=now, settings=s, refresh=False, log=log)
                log(f"опубликовано: {r['dir']} ({r['mode']})")
            else:
                log("новых прогонов нет")
        except Exception as e:  # сеть/API — пробуем на следующем круге
            log(f"ошибка цикла: {type(e).__name__}: {e}")
        time.sleep(interval_s)
