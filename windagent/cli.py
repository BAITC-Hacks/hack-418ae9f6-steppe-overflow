"""Командная строка: `windagent <команда>`."""

from __future__ import annotations

import argparse
import sys


def _cmd_data(args: argparse.Namespace) -> int:
    from windagent.data import scada

    print(scada.summary())
    if args.export:
        from windagent.config import resolve

        out = resolve(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        scada.load_hourly().to_parquet(out)
        print(f"Почасовой датасет сохранён: {out}")
    return 0


def _cmd_weather_status(args: argparse.Namespace) -> int:
    from windagent.data import weather

    print(weather.status())
    return 0


def _cmd_weather_download(args: argparse.Namespace) -> int:
    from windagent.config import load_settings
    from windagent.data import weather

    s = load_settings()
    w = s["weather"]
    start = args.start or w["archive_start"]
    end = args.end or w["archive_end"]
    what = set(args.what.split(","))
    if "single" in what:
        for m in w["single_runs"]:
            print(weather.download_single_runs(m, start, end, workers=args.workers))
    if "previous" in what:
        for m in w["previous_runs"]:
            print(weather.download_previous_runs(m, start, end))
    if "neighbors" in what:
        print(weather.download_neighbors(start, end))
    if "era5" in what:
        print(weather.download_era5(args.era5_start, args.era5_end))
    print(weather.status())
    return 0


def _cmd_features_build(args: argparse.Namespace) -> int:
    from windagent.eval import backtest

    backtest.build_full_dataset()
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    import pandas as pd

    from windagent.config import load_settings, resolve
    from windagent.eval import backtest
    from windagent.models.baselines import default_baselines
    from windagent.models.gbm import EnsembleForecaster

    s = load_settings()
    periods = list(s["validation"]) if args.period == "all" else args.period.split(",")
    ds = backtest.load_dataset(s)
    pd.set_option("display.width", 160)
    all_preds = []
    for name in periods:
        per = s["validation"][name]
        models = default_baselines() + ([] if args.baselines_only else [EnsembleForecaster()])
        preds = backtest.run_backtest(models, ds, per["first_issue"], per["last_issue"], s)
        preds.insert(0, "period", name)
        all_preds.append(preds)
        table = backtest.report(preds)
        print(f"\n=== {name}: выпуски {per['first_issue']} … {per['last_issue']} (ВЭС, p_farm) ===")
        print(table.round(4).to_string(index=False))
    preds = pd.concat(all_preds, ignore_index=True)
    out = resolve("artifacts/backtest/backtest.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    # Компактная выборка для веб-интерфейса: ВЭС, основная модель и две опорные
    keep = ["period", "model", "issue_date", "target_time_utc", "target_time_local", "horizon_h", "lead_day",
            "p_farm", "p_farm_pred"]
    compact = preds[preds["model"].isin(["ensemble", "phys_ifs", "clim"])][keep]
    compact.to_parquet(resolve("artifacts/reports/backtest_preds.parquet"), index=False)
    md = resolve("artifacts/reports/backtest.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(backtest.markdown_summary(preds, "Результаты бэктеста"), encoding="utf-8")
    print(f"\nПрогнозы: {out}\nСводка: {md}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from windagent import forecast

    forecast.train_production()
    return 0


def _cmd_forecast(args: argparse.Namespace) -> int:
    import json

    import pandas as pd

    from windagent import forecast
    from windagent.config import resolve

    model = forecast.load_model()
    f, m = forecast.forecast_issue(args.date, model)
    out = resolve("artifacts/forecasts")
    out.mkdir(parents=True, exist_ok=True)
    f.to_csv(out / f"forecast_{args.date}.csv", index=False)
    (out / f"manifest_{args.date}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 160)
    print(f"Выпуск {args.date}: момент прогноза {m['as_of_utc']} UTC, модель {m['model_version']}")
    latest = {}
    for src in m["sources"]:
        if src.get("model") and src.get("max_published_at"):
            key = (src["source"], src["model"])
            latest[key] = max(latest.get(key, ""), src["max_published_at"])
    for (source, name), pub in latest.items():
        print(f"  {source:<8} {name:<14} самый свежий использованный прогон опубликован {pub}")
    print(f"  все источники опубликованы до момента прогноза: {m['checks']['all_sources_published_before_as_of']}")
    cols = ["target_time_local", "lead_day", "p_farm", "p_farm_q10", "p_farm_q90"]
    print(f.set_index("target_time_local")[cols[1:]].iloc[::3].to_string())
    print(f"Сохранено: {out}")
    return 0


def _cmd_submission(args: argparse.Namespace) -> int:
    from windagent import forecast
    from windagent.config import load_settings

    s = load_settings()
    try:
        model = forecast.load_model()
    except FileNotFoundError:
        model = forecast.train_production(s)
    f = s["forecast"]
    print(forecast.run_period(f["test_first_issue"], f["test_last_issue"], model, s))
    return 0


def _cmd_agent(args: argparse.Namespace) -> int:
    from windagent import forecast, protocol
    from windagent.agent.orchestrator import run_agent
    from windagent.config import load_settings

    s = load_settings()
    model = forecast.load_model()
    if args.period == "test":
        dates = protocol.issue_dates(s["forecast"]["test_first_issue"], s["forecast"]["test_last_issue"])
    else:
        dates = [args.date or s["forecast"]["test_first_issue"]]
    quiet = len(dates) > 1
    for d in dates:
        print(f"=== Агент: выпуск {d:%Y-%m-%d}" if hasattr(d, "strftime") else f"=== Агент: выпуск {d}")
        r = run_agent(d, model, mode=args.mode, recheck=not args.no_recheck, settings=s,
                      log=None if quiet else print)
        print(f"  режим: {r['mode']}, шагов: {r['steps']}, версий: {r['versions']}")
        if not quiet:
            print(f"\nОбъяснение: {r['explanation']}\n\nИтог: {r['summary']}\nЖурнал: {r['run_file']}")
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    from windagent import forecast, live

    model = forecast.load_model()
    if args.watch:
        live.watch(model, interval_s=args.interval)
        return 0
    r = live.run_live(model, mode=args.mode)
    print(f"\nЖивой выпуск {r['issue_date']} (момент прогноза {r['as_of_utc']} UTC, прогон ECMWF {r['ecmwf_run']}, "
          f"режим {r['mode']})\n{r['explanation']}\nЖурнал: {r['dir']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="windagent",
        description="Agentic AI прогноз почасовой выработки ВЭС на 24–48 часов",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("data", help="Сводка по SCADA-данным и экспорт почасового датасета")
    d.add_argument("--export", metavar="PATH", help="сохранить почасовой датасет в parquet")
    d.set_defaults(func=_cmd_data)

    w = sub.add_parser("weather", help="Архив прогнозов погоды (Open-Meteo) в локальном кэше")
    wsub = w.add_subparsers(dest="weather_command", required=True)
    ws = wsub.add_parser("status", help="что лежит в кэше")
    ws.set_defaults(func=_cmd_weather_status)
    wd = wsub.add_parser("download", help="докачать архив прогнозов в кэш (нужен интернет)")
    wd.add_argument("--start", help="первая дата (по умолчанию из конфига)")
    wd.add_argument("--end", help="последняя дата (по умолчанию из конфига)")
    wd.add_argument("--what", default="single,previous", help="single,previous,neighbors,era5")
    wd.add_argument("--workers", type=int, default=4, help="параллельных запросов")
    wd.add_argument("--era5-start", default="2023-03-01")
    wd.add_argument("--era5-end", default="2026-01-31")
    wd.set_defaults(func=_cmd_weather_download)

    f = sub.add_parser("features", help="Признаки по историческим и тестовым выпускам")
    fsub = f.add_subparsers(dest="features_command", required=True)
    fsub.add_parser("build", help="собрать artifacts/features/dataset.parquet").set_defaults(func=_cmd_features_build)

    b = sub.add_parser("backtest", help="Бэктест на контрольных периодах (протокол как у теста)")
    b.add_argument("--period", default="all", help="feb2025,dec2025,jan2026 или all")
    b.add_argument("--baselines-only", action="store_true", help="без основной модели (быстро)")
    b.set_defaults(func=_cmd_backtest)

    sub.add_parser("train", help="Обучить итоговую модель на данных до первого тестового выпуска").set_defaults(func=_cmd_train)

    fc = sub.add_parser("forecast", help="Прогноз одного выпуска (48 ч) с манифестом источников")
    fc.add_argument("--date", required=True, help="дата выпуска, напр. 2026-02-10")
    fc.set_defaults(func=_cmd_forecast)

    ag = sub.add_parser("agent", help="AI-агент: полный цикл выпуска прогноза с анализом и пересчётом")
    ag.add_argument("--date", help="дата выпуска, напр. 2026-02-10 (по умолчанию первая дата теста)")
    ag.add_argument("--period", choices=["test"], help="все 28 выпусков тестового периода")
    ag.add_argument("--mode", choices=["auto", "rules", "llm"], default="auto",
                    help="auto: LLM, если задан OPENAI_API_KEY, иначе правила")
    ag.add_argument("--no-recheck", action="store_true", help="без пересчёта при выходе нового прогона")
    ag.set_defaults(func=_cmd_agent)

    lv = sub.add_parser("live", help="Живой режим: прогноз на настоящее завтра по свежим прогнозам погоды")
    lv.add_argument("--watch", action="store_true", help="планировщик: пересчёт при выходе нового прогона")
    lv.add_argument("--interval", type=int, default=900, help="период проверки, с (по умолчанию 15 мин)")
    lv.add_argument("--mode", choices=["auto", "rules", "llm"], default="auto")
    lv.set_defaults(func=_cmd_live)

    sub.add_parser("submission", help="Прогноз на весь тестовый период (31.01–27.02.2026)").set_defaults(func=_cmd_submission)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
