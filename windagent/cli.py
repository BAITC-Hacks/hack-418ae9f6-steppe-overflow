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

    s = load_settings()
    periods = list(s["validation"]) if args.period == "all" else args.period.split(",")
    ds = backtest.load_dataset(s)
    pd.set_option("display.width", 160)
    all_preds = []
    for name in periods:
        per = s["validation"][name]
        preds = backtest.run_backtest(default_baselines(), ds, per["first_issue"], per["last_issue"], s)
        preds.insert(0, "period", name)
        all_preds.append(preds)
        table = backtest.report(preds)
        print(f"\n=== {name}: выпуски {per['first_issue']} … {per['last_issue']} (ВЭС, p_farm) ===")
        print(table.round(4).to_string(index=False))
    preds = pd.concat(all_preds, ignore_index=True)
    out = resolve("artifacts/backtest/baselines.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    md = resolve("artifacts/reports/baselines.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(backtest.markdown_summary(preds, "Бейзлайны: результаты бэктеста"), encoding="utf-8")
    print(f"\nПрогнозы: {out}\nСводка: {md}")
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
    wd.add_argument("--what", default="single,previous", help="single,previous,era5")
    wd.add_argument("--workers", type=int, default=4, help="параллельных запросов")
    wd.add_argument("--era5-start", default="2023-03-01")
    wd.add_argument("--era5-end", default="2026-01-31")
    wd.set_defaults(func=_cmd_weather_download)

    f = sub.add_parser("features", help="Признаки по историческим и тестовым выпускам")
    fsub = f.add_subparsers(dest="features_command", required=True)
    fsub.add_parser("build", help="собрать artifacts/features/dataset.parquet").set_defaults(func=_cmd_features_build)

    b = sub.add_parser("backtest", help="Бэктест на контрольных периодах (протокол как у теста)")
    b.add_argument("--period", default="all", help="feb2025,dec2025,jan2026 или all")
    b.set_defaults(func=_cmd_backtest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
