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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="windagent",
        description="Agentic AI прогноз почасовой выработки ВЭС на 24–48 часов",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("data", help="Сводка по SCADA-данным и экспорт почасового датасета")
    d.add_argument("--export", metavar="PATH", help="сохранить почасовой датасет в parquet")
    d.set_defaults(func=_cmd_data)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
