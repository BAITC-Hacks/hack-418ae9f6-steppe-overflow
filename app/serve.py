"""Запуск веб-интерфейса с прогревом.

Тяжёлые библиотеки, модель и архив погоды загружаются до первого посетителя, затем в том же
процессе стартует Streamlit — поэтому первая страница после перезапуска сервера открывается сразу.
  python app/serve.py --server.port=8501
"""

import sys
import time
from pathlib import Path

APP = Path(__file__).with_name("streamlit_app.py")


def warm_up() -> None:
    t0 = time.time()
    import plotly.graph_objects  # noqa: F401
    import sklearn.ensemble  # noqa: F401

    from windagent import forecast, protocol
    from windagent.config import load_settings
    from windagent.data import scada
    from windagent.ui import data

    forecast.load_model()
    data.submission()
    data.backtest_preds()
    first = protocol.issue_dates(load_settings()["forecast"]["test_first_issue"], "2026-02-01")[0]
    data.weather_for_issue(first)  # читает и кэширует архив прогнозов погоды
    scada.load_hourly()
    print(f"Прогрев завершён за {time.time() - t0:.1f} с", flush=True)


if __name__ == "__main__":
    warm_up()
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
    sys.exit(cli.main())
