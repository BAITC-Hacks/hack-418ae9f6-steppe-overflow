"""Цвета и оформление графиков. Цвет закреплён за сущностью на всех экранах.

Палитра проверена валидатором (CVD, контраст) для светлой и тёмной темы.
Светлые оттенки (аква, жёлтый) слабо контрастны с фоном — поэтому у графиков
всегда есть легенда и таблица с теми же данными.
"""

FORECAST = "#2a78d6"   # прогноз ВЭС — синий
FACT = "#eb6834"       # факт SCADA — оранжевый
BAND = "rgba(42,120,214,0.14)"  # интервал P10–P90 — синяя «вуаль»
MUTED = "#898781"      # второстепенные опорные линии
GRID = "rgba(137,135,129,0.25)"

# Погодные модели: фиксированный порядок слотов палитры
NWP_COLORS = {
    "ifs": "#2a78d6",
    "icon": "#eb6834",
    "gfs": "#1baf7a",
    "ifs025": "#eda100",
}
NWP_NAMES = {
    "ifs": "ECMWF IFS 9 км (прогон с точным временем)",
    "icon": "ICON (DWD)",
    "gfs": "GFS (NOAA)",
    "ifs025": "ECMWF IFS 0.25°",
}
TURBINE_COLORS = {"p_t1": "#1baf7a", "p_t2": "#eda100"}

MODEL_NAMES = {
    "ensemble": "Основная модель (ансамбль)",
    "phys_ifs": "ECMWF → кривая мощности",
    "clim": "Климатология",
}
MODEL_COLORS = {"ensemble": FORECAST, "phys_ifs": "#1baf7a", "clim": MUTED}


def base_layout(**kw) -> dict:
    """Спокойное оформление: тонкая сетка, легенда сверху, единая подсказка по оси X."""
    layout = dict(
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        xaxis=dict(showgrid=False, ticks="outside", ticklen=4),
        yaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13),
    )
    layout.update(kw)
    return layout
