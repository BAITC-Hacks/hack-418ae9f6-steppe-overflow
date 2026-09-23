"""Фирменный стиль графиков Steppe Wind. Цвет закреплён за сущностью на всех экранах.

Палитра проверена валидатором (цветовая слепота, контраст) на белом фоне:
зелёный прогноз не соседствует с оранжевым/красным — пара «зелёный–оранжевый»
неразличима при дейтеранопии. Жёлтый и розовый слабо контрастны с фоном,
поэтому у графиков всегда есть легенда и табличный вид.
"""

import plotly.graph_objects as go

# Бренд (логотип)
BRAND_DARK = "#0c4741"
BRAND_GREEN = "#00b72b"

# Текст и служебные элементы
INK = "#0c2f2b"
MUTED = "#5b6b66"
GRID = "#e6eee9"
AXIS = "#c9d8cf"
SURFACE = "#ffffff"
FONT = 'Manrope, Roboto, system-ui, -apple-system, "Segoe UI", sans-serif'

# Данные
FORECAST = "#1a9e4b"                 # прогноз ВЭС — зелёный
FACT = "#4a3aa7"                     # факт SCADA — фиолетовый
BLUE_SURFACE = "#ebf1f7"             # второй цвет бренда: меню, интервал, выпадающие списки
BAND = "#ebf1f7"                     # интервал P10–P90 и коридор моделей — второй цвет бренда
D2_SHADE = "rgba(12,71,65,0.035)"    # фон зоны D+2 (прогноз дальше — неопределённее)

NWP_COLORS = {"ifs": "#1a9e4b", "gfs": "#2a78d6", "ifs025": "#eda100", "icon": "#e87ba4"}
NWP_ORDER = ("ifs", "gfs", "ifs025", "icon")
NWP_NAMES = {
    "ifs": "ECMWF IFS 9 км",
    "icon": "ICON (DWD)",
    "gfs": "GFS (NOAA)",
    "ifs025": "ECMWF IFS 0.25°",
}
TURBINE_COLORS = {"p_t1": "#2a78d6", "p_t2": "#eda100"}

MODEL_NAMES = {
    "ensemble": "Основная модель (ансамбль)",
    "phys_ifs": "ECMWF → кривая мощности",
    "clim": "Климатология",
}
MODEL_COLORS = {"ensemble": "#1a9e4b", "phys_ifs": "#2a78d6", "clim": "#b3c2bb"}
VERSION_COLORS = ["#1a9e4b", "#2a78d6", "#eda100"]

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def base_layout(**kw) -> dict:
    """Спокойное оформление: тонкая сетка, легенда сверху слева, единая подсказка."""
    axis = dict(
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=True, linecolor=AXIS,
        ticks="outside", tickcolor=AXIS, ticklen=4, tickfont=dict(color=MUTED, size=12),
        title=dict(font=dict(color=MUTED, size=12), standoff=10), automargin=True,
    )
    layout = dict(
        template="none",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=12, r=16, t=48, b=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None,
                    font=dict(size=12, color=INK), bgcolor="rgba(0,0,0,0)"),
        xaxis={**axis, "showgrid": False},
        yaxis={**axis, "showline": False, "ticks": ""},
        font=dict(family=FONT, size=13, color=INK),
    )
    for key in ("xaxis", "yaxis"):
        if key in kw:
            layout[key] = {**layout[key], **kw.pop(key)}
    layout.update(kw)
    return layout


def style(fig: go.Figure, **kw) -> go.Figure:
    fig.update_layout(**base_layout(**kw))
    return fig
