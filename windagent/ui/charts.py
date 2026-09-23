"""Графики интерфейса (Plotly) в фирменном стиле: одна ось Y, тонкие линии, спокойная сетка,
легенда сверху, выборочные подписи (пик), затонированная зона D+2."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from windagent.ui import theme

HOUR_MS = 3_600_000
TIME_AXIS = dict(tickformat="%H:%M<br>%d.%m", dtick=6 * HOUR_MS, hoverformat="%d.%m %H:%M")


def _day_zones(fig: go.Figure, f: pd.DataFrame) -> None:
    """Сутки D+1 и D+2: лёгкая тонировка D+2 и подписи зон вверху графика."""
    x = pd.to_datetime(f["target_time_local"])
    d2 = x[f["lead_day"].to_numpy() == 2]
    if not len(d2):
        return
    start, end = d2.min() - pd.Timedelta(minutes=30), x.max() + pd.Timedelta(minutes=30)
    fig.add_vrect(x0=start, x1=end, fillcolor=theme.D2_SHADE, line_width=0, layer="below")
    fig.add_vline(x=start, line_width=1, line_color=theme.AXIS)
    for x0, text in ((x.min(), "Завтра · D+1"), (start, "Послезавтра · D+2")):
        fig.add_annotation(x=x0, y=1, yref="paper", xanchor="left", yanchor="top", xshift=6, yshift=-4,
                           text=text, showarrow=False, font=dict(size=11, color=theme.MUTED))


def _peak_label(fig: go.Figure, f: pd.DataFrame, col: str = "p_farm") -> None:
    d1 = f[f["lead_day"] == 1]
    if not len(d1):
        return
    peak = d1.loc[d1[col].idxmax()]
    fig.add_trace(go.Scatter(
        x=[peak["target_time_local"]], y=[peak[col]], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(size=9, color=theme.FORECAST, line=dict(width=2, color=theme.SURFACE)),
    ))
    fig.add_annotation(x=peak["target_time_local"], y=peak[col], text=f"<b>пик {peak[col]:.0%}</b>", showarrow=False,
                       yshift=18, bgcolor="rgba(255,255,255,0.85)", borderpad=2, font=dict(size=12, color=theme.INK))


def forecast_chart(f: pd.DataFrame, fact: pd.Series | None = None, show_turbines: bool = False) -> go.Figure:
    """Прогноз ВЭС на 48 ч: линия + интервал P10–P90, факт (если известен), турбины по желанию."""
    x = f["target_time_local"]
    fig = go.Figure()
    _day_zones(fig, f)
    if "p_farm_q10" in f:
        fig.add_trace(go.Scatter(x=x, y=f["p_farm_q90"], line=dict(width=0), hoverinfo="skip",
                                 showlegend=False, name="P90"))
        fig.add_trace(go.Scatter(x=x, y=f["p_farm_q10"], line=dict(width=0), fill="tonexty",
                                 fillcolor=theme.BAND, name="Интервал P10–P90",
                                 customdata=f[["p_farm_q10", "p_farm_q90"]],
                                 hovertemplate="интервал %{customdata[0]:.0%}–%{customdata[1]:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=f["p_farm"], name="Прогноз ВЭС",
                             line=dict(color=theme.FORECAST, width=2.6, shape="spline", smoothing=0.4),
                             hovertemplate="прогноз <b>%{y:.0%}</b><extra></extra>"))
    if show_turbines:
        for col, name in (("p_t1", "Турбина 1"), ("p_t2", "Турбина 2")):
            fig.add_trace(go.Scatter(x=x, y=f[col], name=name,
                                     line=dict(color=theme.TURBINE_COLORS[col], width=1.4),
                                     hovertemplate=f"{name.lower()} %{{y:.0%}}<extra></extra>"))
    if fact is not None and fact.notna().any():
        fig.add_trace(go.Scatter(x=x, y=fact.to_numpy(), name="Факт SCADA", mode="lines+markers",
                                 line=dict(color=theme.FACT, width=1.8),
                                 marker=dict(size=5, color=theme.FACT, line=dict(width=1.5, color=theme.SURFACE)),
                                 hovertemplate="факт <b>%{y:.0%}</b><extra></extra>"))
    _peak_label(fig, f)
    theme.style(fig, height=430, xaxis=TIME_AXIS,
                yaxis=dict(range=[0, 1.08], tickformat=".0%", dtick=0.25, title="мощность, % номинала"))
    return fig


def weather_chart(w: pd.DataFrame, show_all: bool = False) -> go.Figure:
    """Ветер на 100 м: коридор «минимум–максимум» по четырём моделям и линия главной модели ECMWF.

    Ширина коридора — насколько модели расходятся (неопределённость). show_all — показать все модели.
    """
    fig = go.Figure()
    _day_zones(fig, w)
    x = w["target_time_local"]
    cols = [f"{s}__wind_speed_100m" for s in theme.NWP_ORDER if f"{s}__wind_speed_100m" in w]
    ens = w[cols]
    fig.add_trace(go.Scatter(x=x, y=ens.max(axis=1), line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=ens.min(axis=1), line=dict(width=0), fill="tonexty", fillcolor=theme.BAND,
                             name="Коридор моделей (мин–макс)", customdata=ens.max(axis=1),
                             hovertemplate="коридор %{y:.1f}–%{customdata:.1f} м/с<extra></extra>"))
    for src in theme.NWP_ORDER:
        col = f"{src}__wind_speed_100m"
        if col not in w or (src != "ifs" and not show_all):
            continue
        main = src == "ifs"
        fig.add_trace(go.Scatter(
            x=x, y=w[col], name=theme.NWP_NAMES[src],
            line=dict(color=theme.NWP_COLORS[src], width=2.8 if main else 1.4, shape="spline", smoothing=0.4),
            hovertemplate=f"{theme.NWP_NAMES[src]}: <b>%{{y:.1f}} м/с</b><extra></extra>",
        ))
    # Опорные уровни — подписи справа, за областью графика, чтобы их не перекрывали линии
    for ws, label in ((3, "старт выработки"), (12, "номинал")):
        fig.add_hline(y=ws, line_width=1, line_color=theme.AXIS, layer="below")
        fig.add_annotation(x=1, xref="paper", y=ws, text=f"{label}<br>≈ {ws} м/с", showarrow=False, xanchor="left",
                           align="left", xshift=6, font=dict(size=11, color=theme.MUTED))
    theme.style(fig, height=400, xaxis=TIME_AXIS, yaxis=dict(rangemode="tozero", title="ветер на 100 м, м/с"),
                margin=dict(l=12, r=92, t=48, b=12))
    return fig


def mae_by_model_chart(table: pd.DataFrame) -> go.Figure:
    """MAE по моделям и периодам. Основная модель — зелёным, опорные — синим и серым."""
    fig = go.Figure()
    for model in table.index:
        fig.add_trace(go.Bar(
            x=table.columns, y=table.loc[model], name=theme.MODEL_NAMES.get(model, model),
            marker=dict(color=theme.MODEL_COLORS.get(model, theme.MUTED), cornerradius=4, line=dict(width=0)),
            text=[f"{v:.3f}" for v in table.loc[model]], textposition="outside",
            textfont=dict(size=11, color=theme.MUTED), cliponaxis=False,
            hovertemplate="%{x}: MAE <b>%{y:.3f}</b><extra></extra>",
        ))
    theme.style(fig, height=380, barmode="group", bargap=0.42, bargroupgap=0.14, hovermode="closest",
                yaxis=dict(rangemode="tozero", title="MAE, доля номинала"), xaxis=dict(showline=True, ticks=""))
    return fig


def error_by_horizon_chart(p: pd.DataFrame) -> go.Figure:
    """Средняя абсолютная ошибка по горизонту прогноза (часы от момента выпуска)."""
    fig = go.Figure()
    for model in ("ensemble", "phys_ifs", "clim"):
        g = p[p["model"] == model]
        if not len(g):
            continue
        e = (g["p_farm_pred"] - g["p_farm"]).abs().groupby(g["horizon_h"]).mean()
        fig.add_trace(go.Scatter(x=e.index, y=e.values, name=theme.MODEL_NAMES[model],
                                 line=dict(color=theme.MODEL_COLORS[model], width=2.6 if model == "ensemble" else 1.6,
                                           shape="spline", smoothing=0.4),
                                 hovertemplate=f"{theme.MODEL_NAMES[model]}: <b>%{{y:.3f}}</b><extra></extra>"))
    theme.style(fig, height=340, xaxis=dict(title="горизонт от момента прогноза, ч", dtick=6),
                yaxis=dict(rangemode="tozero", title="MAE"))
    return fig


def versions_chart(versions: list[dict]) -> go.Figure:
    """Версии прогноза агента: v1 (на момент выпуска) и ревизии после новых прогонов погоды."""
    fig = go.Figure()
    first = next((v["forecast"] for v in versions if v.get("forecast") is not None), None)
    if first is not None:
        _day_zones(fig, first)
    for i, v in enumerate(versions):
        f = v.get("forecast")
        if f is None:
            continue
        last = i == len(versions) - 1
        fig.add_trace(go.Scatter(
            x=f["target_time_local"], y=f["p_farm"], name=f"v{v['version']} — данные на {v['as_of_local']}",
            line=dict(color=theme.VERSION_COLORS[i % len(theme.VERSION_COLORS)], width=2.6 if last else 1.6,
                      shape="spline", smoothing=0.4),
            hovertemplate=f"v{v['version']}: <b>%{{y:.0%}}</b><extra></extra>",
        ))
    theme.style(fig, height=380, xaxis=TIME_AXIS,
                yaxis=dict(range=[0, 1.08], tickformat=".0%", dtick=0.25, title="мощность, % номинала"))
    return fig


def evaluation_chart(m: pd.DataFrame) -> go.Figure:
    """Прогноз «сутки вперёд» против загруженного факта за весь период."""
    d = m[m["lead_day"] == 1].sort_values("target_time_local")
    fig = go.Figure()
    if d["p_farm_q10"].notna().any():
        fig.add_trace(go.Scatter(x=d["target_time_local"], y=d["p_farm_q90"], line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=d["target_time_local"], y=d["p_farm_q10"], line=dict(width=0), fill="tonexty",
                                 fillcolor=theme.BAND, name="Интервал P10–P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["target_time_local"], y=d["p_farm"], name="Прогноз на сутки вперёд",
                             line=dict(color=theme.FORECAST, width=2), hovertemplate="прогноз <b>%{y:.0%}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=d["target_time_local"], y=d["fact"], name="Ваши данные (факт)",
                             line=dict(color=theme.FACT, width=1.5), hovertemplate="факт <b>%{y:.0%}</b><extra></extra>"))
    theme.style(fig, height=420, xaxis=dict(tickformat="%d.%m", hoverformat="%d.%m %H:%M"),
                yaxis=dict(range=[0, 1.05], tickformat=".0%", dtick=0.25, title="мощность, % номинала"))
    return fig


def daily_mae_chart(daily: pd.Series) -> go.Figure:
    """Ошибка прогноза «сутки вперёд» по дням — один ряд, один цвет."""
    fig = go.Figure(go.Bar(
        x=daily.index, y=daily.values, marker=dict(color=theme.FORECAST, cornerradius=3, line=dict(width=0)),
        hovertemplate="%{x|%d.%m}: MAE <b>%{y:.3f}</b><extra></extra>", name="MAE за сутки",
    ))
    theme.style(fig, height=280, hovermode="closest", bargap=0.35, showlegend=False,
                xaxis=dict(tickformat="%d.%m"), yaxis=dict(rangemode="tozero", title="MAE за сутки"))
    return fig


def forecast_vs_fact_chart(p: pd.DataFrame) -> go.Figure:
    """Каждая точка — час: прогноз основной модели против факта. Диагональ — идеальный прогноз."""
    d = p[p["model"] == "ensemble"].dropna(subset=["p_farm", "p_farm_pred"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=theme.AXIS, width=1.5),
                             name="идеальный прогноз", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=d["p_farm"], y=d["p_farm_pred"], mode="markers", name="час прогноза",
        marker=dict(size=5, color=theme.FORECAST, opacity=0.22, line=dict(width=0)),
        hovertemplate="факт %{x:.0%} → прогноз %{y:.0%}<extra></extra>",
    ))
    # Средний прогноз для каждого уровня факта — видно, где модель систематически ошибается
    bins = pd.cut(d["p_farm"], bins=[i / 10 for i in range(11)], include_lowest=True)
    mean = d.groupby(bins, observed=True).agg(x=("p_farm", "mean"), y=("p_farm_pred", "mean"))
    fig.add_trace(go.Scatter(x=mean["x"], y=mean["y"], mode="lines+markers", name="средний прогноз",
                             line=dict(color=theme.BRAND_DARK, width=2.4),
                             marker=dict(size=7, color=theme.BRAND_DARK, line=dict(width=2, color=theme.SURFACE)),
                             hovertemplate="при факте ≈ %{x:.0%} прогноз в среднем %{y:.0%}<extra></extra>"))
    theme.style(fig, height=420, hovermode="closest",
                xaxis=dict(range=[0, 1], tickformat=".0%", title="факт, % номинала", showgrid=True),
                yaxis=dict(range=[0, 1], tickformat=".0%", title="прогноз, % номинала", scaleanchor="x"))
    return fig


def error_hist_chart(p: pd.DataFrame) -> go.Figure:
    """Распределение ошибок основной модели (прогноз − факт) по всем часам бэктеста."""
    d = p[p["model"] == "ensemble"].dropna(subset=["p_farm", "p_farm_pred"])
    err = d["p_farm_pred"] - d["p_farm"]
    fig = go.Figure(go.Histogram(x=err, xbins=dict(start=-1, end=1, size=0.05), name="часы",
                                 marker=dict(color=theme.FORECAST, line=dict(width=1, color=theme.SURFACE)),
                                 hovertemplate="ошибка %{x}: <b>%{y}</b> ч<extra></extra>"))
    fig.add_vline(x=0, line_width=1.5, line_color=theme.BRAND_DARK)
    share = float((err.abs() <= 0.1).mean())
    fig.add_annotation(x=0, y=1, yref="paper", yanchor="bottom", text=f"{share:.0%} часов — ошибка не больше 10 п.п.",
                       showarrow=False, font=dict(size=12, color=theme.INK))
    theme.style(fig, height=320, hovermode="closest", bargap=0.05, showlegend=False,
                xaxis=dict(tickformat="+.0%", title="ошибка прогноза (прогноз − факт)", range=[-1, 1]),
                yaxis=dict(title="часов"))
    return fig
