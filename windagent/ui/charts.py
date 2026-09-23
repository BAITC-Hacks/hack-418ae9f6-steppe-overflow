"""Графики интерфейса (Plotly). Одна ось Y на график, тонкие линии, легенда сверху."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from windagent.ui import theme


def _day_divider(fig: go.Figure, f: pd.DataFrame) -> None:
    """Граница D+1 / D+2 — тонкая вертикаль с подписями суток."""
    d2 = f.loc[f["lead_day"] == 2, "target_time_local"]
    if len(d2):
        x = pd.Timestamp(d2.min())
        fig.add_vline(x=x, line_width=1, line_color=theme.MUTED, opacity=0.5)
        for text, anchor in (("  D+2 →", "left"), ("← D+1  ", "right")):
            fig.add_annotation(x=x, y=0.99, yref="paper", text=text, showarrow=False, xanchor=anchor,
                               yanchor="top", font=dict(size=11, color=theme.MUTED))


def forecast_chart(f: pd.DataFrame, fact: pd.Series | None = None, show_turbines: bool = False) -> go.Figure:
    """Прогноз ВЭС на 48 ч: линия + интервал P10–P90, факт (если известен), турбины по желанию."""
    x = f["target_time_local"]
    fig = go.Figure()
    if "p_farm_q10" in f:
        fig.add_trace(go.Scatter(x=x, y=f["p_farm_q90"], line=dict(width=0), hoverinfo="skip",
                                 showlegend=False, name="P90"))
        fig.add_trace(go.Scatter(x=x, y=f["p_farm_q10"], line=dict(width=0), fill="tonexty",
                                 fillcolor=theme.BAND, name="Интервал P10–P90",
                                 customdata=f[["p_farm_q10", "p_farm_q90"]],
                                 hovertemplate="P10–P90: %{customdata[0]:.2f}–%{customdata[1]:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=f["p_farm"], name="Прогноз ВЭС",
                             line=dict(color=theme.FORECAST, width=2.5, shape="spline", smoothing=0.3),
                             hovertemplate="прогноз %{y:.2f}<extra></extra>"))
    if show_turbines:
        for col, name in (("p_t1", "Турбина 1"), ("p_t2", "Турбина 2")):
            fig.add_trace(go.Scatter(x=x, y=f[col], name=name, line=dict(color=theme.TURBINE_COLORS[col], width=1.5),
                                     hovertemplate=f"{name.lower()} %{{y:.2f}}<extra></extra>"))
    if fact is not None and fact.notna().any():
        fig.add_trace(go.Scatter(x=x, y=fact.to_numpy(), name="Факт SCADA", mode="lines+markers",
                                 line=dict(color=theme.FACT, width=2), marker=dict(size=6),
                                 hovertemplate="факт %{y:.2f}<extra></extra>"))
    _day_divider(fig, f)
    fig.update_layout(**theme.base_layout(height=420, yaxis_title="мощность, доля номинала"))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_xaxes(tickformat="%d.%m %H:%M")
    return fig


def weather_chart(w: pd.DataFrame) -> go.Figure:
    """Прогноз ветра на 100 м от каждой погодной модели — разброс = неопределённость."""
    fig = go.Figure()
    for src in ("ifs", "icon", "gfs", "ifs025"):
        col = f"{src}__wind_speed_100m"
        if col in w:
            fig.add_trace(go.Scatter(
                x=w["target_time_local"], y=w[col], name=theme.NWP_NAMES[src],
                line=dict(color=theme.NWP_COLORS[src], width=2.5 if src == "ifs" else 1.5),
                hovertemplate=f"{src.upper()} %{{y:.1f}} м/с<extra></extra>",
            ))
    _day_divider(fig, w)
    fig.update_layout(**theme.base_layout(height=380, yaxis_title="ветер на 100 м, м/с"))
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(tickformat="%d.%m %H:%M")
    return fig


def mae_by_model_chart(table: pd.DataFrame) -> go.Figure:
    """MAE по моделям и периодам. Основная модель — синим, опорные — приглушённо."""
    fig = go.Figure()
    for model in table.index:
        fig.add_trace(go.Bar(
            x=table.columns, y=table.loc[model], name=theme.MODEL_NAMES.get(model, model),
            marker=dict(color=theme.MODEL_COLORS.get(model, theme.MUTED), cornerradius=4),
            text=[f"{v:.3f}" for v in table.loc[model]], textposition="outside",
            hovertemplate="%{x}: MAE %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(**theme.base_layout(height=360, barmode="group", bargap=0.45, bargroupgap=0.12,
                                          hovermode="closest", yaxis_title="MAE (доля номинала)"))
    fig.update_yaxes(rangemode="tozero")
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
                                 line=dict(color=theme.MODEL_COLORS[model], width=2.5 if model == "ensemble" else 1.5),
                                 hovertemplate="%{x} ч: MAE %{y:.3f}<extra></extra>"))
    fig.update_layout(**theme.base_layout(height=340, xaxis_title="горизонт от момента прогноза, ч",
                                          yaxis_title="MAE"))
    fig.update_yaxes(rangemode="tozero")
    return fig


VERSION_COLORS = ["#2a78d6", "#1baf7a", "#eda100"]


def versions_chart(versions: list[dict]) -> go.Figure:
    """Версии прогноза агента на одном графике: v1 (на момент выпуска) и ревизии после новых прогонов."""
    fig = go.Figure()
    for i, v in enumerate(versions):
        f = v.get("forecast")
        if f is None:
            continue
        label = f"v{v['version']} — данные на {v['as_of_local']}"
        fig.add_trace(go.Scatter(
            x=f["target_time_local"], y=f["p_farm"], name=label,
            line=dict(color=VERSION_COLORS[i % len(VERSION_COLORS)], width=2.5 if i == len(versions) - 1 else 1.8,
                      shape="spline", smoothing=0.3),
            hovertemplate=f"v{v['version']} %{{y:.2f}}<extra></extra>",
        ))
    first = next((v["forecast"] for v in versions if v.get("forecast") is not None), None)
    if first is not None:
        _day_divider(fig, first)
    fig.update_layout(**theme.base_layout(height=360, yaxis_title="мощность, доля номинала"))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_xaxes(tickformat="%d.%m %H:%M")
    return fig
