"""Plotly chart factories for the Sorare dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .helpers import (
    LEAGUE_COLS,
    PLAYER_COLS,
    POSITION_COLS,
    TEAM_COLS,
    first_existing,
    metric_options,
    numeric_options,
)


TEMPLATE = "plotly_dark"
COLORWAY = ["#38bdf8", "#22c55e", "#f59e0b", "#f43f5e", "#a78bfa", "#14b8a6"]


def empty_figure(title: str, message: str = "No compatible data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, font={"size": 16})
    fig.update_layout(template=TEMPLATE, title=title, height=420)
    return fig


def style(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        colorway=COLORWAY,
        title=title,
        height=height,
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
    )
    return fig


def histogram(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    if df.empty or column not in df.columns:
        return empty_figure(title)
    fig = px.histogram(df, x=column, nbins=32, marginal="box", opacity=0.85)
    return style(fig, title)


def position_distribution(df: pd.DataFrame) -> go.Figure:
    col = first_existing(df.columns, POSITION_COLS)
    if df.empty or not col:
        return empty_figure("Position Distribution")
    counts = df[col].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["position", "players"]
    fig = px.bar(counts, x="position", y="players", text="players")
    return style(fig, "Position Distribution")


def ranking_bar(df: pd.DataFrame, metric: str, title: str, limit: int = 20) -> go.Figure:
    player_col = first_existing(df.columns, PLAYER_COLS)
    if df.empty or metric not in df.columns or not player_col:
        return empty_figure(title)
    data = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(limit)
    fig = px.bar(data, x=metric, y=player_col, orientation="h", color=metric, color_continuous_scale="Tealgrn")
    fig.update_yaxes(autorange="reversed")
    return style(fig, title, height=max(420, 24 * len(data) + 120))


def expected_vs_actual(df: pd.DataFrame) -> go.Figure:
    player_col = first_existing(df.columns, PLAYER_COLS)
    color_col = first_existing(df.columns, POSITION_COLS)
    if df.empty or "expected_score" not in df.columns or "avg_score" not in df.columns:
        return empty_figure("Expected vs Actual Score")
    fig = px.scatter(
        df,
        x="expected_score",
        y="avg_score",
        color=color_col,
        hover_name=player_col,
        hover_data=[c for c in ["Selected Club", "age", "score_residual"] if c in df.columns],
    )
    lo = np.nanmin([df["expected_score"].min(), df["avg_score"].min()])
    hi = np.nanmax([df["expected_score"].max(), df["avg_score"].max()])
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Expected = Actual", line={"dash": "dash"}))
    return style(fig, "Expected vs Actual Score")


def residual_distribution(df: pd.DataFrame) -> go.Figure:
    return histogram(df, "score_residual", "Residual Distribution")


def feature_importance(df: pd.DataFrame, value_col: str | None = None, title: str = "Feature Importance") -> go.Figure:
    if df.empty:
        return empty_figure(title)
    feature_col = first_existing(df.columns, ["source_feature", "feature", "variable"])
    value_col = value_col or first_existing(df.columns, ["combined_rank_score", "rf_importance", "permutation_importance_mean", "mean_abs_shap"])
    if not feature_col or not value_col:
        return empty_figure(title)
    data = df.dropna(subset=[value_col]).copy()
    ascending = value_col == "combined_rank_score"
    data = data.sort_values(value_col, ascending=ascending).head(25)
    fig = px.bar(data, x=value_col, y=feature_col, orientation="h", color=value_col, color_continuous_scale="Viridis")
    fig.update_yaxes(autorange="reversed")
    return style(fig, title, height=max(420, 24 * len(data) + 120))


def correlation_heatmap(df: pd.DataFrame, target: str = "avg_score") -> go.Figure:
    nums = numeric_options(df)
    if df.empty or len(nums) < 2:
        return empty_figure("Correlation Heatmap")
    corr = df[nums].corr(numeric_only=True)
    if target in corr.columns:
        ordered = corr[target].abs().sort_values(ascending=False).head(16).index.tolist()
        corr = corr.loc[ordered, ordered]
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    return style(fig, "Correlation Heatmap", height=560)


def correlation_bars(corr_df: pd.DataFrame, positive: bool = True) -> go.Figure:
    title = "Top Positive Correlations" if positive else "Top Negative Correlations"
    if corr_df.empty or "feature" not in corr_df.columns or "pearson" not in corr_df.columns:
        return empty_figure(title)
    data = corr_df.dropna(subset=["pearson"]).sort_values("pearson", ascending=not positive).head(12)
    fig = px.bar(data, x="pearson", y="feature", orientation="h", color="pearson", color_continuous_scale="RdBu")
    fig.update_yaxes(autorange="reversed")
    return style(fig, title)


def radar_chart(df: pd.DataFrame, player_name: str | None, metrics: list[str] | None = None) -> go.Figure:
    player_col = first_existing(df.columns, PLAYER_COLS)
    if df.empty or not player_col or not player_name:
        return empty_figure("Player Radar")
    row = df[df[player_col].astype(str) == str(player_name)]
    if row.empty:
        return empty_figure("Player Radar", "Select a player")
    metrics = metrics or [m for m in metric_options(df) if m not in ["expected_score", "score_residual"]][:6]
    if len(metrics) < 3:
        return empty_figure("Player Radar", "Need at least three score metrics")
    values = []
    for metric in metrics:
        series = pd.to_numeric(df[metric], errors="coerce")
        value = pd.to_numeric(row.iloc[0][metric], errors="coerce")
        percentile = series.rank(pct=True).loc[row.index[0]] * 100 if pd.notna(value) else 0
        values.append(float(percentile))
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=metrics + metrics[:1], fill="toself", name=player_name))
    fig.update_polars(radialaxis={"visible": True, "range": [0, 100]})
    return style(fig, f"{player_name} Percentile Radar")


def percentile_bars(df: pd.DataFrame, player_name: str | None, metrics: list[str] | None = None) -> go.Figure:
    player_col = first_existing(df.columns, PLAYER_COLS)
    if df.empty or not player_col or not player_name:
        return empty_figure("Percentile Profile")
    row = df[df[player_col].astype(str) == str(player_name)]
    if row.empty:
        return empty_figure("Percentile Profile", "Select a player")
    metrics = metrics or metric_options(df)[:8]
    rows = []
    for metric in metrics:
        series = pd.to_numeric(df[metric], errors="coerce")
        percentile = series.rank(pct=True).loc[row.index[0]] * 100
        rows.append({"metric": metric, "percentile": percentile})
    plot_df = pd.DataFrame(rows).dropna()
    fig = px.bar(plot_df, x="percentile", y="metric", orientation="h", range_x=[0, 100], text="percentile")
    fig.update_traces(texttemplate="%{text:.0f}")
    fig.update_yaxes(autorange="reversed")
    return style(fig, f"{player_name} Percentiles")


def compare_players(df: pd.DataFrame, player_a: str | None, player_b: str | None, metrics: list[str] | None = None) -> go.Figure:
    player_col = first_existing(df.columns, PLAYER_COLS)
    if df.empty or not player_col or not player_a or not player_b:
        return empty_figure("Player Comparison", "Select two players")
    metrics = metrics or metric_options(df)[:8]
    rows = []
    for name in [player_a, player_b]:
        match = df[df[player_col].astype(str) == str(name)]
        if match.empty:
            continue
        for metric in metrics:
            rows.append({"Player": name, "metric": metric, "value": match.iloc[0][metric]})
    if not rows:
        return empty_figure("Player Comparison")
    fig = px.bar(pd.DataFrame(rows), x="metric", y="value", color="Player", barmode="group")
    return style(fig, "Player Comparison")


def outlier_scatter(df: pd.DataFrame, x: str = "total_minutes", y: str = "avg_score") -> go.Figure:
    player_col = first_existing(df.columns, PLAYER_COLS)
    color_col = "outlier_category" if "outlier_category" in df.columns else first_existing(df.columns, POSITION_COLS)
    if df.empty or x not in df.columns or y not in df.columns:
        return empty_figure("Outlier Scatter")
    fig = px.scatter(df, x=x, y=y, color=color_col, hover_name=player_col, hover_data=[c for c in ["age", "height_cm"] if c in df])
    return style(fig, "Outlier Scatter")


def boxplot(df: pd.DataFrame, metric: str = "avg_score") -> go.Figure:
    group = first_existing(df.columns, POSITION_COLS)
    if df.empty or metric not in df.columns:
        return empty_figure("Outlier Boxplot")
    fig = px.box(df, x=group, y=metric, points="all", color=group)
    return style(fig, f"{metric} by Position")


def team_residual_summary(df: pd.DataFrame) -> go.Figure:
    team_col = first_existing(df.columns, TEAM_COLS)
    if df.empty or not team_col or "score_residual" not in df.columns:
        return empty_figure("Team Overperformance")
    data = df.groupby(team_col, dropna=False)["score_residual"].mean().sort_values(ascending=False).head(20).reset_index()
    fig = px.bar(data, x="score_residual", y=team_col, orientation="h", color="score_residual", color_continuous_scale="RdYlGn")
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Team-Level Average Residual")
