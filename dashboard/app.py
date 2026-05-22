"""Interactive Gradio dashboard for Sorare football player analysis outputs."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.utils import charts
from dashboard.utils.data_loader import load_dashboard_data
from dashboard.utils.filters import age_bounds, apply_player_filters, choices
from dashboard.utils.helpers import (
    LEAGUE_COLS,
    PLAYER_COLS,
    POSITION_COLS,
    TEAM_COLS,
    describe_feature,
    empty_df,
    first_existing,
    make_warning_markdown,
    metric_options,
    numeric_options,
    safe_round,
)


DATA = load_dashboard_data()
FINAL_DF = DATA.get("final")
RANKINGS_DF = DATA.get("rankings") if not DATA.get("rankings").empty else FINAL_DF
OVER_DF = DATA.get("overperformers")
UNDER_DF = DATA.get("underperformers")
OUTLIERS_DF = DATA.get("outliers")
FEATURE_DF = DATA.get("feature_importance")
SHAP_DF = DATA.get("shap")
CORR_DF = DATA.get("correlations")
UNMATCHED_DF = DATA.get("unmatched")

APP_CSS = """
.gradio-container {max-width: 1500px !important}
.app-title h1 {font-size: 2.1rem; margin-bottom: 0.1rem}
.kpi-grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin: 8px 0 16px}
.kpi-card {border:1px solid rgba(148,163,184,.28); border-radius:8px; padding:14px; background:rgba(15,23,42,.55)}
.kpi-card span {display:block; color:#94a3b8; font-size:.82rem; margin-bottom:6px}
.kpi-card strong {font-size:1.35rem; color:#e2e8f0}
.hint {color:#94a3b8; font-size:.92rem}
"""


def _player_col(df: pd.DataFrame) -> str | None:
    return first_existing(df.columns, PLAYER_COLS)


def _team_col(df: pd.DataFrame) -> str | None:
    return first_existing(df.columns, TEAM_COLS)


def _league_col(df: pd.DataFrame) -> str | None:
    return first_existing(df.columns, LEAGUE_COLS)


def _display_cols(df: pd.DataFrame, extra: list[str] | None = None) -> list[str]:
    preferred = [
        "Player",
        "Position",
        "Selected Club",
        "competition_id",
        "leagues_played",
        "age",
        "height_cm",
        "appearances",
        "total_minutes",
        "avg_score",
        "minutes_adjusted_score",
        "position_adjusted_score",
        "score_per_90",
        "consistency_score",
        "expected_score",
        "score_residual",
        "transfermarkt_match_confidence",
        "outlier_category",
    ]
    cols = [c for c in preferred + (extra or []) if c in df.columns]
    return cols or df.columns[:12].tolist()


def _sort_table(df: pd.DataFrame, metric: str, ascending: bool = False, limit: int = 100) -> pd.DataFrame:
    if df.empty:
        return empty_df("No rows available for the current filters.")
    if metric in df.columns:
        df = df.sort_values(metric, ascending=ascending, na_position="last")
    return df[_display_cols(df)].head(limit).reset_index(drop=True)


def _kpi_html(filtered: pd.DataFrame) -> str:
    if filtered.empty:
        return "<div class='kpi-grid'><div class='kpi-card'><span>No data</span><strong>0</strong></div></div>"

    player_col = _player_col(filtered)
    team_col = _team_col(filtered)
    league_col = _league_col(filtered)
    position_col = first_existing(filtered.columns, POSITION_COLS)
    avg_score = filtered["avg_score"].mean() if "avg_score" in filtered else np.nan
    median_score = filtered["avg_score"].median() if "avg_score" in filtered else np.nan
    high_score = filtered["avg_score"].max() if "avg_score" in filtered else np.nan
    avg_age = filtered["age"].mean() if "age" in filtered else np.nan
    avg_height = filtered["height_cm"].mean() if "height_cm" in filtered else np.nan

    kpis = [
        ("Players", filtered[player_col].nunique() if player_col else len(filtered)),
        ("Teams", filtered[team_col].nunique() if team_col else "N/A"),
        ("Leagues", filtered[league_col].dropna().astype(str).str.split(",").explode().str.strip().nunique() if league_col else "N/A"),
        ("Positions", filtered[position_col].nunique() if position_col else "N/A"),
        ("Average Score", safe_round(avg_score)),
        ("Median Score", safe_round(median_score)),
        ("Highest Score", safe_round(high_score)),
        ("Average Age", safe_round(avg_age)),
        ("Average Height", safe_round(avg_height, 0)),
    ]
    cards = "".join(f"<div class='kpi-card'><span>{label}</span><strong>{value}</strong></div>" for label, value in kpis)
    return f"<div class='kpi-grid'>{cards}</div>"


def _age_pair(age_min, age_max) -> list[float]:
    """Normalize free-form number inputs into a valid age range."""
    lo = 0 if age_min is None else float(age_min)
    hi = 100 if age_max is None else float(age_max)
    return [min(lo, hi), max(lo, hi)]


def filter_final(positions, teams, leagues, age_min, age_max, min_minutes):
    age_range = _age_pair(age_min, age_max)
    filtered = apply_player_filters(FINAL_DF, positions, teams, leagues, age_range, min_minutes)
    summary = _sort_table(filtered, "avg_score", limit=25)
    return (
        _kpi_html(filtered),
        summary,
        charts.histogram(filtered, "avg_score", "Score Distribution"),
        charts.histogram(filtered, "age", "Age Distribution"),
        charts.position_distribution(filtered),
    )


def ranking_view(search, positions, teams, leagues, age_min, age_max, min_minutes, metric, selected_player, compare_a, compare_b):
    age_range = _age_pair(age_min, age_max)
    metric = metric or (metric_options(RANKINGS_DF) or ["avg_score"])[0]
    filtered = apply_player_filters(RANKINGS_DF, positions, teams, leagues, age_range, min_minutes)
    player_col = _player_col(filtered)
    if search and player_col:
        filtered = filtered[filtered[player_col].astype(str).str.contains(str(search), case=False, na=False)]

    table = _sort_table(filtered, metric, ascending=False, limit=200)
    top_chart = charts.ranking_bar(filtered, metric, f"Top Players by {metric}", limit=25)
    base = FINAL_DF if not FINAL_DF.empty else filtered
    return (
        table,
        top_chart,
        charts.radar_chart(base, selected_player),
        charts.percentile_bars(base, selected_player),
        charts.compare_players(base, compare_a, compare_b),
    )


def feature_view(target, importance_metric):
    base = FINAL_DF
    feature_table = FEATURE_DF.copy()
    if not feature_table.empty:
        feature_col = first_existing(feature_table.columns, ["source_feature", "feature"])
        if feature_col:
            feature_table["explanation"] = feature_table[feature_col].astype(str).map(describe_feature)
    return (
        charts.feature_importance(FEATURE_DF, importance_metric, "Model Feature Importance"),
        charts.feature_importance(SHAP_DF, "mean_abs_shap", "SHAP Summary" if not SHAP_DF.empty else "SHAP Summary"),
        charts.correlation_heatmap(base, target or "avg_score"),
        charts.correlation_bars(CORR_DF, positive=True),
        charts.correlation_bars(CORR_DF, positive=False),
        feature_table.head(50) if not feature_table.empty else empty_df("No feature importance file found."),
    )


def residual_view(positions, teams, leagues, age_min, age_max):
    age_range = _age_pair(age_min, age_max)
    residual_frames = []
    if not OVER_DF.empty:
        residual_frames.append(OVER_DF.assign(group="Overperformer"))
    if not UNDER_DF.empty:
        residual_frames.append(UNDER_DF.assign(group="Underperformer"))
    combined = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    filtered = apply_player_filters(combined, positions, teams, leagues, age_range, 0)
    over = _sort_table(apply_player_filters(OVER_DF, positions, teams, leagues, age_range, 0), "score_residual", limit=50)
    under = _sort_table(apply_player_filters(UNDER_DF, positions, teams, leagues, age_range, 0), "score_residual", ascending=True, limit=50)
    return (
        over,
        under,
        charts.expected_vs_actual(filtered),
        charts.residual_distribution(filtered),
        charts.ranking_bar(apply_player_filters(OVER_DF, positions, teams, leagues, age_range, 0), "score_residual", "Top Overperformers", 20),
        charts.ranking_bar(apply_player_filters(UNDER_DF, positions, teams, leagues, age_range, 0), "score_residual", "Least Efficient Profiles", 20),
        charts.team_residual_summary(filtered),
    )


def outlier_view(x_metric, y_metric, box_metric):
    df = OUTLIERS_DF if not OUTLIERS_DF.empty else FINAL_DF
    table = _sort_table(df, "avg_score", limit=150)
    explanation = (
        "Outliers combine z-score/IQR flags, Isolation Forest output where available, and category labels. "
        "Review extreme minutes, age, height, and score residuals before treating an outlier as a true player insight."
    )
    return table, charts.outlier_scatter(df, x_metric, y_metric), charts.boxplot(df, box_metric), explanation


def profile_view(player, compare_player):
    df = FINAL_DF if not FINAL_DF.empty else RANKINGS_DF
    player_col = _player_col(df)
    if df.empty or not player_col or not player:
        return empty_df("Select a player."), charts.empty_figure("Player Radar"), charts.empty_figure("Percentiles"), charts.empty_figure("Comparison")
    row = df[df[player_col].astype(str) == str(player)]
    profile = row[_display_cols(row)].T.reset_index()
    profile.columns = ["field", "value"]
    return (
        profile,
        charts.radar_chart(df, player),
        charts.percentile_bars(df, player),
        charts.compare_players(df, player, compare_player),
    )


def quality_view():
    df = FINAL_DF
    confidence_col = "transfermarkt_match_confidence"
    matched_count = int(df[confidence_col].notna().sum()) if confidence_col in df else 0
    unmatched_count = len(UNMATCHED_DF)
    duplicate_subset = [c for c in ["Player", "Selected Club"] if c in df.columns]
    duplicate_count = int(df.duplicated(subset=duplicate_subset).sum()) if duplicate_subset and not df.empty else 0
    summary = pd.DataFrame(
        {
            "metric": ["Matched players", "Unmatched players", "Potential duplicate rows", "Loaded final rows"],
            "value": [matched_count, unmatched_count, duplicate_count, len(df)],
        }
    )
    confidence_chart = charts.histogram(df, confidence_col, "Transfermarkt Match Confidence") if confidence_col in df else charts.empty_figure("Transfermarkt Match Confidence")
    duplicate_cols = [c for c in ["Player", "Selected Club", "Position", "transfermarkt_match_confidence"] if c in df.columns]
    duplicates = df[df.duplicated(subset=duplicate_subset, keep=False)][duplicate_cols] if duplicate_subset and duplicate_cols else empty_df("No duplicate diagnostics available.")
    return summary, UNMATCHED_DF.head(250) if not UNMATCHED_DF.empty else empty_df("No unmatched file found."), confidence_chart, duplicates.head(100)


def export_filtered(positions, teams, leagues, age_min, age_max, min_minutes, metric):
    age_range = _age_pair(age_min, age_max)
    filtered = apply_player_filters(FINAL_DF, positions, teams, leagues, age_range, min_minutes)
    if metric in filtered.columns:
        filtered = filtered.sort_values(metric, ascending=False)
    out_path = Path(tempfile.gettempdir()) / "sorare_filtered_players.csv"
    filtered.to_csv(out_path, index=False)
    return str(out_path)


def export_player_report(player):
    df = FINAL_DF if not FINAL_DF.empty else RANKINGS_DF
    player_col = _player_col(df)
    if df.empty or not player_col or not player:
        return None
    row = df[df[player_col].astype(str) == str(player)]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(player)).strip("_") or "player"
    out_path = Path(tempfile.gettempdir()) / f"sorare_player_report_{safe_name}.csv"
    row.to_csv(out_path, index=False)
    return str(out_path)


def build_dashboard() -> gr.Blocks:
    base_df = FINAL_DF if not FINAL_DF.empty else RANKINGS_DF
    positions = choices(base_df, POSITION_COLS)
    teams = choices(base_df, TEAM_COLS)
    leagues = choices(base_df, LEAGUE_COLS)
    min_age, max_age = age_bounds(base_df)
    metrics = metric_options(FINAL_DF) or metric_options(RANKINGS_DF) or ["avg_score"]
    player_col = _player_col(base_df)
    players = sorted(base_df[player_col].dropna().astype(str).unique().tolist()) if player_col and not base_df.empty else []
    numeric = numeric_options(base_df)
    importance_metrics = [c for c in FEATURE_DF.columns if c != "source_feature"] if not FEATURE_DF.empty else []

    with gr.Blocks(title="Sorare Player Analytics Dashboard") as demo:
        gr.Markdown("# Sorare Player Analytics Dashboard\nExplore rankings, model drivers, outliers, residuals, and Transfermarkt data quality.", elem_classes=["app-title"])
        gr.Markdown(make_warning_markdown(DATA.warnings))

        with gr.Tab("Overview"):
            with gr.Accordion("Filters", open=True):
                with gr.Row():
                    ov_pos = gr.Dropdown(positions, multiselect=True, label="Position")
                    ov_team = gr.Dropdown(teams, multiselect=True, label="Team")
                    ov_league = gr.Dropdown(leagues, multiselect=True, label="League")
                with gr.Row():
                    ov_age_min = gr.Number(value=min_age, label="Minimum age")
                    ov_age_max = gr.Number(value=max_age, label="Maximum age")
                    ov_minutes = gr.Number(value=0, label="Minimum minutes played")
            ov_kpis = gr.HTML()
            ov_refresh = gr.Button("Update Overview", variant="primary")
            ov_table = gr.Dataframe(label="Filtered Summary", interactive=False, wrap=True)
            with gr.Row():
                ov_score = gr.Plot()
                ov_age_plot = gr.Plot()
                ov_position = gr.Plot()
            for comp in [ov_pos, ov_team, ov_league, ov_age_min, ov_age_max, ov_minutes]:
                comp.change(filter_final, [ov_pos, ov_team, ov_league, ov_age_min, ov_age_max, ov_minutes], [ov_kpis, ov_table, ov_score, ov_age_plot, ov_position], show_progress="minimal")
            ov_refresh.click(filter_final, [ov_pos, ov_team, ov_league, ov_age_min, ov_age_max, ov_minutes], [ov_kpis, ov_table, ov_score, ov_age_plot, ov_position], show_progress="minimal")
            demo.load(filter_final, [ov_pos, ov_team, ov_league, ov_age_min, ov_age_max, ov_minutes], [ov_kpis, ov_table, ov_score, ov_age_plot, ov_position], show_progress="minimal")

        with gr.Tab("Player Rankings"):
            with gr.Accordion("Ranking Filters", open=True):
                with gr.Row():
                    search = gr.Textbox(label="Search player")
                    rank_metric = gr.Dropdown(metrics, value=metrics[0], label="Ranking metric")
                    rank_minutes = gr.Number(value=0, label="Minimum minutes")
                with gr.Row():
                    rank_pos = gr.Dropdown(positions, multiselect=True, label="Position")
                    rank_team = gr.Dropdown(teams, multiselect=True, label="Team")
                    rank_league = gr.Dropdown(leagues, multiselect=True, label="League")
                    rank_age_min = gr.Number(value=min_age, label="Minimum age")
                    rank_age_max = gr.Number(value=max_age, label="Maximum age")
                with gr.Row():
                    selected_player = gr.Dropdown(players, value=players[0] if players else None, label="Selected player")
                    compare_a = gr.Dropdown(players, value=players[0] if players else None, label="Compare player A")
                    compare_b = gr.Dropdown(players, value=players[1] if len(players) > 1 else None, label="Compare player B")
                rank_refresh = gr.Button("Update Rankings", variant="primary")
            rank_table = gr.Dataframe(label="Sortable Ranking Table", interactive=False, wrap=True)
            rank_bar = gr.Plot()
            with gr.Row():
                rank_radar = gr.Plot()
                rank_pct = gr.Plot()
            rank_compare = gr.Plot()
            rank_inputs = [search, rank_pos, rank_team, rank_league, rank_age_min, rank_age_max, rank_minutes, rank_metric, selected_player, compare_a, compare_b]
            rank_refresh.click(ranking_view, rank_inputs, [rank_table, rank_bar, rank_radar, rank_pct, rank_compare], show_progress="full")

        with gr.Tab("Feature Importance"):
            with gr.Row():
                target = gr.Dropdown(numeric, value="avg_score" if "avg_score" in numeric else (numeric[0] if numeric else None), label="Target variable")
                imp_metric = gr.Dropdown(importance_metrics, value=importance_metrics[0] if importance_metrics else None, label="Importance metric")
                feature_refresh = gr.Button("Update Feature Analysis", variant="primary")
            with gr.Row():
                imp_plot = gr.Plot()
                shap_plot = gr.Plot()
            heatmap = gr.Plot()
            with gr.Row():
                pos_corr = gr.Plot()
                neg_corr = gr.Plot()
            feature_table = gr.Dataframe(label="Variable Explanations", interactive=False, wrap=True)
            feature_refresh.click(feature_view, [target, imp_metric], [imp_plot, shap_plot, heatmap, pos_corr, neg_corr, feature_table], show_progress="full")

        with gr.Tab("Overperformers & Underperformers"):
            with gr.Row():
                res_pos = gr.Dropdown(positions, multiselect=True, label="Position")
                res_team = gr.Dropdown(teams, multiselect=True, label="Team")
                res_league = gr.Dropdown(leagues, multiselect=True, label="League")
                res_age_min = gr.Number(value=min_age, label="Minimum age")
                res_age_max = gr.Number(value=max_age, label="Maximum age")
                res_refresh = gr.Button("Update Residual Analysis", variant="primary")
            with gr.Row():
                over_table = gr.Dataframe(label="Hidden Gems / Overperformers", interactive=False, wrap=True)
                under_table = gr.Dataframe(label="Underperformers", interactive=False, wrap=True)
            exp_scatter = gr.Plot()
            with gr.Row():
                residual_plot = gr.Plot()
                team_residual = gr.Plot()
            with gr.Row():
                over_bar = gr.Plot()
                under_bar = gr.Plot()
            res_refresh.click(residual_view, [res_pos, res_team, res_league, res_age_min, res_age_max], [over_table, under_table, exp_scatter, residual_plot, over_bar, under_bar, team_residual], show_progress="full")

        with gr.Tab("Outliers"):
            gr.Markdown("Outlier diagnostics flag unusual statistical profiles. Use them as leads for data checks and player-context review.")
            with gr.Row():
                out_x = gr.Dropdown(numeric, value="total_minutes" if "total_minutes" in numeric else (numeric[0] if numeric else None), label="X metric")
                out_y = gr.Dropdown(numeric, value="avg_score" if "avg_score" in numeric else (numeric[1] if len(numeric) > 1 else None), label="Y metric")
                box_metric = gr.Dropdown(numeric, value="avg_score" if "avg_score" in numeric else (numeric[0] if numeric else None), label="Boxplot metric")
                out_refresh = gr.Button("Update Outliers", variant="primary")
            out_table = gr.Dataframe(label="Outlier Table", interactive=False, wrap=True)
            with gr.Row():
                out_scatter = gr.Plot()
                out_box = gr.Plot()
            out_explain = gr.Textbox(label="Interpretation guide", interactive=False)
            out_refresh.click(outlier_view, [out_x, out_y, box_metric], [out_table, out_scatter, out_box, out_explain], show_progress="full")

        with gr.Tab("Player Profile"):
            with gr.Row():
                profile_player = gr.Dropdown(players, value=players[0] if players else None, label="Player")
                profile_compare = gr.Dropdown(players, value=players[1] if len(players) > 1 else None, label="Compare with")
                profile_refresh = gr.Button("Update Profile", variant="primary")
            profile_table = gr.Dataframe(label="Player Details", interactive=False, wrap=True)
            with gr.Row():
                profile_radar = gr.Plot()
                profile_pct = gr.Plot()
            profile_cmp = gr.Plot()
            profile_refresh.click(profile_view, [profile_player, profile_compare], [profile_table, profile_radar, profile_pct, profile_cmp], show_progress="full")

        with gr.Tab("Transfermarkt Data Quality"):
            quality_refresh = gr.Button("Load Data Quality View", variant="primary")
            quality_summary = gr.Dataframe(label="Data Quality Summary", interactive=False)
            unmatched = gr.Dataframe(label="Unmatched Transfermarkt Players", interactive=False, wrap=True)
            confidence = gr.Plot()
            duplicates = gr.Dataframe(label="Potential Duplicate Matches", interactive=False, wrap=True)
            quality_refresh.click(quality_view, None, [quality_summary, unmatched, confidence, duplicates], show_progress="full")

        with gr.Tab("Export"):
            gr.Markdown("Create CSV exports from the current filter selections. Plot PNG export is available from each Plotly chart toolbar.")
            with gr.Row():
                ex_pos = gr.Dropdown(positions, multiselect=True, label="Position")
                ex_team = gr.Dropdown(teams, multiselect=True, label="Team")
                ex_league = gr.Dropdown(leagues, multiselect=True, label="League")
            with gr.Row():
                ex_age_min = gr.Number(value=min_age, label="Minimum age")
                ex_age_max = gr.Number(value=max_age, label="Maximum age")
                ex_minutes = gr.Number(value=0, label="Minimum minutes")
                ex_metric = gr.Dropdown(metrics, value=metrics[0], label="Sort metric")
            export_btn = gr.Button("Generate Filtered Dataset", variant="primary")
            export_file = gr.File(label="Filtered dataset CSV")
            export_btn.click(export_filtered, [ex_pos, ex_team, ex_league, ex_age_min, ex_age_max, ex_minutes, ex_metric], export_file, show_progress="full")
            with gr.Row():
                report_player = gr.Dropdown(players, value=players[0] if players else None, label="Player report")
                report_btn = gr.Button("Generate Player Report")
            report_file = gr.File(label="Player report CSV")
            report_btn.click(export_player_report, report_player, report_file, show_progress="full")

    return demo


def run_dashboard(share: bool = False) -> None:
    """Launch the dashboard with conservative defaults for local machines."""
    port = int(os.getenv("SORARE_DASHBOARD_PORT", "7860"))
    app = build_dashboard()
    app.queue(default_concurrency_limit=2).launch(
        theme=gr.themes.Soft(),
        css=APP_CSS,
        server_name="127.0.0.1",
        server_port=port,
        share=share,
    )


if __name__ == "__main__":
    run_dashboard()
