#!/usr/bin/env python3
"""Interactive Streamlit Experiment Tracking & Visualization Dashboard.

Provides interactive leaderboards, metric comparison charts, feature importance
visualizations, threshold sensitivity curves, and country partition distributions
across all Entity Resolution experiments and runs.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.experiment_tracker import ExperimentTracker

# Page layout & title
st.set_page_config(
    page_title="Amazon ML Challenge 2026 - Entity Resolution Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏆 Business Entity Resolution — Experiment Dashboard")
st.markdown(
    "Track, visualize, and compare model architectures, decision threshold calibrations, "
    "and geographic partition metrics across experiments."
)

# Initialize tracker
tracker = ExperimentTracker(ledger_path=PROJECT_ROOT / "experiments" / "runs.json", use_mlflow=False)
runs = tracker.list_runs()
leaderboard_df = tracker.get_leaderboard()

# Sidebar: Controls & Stats
with st.sidebar:
    st.header("⚙️ Experiment Controls")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Summary Statistics")
    total_runs = len(runs)
    st.metric("Total Experiments Logged", total_runs)

    if not leaderboard_df.empty and "macro_f05" in leaderboard_df.columns:
        best_row = leaderboard_df.iloc[0]
        st.metric("Best Macro F_0.5", f"{best_row.get('macro_f05', 0.0):.4f}")
        st.metric("Best Threshold (τ*)", f"{best_row.get('best_tau', 0.50):.4f}")
        st.caption(f"Top Run: **{best_row.get('run_name')}**")
    else:
        st.info("No runs logged yet. Execute `run_entity_resolution.py` to record experiments.")

    st.markdown("---")
    st.markdown("**Competition Metric:** Macro $F_{0.5}$ (Precision weighted 2x over Recall)")
    st.markdown("**Team:** Outliers")

if leaderboard_df.empty:
    st.warning("No experiment records found in `experiments/runs.json`.")
    st.info("Tip: Run `python scripts/run_entity_resolution.py` to log your first experiment.")
    st.stop()

# Tab Navigation
tab_leaderboard, tab_analysis, tab_partitions, tab_compare = st.tabs([
    "📋 Experiments Leaderboard",
    "📈 Metric & Feature Analysis",
    "🌍 Geographic Partitions",
    "⚖️ Side-by-Side Comparison",
])

# ==============================================================================
# TAB 1: LEADERBOARD
# ==============================================================================
with tab_leaderboard:
    st.subheader("Leaderboard (Ranked by Validation Macro F_0.5)")

    # Color format columns
    display_cols = [
        "run_name",
        "macro_f05",
        "best_tau",
        "precision",
        "recall",
        "sample_train_s1",
        "max_candidates",
        "duration_s",
        "timestamp",
    ]
    present_cols = [c for c in display_cols if c in leaderboard_df.columns]

    st.dataframe(
        leaderboard_df[present_cols].style.highlight_max(
            subset=["macro_f05"] if "macro_f05" in leaderboard_df.columns else [],
            color="#2e7d32",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Metric trend over experiment iterations
    if len(leaderboard_df) > 1:
        st.markdown("### Metric Progression Over Time")
        time_df = leaderboard_df.copy()
        time_df["run_index"] = range(1, len(time_df) + 1)
        fig_trend = px.line(
            time_df,
            x="run_name",
            y="macro_f05",
            markers=True,
            title="Macro F_0.5 Score by Experiment",
            labels={"macro_f05": "Macro F_0.5", "run_name": "Experiment"},
        )
        fig_trend.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_trend, use_container_width=True)

# ==============================================================================
# TAB 2: METRIC & FEATURE ANALYSIS
# ==============================================================================
with tab_analysis:
    st.subheader("Deep-Dive Run Diagnostics")
    run_options = {r["run_name"]: r["run_id"] for r in runs}
    selected_run_name = st.selectbox("Select Experiment Run to Analyze:", list(run_options.keys()))
    selected_run = tracker.get_run(run_options[selected_run_name])

    if selected_run:
        col1, col2, col3, col4 = st.columns(4)
        m = selected_run.get("metrics", {})
        p = selected_run.get("params", {})
        with col1:
            st.metric("Macro F_0.5", f"{m.get('macro_f05', 0.0):.4f}")
        with col2:
            st.metric("Decision Threshold τ*", f"{m.get('best_tau', 0.50):.4f}")
        with col3:
            st.metric("S1 Train Sample", f"{p.get('sample_train_s1', 'Full'):,}" if isinstance(p.get('sample_train_s1'), int) else str(p.get('sample_train_s1', 'Full')))
        with col4:
            st.metric("Run Duration", f"{selected_run.get('duration_seconds', 0.0)} s")

        st.markdown("---")
        feat_col, thresh_col = st.columns([1, 1])

        with feat_col:
            st.markdown("#### 🔍 Feature Importances (Top 22 Pairwise Features)")
            fi = selected_run.get("feature_importances", {})
            if fi:
                fi_df = pd.DataFrame([
                    {"feature": k, "importance": float(v)} for k, v in fi.items()
                ]).sort_values(by="importance", ascending=True)

                fig_fi = px.bar(
                    fi_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"Feature Importances — {selected_run_name}",
                    labels={"importance": "Importance Score (Splits / Gain)", "feature": "Feature"},
                    color="importance",
                    color_continuous_scale="Viridis",
                )
                fig_fi.update_layout(height=550)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importances not recorded for this run.")

        with thresh_col:
            st.markdown("#### 🎯 Threshold Calibration Sensitivity Curve")
            tc = selected_run.get("threshold_curve", {})
            if tc and "thresholds" in tc and "scores" in tc:
                curve_df = pd.DataFrame({
                    "tau": tc["thresholds"],
                    "macro_f05": tc["scores"],
                })
                best_tau = m.get("best_tau", 0.50)
                best_score = m.get("macro_f05", 0.0)

                fig_curve = go.Figure()
                fig_curve.add_trace(
                    go.Scatter(
                        x=curve_df["tau"],
                        y=curve_df["macro_f05"],
                        mode="lines+markers",
                        name="Macro F_0.5",
                        line=dict(color="#1976d2", width=3),
                    )
                )
                # Optimal point marker
                fig_curve.add_trace(
                    go.Scatter(
                        x=[best_tau],
                        y=[best_score],
                        mode="markers",
                        marker=dict(size=14, color="#d32f2f", symbol="star"),
                        name=f"Optimal τ* = {best_tau:.4f}",
                    )
                )
                fig_curve.update_layout(
                    title=f"Threshold vs. Macro F_0.5 (Optimal: τ* = {best_tau:.4f})",
                    xaxis_title="Candidate Probability Threshold (τ)",
                    yaxis_title="Validation Macro F_0.5",
                    height=550,
                )
                st.plotly_chart(fig_curve, use_container_width=True)
            else:
                st.info("Threshold optimization curve not recorded for this run.")

# ==============================================================================
# TAB 3: GEOGRAPHIC PARTITIONS
# ==============================================================================
with tab_partitions:
    st.subheader("Geographic Partition Breakdown & Predictions")
    selected_run_name_p = st.selectbox(
        "Select Run for Partition Diagnostics:", list(run_options.keys()), key="part_run_select"
    )
    p_run = tracker.get_run(run_options[selected_run_name_p])

    if p_run and "partition_summary" in p_run and p_run["partition_summary"]:
        part_data = p_run["partition_summary"]
        part_rows = []
        for country, counts in part_data.items():
            tot = counts.get("total", counts.get("matched", 0) + counts.get("singletons", 0))
            matched = counts.get("matched", 0)
            singletons = counts.get("singletons", tot - matched)
            match_pct = (matched / tot * 100.0) if tot > 0 else 0.0
            part_rows.append({
                "Country": country,
                "Total S1 Entities": tot,
                "Predicted Matched": matched,
                "Predicted Singletons": singletons,
                "Match Ratio (%)": round(match_pct, 2),
            })

        part_df = pd.DataFrame(part_rows)
        st.dataframe(part_df, use_container_width=True, hide_index=True)

        # Plotly grouped bar
        fig_part = px.bar(
            part_df,
            x="Country",
            y=["Predicted Matched", "Predicted Singletons"],
            title=f"Prediction Distribution per Country ({selected_run_name_p})",
            barmode="group",
            color_discrete_map={"Predicted Matched": "#2e7d32", "Predicted Singletons": "#0288d1"},
        )
        st.plotly_chart(fig_part, use_container_width=True)
    else:
        st.info("Partition summary not available for this run.")

# ==============================================================================
# TAB 4: RUN COMPARISON
# ==============================================================================
with tab_compare:
    st.subheader("Side-by-Side Experiment Comparison")
    if len(runs) >= 2:
        col_a, col_b = st.columns(2)
        with col_a:
            run_a_name = st.selectbox("Experiment A:", list(run_options.keys()), index=0)
            run_a = tracker.get_run(run_options[run_a_name])
        with col_b:
            run_b_name = st.selectbox("Experiment B:", list(run_options.keys()), index=1)
            run_b = tracker.get_run(run_options[run_b_name])

        if run_a and run_b:
            # Metrics comparison table
            ma = run_a.get("metrics", {})
            mb = run_b.get("metrics", {})
            pa = run_a.get("params", {})
            pb = run_b.get("params", {})

            cmp_data = [
                {"Dimension": "Macro F_0.5", run_a_name: ma.get("macro_f05"), run_b_name: mb.get("macro_f05"), "Delta": round((mb.get("macro_f05", 0) or 0) - (ma.get("macro_f05", 0) or 0), 4)},
                {"Dimension": "Optimal Threshold τ*", run_a_name: ma.get("best_tau"), run_b_name: mb.get("best_tau"), "Delta": round((mb.get("best_tau", 0) or 0) - (ma.get("best_tau", 0) or 0), 4)},
                {"Dimension": "S1 Sample Size", run_a_name: pa.get("sample_train_s1"), run_b_name: pb.get("sample_train_s1"), "Delta": "-"},
                {"Dimension": "Max Candidates (K)", run_a_name: pa.get("max_candidates"), run_b_name: pb.get("max_candidates"), "Delta": "-"},
                {"Dimension": "Duration (s)", run_a_name: run_a.get("duration_seconds"), run_b_name: run_b.get("duration_seconds"), "Delta": round(run_b.get("duration_seconds", 0) - run_a.get("duration_seconds", 0), 2)},
            ]
            st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)
    else:
        st.info("Log at least two runs to enable side-by-side comparison.")
