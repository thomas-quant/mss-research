import pandas as pd

from mss_research.plots import create_summary_plots


def test_create_summary_plots_writes_core_pngs(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "instrument": "ES",
                "timeframe": "5min",
                "event_type": "mss",
                "swing_tier": "short",
                "closed_through": True,
                "momentum_bucket": "high",
                "relative_volume_bucket": "normal",
                "broken_swing_rsi_divergence": False,
                "broken_swing_volume_divergence": False,
                "horizon": 5,
                "n": 100,
                "win_rate": 0.55,
                "win_rate_ci_low": 0.50,
                "win_rate_ci_high": 0.60,
                "mean_aligned_return": 0.001,
                "mean_aligned_return_ci_low": 0.0001,
                "mean_aligned_return_ci_high": 0.002,
                "median_aligned_return": 0.0005,
            },
            {
                "instrument": "NQ",
                "timeframe": "5min",
                "event_type": "rsi_divergence",
                "swing_tier": "short",
                "closed_through": pd.NA,
                "momentum_bucket": "medium",
                "relative_volume_bucket": "high",
                "broken_swing_rsi_divergence": pd.NA,
                "broken_swing_volume_divergence": pd.NA,
                "horizon": 5,
                "n": 80,
                "win_rate": 0.48,
                "win_rate_ci_low": 0.42,
                "win_rate_ci_high": 0.55,
                "mean_aligned_return": -0.0002,
                "mean_aligned_return_ci_low": -0.001,
                "mean_aligned_return_ci_high": 0.0004,
                "median_aligned_return": -0.0001,
            },
        ]
    )

    paths = create_summary_plots(summary, tmp_path)

    names = {p.name for p in paths}
    assert "win_rate_by_event_type.png" in names
    assert "mean_return_by_event_type.png" in names
    assert "sample_size_by_event_type.png" in names
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_create_mss_distribution_plot_writes_quantile_png(tmp_path):
    from mss_research.plots import create_mss_distribution_plot

    events = pd.DataFrame(
        [
            {"event_type": "mss", "swing_tier": "short", "closed_through": True, "timeframe": "5min", "aligned_return_5": -0.01},
            {"event_type": "mss", "swing_tier": "short", "closed_through": True, "timeframe": "5min", "aligned_return_5": 0.00},
            {"event_type": "mss", "swing_tier": "short", "closed_through": True, "timeframe": "5min", "aligned_return_5": 0.03},
            {"event_type": "rsi_divergence", "swing_tier": "short", "closed_through": pd.NA, "timeframe": "5min", "aligned_return_5": 0.10},
        ]
    )

    path = create_mss_distribution_plot(events, tmp_path, horizons=[5])

    assert path.name == "mss_aligned_return_distribution.png"
    assert path.exists() and path.stat().st_size > 0


def test_create_cisd_distribution_plot_writes_quantile_png(tmp_path):
    from mss_research.plots import create_cisd_distribution_plot

    events = pd.DataFrame(
        [
            {"event_type": "cisd", "cisd_break_level_type": "open", "aligned_return_5": -0.01},
            {"event_type": "cisd", "cisd_break_level_type": "open", "aligned_return_5": 0.00},
            {"event_type": "cisd", "cisd_break_level_type": "extreme", "aligned_return_5": 0.03},
            {"event_type": "mss", "swing_tier": "short", "closed_through": True, "aligned_return_5": 0.10},
        ]
    )

    path = create_cisd_distribution_plot(events, tmp_path, horizons=[5])

    assert path.name == "cisd_aligned_return_distribution.png"
    assert path.exists() and path.stat().st_size > 0


def test_summary_plots_include_leg_context_bucket_charts(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "instrument": "ES",
                "timeframe": "5min",
                "event_type": "mss",
                "swing_tier": "short",
                "closed_through": True,
                "momentum_bucket": "high",
                "relative_volume_bucket": "normal",
                "leg_rsi_momentum_bucket": "high",
                "leg_volume_bucket": "high",
                "broken_swing_rsi_divergence": False,
                "broken_swing_volume_divergence": False,
                "horizon": 5,
                "n": 100,
                "win_rate": 0.56,
                "win_rate_ci_low": 0.50,
                "win_rate_ci_high": 0.61,
                "mean_aligned_return": 0.001,
                "mean_aligned_return_ci_low": 0.0,
                "mean_aligned_return_ci_high": 0.002,
                "p25_aligned_return": -0.001,
                "median_aligned_return": 0.0005,
                "p75_aligned_return": 0.002,
            }
        ]
    )

    paths = create_summary_plots(summary, tmp_path)
    names = {p.name for p in paths}

    assert "win_rate_by_leg_rsi_momentum_bucket.png" in names
    assert "win_rate_by_leg_volume_bucket.png" in names


def test_leg_bucket_plots_exclude_nan_bucket_rows(tmp_path):
    summary = pd.DataFrame(
        [
            {"event_type": "mss", "swing_tier": "short", "closed_through": True, "horizon": 5, "n": 50, "win_rate": 0.55, "mean_aligned_return": 0.001, "leg_rsi_momentum_bucket": "high", "leg_volume_bucket": "high"},
            {"event_type": "rsi_divergence", "swing_tier": "short", "closed_through": pd.NA, "horizon": 5, "n": 500, "win_rate": 0.99, "mean_aligned_return": 0.01, "leg_rsi_momentum_bucket": pd.NA, "leg_volume_bucket": pd.NA},
        ]
    )

    data = create_summary_plots(summary, tmp_path)

    assert (tmp_path / "win_rate_by_leg_rsi_momentum_bucket.png").exists()
    assert (tmp_path / "win_rate_by_leg_volume_bucket.png").exists()


def test_summary_plots_include_time_of_day_and_session_volume_charts(tmp_path):
    summary = pd.DataFrame(
        [
            {"event_session": "ny_am", "event_type": "mss", "swing_tier": "short", "closed_through": True, "horizon": 15, "n": 50, "win_rate": 0.55, "mean_aligned_return": 0.001, "leg_volume_bucket": "high"},
            {"event_session": "london", "event_type": "mss", "swing_tier": "short", "closed_through": True, "horizon": 15, "n": 50, "win_rate": 0.45, "mean_aligned_return": -0.001, "leg_volume_bucket": "low"},
        ]
    )

    paths = create_summary_plots(summary, tmp_path)
    names = {p.name for p in paths}

    assert "win_rate_by_time_of_day_session.png" in names
    assert "win_rate_by_session_and_leg_volume.png" in names


def test_summary_plots_include_right_leg_mean_and_relative_momentum_charts(tmp_path):
    summary = pd.DataFrame(
        [
            {"event_type": "mss", "swing_tier": "short", "closed_through": True, "horizon": 15, "n": 50, "win_rate": 0.55, "mean_aligned_return": 0.001, "right_leg_rsi_mean_bucket": "high", "leg_rsi_mean_delta_bucket": "strengthening"},
            {"event_type": "mss", "swing_tier": "short", "closed_through": True, "horizon": 15, "n": 50, "win_rate": 0.45, "mean_aligned_return": -0.001, "right_leg_rsi_mean_bucket": "low", "leg_rsi_mean_delta_bucket": "weakening"},
        ]
    )

    paths = create_summary_plots(summary, tmp_path)
    names = {p.name for p in paths}

    assert "win_rate_by_right_leg_rsi_mean_bucket.png" in names
    assert "win_rate_by_leg_rsi_mean_delta_bucket.png" in names


def test_summary_plots_include_leg_relative_volume_delta_chart(tmp_path):
    summary = pd.DataFrame(
        [
            {"event_type": "mss", "swing_tier": "short", "closed_through": True, "horizon": 15, "n": 50, "win_rate": 0.55, "mean_aligned_return": 0.001, "leg_relative_volume_delta_bucket": "expanding"},
            {"event_type": "mss", "swing_tier": "short", "closed_through": True, "horizon": 15, "n": 50, "win_rate": 0.45, "mean_aligned_return": -0.001, "leg_relative_volume_delta_bucket": "contracting"},
        ]
    )

    paths = create_summary_plots(summary, tmp_path)
    names = {p.name for p in paths}

    assert "win_rate_by_leg_relative_volume_delta_bucket.png" in names


def test_summary_plots_include_cisd_break_level_type_chart(tmp_path):
    summary = pd.DataFrame(
        [
            {"event_type": "cisd", "cisd_break_level_type": "open", "horizon": 15, "n": 50, "win_rate": 0.55, "mean_aligned_return": 0.001},
            {"event_type": "cisd", "cisd_break_level_type": "extreme", "horizon": 15, "n": 50, "win_rate": 0.45, "mean_aligned_return": -0.001},
        ]
    )

    paths = create_summary_plots(summary, tmp_path)
    names = {p.name for p in paths}

    assert "win_rate_by_cisd_break_level_type.png" in names


def test_summary_plots_include_timeframe_comparison_charts(tmp_path):
    summary = pd.DataFrame(
        [
            {"timeframe": "1min", "event_type": "cisd", "cisd_break_level_type": "open", "horizon": 5, "n": 100, "win_rate": 0.51, "mean_aligned_return": 0.001, "p75_aligned_return": 0.002},
            {"timeframe": "5min", "event_type": "cisd", "cisd_break_level_type": "open", "horizon": 5, "n": 80, "win_rate": 0.49, "mean_aligned_return": -0.001, "p75_aligned_return": 0.003},
            {"timeframe": "15min", "event_type": "mss", "swing_tier": "short", "closed_through": True, "horizon": 5, "n": 60, "win_rate": 0.47, "mean_aligned_return": -0.002, "p75_aligned_return": 0.004},
        ]
    )

    paths = create_summary_plots(summary, tmp_path)
    names = {p.name for p in paths}

    assert "win_rate_by_timeframe_and_event_type.png" in names
    assert "mean_return_by_timeframe_and_event_type.png" in names
    assert "p75_return_by_timeframe_and_event_type.png" in names
    assert "cisd_p75_return_by_timeframe.png" in names
