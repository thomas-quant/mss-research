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
