import pandas as pd
import polars as pl
import pytest

from mss_research.features import (
    add_indicators,
    detect_cisd_events,
    detect_intermediate_swings,
    detect_mss_events,
    detect_swings,
    label_forward_returns,
    resample_ohlcv,
)


def as_pd(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def set_col(df, name, values):
    if isinstance(df, pl.DataFrame):
        return df.with_columns(pl.Series(name, values))
    df.loc[:, name] = values
    return df


def bars(highs, lows=None, closes=None, opens=None, volumes=None):
    n = len(highs)
    lows = lows if lows is not None else [h - 1 for h in highs]
    closes = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    opens = opens if opens is not None else closes
    volumes = volumes if volumes is not None else [100] * n
    return pd.DataFrame(
        {
            "datetime_utc": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )


def test_resample_ohlcv_aggregates_standard_fields():
    df = bars([10, 12, 11, 15, 14], lows=[9, 8, 10, 13, 12], closes=[9.5, 11, 10.5, 14, 13], opens=[9, 9.5, 11, 10.5, 14], volumes=[1, 2, 3, 4, 5])

    out = resample_ohlcv(df, "5min")

    assert len(out) == 1
    row = as_pd(out).iloc[0]
    assert row["Open"] == 9
    assert row["High"] == 15
    assert row["Low"] == 8
    assert row["Close"] == 13
    assert row["Volume"] == 15


def test_detect_swings_marks_strict_fractal_highs_and_lows():
    df = bars([1, 3, 2, 4, 3], lows=[0, 1, -1, 2, 1])

    out = detect_swings(df, k=1)

    out_pd = as_pd(out)
    assert out_pd["swing_high"].tolist() == [False, True, False, True, False]
    assert out_pd["swing_low"].tolist() == [False, False, True, False, False]
    assert out_pd.loc[1, "swing_high_available_idx"] == 2
    assert pd.isna(out_pd.loc[4, "swing_high_available_idx"])


def test_detect_intermediate_swings_uses_swing_of_swings_prominence():
    df = bars(
        [1, 3, 2, 5, 2, 4, 1, 2],
        lows=[0, 1, 0, 2, -2, 2, 0, 1],
    )
    st = detect_swings(df, k=1)

    out = detect_intermediate_swings(st, k=1)

    out_pd = as_pd(out)
    assert out_pd["intermediate_swing_high"].tolist() == [False, False, False, True, False, False, False, False]
    assert out_pd["intermediate_swing_low"].tolist() == [False, False, False, False, True, False, False, False]
    assert out_pd.loc[3, "intermediate_swing_high_available_idx"] == 6
    assert out_pd.loc[4, "intermediate_swing_low_available_idx"] == 7


def test_detect_mss_events_distinguishes_trade_and_close_breaks_and_waits_for_confirmation():
    df = bars(
        [10, 12, 11, 12.5, 13.0, 11],
        lows=[9, 10, 9, 11, 12, 10],
        closes=[9.5, 11.5, 10.5, 11.8, 12.6, 10.5],
        opens=[9.5, 11, 11, 11.6, 12, 12],
        volumes=[100, 100, 100, 200, 300, 100],
    )
    df = detect_swings(df, k=1)
    df = detect_intermediate_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)

    events = detect_mss_events(df, tier="short", k=1)

    bullish = as_pd(events).query("direction == 1").iloc[0]
    assert bullish["event_idx"] == 3
    assert bullish["traded_through"] == True
    assert bullish["closed_through"] == False
    assert bullish["broken_swing_idx"] == 1


def test_label_forward_returns_aligns_with_signal_direction():
    df = bars([10, 11, 12], lows=[9, 10, 11], closes=[10, 12, 9])
    events = pd.DataFrame(
        [
            {"event_idx": 0, "direction": 1, "event_type": "bull"},
            {"event_idx": 1, "direction": -1, "event_type": "bear"},
        ]
    )

    out = label_forward_returns(events, df, horizons=[1])

    out_pd = as_pd(out)
    assert out_pd.loc[0, "fwd_return_1"] == pytest.approx(0.2)
    assert out_pd.loc[0, "aligned_return_1"] == pytest.approx(0.2)
    assert out_pd.loc[0, "win_1"] == True
    assert out_pd.loc[1, "fwd_return_1"] == pytest.approx(-0.25)
    assert out_pd.loc[1, "aligned_return_1"] == pytest.approx(0.25)
    assert out_pd.loc[1, "win_1"] == True


def test_intermediate_mss_waits_until_after_prominence_confirmation_bar():
    df = bars(
        [1, 3, 2, 5, 2, 4, 3, 6.5],
        lows=[0, 1, 0, 2, -2, 2, 0, 1],
        closes=[1, 2.5, 1.5, 4.5, 1, 3, 5.5, 6.2],
    )
    df = detect_swings(df, k=1)
    df = detect_intermediate_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)

    events = detect_mss_events(df, tier="intermediate", k=1)

    bullish = as_pd(events).query("direction == 1").iloc[0]
    assert bullish["broken_swing_idx"] == 3
    assert bullish["event_idx"] == 7


def test_summarize_events_includes_p25_and_p75_aligned_returns():
    from mss_research.features import summarize_events

    events = pd.DataFrame(
        {
            "instrument": ["ES"] * 4,
            "timeframe": ["5min"] * 4,
            "event_type": ["mss"] * 4,
            "swing_tier": ["short"] * 4,
            "closed_through": [True] * 4,
            "momentum_bucket": ["high"] * 4,
            "relative_volume_bucket": ["normal"] * 4,
            "broken_swing_rsi_divergence": [False] * 4,
            "broken_swing_volume_divergence": [False] * 4,
            "aligned_return_5": [-0.02, -0.01, 0.01, 0.04],
            "win_5": [False, False, True, True],
        }
    )

    summary = summarize_events(events, horizons=[5], bootstrap_iterations=5)

    summary_pd = as_pd(summary)
    assert summary_pd.loc[0, "p25_aligned_return"] == pytest.approx(-0.0125)
    assert summary_pd.loc[0, "p75_aligned_return"] == pytest.approx(0.0175)


def test_bootstrap_ci_preserves_seeded_sampling_sequence():
    from mss_research.features import bootstrap_ci

    lo, hi = bootstrap_ci(pd.Series([0.0, 1.0, 1.0, 0.0]), iterations=5, seed=7, statistic="mean")

    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(0.5)


def test_summarize_events_groups_by_leg_relative_volume_delta_bucket():
    from mss_research.features import summarize_events

    events = pd.DataFrame(
        {
            "instrument": ["ES", "ES"],
            "timeframe": ["5min", "5min"],
            "event_type": ["mss", "mss"],
            "swing_tier": ["short", "short"],
            "closed_through": [True, True],
            "leg_relative_volume_delta_bucket": ["expanding", "contracting"],
            "aligned_return_5": [0.02, -0.01],
            "win_5": [True, False],
        }
    )

    summary = summarize_events(events, horizons=[5], bootstrap_iterations=5)

    assert set(as_pd(summary)["leg_relative_volume_delta_bucket"]) == {"expanding", "contracting"}


def test_mss_event_includes_leg_volume_and_rsi_momentum_context():
    df = bars(
        [10, 12, 11, 12.5, 13.0, 11],
        lows=[9, 10, 9, 11, 12, 10],
        closes=[9.5, 11.5, 10.5, 11.8, 12.6, 10.5],
        opens=[9.5, 11, 11, 11.6, 12, 12],
        volumes=[100, 100, 100, 200, 300, 100],
    )
    df = detect_swings(df, k=1)
    df = detect_intermediate_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)
    df = set_col(df, "rsi", [50, 55, 35, 72, 68, 45])

    events = detect_mss_events(df, tier="short", k=1)

    bullish = as_pd(events).query("direction == 1").iloc[0]
    assert bullish["leg_start_idx"] == 2
    assert bullish["leg_bar_count"] == 2
    assert bullish["leg_volume_sum"] == 300
    assert bullish["leg_relative_volume"] == pytest.approx(1.5)
    assert bullish["leg_volume_bucket"] == "high"
    assert bullish["leg_rsi_extreme"] == 72
    assert bullish["leg_rsi_aligned"] == 72
    assert bullish["leg_rsi_momentum_bucket"] == "high"


def test_bullish_mss_breaks_high_left_of_low_extremity_not_post_low_swing_high():
    df = bars(
        [15, 20, 16, 18, 17, 19.5, 20.5],
        lows=[14, 15, 10, 12, 11, 13, 14],
        closes=[14.5, 19, 11, 17, 16, 19, 20.2],
        volumes=[100] * 7,
    )
    df = detect_swings(df, k=1)
    df = detect_intermediate_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)

    events = detect_mss_events(df, tier="short", k=1)
    bullish = as_pd(events).query("direction == 1")

    assert bullish["event_idx"].tolist() == [6]
    assert bullish.iloc[0]["broken_swing_idx"] == 1
    assert bullish.iloc[0]["leg_start_idx"] == 2


def test_assigns_ict_style_time_of_day_sessions_in_new_york_time():
    from mss_research.features import assign_time_of_day_session

    times = pd.to_datetime(
        [
            "2024-01-02 02:00:00Z",  # 21:00 prior day NY
            "2024-01-02 08:00:00Z",  # 03:00 NY
            "2024-01-02 14:30:00Z",  # 09:30 NY
            "2024-01-02 19:00:00Z",  # 14:00 NY
            "2024-01-02 22:00:00Z",  # 17:00 NY
        ]
    )

    assert [assign_time_of_day_session(t) for t in times] == ["asia", "london", "ny_am", "ny_pm", "other"]


def test_mss_event_includes_time_of_day_session():
    df = bars(
        [10, 12, 11, 12.5, 13.0, 11],
        lows=[9, 10, 9, 11, 12, 10],
        closes=[9.5, 11.5, 10.5, 11.8, 12.6, 10.5],
        opens=[9.5, 11, 11, 11.6, 12, 12],
        volumes=[100, 100, 100, 200, 300, 100],
    )
    df["datetime_utc"] = pd.date_range("2024-01-02 14:27:00Z", periods=len(df), freq="min")
    df = detect_swings(df, k=1)
    df = detect_intermediate_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)

    events = detect_mss_events(df, tier="short", k=1)

    assert as_pd(events).iloc[0]["event_session"] == "ny_am"


def test_mss_event_includes_right_left_leg_rsi_mean_and_relative_momentum():
    df = bars(
        [15, 20, 16, 18, 17, 20.5],
        lows=[14, 15, 10, 12, 11, 13],
        closes=[14.5, 19, 11, 17, 16, 20.2],
        volumes=[100] * 6,
    )
    df = detect_swings(df, k=1)
    df = detect_intermediate_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)
    df = set_col(df, "rsi", [60, 70, 30, 55, 50, 75])

    events = detect_mss_events(df, tier="short", k=1)
    bullish = as_pd(events).query("direction == 1").iloc[0]

    assert bullish["leg_start_idx"] == 2
    assert bullish["left_leg_start_idx"] == 1
    assert bullish["right_leg_rsi_mean_aligned"] == pytest.approx((30 + 55 + 50 + 75) / 4)
    assert bullish["left_leg_rsi_mean_aligned"] == pytest.approx(100 - ((70 + 30) / 2))
    assert bullish["leg_rsi_mean_delta"] == pytest.approx(((30 + 55 + 50 + 75) / 4) - (100 - ((70 + 30) / 2)))
    assert bullish["right_leg_rsi_mean_bucket"] == "low"
    assert bullish["leg_rsi_mean_delta_bucket"] == "neutral"


def test_mss_event_includes_right_left_leg_relative_volume_delta():
    df = bars(
        [15, 20, 16, 18, 17, 20.5],
        lows=[14, 15, 10, 12, 11, 13],
        closes=[14.5, 19, 11, 17, 16, 20.2],
        volumes=[100, 200, 300, 600, 600, 600],
    )
    df = detect_swings(df, k=1)
    df = detect_intermediate_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)
    df = set_col(df, "rsi", [60, 70, 30, 55, 50, 75])

    events = detect_mss_events(df, tier="short", k=1)
    bullish = as_pd(events).query("direction == 1").iloc[0]

    assert bullish["leg_start_idx"] == 2
    assert bullish["left_leg_start_idx"] == 1
    assert bullish["left_leg_bar_count"] == 2
    assert bullish["left_leg_volume_sum"] == 500
    assert bullish["leg_volume_sum"] == 2100
    assert bullish["left_leg_relative_volume"] == pytest.approx(500 / (600 * 2))
    assert bullish["leg_relative_volume"] == pytest.approx(2100 / (600 * 4))
    assert bullish["leg_relative_volume_delta"] == pytest.approx((2100 / (600 * 4)) - (500 / (600 * 2)))
    assert bullish["leg_relative_volume_delta_bucket"] == "expanding"


def test_mss_event_volume_delta_is_nan_without_prior_left_leg():
    from mss_research.features import _leg_context

    df = bars(
        [10, 11, 12],
        lows=[9, 8, 10],
        closes=[9.5, 8.5, 11.5],
        volumes=[100, 200, 300],
    )
    df = detect_swings(df, k=1)
    df = detect_intermediate_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)

    context = _leg_context(df, event_idx=2, direction=1, leg_start_idx=1)

    assert pd.isna(context["left_leg_volume_sum"])
    assert pd.isna(context["left_leg_relative_volume"])
    assert pd.isna(context["leg_relative_volume_delta"])
    assert pd.isna(context["leg_relative_volume_delta_bucket"])


def test_detect_cisd_events_fires_bullish_open_and_extreme_breaks_after_confirmed_swing_low():
    df = bars(
        [12.2, 12.1, 11.0, 10.2, 11.0, 12.0, 12.6],
        lows=[11.8, 10.8, 9.8, 8.5, 9.5, 10.7, 11.8],
        closes=[12.0, 11.0, 10.0, 9.0, 10.5, 11.8, 12.3],
        opens=[12.0, 11.6, 10.8, 9.8, 9.2, 11.0, 12.0],
        volumes=[100] * 7,
    )
    df = detect_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)

    events = detect_cisd_events(df, k=1)

    out = as_pd(events).sort_values("cisd_break_level_type").reset_index(drop=True)
    assert out["event_type"].tolist() == ["cisd", "cisd"]
    assert out["direction"].tolist() == [1, 1]
    assert out["cisd_break_level_type"].tolist() == ["extreme", "open"]
    assert out["event_idx"].tolist() == [6, 5]
    assert out["cisd_anchor_idx"].tolist() == [3, 3]
    assert out["cisd_run_start_idx"].tolist() == [1, 1]
    assert out["cisd_run_end_idx"].tolist() == [3, 3]
    assert out["cisd_run_length"].tolist() == [3, 3]
    assert out.loc[0, "cisd_break_level"] == pytest.approx(12.1)
    assert out.loc[1, "cisd_break_level"] == pytest.approx(11.6)


def test_detect_cisd_events_fires_bearish_open_and_extreme_breaks_after_confirmed_swing_high():
    df = bars(
        [10.2, 11.2, 12.2, 13.5, 12.8, 11.5, 10.5],
        lows=[9.8, 10.0, 11.0, 12.0, 11.2, 10.1, 9.4],
        closes=[10.0, 11.0, 12.0, 13.0, 12.4, 10.2, 9.8],
        opens=[10.0, 10.4, 11.2, 12.6, 12.8, 12.0, 10.5],
        volumes=[100] * 7,
    )
    df = detect_swings(df, k=1)
    df = add_indicators(df, rsi_period=2, rolling_window=2)

    events = detect_cisd_events(df, k=1)

    out = as_pd(events).sort_values("cisd_break_level_type").reset_index(drop=True)
    assert out["event_type"].tolist() == ["cisd", "cisd"]
    assert out["direction"].tolist() == [-1, -1]
    assert out["cisd_break_level_type"].tolist() == ["extreme", "open"]
    assert out["event_idx"].tolist() == [6, 5]
    assert out["cisd_anchor_idx"].tolist() == [3, 3]
    assert out["cisd_run_start_idx"].tolist() == [1, 1]
    assert out["cisd_run_end_idx"].tolist() == [3, 3]
    assert out["cisd_run_length"].tolist() == [3, 3]
    assert out.loc[0, "cisd_break_level"] == pytest.approx(10.0)
    assert out.loc[1, "cisd_break_level"] == pytest.approx(10.4)


def test_detect_cisd_events_requires_three_candle_run_and_close_break():
    two_candle_run = bars(
        [12.0, 11.0, 10.2, 11.0, 12.0],
        lows=[11.5, 10.0, 8.5, 9.5, 10.5],
        closes=[12.0, 11.0, 10.0, 10.5, 12.2],
        opens=[12.0, 11.6, 10.8, 9.5, 11.0],
    )
    two_candle_run = add_indicators(detect_swings(two_candle_run, k=1), rsi_period=2, rolling_window=2)

    assert detect_cisd_events(two_candle_run, k=1).is_empty()

    intrabar_only = bars(
        [12.2, 12.1, 11.0, 10.2, 11.0, 12.5],
        lows=[11.8, 10.8, 9.8, 8.5, 9.5, 10.7],
        closes=[12.0, 11.0, 10.0, 9.0, 10.5, 11.4],
        opens=[12.0, 11.6, 10.8, 9.8, 9.2, 11.0],
    )
    intrabar_only = add_indicators(detect_swings(intrabar_only, k=1), rsi_period=2, rolling_window=2)

    assert detect_cisd_events(intrabar_only, k=1).is_empty()


def test_summarize_events_groups_by_cisd_break_level_type():
    from mss_research.features import summarize_events

    events = pd.DataFrame(
        {
            "instrument": ["ES", "ES"],
            "timeframe": ["5min", "5min"],
            "event_type": ["cisd", "cisd"],
            "cisd_break_level_type": ["open", "extreme"],
            "aligned_return_5": [0.02, -0.01],
            "win_5": [True, False],
        }
    )

    summary = summarize_events(events, horizons=[5], bootstrap_iterations=5)

    assert set(as_pd(summary)["cisd_break_level_type"]) == {"open", "extreme"}


def test_summarize_events_handles_mixed_cisd_and_non_cisd_rows():
    from mss_research.features import summarize_events

    n_mss = 110
    events = pd.DataFrame(
        {
            "instrument": ["ES"] * (n_mss + 1),
            "timeframe": ["5min"] * (n_mss + 1),
            "event_type": ["mss"] * n_mss + ["cisd"],
            "event_session": [f"s{i}" for i in range(n_mss)] + ["s_cisd"],
            "swing_tier": ["short"] * (n_mss + 1),
            "closed_through": [True] * (n_mss + 1),
            "cisd_break_level_type": [pd.NA] * n_mss + ["open"],
            "aligned_return_5": [0.01] * n_mss + [-0.01],
            "win_5": [True] * n_mss + [False],
        }
    )

    summary = summarize_events(events, horizons=[5], bootstrap_iterations=5)

    out = as_pd(summary)
    assert set(out["event_type"]) == {"mss", "cisd"}
    assert out.loc[out["event_type"] == "cisd", "cisd_break_level_type"].iloc[0] == "open"
