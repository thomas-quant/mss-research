import pandas as pd
import pytest

from mss_research.features import (
    add_indicators,
    detect_intermediate_swings,
    detect_mss_events,
    detect_swings,
    label_forward_returns,
    resample_ohlcv,
)


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
    row = out.iloc[0]
    assert row["Open"] == 9
    assert row["High"] == 15
    assert row["Low"] == 8
    assert row["Close"] == 13
    assert row["Volume"] == 15


def test_detect_swings_marks_strict_fractal_highs_and_lows():
    df = bars([1, 3, 2, 4, 3], lows=[0, 1, -1, 2, 1])

    out = detect_swings(df, k=1)

    assert out["swing_high"].tolist() == [False, True, False, True, False]
    assert out["swing_low"].tolist() == [False, False, True, False, False]
    assert out.loc[1, "swing_high_available_idx"] == 2
    assert pd.isna(out.loc[4, "swing_high_available_idx"])


def test_detect_intermediate_swings_uses_swing_of_swings_prominence():
    df = bars(
        [1, 3, 2, 5, 2, 4, 1, 2],
        lows=[0, 1, 0, 2, -2, 2, 0, 1],
    )
    st = detect_swings(df, k=1)

    out = detect_intermediate_swings(st, k=1)

    assert out["intermediate_swing_high"].tolist() == [False, False, False, True, False, False, False, False]
    assert out["intermediate_swing_low"].tolist() == [False, False, False, False, True, False, False, False]
    assert out.loc[3, "intermediate_swing_high_available_idx"] == 6
    assert out.loc[4, "intermediate_swing_low_available_idx"] == 7


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

    bullish = events[events["direction"] == 1].iloc[0]
    assert bullish["event_idx"] == 3
    assert bullish["traded_through"] is True
    assert bullish["closed_through"] is False
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

    assert out.loc[0, "fwd_return_1"] == pytest.approx(0.2)
    assert out.loc[0, "aligned_return_1"] == pytest.approx(0.2)
    assert out.loc[0, "win_1"] is True
    assert out.loc[1, "fwd_return_1"] == pytest.approx(-0.25)
    assert out.loc[1, "aligned_return_1"] == pytest.approx(0.25)
    assert out.loc[1, "win_1"] is True


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

    bullish = events[events["direction"] == 1].iloc[0]
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

    assert summary.loc[0, "p25_aligned_return"] == pytest.approx(-0.0125)
    assert summary.loc[0, "p75_aligned_return"] == pytest.approx(0.0175)


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
    df.loc[:, "rsi"] = [50, 55, 35, 72, 68, 45]

    events = detect_mss_events(df, tier="short", k=1)

    bullish = events[events["direction"] == 1].iloc[0]
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
    bullish = events[events["direction"] == 1]

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

    assert events.iloc[0]["event_session"] == "ny_am"
