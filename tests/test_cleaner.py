import pandas as pd

from src.data.cleaner import TS_VALID_MAX, TS_VALID_MIN, clean

VALID = {"pv", "cart", "fav", "buy"}


def _sample() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 2],
        "item_id": [10, 10, 20, 20, 30],
        "category_id": [100, 100, 200, 200, 300],
        "behavior_type": ["pv", "pv", "pv", "bad", "buy"],
        "timestamp": [1511600000, 1511600000, 1511700000, 1511700001, 1511800000],
    })


def test_clean_removes_duplicates():
    out = clean(_sample())
    assert out.duplicated(subset=["user_id", "item_id", "behavior_type", "timestamp"]).sum() == 0


def test_clean_drops_invalid_behavior_type():
    out = clean(_sample())
    assert set(out["behavior_type"].unique()) <= VALID
    assert len(out) == 3


def test_clean_adds_ts_column():
    out = clean(_sample())
    assert "ts" in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["ts"])


def test_clean_drops_out_of_window_ts():
    df = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 20],
        "category_id": [100, 200],
        "behavior_type": ["pv", "pv"],
        "timestamp": [1511600000, 1535416235],   # 后者 = 2018-08-28，越界
    })
    out = clean(df)
    assert len(out) == 1
    assert out.iloc[0]["user_id"] == 1


def test_clean_window_boundaries_are_half_open():
    """半开区间 [MIN, MAX)：ts == MIN 保留，ts == MAX 剔除。"""
    df = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 20],
        "category_id": [100, 200],
        "behavior_type": ["pv", "pv"],
        "timestamp": [TS_VALID_MIN, TS_VALID_MAX],
    })
    out = clean(df)
    assert len(out) == 1
    assert out.iloc[0]["user_id"] == 1
