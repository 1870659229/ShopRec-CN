import pandas as pd
import pytest

from src.features.builder import audit_no_leakage, build

LEAKY = ["user_click_mean", "item_sales_max", "user_avg_rating"]


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [1, 1, 2], "item_id": [10, 11, 10],
        "category_id": [100, 100, 200], "behavior_type": ["pv", "buy", "cart"],
        "timestamp": [1, 2, 3], "ts": pd.date_range("2017-11-25", periods=3),
    })


def test_build_excludes_raw_ids_from_features():
    feats = build(_df())
    assert "user_id" not in feats.columns
    assert "behavior_type" not in feats.columns


def test_audit_rejects_aggregate_features():
    with pytest.raises(ValueError, match="聚合统计特征"):
        audit_no_leakage(LEAKY)


def test_audit_passes_clean_features():
    audit_no_leakage(["user_id", "item_id", "category_id"])


def test_build_output_columns_are_exactly_whitelisted():
    """输出列必须恰好是白名单 —— 防止以后有人悄悄加列引入泄露。"""
    feats = build(_df())
    assert set(feats.columns) == {"item_id", "category_id", "label"}


def test_build_label_mapping():
    """_df() 的 behavior_type 为 pv/buy/cart ⇒ label 应为 0/1/1。"""
    feats = build(_df())
    assert list(feats["label"]) == [0, 1, 1]
