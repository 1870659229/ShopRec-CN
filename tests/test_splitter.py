import pandas as pd
import pytest

from src.data.splitter import time_split, to_db
from src.db import repo


def _df() -> pd.DataFrame:
    # ts 故意用固定置换打乱顺序（非随机），使用例能鉴别 time_split 是否真的按时间排序；
    # 若直接用单调递增的 ts，去掉排序后用例仍会通过，失去鉴别力。
    hours = [5, 0, 9, 3, 7, 1, 8, 2, 6, 4]
    base = pd.Timestamp("2017-11-25")
    return pd.DataFrame({
        "user_id": [1] * 10,
        "item_id": range(10),
        "behavior_type": ["pv"] * 10,
        "timestamp": list(range(10)),
        "ts": [base + pd.Timedelta(hours=h) for h in hours],
    })


def test_split_by_time_not_random():
    train, test = time_split(_df(), ratio=0.8)
    assert len(train) == 8 and len(test) == 2
    assert train["ts"].max() <= test["ts"].min()


def test_split_preserves_all_rows():
    train, test = time_split(_df(), ratio=0.8)
    assert len(train) + len(test) == 10


# --- to_db（连真实 MySQL，库 shoprec_ci_test（由 tests/conftest.py 隔离）；用合成小数据，不读真实大文件）---

_TEST_USER = 900000001
_TEST_ITEMS = (800001, 800002)


def _behavior_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [_TEST_USER] * 3,
        "item_id": [800001, 800002, 800003],
        "behavior_type": ["pv", "buy", "cart"],
        "timestamp": [1511600000, 1511603600, 1511607200],  # epoch 秒，必须被丢弃
        "ts": pd.to_datetime([1511600000, 1511603600, 1511607200], unit="s"),
    })


def test_to_db_behavior_writes_rows_without_timestamp_column():
    # 先清空表，使 COUNT(*) 可确定
    repo.execute("DELETE FROM behavior")
    df = _behavior_df()
    n = to_db(df, "behavior")
    assert n == 3

    count = int(repo.query_df("SELECT COUNT(*) AS c FROM behavior").iloc[0]["c"])
    assert count == 3

    # behavior 表没有 timestamp 列：说明 epoch 秒列未被写入
    back = repo.query_df("SELECT * FROM behavior")
    assert "timestamp" not in back.columns

    # 回读的 ts 与输入一致
    got = repo.query_df("SELECT ts FROM behavior ORDER BY ts")["ts"].tolist()
    want = sorted(df["ts"].tolist())
    assert got == want

    repo.execute("DELETE FROM behavior")


def test_to_db_item_dedups_and_is_idempotent():
    repo.execute(
        "DELETE FROM item WHERE item_id IN (:a, :b)",
        {"a": _TEST_ITEMS[0], "b": _TEST_ITEMS[1]},
    )
    df = pd.DataFrame({
        "item_id": [800001, 800001, 800002],   # (800001, 400001) 重复两次
        "category_id": [400001, 400001, 400002],
    })
    to_db(df, "item")

    def _count() -> int:
        return int(
            repo.query_df(
                "SELECT COUNT(*) AS c FROM item WHERE item_id IN (:a, :b)",
                {"a": _TEST_ITEMS[0], "b": _TEST_ITEMS[1]},
            ).iloc[0]["c"]
        )

    assert _count() == 2  # 去重后 2 行
    to_db(df, "item")     # INSERT IGNORE 再写一次不应报错
    assert _count() == 2

    repo.execute(
        "DELETE FROM item WHERE item_id IN (:a, :b)",
        {"a": _TEST_ITEMS[0], "b": _TEST_ITEMS[1]},
    )


def test_to_db_unknown_table_raises():
    with pytest.raises(ValueError, match="未知表名"):
        to_db(_behavior_df(), "no_such_table")

