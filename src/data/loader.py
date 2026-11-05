import logging

import pandas as pd

logger = logging.getLogger(__name__)
COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]


def load_raw(path: str, nrows: int | None = None) -> pd.DataFrame:
    """加载天池 UserBehavior CSV（无表头，逗号分隔）。"""
    df = pd.read_csv(path, header=None, names=COLUMNS, nrows=nrows)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s")
    logger.info("加载 %d 行，时间范围 %s ~ %s", len(df), df["ts"].min(), df["ts"].max())
    return df
