import logging

import pandas as pd

logger = logging.getLogger(__name__)
VALID_TYPES = {"pv", "cart", "fav", "buy"}

# 天池 UserBehavior 主时段（UTC）：2017-11-24 00:00 ~ 2017-12-04 00:00（半开区间）
# 该窗口外实测 15,719 行（占 0.0157%）为时间戳异常值，极端达 1902 年与 2037 年
TS_VALID_MIN = 1511481600
TS_VALID_MAX = 1512345600


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """过滤非法行为类型、去重、补 ts 列，并按有效时间窗口过滤越界时间戳。"""
    before = len(df)
    out = df[df["behavior_type"].isin(VALID_TYPES)].copy()
    n_bad_type = before - len(out)
    out = out.drop_duplicates(subset=["user_id", "item_id", "behavior_type", "timestamp"])
    n_dup = before - n_bad_type - len(out)
    if "ts" not in out.columns:
        out["ts"] = pd.to_datetime(out["timestamp"], unit="s")
    in_window = (out["ts"] >= pd.Timestamp(TS_VALID_MIN, unit="s")) & (
        out["ts"] < pd.Timestamp(TS_VALID_MAX, unit="s")
    )
    n_oob = int((~in_window).sum())
    out = out[in_window]
    logger.info(
        "清洗：%d -> %d 行（非法类型剔除 %d，重复剔除 %d，时间戳越界剔除 %d）",
        before,
        len(out),
        n_bad_type,
        n_dup,
        n_oob,
    )
    return out.reset_index(drop=True)
