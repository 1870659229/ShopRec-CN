import logging

import pandas as pd

logger = logging.getLogger(__name__)
_FORBIDDEN_PATTERNS = ("_mean", "_max", "_min", "_avg", "_std", "_sum")
POSITIVE_TYPES = ("buy", "cart", "fav")


def audit_no_leakage(feature_cols: list[str]) -> None:
    """拒绝任何聚合统计特征，防止标签泄露。

    注意：这是**命名约定审计** —— 仅依据列名是否含 _mean/_max/_min/_avg/_std/_sum
    判断，**不是语义审计**。把聚合特征改成不含这些后缀的列名（例如 click_cnt）即可绕过。
    因此产出特征的地方必须同时保证只生成白名单内的列。

    匹配区分大小写，只匹配带前导下划线的子串（如 `_mean`），
    `x_SUM`/`A_Mean` 等不会被拦截。
    """
    bad = [c for c in feature_cols if any(p in c for p in _FORBIDDEN_PATTERNS)]
    if bad:
        raise ValueError(f"检测到疑似聚合统计特征，存在泄露风险：{bad}")
    logger.info("泄露审计通过，共 %d 个特征", len(feature_cols))


def build(df: pd.DataFrame) -> pd.DataFrame:
    """产出可直接喂给模型的 item 侧特征（标识与静态属性 + 标签）。

    注意：user_id 是分组 / 时间切分的键，不作为特征输出 ——
    入特征会让模型记住用户而非泛化，属用户侧泄露。

    返回值保留输入 DataFrame 的行索引，可用 index 与输入对齐。
    """
    feats = df[["item_id", "category_id"]].copy()
    feats["label"] = df["behavior_type"].isin(POSITIVE_TYPES).astype(int)
    audit_no_leakage(list(feats.columns))
    return feats
