"""召回层：品类内热度召回与跨品类迁移召回。"""

import logging
from typing import Final

from src.db import repo

logger = logging.getLogger(__name__)

# 负反馈行为类型：命中这些行为的商品不再重复推荐。
# ORD=已购买（强负信号，已拥有）；CART=已加购（弱负信号，意向已表达）。
# 若产品侧希望"加购商品仍可再推"，把本常量改为 ("ORD",) 即可。
NEGATIVE_BEHAVIORS: Final[tuple[str, ...]] = ("ORD", "CART")

# 排除集合上限：真实库重度用户的历史可能很长，NOT IN 列表过长会拖慢查询，
# 故只取最近的若干条负反馈（按行为时间倒序）。
_MAX_NEGATIVE_ITEMS: Final[int] = 500


def negative_items(user_id: int | None) -> list[int]:
    """取该用户的负反馈商品（已购买 / 已加购），供召回层排除。

    返回去重后的 item_id 列表；user_id 为空或无负反馈时返回空列表
    （空列表 = 不做排除，与改造前行为完全一致）。
    """
    if not user_id:
        return []
    binds = {f"b{i}": b for i, b in enumerate(NEGATIVE_BEHAVIORS)}
    placeholders = ", ".join(f":{name}" for name in binds)
    df = repo.query_df(
        "SELECT item_id FROM behavior"
        f" WHERE user_id=:u AND behavior_type IN ({placeholders})"
        " GROUP BY item_id ORDER BY MAX(ts) DESC LIMIT :n",
        {"u": user_id, "n": _MAX_NEGATIVE_ITEMS, **binds},
    )
    if df.empty:
        return []
    return [int(x) for x in df["item_id"].tolist()]


def _exclude_clause(exclude: list[int] | None) -> tuple[str, dict[str, int]]:
    """生成 NOT IN 子句与命名绑定参数。

    Args:
        exclude: 需排除的 item_id 列表；为空 / None 时返回空子句（不拼接），
            保证与改造前 SQL 完全一致。

    Returns:
        ``(sql_fragment, binds)``，可直接拼在 WHERE 之后。
    """
    if not exclude:
        return "", {}
    binds = {f"e{i}": int(v) for i, v in enumerate(exclude)}
    placeholders = ", ".join(f":{name}" for name in binds)
    return f" AND item_id NOT IN ({placeholders})", binds


def _hot_in_category(
    cate_id: int, n: int, exclude: list[int] | None = None
) -> list[tuple[int, int, float]]:
    """同品类内按热度取 Top-N（排除负反馈商品），返回 (item_id, category_id, pop_score)。

    排除发生在 **LIMIT 之前**，即"先剔除已购/已加购、再取 Top-N"，
    因此被排除的名额会由该品类后续商品递补，不会减少候选池规模
    （除非该品类商品被买光）。
    """
    cond, binds = _exclude_clause(exclude)
    df = repo.query_df(
        "SELECT item_id, category_id, pop_score FROM item"
        f" WHERE category_id=:c{cond} ORDER BY pop_score DESC LIMIT :n",
        {"c": cate_id, "n": n, **binds},
    )
    return [(int(r.item_id), int(r.category_id), float(r.pop_score or 0.0)) for r in df.itertuples()]


def _hot_global(n: int, exclude: list[int] | None = None) -> list[tuple[int, int, float]]:
    """全局热门兜底召回（排除负反馈商品）。"""
    cond, binds = _exclude_clause(exclude)
    df = repo.query_df(
        f"SELECT item_id, category_id, pop_score FROM item WHERE 1=1{cond}"
        " ORDER BY pop_score DESC LIMIT :n",
        {"n": n, **binds},
    )
    return [(int(r.item_id), int(r.category_id), float(r.pop_score or 0.0)) for r in df.itertuples()]


def category_internal(
    cate_id: int, k: int, exclude: list[int] | None = None
) -> list[tuple[int, int, float]]:
    """品类内召回（目标域）：同 category 内按热度取 Top-K。

    在函数内再截断一次：LIMIT 只是下推到库的优化，返回条数以本函数契约为准。
    exclude 为负反馈商品列表（见 :func:`negative_items`），默认 None 不排除。
    """
    return _hot_in_category(cate_id, k, exclude)[:k]


# 辅助路最多迁移的来源类目数：按类目热度取头部类目，控制探索噪声。
_MAX_AUX_CATES = 5


def cross_category(
    recent_cates: list[int], k: int, exclude: list[int] | None = None
) -> list[tuple[int, int, float]]:
    """跨品类迁移召回（辅助域）：从用户最近品类**之外**补充热门。

    先按类目热度取「排除 recent_cates 后」最热门的前 _MAX_AUX_CATES 个类目，
    每个来源品类只取 k 条，控制迁移信号噪声。recent_cates 为空时退化为
    全局热门兜底；非空但品类外无候选时返回空列表（不违背品类外约束）。
    exclude 为负反馈商品列表（见 :func:`negative_items`），默认 None 不排除。
    """
    if not recent_cates:
        return _hot_global(k, exclude)
    # NOT IN 动态占位符：SQLAlchemy text 下逐类目命名绑定，避免字符串拼接注入。
    binds: dict[str, int] = {f"c{i}": int(c) for i, c in enumerate(recent_cates)}
    placeholders = ", ".join(f":{name}" for name in binds)
    df = repo.query_df(
        "SELECT category_id, MAX(pop_score) AS cate_heat FROM item"
        f" WHERE category_id NOT IN ({placeholders})"
        " GROUP BY category_id ORDER BY cate_heat DESC LIMIT :n",
        {**binds, "n": _MAX_AUX_CATES},
    )
    pool: list[tuple[int, int, float]] = []
    for row in df.itertuples():
        pool.extend(_hot_in_category(int(row.category_id), k, exclude))
    return pool
