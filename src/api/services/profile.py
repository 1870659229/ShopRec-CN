"""画像服务：把用户近期行为实时折叠进 user_profile（秒级生效，不改模型参数）。"""

import json
import logging

from src.db import repo

logger = logging.getLogger(__name__)

# 折叠窗口：recent_items 取最近 50 条行为；interest_tags 取最近互动的 10 个不同品类
MAX_RECENT = 50
MAX_INTEREST_TAGS = 10


def should_retrain(new_count: int, threshold: int) -> bool:
    """本地累计新行为数达到阈值（含等于）即应触发重训。"""
    return new_count >= threshold


def refresh_profile(user_id: int) -> None:
    """按 user_id 重算并 UPSERT 实时画像。

    - recent_items：最近 MAX_RECENT 条行为的 item_id，最新在前。
      同秒并列时用自增 id 做次序打破，保证顺序确定。
    - interest_tags：最近互动过的最多 MAX_INTEREST_TAGS 个不同品类，按
      该品类最近一次互动时间倒序（同秒并列用 MAX(id) 打破）。JOIN 后用
      GROUP BY + MAX(ts) 排序，而非 DISTINCT + ORDER BY b.ts
      （后者在 ONLY_FULL_GROUP_BY 下报 3065）。
    """
    df = repo.query_df(
        "SELECT item_id FROM behavior WHERE user_id=:u"
        " ORDER BY ts DESC, id DESC LIMIT :n",
        {"u": user_id, "n": MAX_RECENT},
    )
    recent: list[int] = [int(x) for x in df["item_id"].tolist()] if not df.empty else []

    cate_df = repo.query_df(
        "SELECT i.category_id FROM behavior b JOIN item i ON b.item_id=i.item_id"
        " WHERE b.user_id=:u AND i.category_id IS NOT NULL"
        " GROUP BY i.category_id ORDER BY MAX(b.ts) DESC, MAX(b.id) DESC LIMIT :n",
        {"u": user_id, "n": MAX_INTEREST_TAGS},
    )
    tags: list[int] = (
        [int(x) for x in cate_df["category_id"].tolist()] if not cate_df.empty else []
    )

    repo.execute(
        "INSERT INTO user_profile (user_id, recent_items, interest_tags) VALUES (:u, :r, :t)"
        " ON DUPLICATE KEY UPDATE recent_items=:r, interest_tags=:t",
        {"u": user_id, "r": json.dumps(recent), "t": json.dumps(tags)},
    )
