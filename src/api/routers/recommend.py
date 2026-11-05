"""推荐接口：读画像 → 召回 → 融合排序 → 落 recommend_log。"""

import json
import logging
import uuid

from fastapi import APIRouter, Query

from src.api.services import rank as rank_svc
from src.api.services.recall import category_internal, cross_category, negative_items
from src.db import repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["recommend"])

# 目标域召回的品类数上限：仅取画像中兴趣权重最高的前几个品类，
# 防止画像品类过多时目标域候选池被冲淡（与跨品类迁移路区分）。
_MAX_TGT_CATES = 3


@router.get("/recommend")
def recommend(
    user_id: int = Query(...),
    k: int = Query(10, ge=0, le=100),
) -> dict:
    """为用户生成 Top-K 推荐；无画像或候选池为空时优雅返回空 items。"""
    prof = repo.query_df(
        "SELECT interest_tags FROM user_profile WHERE user_id=:u", {"u": user_id}
    )
    cates: list[int] = []
    if not prof.empty and prof.iloc[0]["interest_tags"]:
        try:
            tags = json.loads(prof.iloc[0]["interest_tags"])
            cates = [int(x) for x in tags]
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            # 画像脏数据（非法 JSON / 非数字元素 / 顶层数字）：降级为空画像，
            # 走全局热门兜底，而不是让整个推荐接口 500。
            logger.warning(
                "user_id=%s interest_tags 脏数据（%s: %s），降级为空画像走全局热门兜底",
                user_id,
                type(exc).__name__,
                exc,
            )
            cates = []

    # 负反馈：该用户已购买 / 已加购的商品不再重复推荐。
    # 召回层在 LIMIT 之前剔除，被排除的名额由同品类后续商品递补，
    # 因此候选池规模不变、只换内容（买了 A 款表 → 推荐同品类其他表）。
    exclude = negative_items(user_id)

    candidates_tgt: list[tuple[int, int, float]] = []
    for c in cates[:_MAX_TGT_CATES]:
        candidates_tgt.extend(category_internal(c, k, exclude))
    candidates_aux = cross_category(cates, k, exclude)

    items = rank_svc.rank_with_sources(candidates_tgt, candidates_aux, k=k)

    # 无画像时 aux 路实为全局热门兜底，来源语义改写为 "hot"，
    # 与前端「冷启动走全局热门兜底」文案一致，避免误导性标签。
    if not cates:
        items = [(i, "hot") for i, _ in items]

    # 批量写入：原先逐条 INSERT（每条一个事务），k=40 时占端到端 94% 延迟。
    # 改为单事务批量后，网络往返与提交次数由 k 次降为 1 次。
    req_id = str(uuid.uuid4())
    if items:
        repo.execute_many(
            "INSERT INTO recommend_log"
            " (request_id, user_id, item_id, recall_source, rank_score, position)"
            " VALUES (:r, :u, :i, :s, :sc, :p)",
            [
                {
                    "r": req_id, "u": user_id, "i": item_id,
                    "s": source, "sc": 0.0, "p": pos + 1,
                }
                for pos, (item_id, source) in enumerate(items)
            ],
        )

    # 标题仅 demo 段位有值（演示标题），真实数据为空串，前端回落编号展示；
    # category_name 供前端商品卡标注类目（与 /api/item/search 字段对齐），
    # 无 category 行时 LEFT JOIN 得 NULL → 空串，前端回落原有占位行为；
    # demo_price 同理：演示价格仅 demo 段位有值，NULL → None（NaN 进 JSON
    # 是非法字面量，前端 JSON.parse 会静默失败）。
    titles: dict[int, str] = {}
    cate_names: dict[int, str] = {}
    prices: dict[int, float | None] = {}
    if items:
        id_ph = ", ".join(f":i{p}" for p in range(len(items)))
        tdf = repo.query_df(
            f"SELECT i.item_id, i.title, c.name AS category_name, i.demo_price"
            f" FROM item i LEFT JOIN category c ON c.category_id = i.category_id"
            f" WHERE i.item_id IN ({id_ph})",
            {f"i{p}": i for p, (i, _) in enumerate(items)},
        )
        titles = dict(zip(tdf["item_id"].astype(int), tdf["title"].fillna("")))
        cate_names = dict(zip(tdf["item_id"].astype(int), tdf["category_name"].fillna("")))
        prices = dict(
            zip(
                tdf["item_id"].astype(int),
                tdf["demo_price"].where(tdf["demo_price"].notna(), None),
            )
        )

    return {
        "request_id": req_id,
        "items": [
            {
                "item_id": i,
                "source": s,
                "title": titles.get(i, ""),
                "category_name": cate_names.get(i, ""),
                "demo_price": prices.get(i),
            }
            for i, s in items
        ],
    }
