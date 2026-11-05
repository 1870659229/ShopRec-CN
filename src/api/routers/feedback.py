"""反馈与闭环更新接口：写 behavior 并刷新实时画像；累计达标则登记重训。"""

import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.services.profile import refresh_profile, should_retrain
from src.db import repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["feedback"])

# 默认重训阈值：本地累计新行为达到该量级才登记一次重训。
# /api/update/check 支持按请求注入 threshold（运维调参 / 测试用小阈值），默认保持 500。
RETRAIN_THRESHOLD = 500


class FeedbackIn(BaseModel):
    user_id: int
    item_id: int
    # 行为枚举与 JDsearch 对齐 + EXPOSE 曝光（6 字符 < VARCHAR(8)）：
    # 曝光只作行为量统计与漏斗分析，不参与画像折叠（弱信号不稀释兴趣）。
    behavior_type: Literal["CLICK", "CART", "ORD", "FLW", "EXPOSE"]
    # 5 分制评分，None 表示未评分
    rating: int | None = Field(None, ge=1, le=5)


def _row_exists(sql: str, params: dict) -> bool:
    """轻量存在性探测：查询结果非空即视为存在。"""
    return not repo.query_df(sql, params).empty


@router.post("/feedback")
def feedback(body: FeedbackIn) -> dict:
    """写入一条 behavior 并刷新实时画像。

    schema 无外键，故先查后插：user_id / item_id 任一不存在返回 404，
    避免脏行静默落库；枚举与评分范围非法由 pydantic 返回 422。
    """
    if not _row_exists(
        "SELECT 1 AS ok FROM user WHERE user_id=:u LIMIT 1", {"u": body.user_id}
    ):
        raise HTTPException(status_code=404, detail="用户不存在")
    if not _row_exists(
        "SELECT 1 AS ok FROM item WHERE item_id=:i LIMIT 1", {"i": body.item_id}
    ):
        raise HTTPException(status_code=404, detail="商品不存在")
    # schema 约定 ts 为 UTC 朴素 DATETIME：取 UTC 当前时刻并剥掉 tzinfo，
    # 避免依赖驱动对 aware datetime 的静默格式化行为。
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    repo.execute(
        "INSERT INTO behavior (user_id, item_id, behavior_type, rating, ts, source)"
        " VALUES (:u, :i, :b, :r, :t, 'local')",
        {
            "u": body.user_id,
            "i": body.item_id,
            "b": body.behavior_type,
            "r": body.rating,
            "t": ts,
        },
    )
    # EXPOSE 是弱信号（看到≠感兴趣）：落库统计行为量，但不折叠进画像，
    # 避免"刷一遍推荐流"就把兴趣品类刷成推荐结果自身（自证循环）。
    if body.behavior_type != "EXPOSE":
        refresh_profile(body.user_id)
    return {"status": "ok"}


@router.post("/update/check")
def update_check(threshold: int = Query(RETRAIN_THRESHOLD, ge=1)) -> dict:
    """统计 source='local' 的新行为数，达标则写一条未激活的 model_version。

    幂等：同一 data_range（local_{count}）只登记一次，重复调用不再 INSERT，
    此时仍返回 retrain_triggered=true 且 already_registered=true。
    """
    df = repo.query_df("SELECT COUNT(*) AS c FROM behavior WHERE source='local'")
    new_count = int(df.iloc[0]["c"])
    if should_retrain(new_count, threshold):
        data_range = f"local_{new_count}"
        already_registered = _row_exists(
            "SELECT 1 AS ok FROM model_version"
            " WHERE model_type='retrain' AND data_range=:d LIMIT 1",
            {"d": data_range},
        )
        if not already_registered:
            repo.execute(
                "INSERT INTO model_version (model_type, data_range, metrics, path, is_active)"
                " VALUES ('retrain', :d, JSON_OBJECT('trigger','threshold'), '', 0)",
                {"d": data_range},
            )
        return {
            "retrain_triggered": True,
            "count": new_count,
            "already_registered": already_registered,
        }
    return {"retrain_triggered": False, "count": new_count}
