"""用户注册与登录路由。"""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import LoginIn, RegisterIn
from src.api.security import create_token, hash_password, verify_password
from src.db import repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/user", tags=["user"])

# 用户端清单的行为类型（与 FeedbackIn 枚举对齐）：
# 已买到的宝贝 / 购物车 / 收藏夹 三个淘宝式入口的数据源。
_LIST_BEHAVIOR_TYPES = ("ORD", "CART", "FLW")


@router.post("/register")
def register(body: RegisterIn) -> dict:
    exists = repo.query_df("SELECT user_id FROM user WHERE username=:u", {"u": body.username})
    if not exists.empty:
        raise HTTPException(status_code=409, detail="用户名已存在")
    repo.execute(
        "INSERT INTO user (username, password_hash, age_bucket, gender, city_level)"
        " VALUES (:u, :p, :a, :g, :c)",
        {"u": body.username, "p": hash_password(body.password),
         "a": body.age_bucket, "g": body.gender, "c": body.city_level},
    )
    row = repo.query_df("SELECT user_id FROM user WHERE username=:u", {"u": body.username})
    # 返回 username：前端注册后直接建立登录态，无需再查一次 info
    return {"user_id": int(row.iloc[0]["user_id"]), "username": body.username}


@router.post("/login")
def login(body: LoginIn) -> dict:
    row = repo.query_df(
        "SELECT user_id, password_hash FROM user WHERE username=:u", {"u": body.username}
    )
    if row.empty or not verify_password(body.password, row.iloc[0]["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    uid = int(row.iloc[0]["user_id"])
    return {"token": create_token(uid), "user_id": uid, "username": body.username}


@router.get("/info")
def user_info(user_id: int = Query(..., ge=1, description="用户 ID")) -> dict:
    """按 user_id 查用户名：顶栏展示与旧 session 补查（localStorage 无 username 时）。"""
    row = repo.query_df(
        "SELECT user_id, username FROM user WHERE user_id=:u", {"u": user_id}
    )
    if row.empty:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {
        "user_id": int(row.iloc[0]["user_id"]),
        "username": str(row.iloc[0]["username"]),
    }


@router.get("/behaviors")
def user_behaviors(
    user_id: int = Query(..., ge=1, description="用户 ID"),
    type: Literal["ORD", "CART", "FLW"] = Query(  # noqa: A002 - 查询参数名与前端约定一致
        ..., description="清单类型：ORD 已买到 / CART 购物车 / FLW 收藏夹"
    ),
    limit: int = Query(60, ge=1, le=200, description="去重后的商品条数上限"),
) -> dict:
    """用户某类行为的商品清单（已买到的宝贝 / 购物车 / 收藏夹）。

    behavior 是 append-only 流水，同一商品可能有多条记录（重复点击/
    重复加购），故按 item_id 去重并保留最近一次行为时间。去重发生在
    SQL 取回之后：按 limit*5 的窗口过量捞取，再在内存里去重截断，
    避免窗口函数在旧版 MySQL 上的兼容问题。
    """
    df = repo.query_df(
        "SELECT b.item_id, b.ts, i.title, i.category_id,"
        " c.name AS category_name, i.demo_price"
        " FROM behavior b"
        " JOIN item i ON i.item_id = b.item_id"
        " LEFT JOIN category c ON c.category_id = i.category_id"
        " WHERE b.user_id = :u AND b.behavior_type = :t"
        " ORDER BY b.ts DESC LIMIT :n",
        {"u": user_id, "t": type, "n": int(limit) * 5},
    )
    seen: set[int] = set()
    items: list[dict] = []
    for r in df.itertuples():
        iid = int(r.item_id)
        if iid in seen:
            continue
        seen.add(iid)
        ts = r.ts
        items.append(
            {
                "item_id": iid,
                "title": None if r.title is None else str(r.title),
                "category_id": None if r.category_id is None else int(r.category_id),
                "category_name": r.category_name,
                "demo_price": None if r.demo_price is None else float(r.demo_price),
                # schema 约定 ts 为 UTC 朴素 DATETIME：转 ISO 串交付，
                # 前端按本地时区渲染（演示库行为多为本会话产生，时差可忽略）
                "last_action": None if ts is None else ts.isoformat(sep=" "),
            }
        )
        if len(items) >= limit:
            break
    return {"items": items, "total": len(items)}
