"""商品检索接口：类目列表、关键词搜索、输入联想。

搜索范围：商品标题关键词 + 商品编号（前缀）+ 类目名（模糊包含）+ 类目 ID 精确筛选。
标题字段仅 demo 段位（9300 段）有值（演示标题，非 JDsearch 数据内容），
真实数据行为空串，不参与离线指标计算。
"""

from fastapi import APIRouter, Query

from src.db import repo

router = APIRouter(prefix="/api", tags=["item"])


@router.get("/category/list")
def category_list() -> dict:
    """全部演示类目及其商品数（供搜索条类目下拉与联想）。"""
    df = repo.query_df(
        "SELECT c.category_id, c.name AS category_name,"
        " COUNT(i.item_id) AS item_count"
        " FROM category c LEFT JOIN item i ON i.category_id = c.category_id"
        " GROUP BY c.category_id, c.name ORDER BY c.category_id"
    )
    return {"categories": df.to_dict(orient="records")}


@router.get("/item/search")
def search_items(
    q: str = Query("", description="商品标题关键词 / 编号前缀 / 类目名关键词"),
    category_id: int | None = Query(None, ge=0, description="类目精确筛选"),
    k: int = Query(40, ge=1, le=200),
) -> dict:
    """按标题关键词 / 编号前缀 / 类目名模糊 / 类目 ID 检索商品，按热度降序。

    q 与 category_id 同时为空时返回全部商品（即"按类目浏览"）。
    """
    clauses: list[str] = []
    params: dict = {}
    if q.strip():
        q = q.strip()
        clauses.append(
            "(i.title LIKE :qany OR CAST(i.item_id AS CHAR) LIKE :qpre"
            " OR c.name LIKE :qany)"
        )
        params["qpre"] = f"{q}%"
        params["qany"] = f"%{q}%"
    if category_id is not None:
        clauses.append("i.category_id = :cid")
        params["cid"] = category_id
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    df = repo.query_df(
        "SELECT i.item_id, i.title, i.category_id,"
        " c.name AS category_name, i.pop_score, i.demo_price"
        " FROM item i LEFT JOIN category c ON c.category_id = i.category_id"
        f" {where} ORDER BY i.pop_score DESC LIMIT {int(k)}",
        params,
    )
    return {"items": repo.jsonable_records(df), "total": int(len(df))}


@router.get("/item/suggest")
def suggest_items(q: str = Query(..., min_length=1, description="联想前缀/关键词")) -> dict:
    """输入联想：匹配的类目（含商品数）+ 商品（含类目名），淘宝式混排的原料。"""
    q = q.strip()
    cates = repo.query_df(
        "SELECT c.category_id, c.name AS category_name,"
        " COUNT(i.item_id) AS item_count"
        " FROM category c LEFT JOIN item i ON i.category_id = c.category_id"
        " WHERE c.name LIKE :any GROUP BY c.category_id, c.name ORDER BY c.category_id",
        {"any": f"%{q}%"},
    )
    items = repo.query_df(
        "SELECT i.item_id, i.title, i.category_id, c.name AS category_name"
        " FROM item i LEFT JOIN category c ON c.category_id = i.category_id"
        " WHERE (i.title LIKE :any OR CAST(i.item_id AS CHAR) LIKE :pre)"
        " ORDER BY i.pop_score DESC LIMIT 8",
        {"any": f"%{q}%", "pre": f"{q}%"},
    )
    return {
        "categories": repo.jsonable_records(cates),
        "items": repo.jsonable_records(items),
    }
