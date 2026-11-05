"""用户清单接口测试：/api/user/info 与 /api/user/behaviors（已买到/购物车/收藏夹）。"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.db import repo

client = TestClient(app)

# 测试数据段位：user 9300xx、item 9300xxx，避免与其他测试文件冲突
_U_USER_LO, _U_USER_HI = 930000, 939999
_U_ITEM_LO, _U_ITEM_HI = 9300000, 9399999


def _clean_user_seed() -> None:
    """清理用户清单测试段位数据，保证测试可重复运行。"""
    repo.execute(
        "DELETE FROM user_profile WHERE user_id BETWEEN :lo AND :hi",
        {"lo": _U_USER_LO, "hi": _U_USER_HI},
    )
    repo.execute(
        "DELETE FROM behavior WHERE user_id BETWEEN :lo AND :hi",
        {"lo": _U_USER_LO, "hi": _U_USER_HI},
    )
    repo.execute(
        "DELETE FROM user WHERE user_id BETWEEN :lo AND :hi",
        {"lo": _U_USER_LO, "hi": _U_USER_HI},
    )
    repo.execute(
        "DELETE FROM item WHERE item_id BETWEEN :lo AND :hi",
        {"lo": _U_ITEM_LO, "hi": _U_ITEM_HI},
    )


@pytest.fixture(autouse=True)
def _user_seed_guard() -> None:
    _clean_user_seed()
    yield
    _clean_user_seed()


def _seed_item(item_id: int, cate_id: int, title: str = None, price: float = None) -> None:
    repo.execute(
        "INSERT INTO item (item_id, category_id, source, pop_score, title, demo_price)"
        " VALUES (:i, :c, 'tianchi', 0.5, :t, :p)",
        {"i": item_id, "c": cate_id, "t": title, "p": price},
    )


def _seed_user(user_id: int, username: str) -> None:
    repo.execute(
        "INSERT INTO user (user_id, username, password_hash) VALUES (:u, :n, 'x')",
        {"u": user_id, "n": username},
    )


def _seed_behavior(user_id: int, item_id: int, behavior_type: str) -> None:
    """直接写库（绕开 /api/feedback 的画像刷新），只测清单查询语义。"""
    repo.execute(
        "INSERT INTO behavior (user_id, item_id, behavior_type, ts, source)"
        " VALUES (:u, :i, :b, NOW(6), 'local')",
        {"u": user_id, "i": item_id, "b": behavior_type},
    )


# ---------- GET /api/user/info ----------


def test_user_info_returns_username() -> None:
    _seed_user(930001, "info_user")
    r = client.get("/api/user/info", params={"user_id": 930001})
    assert r.status_code == 200
    assert r.json() == {"user_id": 930001, "username": "info_user"}


def test_user_info_unknown_user_404() -> None:
    r = client.get("/api/user/info", params={"user_id": 999999999})
    assert r.status_code == 404
    assert r.json()["detail"] == "用户不存在"


# ---------- GET /api/user/behaviors ----------


def test_behaviors_dedupes_by_item_keeps_latest() -> None:
    """同一商品多条同类行为 → 列表只出现一次（去重按 item_id）。"""
    _seed_user(930002, "dup_user")
    _seed_item(9300001, 8801, "机械表", 1399.0)
    _seed_item(9300002, 8801, "电子表", 899.0)
    # 9300001 加购两次 + 收藏一次：CART 清单里只能出现一次
    _seed_behavior(930002, 9300001, "CART")
    _seed_behavior(930002, 9300001, "CART")
    _seed_behavior(930002, 9300001, "FLW")
    _seed_behavior(930002, 9300002, "CART")

    r = client.get("/api/user/behaviors", params={"user_id": 930002, "type": "CART"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    ids = [i["item_id"] for i in body["items"]]
    assert ids == [9300002, 9300001]  # ts DESC：后插入的 9300002 在前


def test_behaviors_filters_by_type() -> None:
    """type 参数严格过滤：CART 清单不含 FLW/ORD 商品。"""
    _seed_user(930003, "type_user")
    _seed_item(9300003, 8802, "牛仔裤", 59.0)
    _seed_item(9300004, 8802, "衬衫", 99.0)
    _seed_behavior(930003, 9300003, "CART")
    _seed_behavior(930003, 9300004, "FLW")

    r = client.get("/api/user/behaviors", params={"user_id": 930003, "type": "FLW"})
    body = r.json()
    assert [i["item_id"] for i in body["items"]] == [9300004]
    assert body["items"][0]["title"] == "衬衫"


def test_behaviors_joins_item_fields() -> None:
    """清单携带商品展示字段：标题 / 类目名 / 演示价格 / 最近行为时间。"""
    _seed_user(930004, "join_user")
    _seed_item(9300005, 8803, "墨镜", 299.0)
    _seed_behavior(930004, 9300005, "ORD")

    r = client.get("/api/user/behaviors", params={"user_id": 930004, "type": "ORD"})
    it = r.json()["items"][0]
    assert it["title"] == "墨镜"
    assert it["demo_price"] == 299.0
    assert it["last_action"] is not None
    assert "category_name" in it  # 类目名允许为 None（LEFT JOIN），键必须存在


def test_behaviors_empty_for_new_user() -> None:
    """无行为用户 → 空清单（前端空态的数据基础）。"""
    _seed_user(930005, "empty_user")
    r = client.get("/api/user/behaviors", params={"user_id": 930005, "type": "ORD"})
    assert r.status_code == 200
    assert r.json() == {"items": [], "total": 0}


def test_behaviors_invalid_type_422() -> None:
    """type 不在 ORD/CART/FLW 枚举内 → 422。"""
    _seed_user(930006, "bad_type_user")
    r = client.get(
        "/api/user/behaviors", params={"user_id": 930006, "type": "CLICK"}
    )
    assert r.status_code == 422


def test_behaviors_respects_limit_after_dedupe() -> None:
    """limit 作用于去重之后：3 个商品各 2 条行为，limit=2 → 返回 2 个。"""
    _seed_user(930007, "limit_user")
    for n in range(3):
        _seed_item(9300010 + n, 8804, f"商品{n}", float(n))
        _seed_behavior(930007, 9300010 + n, "CART")
        _seed_behavior(930007, 9300010 + n, "CART")

    r = client.get(
        "/api/user/behaviors", params={"user_id": 930007, "type": "CART", "limit": 2}
    )
    body = r.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
