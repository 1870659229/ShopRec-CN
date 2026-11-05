"""Task 11：反馈服务与闭环更新服务测试（写 behavior、刷新画像、重训判阈）。"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.services.profile import should_retrain
from src.db import repo

client = TestClient(app)

# 测试数据段位：user 9200xx（6 位）、item 9200xxx（7 位），避免与其他测试冲突
_FB_USER_LO, _FB_USER_HI = 920000, 929999
_FB_ITEM_LO, _FB_ITEM_HI = 9200000, 9299999

RETRAIN_THRESHOLD = 500


def _clean_fb_seed() -> None:
    """清理反馈测试段位数据，保证测试可重复运行。

    model_version 无用户维度列；隔离测试库中 model_type='retrain' 的行
    仅由本文件的 /api/update/check 用例产生，故按类型整删安全。
    """
    repo.execute(
        "DELETE FROM model_version WHERE model_type='retrain' AND data_range LIKE 'local_%'"
    )
    repo.execute(
        "DELETE FROM user_profile WHERE user_id BETWEEN :lo AND :hi",
        {"lo": _FB_USER_LO, "hi": _FB_USER_HI},
    )
    repo.execute(
        "DELETE FROM behavior WHERE user_id BETWEEN :lo AND :hi",
        {"lo": _FB_USER_LO, "hi": _FB_USER_HI},
    )
    repo.execute(
        "DELETE FROM user WHERE user_id BETWEEN :lo AND :hi",
        {"lo": _FB_USER_LO, "hi": _FB_USER_HI},
    )
    repo.execute(
        "DELETE FROM item WHERE item_id BETWEEN :lo AND :hi",
        {"lo": _FB_ITEM_LO, "hi": _FB_ITEM_HI},
    )


@pytest.fixture(autouse=True)
def _fb_seed_guard() -> None:
    """每条测试前后各清理一次测试段位数据。"""
    _clean_fb_seed()
    yield
    _clean_fb_seed()


def _seed_item(item_id: int, cate_id: int) -> None:
    repo.execute(
        "INSERT INTO item (item_id, category_id, source, pop_score)"
        " VALUES (:i, :c, 'tianchi', 0.5)",
        {"i": item_id, "c": cate_id},
    )


def _seed_user(user_id: int) -> None:
    """反馈接口现校验 user 存在性，需先落一条测试用户。"""
    repo.execute(
        "INSERT INTO user (user_id, username, password_hash)"
        " VALUES (:u, :n, 'x')",
        {"u": user_id, "n": f"fb_{user_id}"},
    )


def _post_feedback(
    user_id: int,
    item_id: int,
    behavior_type: str = "ORD",
    rating: int | None = 5,
) -> dict:
    r = client.post(
        "/api/feedback",
        json={
            "user_id": user_id,
            "item_id": item_id,
            "behavior_type": behavior_type,
            "rating": rating,
        },
    )
    assert r.status_code == 200
    return r.json()


def _profile(user_id: int) -> tuple[list[int], list[int]]:
    df = repo.query_df(
        "SELECT recent_items, interest_tags FROM user_profile WHERE user_id=:u",
        {"u": user_id},
    )
    assert not df.empty, "user_profile 未被刷新"
    return (
        json.loads(df.iloc[0]["recent_items"]),
        json.loads(df.iloc[0]["interest_tags"]),
    )


# ---------- 纯函数：重训判阈（计划原有用例） ----------


def test_should_retrain_below_threshold() -> None:
    assert should_retrain(new_count=499, threshold=RETRAIN_THRESHOLD) is False


def test_should_retrain_at_threshold() -> None:
    assert should_retrain(new_count=500, threshold=RETRAIN_THRESHOLD) is True


# ---------- 路由层：POST /api/feedback ----------


def test_post_feedback_writes_behavior_and_refreshes_profile() -> None:
    _seed_user(920001)
    _seed_item(9200001, 8801)
    body = _post_feedback(920001, 9200001)
    assert body == {"status": "ok"}

    df = repo.query_df(
        "SELECT item_id, behavior_type, rating, ts, source FROM behavior WHERE user_id=:u",
        {"u": 920001},
    )
    assert len(df) == 1
    row = df.iloc[0]
    assert int(row["item_id"]) == 9200001
    assert row["behavior_type"] == "ORD"
    assert int(row["rating"]) == 5
    assert row["source"] == "local"
    # schema 约定 ts 为 UTC 朴素 DATETIME：若误写本地时间（UTC+8）会越过上界
    ts = row["ts"].to_pydatetime()
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    assert now_utc - timedelta(minutes=1) <= ts <= now_utc + timedelta(minutes=1)

    recent, tags = _profile(920001)
    assert recent == [9200001]
    assert tags == [8801]


def test_two_feedbacks_recent_items_reflect_latest_order() -> None:
    """连续两次反馈走 UPSERT 更新同一行画像，recent_items 最新在前。"""
    _seed_user(920002)
    _seed_item(9200002, 8801)
    _seed_item(9200003, 8802)
    _post_feedback(920002, 9200002)
    _post_feedback(920002, 9200003)
    recent, tags = _profile(920002)
    assert recent == [9200003, 9200002]
    assert tags == [8802, 8801]


def test_post_feedback_unknown_user_404() -> None:
    """user 不存在（段位外假 id）→ 404，且不落脏行。"""
    _seed_item(9200005, 8801)
    r = client.post(
        "/api/feedback",
        json={"user_id": 999999999, "item_id": 9200005, "behavior_type": "ORD", "rating": 5},
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "用户不存在"
    df = repo.query_df(
        "SELECT item_id FROM behavior WHERE user_id=:u", {"u": 999999999}
    )
    assert df.empty


def test_post_feedback_unknown_item_404() -> None:
    """item 不存在（段位外假 id）→ 404，且不落脏行。"""
    _seed_user(920004)
    r = client.post(
        "/api/feedback",
        json={"user_id": 920004, "item_id": 999999999, "behavior_type": "ORD", "rating": 5},
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "商品不存在"
    df = repo.query_df(
        "SELECT item_id FROM behavior WHERE user_id=:u", {"u": 920004}
    )
    assert df.empty


def test_post_feedback_invalid_behavior_type_422() -> None:
    """behavior_type 不在四类枚举内 → 422。"""
    _seed_user(920005)
    _seed_item(9200006, 8801)
    r = client.post(
        "/api/feedback",
        json={"user_id": 920005, "item_id": 9200006, "behavior_type": "xyz", "rating": 5},
    )
    assert r.status_code == 422


def test_post_feedback_rating_out_of_range_422() -> None:
    """rating 超出 5 分制范围 → 422。"""
    _seed_user(920006)
    _seed_item(9200007, 8801)
    r = client.post(
        "/api/feedback",
        json={"user_id": 920006, "item_id": 9200007, "behavior_type": "ORD", "rating": 200},
    )
    assert r.status_code == 422


# ---------- 路由层：POST /api/update/check ----------


def test_update_check_below_threshold_no_retrain() -> None:
    r = client.post("/api/update/check", params={"threshold": 10**9})
    assert r.status_code == 200
    body = r.json()
    assert body["retrain_triggered"] is False
    mv = repo.query_df("SELECT version_id FROM model_version WHERE model_type='retrain'")
    assert mv.empty


def test_update_check_at_threshold_triggers_retrain() -> None:
    """先探测当前 local 计数再定阈值，使判阈与库内存量数据无关、可重复运行。"""
    base = client.post("/api/update/check", params={"threshold": 10**9}).json()["count"]
    threshold = base + 1
    _seed_user(920003)
    _seed_item(9200004, 8803)
    _post_feedback(920003, 9200004)  # 新增 1 条 local 行：计数 = base + 1 = threshold

    r = client.post("/api/update/check", params={"threshold": threshold})
    assert r.status_code == 200
    body = r.json()
    assert body["retrain_triggered"] is True
    assert body["count"] == threshold

    mv = repo.query_df(
        "SELECT data_range, is_active FROM model_version"
        " WHERE model_type='retrain' AND data_range=:d",
        {"d": f"local_{threshold}"},
    )
    assert len(mv) == 1
    assert int(mv.iloc[0]["is_active"]) == 0
    assert str(mv.iloc[0]["data_range"]) == f"local_{threshold}"


def test_update_check_idempotent_no_duplicate_insert() -> None:
    """同一 data_range 重复 check 只登记一次：第二次仍 triggered=true 但行数不增。"""
    base = client.post("/api/update/check", params={"threshold": 10**9}).json()["count"]
    threshold = base + 1
    _seed_user(920007)
    _seed_item(9200008, 8804)
    _post_feedback(920007, 9200008)  # 计数 = base + 1 = threshold

    r1 = client.post("/api/update/check", params={"threshold": threshold})
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["retrain_triggered"] is True
    assert body1["already_registered"] is False

    r2 = client.post("/api/update/check", params={"threshold": threshold})
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["retrain_triggered"] is True
    assert body2["already_registered"] is True

    mv = repo.query_df(
        "SELECT version_id FROM model_version"
        " WHERE model_type='retrain' AND data_range=:d",
        {"d": f"local_{threshold}"},
    )
    assert len(mv) == 1
