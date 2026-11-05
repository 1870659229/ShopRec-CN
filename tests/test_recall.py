"""Task 10：召回与排序（含 transfer.fuse 接线）测试；Task 11 复审补 /api/recommend 路由覆盖。"""

import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.services.rank import rank
from src.api.services.recall import category_internal, cross_category, negative_items
from src.db import repo


# ---------- 召回层（计划原有用例，假数据源） ----------


def test_category_internal_returns_at_most_k(monkeypatch):
    fake = [(1, 100, 0.9), (2, 100, 0.8), (3, 100, 0.7)]
    monkeypatch.setattr(
        "src.api.services.recall._hot_in_category", lambda c, n, exclude=None: fake
    )
    out = category_internal(100, k=2)
    assert len(out) == 2


def test_category_internal_empty_when_no_data(monkeypatch):
    monkeypatch.setattr(
        "src.api.services.recall._hot_in_category", lambda c, n, exclude=None: []
    )
    assert category_internal(999, k=5) == []


# ---------- 召回层：cross_category 辅路语义（品类之外补充） ----------


def test_cross_category_excludes_recent_categories(monkeypatch):
    """辅路只从 recent_cates 之外的类目召回，不与目标域（品类内）重叠。"""
    heat_df = pd.DataFrame({"category_id": [300, 200], "cate_heat": [0.9, 0.8]})
    monkeypatch.setattr("src.db.repo.query_df", lambda sql, params=None: heat_df)
    called: list[int] = []

    def fake_hot(
        cate_id: int, n: int, exclude: list[int] | None = None
    ) -> list[tuple[int, int, float]]:
        called.append(cate_id)
        return [(9000 + cate_id, cate_id, 0.5)]

    monkeypatch.setattr("src.api.services.recall._hot_in_category", fake_hot)
    out = cross_category([100], k=3)
    assert called == [300, 200]  # 类目热度查询结果按序各取热门
    assert all(cate != 100 for _, cate, _ in out)


def test_cross_category_empty_cates_falls_back_global(monkeypatch):
    """recent_cates 为空时保持全局热门兜底。"""
    sentinel = [(1, 10, 0.9)]
    monkeypatch.setattr(
        "src.api.services.recall._hot_global", lambda n, exclude=None: sentinel
    )
    assert cross_category([], k=3) == sentinel


# ---------- 召回层：负反馈过滤（已购 / 已加购不再推荐） ----------


def test_negative_items_returns_purchased_and_carted():
    """负反馈集合 = 已购买(ORD) + 已加购(CART)；CLICK / EXPOSE 不算负反馈。"""
    _seed_rec_items()
    uid = 910010
    for item_id, btype in (
        (9100001, "ORD"),
        (9100002, "CART"),
        (9100003, "CLICK"),
    ):
        repo.execute(
            "INSERT INTO behavior (user_id, item_id, behavior_type, ts, source)"
            " VALUES (:u, :i, :b, NOW(), 'test')",
            {"u": uid, "i": item_id, "b": btype},
        )
    got = set(negative_items(uid))
    assert got == {9100001, 9100002}
    assert 9100003 not in got  # 仅点击不算负反馈


def test_negative_items_empty_for_user_without_behavior():
    """无行为用户返回空集合（空集合 = 不排除，与改造前完全一致）。"""
    assert negative_items(910011) == []
    assert negative_items(None) == []


def test_category_internal_excludes_negative_and_backfills():
    """排除发生在 LIMIT 之前：被排除的名额由同品类后续商品递补，候选池规模不变。"""
    _seed_rec_items()  # 类目 100：9100001(0.9) / 9100002(0.8)
    # 再补一件同类目商品，使「排除 1 件后仍可凑满 k=2」的递补语义可测
    repo.execute(
        "INSERT INTO item (item_id, category_id, source, pop_score)"
        " VALUES (:i, :c, 'tianchi', :p)",
        {"i": 9100004, "c": 100, "p": 0.7},
    )
    out = category_internal(100, k=2, exclude=[9100001])
    ids = [i for i, _, _ in out]
    assert 9100001 not in ids, "已购商品不应再被召回"
    assert len(ids) == 2, "排除后名额应由同品类后续商品递补，而非减少候选"
    assert ids == [9100002, 9100004], "递补应按热度序取后续商品"


def test_recommend_excludes_purchased_items():
    """端到端：已购买的商品不再出现在 /api/recommend 返回结果中。"""
    _seed_rec_items()
    uid = 910012
    repo.execute(
        "INSERT INTO user_profile (user_id, interest_tags) VALUES (:u, :t)",
        {"u": uid, "t": "[100]"},
    )
    repo.execute(
        "INSERT INTO behavior (user_id, item_id, behavior_type, ts, source)"
        " VALUES (:u, :i, 'ORD', NOW(), 'test')",
        {"u": uid, "i": 9100001},
    )
    r = client.get("/api/recommend", params={"user_id": uid, "k": 3})
    assert r.status_code == 200
    ids = [it["item_id"] for it in r.json()["items"]]
    assert 9100001 not in ids, "已购买商品不应再被推荐"
    assert ids, "排除一件后仍应有其他商品被推荐"


def test_cross_category_db_excludes_tgt_category():
    """DB 实测：recent=[100] 且类目 100/200 都有热门商品时，辅路不含类目 100。"""
    _seed_rec_items()
    out = cross_category([100], k=5)
    assert out, "类目 200 存在热门商品，辅路不应为空"
    assert all(cate != 100 for _, cate, _ in out)
    assert any(cate == 200 for _, cate, _ in out)


# ---------- 排序层：融合（transfer.fuse 接线） ----------


def test_rank_prefers_tgt_when_confidence_high():
    """conf=1 时重叠候选纯取 tgt 分：顺序跟随 tgt，而非 aux。"""
    tgt = [(1, 10, 0.6), (2, 10, 0.8)]
    aux = [(1, 10, 0.8), (2, 10, 0.6)]
    assert rank(tgt, aux, k=2, alpha=0.0, conf=1.0) == [2, 1]


def test_rank_prefers_aux_when_confidence_low():
    """conf=0 时重叠候选纯取 aux 分：顺序跟随 aux，而非 tgt。"""
    tgt = [(1, 10, 0.6), (2, 10, 0.8)]
    aux = [(1, 10, 0.8), (2, 10, 0.6)]
    assert rank(tgt, aux, k=2, alpha=0.0, conf=0.0) == [1, 2]


def test_rank_keeps_aux_only_candidates():
    """只在辅助路出现的候选不被丢弃，按该路分数补入排序。"""
    tgt = [(1, 10, 0.9)]
    aux = [(2, 20, 0.5)]
    assert rank(tgt, aux, k=5, alpha=0.0) == [1, 2]


def test_rank_empty_candidates_returns_empty():
    assert rank([], [], k=5) == []


# ---------- 排序层：α 多样性重排（diversity.rerank 接线） ----------


def test_rank_alpha_zero_is_pure_score_order():
    """alpha=0 不被多样性惩罚干扰：单类目下品类内名次与热度同序，融合后序不变。"""
    cands = [(1, 10, 0.9), (2, 10, 0.8), (3, 10, 0.7)]
    assert rank(cands, [], k=3, alpha=0.0) == [1, 2, 3]


def test_rank_fusion_blends_aux_and_tgt_signals():
    """融合真实生效的回归证据（P0-1 修复）：目标域候选的最终分是「全局热度（辅助域）」
    与「品类内名次（目标域）」两个不同信号的加权，而非等同于任一路原始分。

    构造：候选 3 热度最高但与 2 同类目且居次位（品类内名次 1）；候选 1 热度最低
    但居其类目首位（品类内名次 0）。纯热度序为 3>2>1，融合后（conf 偏目标域）
    品类内首位被提权，序变为 3>1>2 —— 若 fuse 未真实执行则不会出现此变化。
    """
    tgt = [(1, 10, 0.5), (2, 20, 0.9), (3, 20, 0.95)]
    assert rank(tgt, [], k=3, alpha=0.0, conf=0.7) == [3, 1, 2]


def test_rank_fusion_conf_zero_keeps_raw_popularity_order():
    """conf=0 时目标域候选纯取辅助域分（热度），序退回热度序 3>2>1。"""
    tgt = [(1, 10, 0.5), (2, 20, 0.9), (3, 20, 0.95)]
    assert rank(tgt, [], k=3, alpha=0.0, conf=0.0) == [3, 2, 1]


def test_rank_alpha_one_prefers_new_category():
    """alpha=1 时同品类候选被扣 1 分，新品类候选 0.7 反超 0.8。"""
    cands = [(1, 10, 0.9), (2, 10, 0.8), (3, 20, 0.7)]
    assert rank(cands, [], k=3, alpha=1.0) == [1, 3, 2]


# ---------- 路由层：/api/recommend（TestClient + shoprec_ci_test 隔离） ----------

# 测试数据段位：user 9100xx（6 位）、item 9100xxx（7 位），避免与其他测试冲突
_REC_USER_LO, _REC_USER_HI = 910000, 919999
_REC_ITEM_LO, _REC_ITEM_HI = 9100000, 9199999
client = TestClient(app)


def _clean_rec_seed() -> None:
    """清理推荐测试段位的全部数据，保证测试可重复运行。"""
    repo.execute(
        "DELETE FROM behavior WHERE user_id BETWEEN :lo AND :hi",
        {"lo": _REC_USER_LO, "hi": _REC_USER_HI},
    )
    repo.execute(
        "DELETE FROM recommend_log WHERE user_id BETWEEN :lo AND :hi",
        {"lo": _REC_USER_LO, "hi": _REC_USER_HI},
    )
    repo.execute(
        "DELETE FROM user_profile WHERE user_id BETWEEN :lo AND :hi",
        {"lo": _REC_USER_LO, "hi": _REC_USER_HI},
    )
    repo.execute(
        "DELETE FROM item WHERE item_id BETWEEN :lo AND :hi",
        {"lo": _REC_ITEM_LO, "hi": _REC_ITEM_HI},
    )


@pytest.fixture(autouse=True)
def _rec_seed_guard() -> None:
    """每条测试前后各清理一次测试段位数据。"""
    _clean_rec_seed()
    yield
    _clean_rec_seed()


def _seed_rec_items() -> None:
    """类目 100 两件热门 + 类目 200 一件更高热度（供辅路/兜底断言）。"""
    for item_id, cate, pop in ((9100001, 100, 0.9), (9100002, 100, 0.8), (9100003, 200, 9.9)):
        repo.execute(
            "INSERT INTO item (item_id, category_id, source, pop_score)"
            " VALUES (:i, :c, 'tianchi', :p)",
            {"i": item_id, "c": cate, "p": pop},
        )


def test_recommend_with_profile_returns_ranked_items():
    """有画像用户：200 + 合法 uuid + items ≤ k + recommend_log 落行。"""
    _seed_rec_items()
    repo.execute(
        "INSERT INTO user_profile (user_id, interest_tags) VALUES (:u, :t)",
        {"u": 910001, "t": "[100]"},
    )
    r = client.get("/api/recommend", params={"user_id": 910001, "k": 2})
    assert r.status_code == 200
    body = r.json()
    uuid.UUID(body["request_id"])  # 非法 uuid 会抛 ValueError 使测试失败
    assert len(body["items"]) <= 2
    log_df = repo.query_df(
        "SELECT item_id, recall_source, position FROM recommend_log WHERE user_id = :u",
        {"u": 910001},
    )
    assert len(log_df) == len(body["items"])
    # 有画像（interest_tags=[100]）时：类目内（tgt）+ 跨品类迁移（aux），
    # 契约变化后 recall_source 记录真实召回路而非写死 "hybrid"。
    assert set(log_df["recall_source"]) <= {"fused", "tgt", "aux", "hot"}
    assert "hybrid" not in set(log_df["recall_source"])
    assert sorted(log_df["item_id"].tolist()) == sorted(
        it["item_id"] for it in body["items"]
    )


def test_recommend_without_profile_falls_back_global():
    """无画像用户：200 且走全局热门兜底（item 已 seed，items 非空）。"""
    _seed_rec_items()
    r = client.get("/api/recommend", params={"user_id": 910002, "k": 3})
    assert r.status_code == 200
    items = r.json()["items"]
    assert 0 < len(items) <= 3
    # 契约变化后 items 为 [{item_id, source, title, category_name}]；无画像 → 来源应全部是 "hot"
    assert all(it["item_id"] in (9100001, 9100002, 9100003) for it in items)
    assert all(it["source"] == "hot" for it in items)


def test_recommend_k_out_of_range_returns_422():
    r = client.get("/api/recommend", params={"user_id": 910001, "k": 1000})
    assert r.status_code == 422


def test_recommend_dirty_json_degrades_to_fallback(monkeypatch):
    """interest_tags 为非法 JSON：降级走兜底而非 500（验证问题 1 修复）。"""
    _seed_rec_items()
    real_query_df = repo.query_df

    def fake_query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
        if "interest_tags" in sql:
            return pd.DataFrame({"interest_tags": ["not-json{{"]})
        return real_query_df(sql, params)

    monkeypatch.setattr("src.db.repo.query_df", fake_query_df)
    r = client.get("/api/recommend", params={"user_id": 910004, "k": 3})
    assert r.status_code == 200
    assert len(r.json()["items"]) > 0


def test_recommend_dirty_tag_elements_degrade():
    """JSON 列只能存合法 JSON，故用 JSON 字符串/数字制造元素级脏数据：
    '"abc"' 逐元素 int() 抛 ValueError；'100' 顶层迭代抛 TypeError——均应降级为 200。"""
    _seed_rec_items()
    for user_id, tags in ((910003, '"abc"'), (910004, "100")):
        repo.execute(
            "INSERT INTO user_profile (user_id, interest_tags) VALUES (:u, :t)",
            {"u": user_id, "t": tags},
        )
        r = client.get("/api/recommend", params={"user_id": user_id, "k": 3})
        assert r.status_code == 200, (user_id, tags)
        assert len(r.json()["items"]) > 0
