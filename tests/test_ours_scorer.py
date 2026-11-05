"""ours 公共打分模块的回归测试。

保护的是「离线评估与在线服务共用同一实现」这一契约：候选池组装顺序、
冷区注入、三条打分线。任何改动若破坏了这些行为，本文件的用例应先失败。
"""

import numpy as np

from src.models.ours_scorer import COLD_BOOST, OursScorer, user_categories


def _make_scorer(cold_cats: set[int] | None = None) -> OursScorer:
    """构造一个最小可测的打分器：两个类目，默认类目 20 为冷区。

    ``global_top`` 刻意不含冷区商品（4/5），否则全局热门会先把它们放进池，
    冷区注入将无从发生（离线真实数据里全局热门只有 300 条，不会覆盖，
    但小样本下必须隔离该路径才能测到注入本身）。
    """
    return OursScorer(
        pop={1: 100.0, 2: 90.0, 3: 80.0, 4: 70.0, 5: 60.0},
        cat_top={10: [1, 2, 3], 20: [4, 5]},
        cat_of={1: 10, 2: 10, 3: 10, 4: 20, 5: 20},
        cold_cats={20} if cold_cats is None else cold_cats,
        global_top=[1, 2, 3],
    )


def test_user_categories_returns_history_categories() -> None:
    assert user_categories([1, 2, 4], {1: 10, 2: 10, 4: 20}) == {10, 20}
    # 未知商品不产生类目，避免 cat_of.get 默认值 0 污染 ucats
    assert user_categories([999], {1: 10}) == set()


def test_build_pool_injects_cold_items() -> None:
    """用户历史全在热类目时，冷区类目商品仍应被注入候选池。"""
    scorer = _make_scorer()
    pool, n_injected = scorer.build_pool(
        hist_set={1}, ucats={10}, cf_items=[2], icf_items=[3]
    )
    assert n_injected == 2, f"冷区注入条数异常：{n_injected}"
    assert {4, 5} <= set(pool), f"冷区商品未注入，pool={pool}"


def test_build_pool_excludes_history_items() -> None:
    scorer = _make_scorer()
    hist_set = {1, 4}
    pool, _ = scorer.build_pool(
        hist_set=hist_set, ucats={10, 20}, cf_items=[1, 2], icf_items=[4, 5]
    )
    assert not hist_set & set(pool), f"历史商品泄漏进候选池：{pool}"


def test_score_cold_boost_is_exactly_cold_boost() -> None:
    """冷区加成只作用于冷区商品，且量级恰好为 COLD_BOOST。

    用「冷区集合为空」的同构打分器作对照，两次打分之差即为纯粹的冷区加成。
    不断言冷区商品一定能反超——COLD_BOOST 是加分项而非保序项，
    冷区商品通常热度排末位，0.20 的加分不足以翻转名次。
    """
    pool = [1, 4]  # 类目 10（非冷区）与类目 20（冷区）
    pc, lines = _make_scorer().score(pool, cf_map={}, icf_map={})
    _, lines_nocold = _make_scorer(cold_cats=set()).score(pool, cf_map={}, icf_map={})

    assert pc == [10, 20]
    delta = lines["ours"] - lines_nocold["ours"]
    assert np.allclose(delta, [0.0, COLD_BOOST]), f"冷区加成量级异常：{delta}"
