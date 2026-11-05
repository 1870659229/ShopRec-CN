"""UserCF 加权历史的回归测试。

覆盖两条不变量：
1. **锚点等价**：``list[int]`` 入参必须与全 1 的 ``dict`` 入参输出逐位一致
   （保证加权改造没有偷改默认行为，评估基线可比）。
2. **加权生效**：提升某个历史商品的意图权重，必须**改变邻居选择与最终排序**，
   而不只是把分数等比例放大——否则"多行为加权"在数学上成立但实际无效。

fixture 设计：user 0 只持有 2 个商品，user 1 持有 4 个商品。
调整余弦分母 ``sqrt(|items(u)|)`` 让少商品用户天然占优，
因此只有"高意图商品恰好只被多商品用户持有"时，权重才能翻转排序。
"""
from __future__ import annotations

import numpy as np

from src.eval.user_cf import UserCF


def _build() -> UserCF:
    """user 0 → {10, 11}；user 1 → {20, 21, 22, 23}；user 2 → {99}（查询者）。"""
    items = np.array([10, 11, 20, 21, 22, 23, 99], dtype=np.int64)
    users = np.array([0, 0, 1, 1, 1, 1, 2], dtype=np.int64)
    return UserCF(items, users, max_item_users=2000)


def test_list_history_equivalent_to_unit_weights() -> None:
    """锚点：list 入参 == 全 1 dict 入参（逐位一致）。"""
    ucf = _build()
    a_list, s_list = ucf.recommend(2, [10, 20], top_items=10)
    a_dict, s_dict = ucf.recommend(2, {10: 1.0, 20: 1.0}, top_items=10)
    assert a_list.tolist() == a_dict.tolist()
    assert np.allclose(s_list, s_dict)


def test_weighted_history_flips_ranking() -> None:
    """加权必须改变最终排序，而非仅缩放分数。

    全 1 时：sim[0] = 1/sqrt(2) ≈ 0.707 > sim[1] = 1/sqrt(4) = 0.5
      → user 0 的候选商品 11 排在最前。
    把商品 20 提权到 5（ORD 强意图）后：
      sim[1] = 5/sqrt(4) = 2.5 > sim[0] = 0.707
      → user 1 的候选商品 21/22/23 反超，11 掉到末尾。
    """
    ucf = _build()
    a_click, s_click = ucf.recommend(2, {10: 1.0, 20: 1.0}, top_items=10)
    a_ord, s_ord = ucf.recommend(2, {10: 1.0, 20: 5.0}, top_items=10)

    assert a_click.tolist() != a_ord.tolist(), "提权后排序必须翻转"
    assert a_click[0] == 11, "全 1 时应由少商品邻居（user 0）占优"
    assert a_ord[0] in (21, 22, 23), "提权 20 后应由其持有者（user 1）占优"
    assert s_ord[0] > s_click[0], "头部分数必须更高"


def test_zero_weight_history_is_ignored() -> None:
    """权重 0 的历史商品不得贡献相似度（等价于不在历史中）。"""
    ucf = _build()
    a_zero, s_zero = ucf.recommend(2, {10: 0.0, 20: 1.0}, top_items=10)
    a_only20, s_only20 = ucf.recommend(2, {20: 1.0}, top_items=10)
    assert a_zero.tolist() == a_only20.tolist()
    assert np.allclose(s_zero, s_only20)
