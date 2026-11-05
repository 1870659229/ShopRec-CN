"""ItemCF 历史输入的回归测试：去重会静默改变召回。

背景（真实事故）
----------------
``src/models/ours_scorer.py`` 抽取公共打分模块时，评估脚本顺手把用户历史
经 ``dict`` 去重后一并传给了 ItemCF。而 ``ItemCF.recommend`` 是

    for it in history: acc[nb] += w

同一商品的多条行为行会被**重复累加**（等价于隐式频次加权）。去重后候选池
组成整体漂移：150 用户冒烟 pool 均值 2644.0→2643.7，全量 ours R@10
0.0162→0.0163，连与 ItemCF 无直接关系的 ``pop`` 行都跟着变（Div@10
0.3929→0.3930）——因为池子变了，所有方法都受影响。

结论：**ItemCF 侧必须吃未去重的原始历史**；只有 UserCF 侧才用去重后的
加权字典。本文件把这条纪律钉死。
"""

from __future__ import annotations

import numpy as np

from src.eval.light_item_cf import ItemCF
from src.eval.user_cf import UserCF


def _tiny_item_cf() -> ItemCF:
    """用 5 用户 × 6 商品的合成交互建一个真实 ItemCF（不走 _build 的桩）。

    交互设计使得 1 与 4 的相似度最高，但 3 同时是 1 和 2 的邻居——
    去重与否会改变"1 的贡献"权重，从而翻转 Top-1。
    """
    items = np.array([1, 2, 3, 1, 2, 4, 1, 3, 5, 2, 3, 6, 1, 4, 6], dtype=np.int64)
    users = np.array([10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14],
                     dtype=np.int64)
    ucf = UserCF(items, users, max_item_users=2000)
    return ItemCF(ucf, top_m=6, top_similar=3)


def _as_map(rec: tuple[np.ndarray, np.ndarray]) -> dict[int, float]:
    return {int(i): float(s) for i, s in zip(rec[0], rec[1])}


def test_duplicate_history_accumulates() -> None:
    """重复历史商品的邻居分应被累加（隐式频次加权），不是去重后取一次。"""
    icf = _tiny_item_cf()
    dedup = _as_map(icf.recommend([1], top_items=6))
    dup = _as_map(icf.recommend([1, 1], top_items=6))

    assert dedup, "单商品历史应召回到邻居"
    assert set(dup) == set(dedup), "重复不应改变召回集合，只改变分数"
    for item, score in dedup.items():
        assert abs(dup[item] - 2.0 * score) < 1e-12, (
            f"商品 {item} 重复两次的分数应为 2×：{dup[item]} vs {score}"
        )


def test_deduping_history_flips_top1() -> None:
    """去重会翻转 Top-1——这正是候选池漂移的可见证据。

    未去重 ``[1,1,2]``：1 的贡献翻倍 → 1 的最近邻 4 登顶。
    去重 ``[1,2]``：1 与 2 贡献均等 → 二者共同邻居 3 登顶。
    """
    icf = _tiny_item_cf()
    top_dup = int(icf.recommend([1, 1, 2], top_items=6)[0][0])
    top_dedup = int(icf.recommend([1, 2], top_items=6)[0][0])

    assert top_dup == 4, f"未去重历史 Top-1 应为 4，实际 {top_dup}"
    assert top_dedup == 3, f"去重历史 Top-1 应为 3，实际 {top_dedup}"
    assert top_dup != top_dedup, "若相等，说明 ItemCF 已改为去重语义，本用例需重写"


def test_history_item_itself_never_recalled() -> None:
    """历史商品自身不回灌：这是 hist_set 的作用，与去重无关。"""
    icf = _tiny_item_cf()
    for hist in ([1], [1, 1], [1, 1, 2], [1, 2]):
        recalled = set(icf.recommend(hist, top_items=6)[0].tolist())
        assert not (set(hist) & recalled), f"历史商品被回灌：hist={hist}, rec={recalled}"
