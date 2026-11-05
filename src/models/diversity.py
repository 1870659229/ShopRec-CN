"""可控品类多样性重排（创新点①）。

通过 alpha 在「纯分数排序」与「最大品类多样性」之间连续调节重排结果。

实现与提速
----------
贪心语义：每轮从剩余候选中挑「分数 − alpha·(类目已出现)」最大者。
朴素实现是 O(k·|pool|) 纯 Python 循环（每步扫全池），实测单次 1.76 ms
（pool=2711、k=10），全量 20000 用户 × 15 次 ≈ **8.8 分钟，占端到端 53%**。

本实现利用「分数已降序」这一前提做等价化简：

* 未惩罚组的最大者 = 降序序列中第一个「类目未出现」的未选元素；
* 已惩罚组的最大者 = 降序序列中第一个「类目已出现」的未选元素（值再减 alpha）。

全局最大必在这两者之中，因此每步只需对全池做常数次向量化运算，
复杂度 O(|pool|)。配合 ``prepare`` 把**排序与类目编码在多个 alpha 间复用**，
同一方法的 5 个 α 只排一次序。

等价性由 ``tests/test_diversity.py`` 的随机对照（与朴素实现逐位比对，
含并列时的 tie-breaking）钉死——任何改动若破坏等价，该测试必失败。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

# 与其他模块保持一致：定义模块级 logger；当前无日志输出，保留备用。
logger = logging.getLogger(__name__)


class Prepared(NamedTuple):
    """按分数降序排好的候选，供多个 alpha 复用同一次排序。"""

    items: np.ndarray
    codes: np.ndarray      # 类目编码（同类目同码，仅池内有效）
    values: np.ndarray     # 分数，降序
    n_cats: int


def encode_categories(cates: Sequence[int]) -> tuple[np.ndarray, int]:
    """把类目 id 编码为池内连续下标，返回 ``(codes, n_cats)``。

    只依赖候选池的类目构成，与打分无关，因此**同一用户对三个方法只编码一次**。
    """
    arr = np.asarray(cates, dtype=np.int64)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64), 0
    codes = np.unique(arr, return_inverse=True)[1]
    return codes.astype(np.int64, copy=False), int(codes.max()) + 1


def prepare(
    items: Sequence[int],
    codes: np.ndarray,
    values: Sequence[float],
    n_cats: int,
) -> Prepared:
    """按分数降序排序（稳定），供 ``rerank_prepared`` 多次复用。"""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return Prepared(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), 0)
    order = np.argsort(-vals, kind="stable")
    return Prepared(
        np.asarray(items, dtype=np.int64)[order],
        np.asarray(codes, dtype=np.int64)[order],
        vals[order],
        n_cats,
    )


def rerank_prepared(prep: Prepared, alpha: float, k: int) -> list[int]:
    """在 ``prepare`` 的结果上按 alpha 贪心选出 k 个。"""
    n = prep.values.size
    if n == 0 or k <= 0:
        return []
    alpha = max(0.0, min(1.0, alpha))
    items, codes, values = prep.items, prep.codes, prep.values

    # alpha=0 时惩罚恒为 0，贪心退化为取分数最高的前 k 个（序列已降序）
    if alpha == 0.0:
        return [int(i) for i in items[: min(k, n)]]

    seen = np.zeros(prep.n_cats, dtype=bool)
    used = np.zeros(n, dtype=bool)
    val = np.empty(n, dtype=np.float64)
    out: list[int] = []
    for _ in range(min(k, n)):
        np.subtract(values, alpha * seen[codes], out=val)
        val[used] = -np.inf
        j = int(np.argmax(val))
        used[j] = True
        seen[codes[j]] = True
        out.append(int(items[j]))
    return out


def rerank(scores: list[tuple[int, int, float]], alpha: float, k: int) -> list[int]:
    """按 alpha 控制品类多样性：alpha=0 纯分数排序，alpha=1 最大多样性。

    scores: [(item_id, category_id, score)]
    返回按重排顺序取前 k 个 item_id。

    贪心策略：每轮从剩余候选中挑「分数 − alpha·(类目已出现)」最大者，
    已出现类目被扣 alpha 分，从而在 alpha=1 时优先探索新类目。

    单次调用入口（在线服务用），语义与朴素实现逐位一致。
    批量评估请用 ``encode_categories`` + ``prepare`` + ``rerank_prepared``，
    以复用排序与类目编码。
    """
    if not scores:
        return []
    codes, n_cats = encode_categories([t[1] for t in scores])
    return rerank_prepared(
        prepare([t[0] for t in scores], codes, [t[2] for t in scores], n_cats), alpha, k
    )
