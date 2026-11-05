"""用户协同过滤（user-based CF）——为离线评估补充"协同"召回路。

为什么必须有 CF：JDsearch 极稀疏（商品交互数中位数 = 1），
纯热度类召回（全局热门 / 品类热门）的候选池 ceiling 只有 1.9%，
**捞不到"全球稀有但与该用户相关"的商品**。CF 正是补这块的。

实现：倒排索引（item → users）+ 调整余弦相似度，取 top-M 邻居后聚合邻居的历史商品打分。

**内存策略**：2000 万+ 交互若用 Python dict/list 存 user→items 会吃掉数 GB，
故用 numpy 排序 + offsets 的**双索引**（按 item 找邻居、按 user 取其历史）紧凑存储；
索引内部一律用**压缩后的整数下标**（不是原始 id），避免查表开销与维度错配。
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# 热门物品在找邻居时贡献噪声大且成本高，按被多少用户交互过设上限剔除
DEFAULT_MAX_ITEM_USERS = 2000
DEFAULT_TOP_NEIGHBORS = 100
DEFAULT_TOP_ITEMS = 100


class UserCF:
    """基于共现的用户协同过滤（numpy 双索引，支持千万级交互）。"""

    def __init__(
        self,
        item_ids: np.ndarray,
        user_ids: np.ndarray,
        max_item_users: int = DEFAULT_MAX_ITEM_USERS,
    ) -> None:
        item_ids = np.asarray(item_ids, dtype=np.int64)
        user_ids = np.asarray(user_ids, dtype=np.int64)
        self._n_pairs = len(item_ids)
        self._max_item_users = max_item_users

        # 把 user 压成连续下标 uinv（后续索引内部一律用下标）
        uniq_users, uinv = np.unique(user_ids, return_inverse=True)
        self._usr_ids = uniq_users
        self._uidx = {int(u): i for i, u in enumerate(uniq_users)}
        self._n_users = len(uniq_users)
        counts = np.bincount(uinv, minlength=self._n_users)
        self._ulen = counts.astype(np.float64)
        self._usr_end = np.cumsum(counts)
        self._usr_start = np.concatenate(([0], self._usr_end[:-1]))

        # 按 user 排序 → 正排索引（user 下标 → 其 item 列表）
        ou = np.argsort(uinv, kind="stable")
        self._usr_items = item_ids[ou]

        # 按 item 排序 → 倒排索引（item → user 下标）
        oi = np.argsort(item_ids, kind="stable")
        sorted_items = item_ids[oi]
        uniq_items, start = np.unique(sorted_items, return_index=True)
        end = np.append(start[1:], len(sorted_items))
        self._inv_items = uniq_items
        self._inv_start = start
        self._inv_end = end
        self._inv_users = uinv[oi]          # 存的是 user 下标
        self._usable = (end - start) <= max_item_users

        logger.info(
            "UserCF: 交互 %d / 用户 %d / 物品 %d（可用物品 %d，剔除过热 > %d 用户）",
            self._n_pairs, self._n_users, len(uniq_items),
            int(self._usable.sum()), max_item_users,
        )

    @classmethod
    def from_user_items(
        cls, user_items: dict[int, list[int]], max_item_users: int = DEFAULT_MAX_ITEM_USERS
    ) -> "UserCF":
        """只索引传入的用户（transductive 调试用）。"""
        its, us = [], []
        for u, items in user_items.items():
            for i in set(items):
                its.append(int(i)); us.append(int(u))
        return cls(np.array(its, dtype=np.int64), np.array(us, dtype=np.int64), max_item_users)

    def _users_of_item(self, item: int) -> np.ndarray | None:
        """返回拥有该 item 的**用户下标**数组（过热 item 返回 None）。"""
        j = np.searchsorted(self._inv_items, item)
        if j >= len(self._inv_items) or self._inv_items[j] != item or not self._usable[j]:
            return None
        return self._inv_users[self._inv_start[j] : self._inv_end[j]]

    def _items_of_user_idx(self, v: int) -> np.ndarray:
        """按用户下标取其历史 item。"""
        return self._usr_items[self._usr_start[v] : self._usr_end[v]]

    def recommend(
        self,
        user: int,
        history: "list[int] | dict[int, float]",
        top_neighbors: int = DEFAULT_TOP_NEIGHBORS,
        top_items: int = DEFAULT_TOP_ITEMS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(item_ids, scores)``：对该用户的推荐商品及其分数（降序）。

        ``history`` 为 ``list[int]`` 时按全 1 处理（**锚点等价**：与改造前逐位一致）；
        为 ``dict[int, float]`` 时按值加权——值越大表示该历史商品意图越强
        （如 ORD=5 / CART=3 / CLICK=1），在找邻居时贡献更大相似度。
        权重 ≤0 的商品不参与找邻居（等价于不在历史中）。
        """
        hist_w: dict[int, float] = (
            {int(i): 1.0 for i in history}
            if not isinstance(history, dict)
            else {int(k): float(v) for k, v in history.items()}
        )
        sim = np.zeros(self._n_users, dtype=np.float64)
        touched = False
        for it, w in hist_w.items():
            if w <= 0:
                continue
            us = self._users_of_item(it)
            if us is None or us.size == 0:
                continue
            # 调整余弦：贡献 = w / sqrt(|items(u)| * |items(v)|)
            np.add.at(sim, us, w / np.sqrt(np.maximum(self._ulen[us], 1.0)))
            touched = True
        if not touched:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        iu = self._uidx.get(int(user))
        if iu is not None:
            sim[iu] = 0.0

        nz = np.flatnonzero(sim)
        if nz.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        if nz.size > top_neighbors:
            nz = nz[np.argpartition(sim[nz], -top_neighbors)[-top_neighbors:]]

        hist_set = set(hist_w)
        acc: dict[int, float] = defaultdict(float)
        for v in nz.tolist():
            w = sim[v]
            if w <= 0:
                continue
            for it in self._items_of_user_idx(v).tolist():
                if it not in hist_set:
                    acc[it] += w
        if not acc:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        items = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
        scores = np.fromiter(acc.values(), dtype=np.float64, count=len(acc))
        order = np.argsort(-scores)[:top_items]
        return items[order], scores[order]


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """把分数转成 ``[0,1]`` 的**排名分**（1 = 最高分）。

    融合多路分数前必须统一量纲：全局热度可达数千、类目内排名分 ≤1，
    直接加权会让热度完全淹没其它路（实测导致 ours 退化成 pop）。
    """
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr
    if n == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(-arr, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return 1.0 - ranks / (n - 1)
