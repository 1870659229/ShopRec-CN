"""预计算 ItemCF：只对 top-M 高频物品建共现相似度表，推荐时 O(1) 查表。

内存估算（M=15000, n_users≈154000）：
- mat (M×n_users, sparse): ~23M nnz × 8B ≈ 184 MB
- cooccur (M×M, dense float64): 15000²×8B ≈ 1.8 GB（构建后取 top-K 即释放）
- 邻居表 (M, K): 15000×20×8B ≈ 2.4 MB
- 峰值 ≈ 2.5 GB，构建后稳态 ~500 MB

不在 top-M 内的 item 不参与 ItemCF 召回（由 UserCF 路兜底）。
"""
from __future__ import annotations

import logging
import time

import numpy as np
from scipy.sparse import csr_matrix

from src.eval.user_cf import UserCF

logger = logging.getLogger(__name__)

DEFAULT_TOP_M = 15000       # 取交互数最大的 M 个 item
DEFAULT_TOP_SIMILAR = 20    # 每个 item 存 top-K 相似 item
DEFAULT_TOP_ITEMS = 200     # 推荐时返回 top-N


class ItemCF:
    """预计算物品-物品协同过滤（scipy sparse 共现 + 查表）。"""

    def __init__(
        self,
        ucf: UserCF,
        top_m: int = DEFAULT_TOP_M,
        top_similar: int = DEFAULT_TOP_SIMILAR,
    ) -> None:
        self._top_similar = top_similar
        self._neighbors: dict[int, list[tuple[int, float]]] = {}

        t0 = time.perf_counter()
        self._build(ucf, top_m, top_similar)
        logger.info(
            "ItemCF 就绪: top_m=%d, 有邻居 item=%d, 耗时 %.1fs",
            top_m, len(self._neighbors), time.perf_counter() - t0,
        )

    def _build(self, ucf: UserCF, top_m: int, top_similar: int) -> None:
        """从 UserCF 倒排索引提取 top-M 高频 item，建共现 + 调整余弦。"""
        inv_items: np.ndarray = ucf._inv_items
        inv_start: np.ndarray = ucf._inv_start
        inv_end: np.ndarray = ucf._inv_end
        inv_users: np.ndarray = ucf._inv_users

        # 取交互数最大的 M 个 item（按倒排段长度降序）
        seg_len = inv_end - inv_start
        order = np.argsort(-seg_len, kind="stable")[:top_m]
        self._item_ids = inv_items[order]  # shape (M,)
        m = len(self._item_ids)
        m = min(m, top_m)

        # 每个 top-M item 的用户数（调整余弦归一化用）
        deg = np.maximum(seg_len[order[:m]].astype(np.float64), 1.0)

        # ---- 构建 M × n_users 二值矩阵 ----
        # 对每个 top-M item k（在 order 中），取其 inv_users 段
        rows_parts = []
        cols_parts = []
        for idx in range(m):
            k = order[idx]
            n_u = int(seg_len[k])
            rows_parts.append(np.full(n_u, idx, dtype=np.int32))
            cols_parts.append(inv_users[inv_start[k]: inv_end[k]].astype(np.int32))
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        mat = csr_matrix((np.ones(len(cols), dtype=np.int8), (rows, cols)),
                         shape=(m, ucf._n_users))
        del rows, cols

        # ---- 共现 M×M ----
        logger.info("  构建共现矩阵 (%d×%d, %.1f MB 矩阵)...",
                     m, ucf._n_users, mat.nnz * 8 / 1e6)
        cooccur = (mat @ mat.T).toarray().astype(np.float64)
        del mat
        np.fill_diagonal(cooccur, 0.0)

        # ---- 调整余弦归一化 ----
        # cooccur[i,j] / sqrt(deg_i * deg_j)
        norm = np.sqrt(np.outer(deg, deg))
        cooccur /= norm

        # ---- 取每个 item 的 top-K 相似（向量化 argpartition）----
        logger.info("  提取 top-%d 邻居...", top_similar)
        k = min(top_similar, m)
        # 每行取相似度最大的 K 个列下标（argpartition 快于全排序）
        part = np.argpartition(-cooccur, k, axis=1)[:, :k]        # (m, k)
        row_idx = np.repeat(np.arange(m), k)                       # (m*k,)
        col_idx = part.reshape(-1)                                   # (m*k,)
        weights = cooccur[row_idx, col_idx]                          # (m*k,)
        del cooccur

        valid = weights > 0
        row_idx, col_idx, weights = (row_idx[valid], col_idx[valid], weights[valid])
        # 按行（item）分组写入邻居表
        for r, j, w in zip(row_idx.tolist(), col_idx.tolist(), weights.tolist()):
            self._neighbors.setdefault(int(self._item_ids[r]), []).append(
                (int(self._item_ids[j]), float(w))
            )

    def recommend(
        self,
        history: list[int],
        top_items: int = DEFAULT_TOP_ITEMS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """查表推荐：遍历历史 item，累加相似 item 分数。O(|hist| × K)"""
        hist_set = set(history)
        acc: dict[int, float] = {}
        for it in history:
            nbrs = self._neighbors.get(int(it))
            if not nbrs:
                continue
            for nb, w in nbrs:
                if nb not in hist_set:
                    acc[nb] = acc.get(nb, 0.0) + w
        if not acc:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        items = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
        scores = np.fromiter(acc.values(), dtype=np.float64, count=len(acc))
        order = np.argsort(-scores)[:top_items]
        return items[order], scores[order]


# 兼容旧接口名
LightItemCF = ItemCF
