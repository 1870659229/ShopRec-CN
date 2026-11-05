"""离线评估指标：Recall@K / NDCG@K / 品类多样性。

本模块仅依赖标准库，可独立单测。
"""

import logging
import math

# 与其他模块保持一致：定义模块级 logger。当前版本没有任何函数需要记日志，
# 保留供后续接入评估流水线时直接使用，暂不产生日志输出。
logger = logging.getLogger(__name__)


def recall_at_k(in_top_k: bool, k: int) -> float:
    """返回单条样本的 Recall@K。

    契约：`k` 参数保留但当前未使用；是否命中 top-K 由调用方判定
    （本函数只把布尔命中转成 1.0/0.0）。
    """
    return 1.0 if in_top_k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """单命中场景下的 NDCG@K。

    rank 为命中文档在推荐列表中的 1-based 位置；rank 落在 [1, k] 之外记 0。
    """
    if rank < 1 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def category_diversity(rec_lists: list[list[str]]) -> float:
    """每个用户推荐列表内去重类目占比的平均值。"""
    if not rec_lists:
        return 0.0
    scores = []
    for recs in rec_lists:
        if not recs:
            scores.append(0.0)
            continue
        scores.append(len(set(recs)) / len(recs))
    return sum(scores) / len(scores)
