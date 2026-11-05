import numpy as np

from src.models.diversity import rerank


def _scores():
    # (item_id, category_id, score)
    return [
        (1, 100, 0.9), (2, 100, 0.85), (3, 100, 0.8),
        (4, 200, 0.5), (5, 300, 0.4),
    ]


def test_alpha_zero_is_pure_score_order():
    out = rerank(_scores(), alpha=0.0, k=3)
    assert out == [1, 2, 3]


def test_alpha_one_promotes_category_diversity():
    out = rerank(_scores(), alpha=1.0, k=3)
    assert len(set(out)) == 3
    assert 4 in out and 5 in out


def test_returns_at_most_k_items():
    assert len(rerank(_scores(), alpha=0.5, k=2)) == 2


def test_empty_input_returns_empty():
    assert rerank([], alpha=0.5, k=3) == []


def _clamp_probe():
    """分数幅度大于 1 的数据集，用于区分 alpha>1 是否被 clamp。

    第二轮贪心比较：已见类目候选 2 的分值 = 3.0 - alpha，新类目候选 3 = 1.5。
      alpha=1.0        -> 2.0 > 1.5  选 2
      alpha=2.0 若 clamp -> 同 alpha=1.0，选 2
      alpha=2.0 不 clamp -> 1.0 < 1.5 错选 3（结果变为 [1, 3, 2]）
    """
    return [(1, 100, 5.0), (2, 100, 3.0), (3, 200, 1.5)]


def test_alpha_above_one_is_clamped_to_one():
    """alpha=2.0 应被 clamp 到 1.0：结果与 alpha=1.0 一致，且第二轮仍选已见类目。"""
    assert rerank(_clamp_probe(), alpha=2.0, k=3) == rerank(_clamp_probe(), alpha=1.0, k=3)
    assert rerank(_clamp_probe(), alpha=2.0, k=3) == [1, 2, 3]


def test_negative_alpha_is_clamped_to_zero():
    """alpha=-1.0 应被 clamp 到 0.0：已见类目不被奖励，第二轮改选新类目。"""
    probe = [(1, 100, 0.9), (2, 100, 0.0), (3, 200, 0.5)]
    # alpha=0.0：第二轮 2->0.0 vs 3->0.5，选 3
    assert rerank(probe, alpha=0.0, k=3) == [1, 3, 2]
    # alpha=-1.0 不 clamp 时会给已见类目 +1 奖励（2->1.0 > 3->0.5，错选 2）
    assert rerank(probe, alpha=-1.0, k=3) == rerank(probe, alpha=0.0, k=3)


# ---- 向量化重写后的等价性护栏 ----
# diversity.py 由 O(k·n) 纯 Python 贪心改为 O(n) 向量化实现以提速（rerank 曾占
# 端到端 53%）。以下用例把**朴素实现原文照抄**作参照物，随机对照逐位比对，
# 含并列分数的 tie-breaking——这是"提速没改语义"的唯一凭据。

def _naive_rerank(scores, alpha, k):
    """提速前的原始实现，逐字保留作为等价性基准（勿"顺手优化"）。"""
    if not scores:
        return []
    alpha = max(0.0, min(1.0, alpha))
    ordered = sorted(scores, key=lambda x: x[2], reverse=True)
    selected = []
    seen_cate = set()
    while ordered and len(selected) < k:
        best_idx, best_val = None, float("-inf")
        for idx, (item, cate, score) in enumerate(ordered):
            penalty = alpha if cate in seen_cate else 0.0
            val = score - penalty
            if val > best_val:
                best_idx, best_val = idx, val
        item, cate, _ = ordered.pop(best_idx)
        selected.append(item)
        seen_cate.add(cate)
    return selected


def _random_case(rng, n, n_cats, dup_scores):
    cats = [int(rng.integers(0, n_cats)) for _ in range(n)]
    if dup_scores:
        vals = [float(rng.integers(0, 4)) for _ in range(n)]   # 制造大量并列
    else:
        vals = [float(rng.random()) for _ in range(n)]
    return [(i + 1, cats[i], vals[i]) for i in range(n)]


def test_vectorized_matches_naive_on_random_cases():
    """随机对照：含并列分数、k 大于类目数、单类目等边界。"""
    rng = np.random.default_rng(20261013)
    checked = 0
    for trial in range(60):
        n = int(rng.integers(1, 120))
        n_cats = int(rng.integers(1, 12))
        scores = _random_case(rng, n, n_cats, dup_scores=bool(trial % 2))
        for alpha in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.7, -0.3):
            k = int(rng.integers(1, 13))
            got = rerank(scores, alpha=alpha, k=k)
            want = _naive_rerank(list(scores), alpha, k)
            assert got == want, (
                f"trial={trial} alpha={alpha} k={k} 不一致\n got={got}\nwant={want}"
            )
            checked += 1
    assert checked == 60 * 9


def test_prepare_reuse_matches_single_call():
    """跨 alpha 复用排序不得改变结果（评估脚本依赖这条才能提速）。"""
    from src.models.diversity import encode_categories, prepare, rerank_prepared

    rng = np.random.default_rng(7)
    n = 300
    cats = [int(rng.integers(0, 25)) for _ in range(n)]
    vals = [float(rng.random()) for _ in range(n)]
    scores = [(i + 1, cats[i], vals[i]) for i in range(n)]

    codes, n_cats = encode_categories(cats)
    prep = prepare([s[0] for s in scores], codes, [s[2] for s in scores], n_cats)
    for alpha in (0.0, 0.02, 0.05, 0.1, 0.2):
        assert rerank_prepared(prep, alpha, 10) == _naive_rerank(scores, alpha, 10)


def test_all_same_category_falls_back_to_score_order():
    """只有一个类目时，任何 alpha 都退化为纯分数排序。"""
    scores = [(1, 9, 0.3), (2, 9, 0.9), (3, 9, 0.6)]
    for alpha in (0.0, 0.5, 1.0):
        assert rerank(scores, alpha=alpha, k=3) == [2, 3, 1]
