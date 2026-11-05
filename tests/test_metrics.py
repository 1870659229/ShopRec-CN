from src.eval.metrics import category_diversity, ndcg_at_k, recall_at_k


# ---- 计划给定的 5 条基础断言 ----
def test_recall_at_k_hit_returns_one():
    assert recall_at_k(True, 5) == 1.0


def test_recall_at_k_miss_returns_zero():
    assert recall_at_k(False, 5) == 0.0


def test_ndcg_at_k_top1_is_one():
    """rank=1 时 1/log2(2)=1。"""
    assert ndcg_at_k(1, 10) == 1.0


def test_ndcg_at_k_out_of_range_is_zero():
    assert ndcg_at_k(11, 10) == 0.0


def test_category_diversity_basic_average():
    """用户1: 2/2=1.0；用户2: 1/2=0.5 ⇒ 平均 0.75。"""
    assert category_diversity([["a", "b"], ["a", "a"]]) == 0.75


# ---- 额外边界覆盖 ----
def test_ndcg_at_k_rank_equals_k_is_nonzero():
    """命中边界：rank==k 应仍计入（1/log2(k+1) > 0）。"""
    assert ndcg_at_k(3, 3) > 0.0


def test_ndcg_at_k_rank_k_plus_1_is_zero():
    assert ndcg_at_k(4, 3) == 0.0


def test_ndcg_at_k_rank_zero_is_zero():
    assert ndcg_at_k(0, 3) == 0.0


def test_ndcg_at_k_negative_rank_is_zero():
    assert ndcg_at_k(-1, 3) == 0.0


def test_category_diversity_empty_list_is_zero():
    assert category_diversity([]) == 0.0


def test_category_diversity_list_of_empty_list_is_zero():
    assert category_diversity([[]]) == 0.0


def test_category_diversity_all_unique_is_one():
    """全唯一类目 ⇒ 去重占比为 1。"""
    assert category_diversity([["a", "b", "c"]]) == 1.0
