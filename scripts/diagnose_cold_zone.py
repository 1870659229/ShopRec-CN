"""诊断冷区召回失效（ColdHit=0 根因定位）。

问题：离线评估 v2 全量结果中，所有方法的 cold_category_hit_rate 均为 0。
本脚本回答两个问题：
1. 候选池里有没有冷区商品？（如果没有，召回策略本身就没覆盖冷区）
2. 测试正样本里有多少冷区商品？（如果测试集里也没有，指标分母为 0 是正常的）

输出：JSON 报告 + 控制台摘要。
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402

logger = logging.getLogger("diagnose_cold")

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"
COLD_LO, COLD_HI = 20, 100


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()

    # ---- 载入评估数据 ----
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    stats = pd.read_parquet(SPLIT_DIR / "stats_cache.parquet")

    # ---- 冷区类目集合 ----
    cate_n = stats[stats["kind"] == "cat"].set_index("cid")["n"]
    cold_cats = set(cate_n[(cate_n >= COLD_LO) & (cate_n < COLD_HI)].index.tolist())
    logger.info("冷区类目 [20,100): %d 个", len(cold_cats))

    # ---- item → 叶子类 ----
    conn = pymysql.connect(
        host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
        password=settings.DB_PASSWORD, database=settings.DB_NAME,
        charset="utf8mb4", connect_timeout=10,
    )
    item2cat = pd.read_sql("SELECT item_id, category_id FROM item WHERE category_id<>0", conn)
    conn.close()
    cat_of = {int(i): int(c) for i, c in zip(item2cat["item_id"], item2cat["category_id"])}
    del item2cat

    # ---- 1) 测试正样本中冷区商品占比 ----
    test_items = test["item_id"].tolist()
    test_cold = [i for i in test_items if cat_of.get(i, 0) in cold_cats]
    n_test_cold = len(test_cold)
    n_test_total = len(test_items)
    # 有多少用户至少有 1 个冷区测试正样本
    test["is_cold"] = test["item_id"].map(lambda i: cat_of.get(i, 0) in cold_cats)
    users_with_cold_truth = int(test[test["is_cold"]]["user_id"].nunique())
    users_total = int(test["user_id"].nunique())

    logger.info("测试正样本: %d, 冷区: %d (%.2f%%)", n_test_total, n_test_cold, 100*n_test_cold/n_test_total)
    logger.info("有冷区正样本的用户: %d / %d (%.1f%%)", users_with_cold_truth, users_total, 100*users_with_cold_truth/users_total)

    # ---- 2) 候选池中冷区商品占比 ----
    # 复用 v2 的候选池构建逻辑：CF ∪ 品类内热门 ∪ 全局热门
    pop = stats[stats["kind"] == "pop"].set_index("item_id")["n"]
    cat_top = {
        int(c): g["item_id"].tolist()
        for c, g in stats[stats["kind"] == "cat_item"].sort_values(["cid", "n"], ascending=[True, False]).groupby("cid")
    }
    global_top = pop.sort_values(ascending=False).index.tolist()

    # 载入 CF 缓存
    cf_pairs_path = SPLIT_DIR / "cf_pairs_2000.parquet"
    pdf = pd.read_parquet(cf_pairs_path, columns=["item_id", "user_id"])

    from src.eval.user_cf import UserCF
    train_by_user = {int(u): [int(x) for x in g] for u, g in train.groupby("user_id")["item_id"]}
    eval_users = [int(u) for u in test["user_id"].unique()]
    cf = UserCF(pdf["item_id"].to_numpy(np.int64), pdf["user_id"].to_numpy(np.int64), max_item_users=2000)
    del pdf

    test_by_user = test.groupby("user_id")["item_id"].apply(set).to_dict()
    users = sorted(test_by_user)

    # 参数与 v2_full 一致
    CF_ITEMS = 1200
    PER_CAT = 60
    GLOBAL_TOP = 300
    POOL_CAP = 2000

    pool_cold_counts = []
    pool_sizes = []
    pool_cold_items_total = 0
    pool_total_items = 0
    cf_cold_items_total = 0
    cf_total_items = 0

    # 也统计：冷区商品在候选池中的来源分布
    cold_from_cf = 0
    cold_from_cat = 0
    cold_from_global = 0

    for n_done, u in enumerate(users, 1):
        if n_done % 5000 == 0:
            logger.info("  %d/%d 用户 (%.0fs)", n_done, len(users), time.perf_counter() - t0)
        truth = test_by_user[u]
        hist = train_by_user.get(u, [])
        hist_set = set(hist)
        ucats = {cat_of[i] for i in hist if i in cat_of}

        cf_items, _ = cf.recommend(u, hist, top_items=CF_ITEMS)
        cf_set = set(cf_items.tolist())

        pool = []
        mark = set()
        for i in cf_items.tolist():
            if i not in hist_set and i not in mark:
                mark.add(i)
                pool.append(int(i))
        for c in ucats:
            for i in cat_top.get(c, ())[:PER_CAT]:
                if i not in hist_set and i not in mark:
                    mark.add(i)
                    pool.append(int(i))
        for i in global_top[:GLOBAL_TOP]:
            if i not in hist_set and i not in mark:
                mark.add(i)
                pool.append(int(i))
        pool = pool[:POOL_CAP]

        pool_sizes.append(len(pool))
        cold_in_pool = [i for i in pool if cat_of.get(i, 0) in cold_cats]
        pool_cold_counts.append(len(cold_in_pool))
        pool_cold_items_total += len(cold_in_pool)
        pool_total_items += len(pool)

        # CF 路贡献的冷区商品
        cold_cf = [i for i in cold_in_pool if i in cf_set]
        cold_from_cf += len(cold_cf)
        # 品类内热门路贡献
        cold_not_cf = set(cold_in_pool) - cf_set
        cold_from_cat += sum(1 for i in cold_not_cf if any(i in cat_top.get(c, [])[:PER_CAT] for c in ucats))
        cold_from_global += sum(1 for i in cold_not_cf if i in set(global_top[:GLOBAL_TOP]))

    mean_pool = float(np.mean(pool_sizes))
    mean_cold = float(np.mean(pool_cold_counts))
    cold_ratio = pool_cold_items_total / max(1, pool_total_items)

    report = {
        "cold_categories_count": len(cold_cats),
        "test": {
            "total_items": n_test_total,
            "cold_items": n_test_cold,
            "cold_ratio_pct": round(100 * n_test_cold / n_test_total, 2),
            "users_with_cold_truth": users_with_cold_truth,
            "users_total": users_total,
            "pct_users_with_cold_truth": round(100 * users_with_cold_truth / users_total, 1),
        },
        "pool": {
            "mean_pool_size": round(mean_pool, 1),
            "mean_cold_items_per_pool": round(mean_cold, 2),
            "cold_ratio_in_pool_pct": round(100 * cold_ratio, 3),
            "total_cold_items_seen": pool_cold_items_total,
            "total_items_seen": pool_total_items,
        },
        "cold_source_breakdown": {
            "from_cf": cold_from_cf,
            "from_cat_pop": cold_from_cat,
            "from_global_pop": cold_from_global,
        },
        "diagnosis": "",
    }

    # ---- 诊断结论 ----
    diag_lines = []
    if n_test_cold == 0:
        diag_lines.append("测试集无冷区正样本 ⇒ ColdHit 分母为 0 是正常现象（指标无意义）。")
    elif users_with_cold_truth == 0:
        diag_lines.append("无用户有冷区正样本 ⇒ ColdHit 无法命中。")
    else:
        diag_lines.append(f"测试集有 {n_test_cold} 个冷区正样本，{users_with_cold_truth}/{users_total} 用户有冷区正样本。")

    if cold_ratio < 0.01:
        diag_lines.append(f"候选池冷区商品占比仅 {100*cold_ratio:.3f}%（极低）⇒ 冷区商品几乎不进池。")
    elif cold_ratio < 0.05:
        diag_lines.append(f"候选池冷区商品占比 {100*cold_ratio:.2f}%（偏低）⇒ 冷区商品进池不足。")
    else:
        diag_lines.append(f"候选池冷区商品占比 {100*cold_ratio:.2f}%。")

    if cold_from_cf == 0 and cold_from_cat == 0:
        diag_lines.append("CF 与品类热门均未贡献冷区商品 ⇒ 召回策略需专门增加冷区路。")

    report["diagnosis"] = "\n".join(diag_lines)

    out = SPLIT_DIR / "cold_zone_diagnosis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("冷区诊断报告")
    print("=" * 72)
    print(f"冷区类目数: {len(cold_cats)}")
    print(f"\n测试集: {n_test_total} 正样本, 冷区 {n_test_cold} ({100*n_test_cold/n_test_total:.2f}%)")
    print(f"有冷区正样本的用户: {users_with_cold_truth} / {users_total} ({100*users_with_cold_truth/users_total:.1f}%)")
    print(f"\n候选池: 平均 {mean_pool:.1f} 商品/用户, 冷区 {mean_cold:.2f}/池")
    print(f"候选池冷区占比: {100*cold_ratio:.3f}%")
    print(f"\n冷区来源: CF={cold_from_cf}, 品类热门={cold_from_cat}, 全局热门={cold_from_global}")
    print(f"\n诊断:\n{report['diagnosis']}")
    print(f"\n报告: {out}    耗时 {time.perf_counter()-t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
