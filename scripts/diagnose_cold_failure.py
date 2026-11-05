"""冷启动失败分层诊断（只读，不重跑主循环，复用 v3 的召回结构）。

回答三个问题，定位 ColdHit=0 到底卡在哪一层：
  L1 分母：多少测试用户的 5 个 holdout 正样本里，落在冷区类目（[20,100) 交互量）？
  L2 可达性：这些冷区正样本，有多少其实在 1995 候选池里？
  L3 排序：在池子里的冷区正样本，ours_v3 / cf 把它们排到第几（top-10 内?）？

结论导向：
  - L1 分母小 → 冷区正样本本底就少，ColdHit 破 0 难（数据属性）
  - L2 低 → ceiling 问题（池子捞不到），需扩池（方案 B）
  - L3 低 → 排序问题，调融合/加分权重
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pymysql

from config import settings
from src.eval.user_cf import UserCF, rank_normalize
from src.eval.light_item_cf import ItemCF
from src.models.diversity import rerank

logger = logging.getLogger("diag_cold")

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"
COLD_LO, COLD_HI = 20, 100


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()

    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    stats = pd.read_parquet(SPLIT_DIR / "stats_cache.parquet")
    meta = json.loads((SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))

    pop = stats[stats["kind"] == "pop"].set_index("item_id")["n"]
    cat_top = {
        int(c): g["item_id"].tolist()
        for c, g in stats[stats["kind"] == "cat_item"].sort_values(["cid", "n"],
                                                                    ascending=[True, False]).groupby("cid")
    }
    cate_n = stats[stats["kind"] == "cat"].set_index("cid")["n"]
    cold_cats = set(cate_n[(cate_n >= COLD_LO) & (cate_n < COLD_HI)].index.tolist())
    global_top = pop.sort_values(ascending=False).index.tolist()

    conn = pymysql.connect(
        host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
        password=settings.DB_PASSWORD, database=settings.DB_NAME,
        charset="utf8mb4", connect_timeout=10,
    )
    item2cat = pd.read_sql("SELECT item_id, category_id FROM item WHERE category_id<>0", conn)
    conn.close()
    cat_of = {int(i): int(c) for i, c in zip(item2cat["item_id"], item2cat["category_id"])}
    del item2cat

    # 冷区定向召回
    cold_items_by_cat: dict[int, list[int]] = {}
    for c in cold_cats:
        if c in cat_top:
            cold_items_by_cat[c] = cat_top[c][:20]

    # UserCF + ItemCF
    pdf = pd.read_parquet(SPLIT_DIR / "cf_pairs_2000.parquet", columns=["item_id", "user_id"])
    ucf = UserCF(pdf["item_id"].to_numpy(np.int64), pdf["user_id"].to_numpy(np.int64),
                  max_item_users=2000)
    icf = ItemCF(ucf, top_m=15000, top_similar=20)
    del pdf
    logger.info("CF 就绪 (%.0fs)", time.perf_counter() - t0)

    train_by_user = {int(u): [int(x) for x in g] for u, g in train.groupby("user_id")["item_id"]}
    test_by_user = test.groupby("user_id")["item_id"].apply(set).to_dict()
    users = sorted(test_by_user)

    CF_ITEMS, PER_CAT, GLOBAL_TOP, COLD_INJECT_PER_CAT, POOL_CAP, K, ALPHA = 1200, 60, 300, 5, 2000, 10, 0.05

    # ---- 分层统计 ----
    n_users_with_cold_pos = 0          # L1 分母
    total_cold_pos = 0
    cold_pos_in_pool = 0              # L2
    cold_pos_in_pool_ours_top10 = 0   # L3 ours
    cold_pos_in_pool_cf_top10 = 0
    rank_ours: list[int] = []         # 在池里的冷区正样本，ours 排名分布
    rank_cf: list[int] = []
    sample_rank_gaps: list[dict] = []  # 前 10 个 case 的具体证据

    for n_done, u in enumerate(users, 1):
        if n_done % 5000 == 0:
            logger.info("  %d/%d (%.0fs)", n_done, len(users), time.perf_counter() - t0)
        truth = test_by_user[u]
        hist = train_by_user.get(u, [])
        hist_set = set(hist)
        ucats = {cat_of[i] for i in hist if i in cat_of}

        # L1：该用户冷区正样本
        cold_pos = {i for i in truth if cat_of.get(i, 0) in cold_cats}
        if cold_pos:
            n_users_with_cold_pos += 1
            total_cold_pos += len(cold_pos)

        cf_items, cf_scores = ucf.recommend(u, hist, top_items=CF_ITEMS)
        icf_items, icf_scores = icf.recommend(hist, top_items=200)
        cf_map = dict(zip(cf_items.tolist(), cf_scores.tolist()))
        icf_map = dict(zip(icf_items.tolist(), icf_scores.tolist()))

        pool: list[int] = []
        mark: set[int] = set()
        for i in cf_items.tolist():
            if i not in hist_set and i not in mark:
                mark.add(i); pool.append(int(i))
        for i in icf_items.tolist():
            if i not in hist_set and i not in mark:
                mark.add(i); pool.append(int(i))
        for c in ucats:
            for i in cat_top.get(c, ())[:PER_CAT]:
                if i not in hist_set and i not in mark:
                    mark.add(i); pool.append(int(i))
        for i in global_top[:GLOBAL_TOP]:
            if i not in hist_set and i not in mark:
                mark.add(i); pool.append(int(i))
        ni = 0
        for c in list(cold_cats)[:200]:
            for i in cold_items_by_cat.get(c, [])[:COLD_INJECT_PER_CAT]:
                if i not in hist_set and i not in mark:
                    mark.add(i); pool.append(int(i)); ni += 1
                if ni >= 300:
                    break
        pool = pool[:POOL_CAP]
        pset = set(pool)

        # L2：冷区正样本在不在池里
        cold_in_pool = cold_pos & pset
        cold_pos_in_pool += len(cold_in_pool)
        if cold_in_pool:
            # 复用主循环融合公式
            pc = [cat_of.get(i, 0) for i in pool]
            raw_pop = np.array([float(pop.get(i, 0)) for i in pool])
            raw_cf = np.array([float(cf_map.get(i, 0.0)) for i in pool])
            raw_icf = np.array([float(icf_map.get(i, 0.0)) for i in pool])
            cat_pos: dict[int, float] = {}
            cat_rank: dict[int, list[int]] = {}
            for i, c in zip(pool, pc):
                cat_rank.setdefault(c, []).append(i)
            for c, items in cat_rank.items():
                if c in cat_top:
                    pos = {it: r for r, it in enumerate(cat_top[c])}
                    for it in items:
                        if it in pos:
                            cat_pos[it] = 1.0 / (1.0 + pos[it])
            raw_cat = np.array([cat_pos.get(i, 0.0) for i in pool])
            s_pop = rank_normalize(raw_pop)
            s_cf = rank_normalize(raw_cf) if raw_cf.max() > 0 else raw_cf
            s_icf = rank_normalize(raw_icf) if raw_icf.max() > 0 else raw_icf
            s_cat = rank_normalize(raw_cat)
            aux = 0.5 * s_pop + 0.5 * s_icf
            coldness = np.array([1.0 if c in cold_cats else 0.0 for c in pc])
            final_score = 0.5 * s_cf + 0.3 * aux + 0.2 * s_cat + 0.10 * coldness
            order_ours = rerank([(pool[j], pc[j], float(final_score[j]))
                                 for j in range(len(pool))], alpha=ALPHA, k=len(pool))
            order_cf = [pool[j] for j in np.argsort(-s_cf)]
            for cp in cold_in_pool:
                # ours 排名（top-K 内?）
                r_o = order_ours.index(cp) if cp in order_ours else len(order_ours)
                r_c = order_cf.index(cp) if cp in order_cf else len(order_cf)
                if r_o < K:
                    cold_pos_in_pool_ours_top10 += 1
                if r_c < K:
                    cold_pos_in_pool_cf_top10 += 1
                rank_ours.append(r_o)
                rank_cf.append(r_c)
                if len(sample_rank_gaps) < 10:
                    sample_rank_gaps.append({
                        "user": u, "cold_item": cp, "cat": cat_of.get(cp, 0),
                        "ours_rank": r_o, "cf_rank": r_c,
                    })

    denom = n_users_with_cold_pos
    out = {
        "protocol": {"n_users": len(users), "holdout": meta["holdout"],
                     "cold_zone": [COLD_LO, COLD_HI], "n_cold_cats": len(cold_cats)},
        "L1_denominator": {
            "users_with_cold_positive": denom,
            "user_ratio": round(denom / len(users), 4),
            "total_cold_positives": total_cold_pos,
        },
        "L2_reachability": {
            "cold_pos_in_pool": cold_pos_in_pool,
            "cold_pos_total": total_cold_pos,
            "pool_coverage": round(cold_pos_in_pool / max(1, total_cold_pos), 4),
        },
        "L3_ranking": {
            "cold_pos_in_pool": cold_pos_in_pool,
            "ours_top10_hit": cold_pos_in_pool_ours_top10,
            "cf_top10_hit": cold_pos_in_pool_cf_top10,
            "ours_rank_mean": round(float(np.mean(rank_ours)), 1) if rank_ours else None,
            "cf_rank_mean": round(float(np.mean(rank_cf)), 1) if rank_cf else None,
            "ours_rank_median": int(np.median(rank_ours)) if rank_ours else None,
            "cf_rank_median": int(np.median(rank_cf)) if rank_cf else None,
            "sample_gaps": sample_rank_gaps,
        },
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (SPLIT_DIR / "cold_failure_diagnosis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"L1 分母: {denom}/{len(users)} 用户有冷区正样本 ({out['L1_denominator']['user_ratio']:.1%})")
    print(f"     冷区正样本总数 = {total_cold_pos}")
    print(f"L2 可达: 池内冷区正样本 {cold_pos_in_pool}/{total_cold_pos} "
          f"({out['L2_reachability']['pool_coverage']:.1%})")
    print(f"L3 排序: 池内冷区正样本 {cold_pos_in_pool} 个")
    print(f"     ours top10 命中 {cold_pos_in_pool_ours_top10} | cf top10 命中 {cold_pos_in_pool_cf_top10}")
    print(f"     ours 排名 中位 {out['L3_ranking']['ours_rank_median']} | cf 中位 {out['L3_ranking']['cf_rank_median']}")
    print(f"结果: {SPLIT_DIR / 'cold_failure_diagnosis.json'}  耗时 {out['elapsed_sec']}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
