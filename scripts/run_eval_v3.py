"""改进版离线评估 v3：ItemCF + 冷区定向注入 + 修 fuse 策略。

改进点（vs v2）：
1. 加 ItemCF 召回路（物-物共现相似度，transductive）
2. 冷区定向召回：对 1026 个冷区类目，每类注入 top-N 商品到候选池
3. 修 fuse：CF 为主路，fuse 只对冷区商品生效（热区不融合，不稀释 CF 信号）
4. α rerank 保持不变

目标：ours 的 R@10 追平或超过 CF 基线，ColdHit > 0
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
from src.eval.user_cf import UserCF, rank_normalize  # noqa: E402
from src.eval.light_item_cf import ItemCF  # noqa: E402
from src.models.diversity import rerank  # noqa: E402

logger = logging.getLogger("run_eval_v3")

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"
COLD_LO, COLD_HI = 20, 100


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()

    # ---- 载入数据 ----
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    stats = pd.read_parquet(SPLIT_DIR / "stats_cache.parquet")
    meta = json.loads((SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))

    pop = stats[stats["kind"] == "pop"].set_index("item_id")["n"]
    cat_top = {
        int(c): g["item_id"].tolist()
        for c, g in stats[stats["kind"] == "cat_item"].sort_values(["cid", "n"], ascending=[True, False]).groupby("cid")
    }
    cate_n = stats[stats["kind"] == "cat"].set_index("cid")["n"]
    cold_cats = set(cate_n[(cate_n >= COLD_LO) & (cate_n < COLD_HI)].index.tolist())
    global_top = pop.sort_values(ascending=False).index.tolist()
    logger.info("缓存载入: 热门 %d / 类目 %d / 冷区 %d", len(pop), len(cat_top), len(cold_cats))

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

    # ---- 冷区类目 → 冷区商品列表（每类 top 20）----
    cold_items_by_cat: dict[int, list[int]] = {}
    for c in cold_cats:
        if c in cat_top:
            cold_items_by_cat[c] = cat_top[c][:20]
    all_cold_items = set()
    for items in cold_items_by_cat.values():
        all_cold_items.update(items)
    logger.info("冷区定向召回: %d 类目 → %d 商品", len(cold_items_by_cat), len(all_cold_items))

    # ---- UserCF ----
    train_by_user = {int(u): [int(x) for x in g] for u, g in train.groupby("user_id")["item_id"]}
    eval_users = [int(u) for u in test["user_id"].unique()]
    pairs_path = SPLIT_DIR / "cf_pairs_2000.parquet"
    pdf = pd.read_parquet(pairs_path, columns=["item_id", "user_id"])
    ucf = UserCF(pdf["item_id"].to_numpy(np.int64), pdf["user_id"].to_numpy(np.int64), max_item_users=2000)
    del pdf
    logger.info("UserCF 就绪 (%.0fs)", time.perf_counter() - t0)

    # ---- LightItemCF（在线计算，复用 UserCF 倒排索引，零额外内存）----
    icf = ItemCF(ucf, top_m=15000, top_similar=20)
    logger.info("ItemCF 就绪 (%.0fs)", time.perf_counter() - t0)

    test_by_user = test.groupby("user_id")["item_id"].apply(set).to_dict()
    users = sorted(test_by_user)

    # ---- 参数 ----
    CF_ITEMS = 1200
    ITEM_CF_ITEMS = 200
    PER_CAT = 60
    GLOBAL_TOP = 300
    COLD_INJECT_PER_CAT = 12  # v4：用户命中的冷区类目每类注入 12 个（提升 L2 可达性）
    COLD_INJECT_CAT_CAP = 40  # v4：每用户最多注入 40 个冷区类目（用户相关）
    POOL_CAP = 3000           # v4：扩池，给冷区腾位置
    K = 10
    ALPHA = 0.05              # 方案 A：α 降到饱和区 0~0.1 内
    COLD_BOOST = 0.20         # v4：池内冷区商品加分 0.10→0.20，弥补基数小

    METHODS = ("pop", "cf", "ours_v4")
    hits = {m: [] for m in METHODS}
    ndcgs = {m: [] for m in METHODS}
    divs = {m: [] for m in METHODS}
    cold_hits = {m: [] for m in METHODS}
    ceiling = []
    pool_sizes = []
    cold_injected_total = 0

    for n_done, u in enumerate(users, 1):
        if n_done % 5000 == 0:
            logger.info("  %d/%d 用户 (%.0fs)", n_done, len(users), time.perf_counter() - t0)
        truth = test_by_user[u]
        hist = train_by_user.get(u, [])
        hist_set = set(hist)
        ucats = {cat_of[i] for i in hist if i in cat_of}

        # ---- UserCF 召回 ----
        cf_items, cf_scores = ucf.recommend(u, hist, top_items=CF_ITEMS)
        cf_map = dict(zip(cf_items.tolist(), cf_scores.tolist()))

        # ---- ItemCF 召回 ----
        icf_items, icf_scores = icf.recommend(hist, top_items=ITEM_CF_ITEMS)
        icf_map = dict(zip(icf_items.tolist(), icf_scores.tolist()))

        # ---- 候选池：UserCF ∪ ItemCF ∪ 品类热门 ∪ 全局热门 ∪ 冷区定向注入 ----
        pool: list[int] = []
        mark = set()
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
        # 冷区定向注入（v4）：只对**用户历史命中的冷区类目**注入，让冷区商品与用户相关
        # 诊断显示全局取前 200 类目（与用户无关）命中率极低（L2 可达仅 2.2%），改为用户相关
        n_cold_injected = 0
        user_cold_cats = [c for c in ucats if c in cold_cats]
        if not user_cold_cats:
            # 用户没有命中任何冷区类目时，退回注入全局头部冷区类目（少量兜底）
            user_cold_cats = list(cold_cats)[:10]
        for c in user_cold_cats[:COLD_INJECT_CAT_CAP]:
            for i in cold_items_by_cat.get(c, [])[:COLD_INJECT_PER_CAT]:
                if i not in hist_set and i not in mark:
                    mark.add(i); pool.append(int(i))
                    n_cold_injected += 1
        cold_injected_total += n_cold_injected
        pool = pool[:POOL_CAP]
        pool_sizes.append(len(pool))
        if not pool:
            continue
        pset = set(pool)
        ceiling.append(len(truth & pset) / max(1, len(truth)))

        pc = [cat_of.get(i, 0) for i in pool]
        raw_pop = np.array([float(pop.get(i, 0)) for i in pool])
        raw_cf = np.array([float(cf_map.get(i, 0.0)) for i in pool])
        raw_icf = np.array([float(icf_map.get(i, 0.0)) for i in pool])

        # 品类内排名分
        cat_pos: dict[int, float] = {}
        cat_rank: dict[int, list[int]] = {}
        for i, c in zip(pool, pc):
            cat_rank.setdefault(c, []).append(i)
        for c, items in cat_rank.items():
            if c not in cat_top:
                continue
            pos = {it: r for r, it in enumerate(cat_top[c])}
            for it in items:
                if it in pos:
                    cat_pos[it] = 1.0 / (1.0 + pos[it])
        raw_cat = np.array([cat_pos.get(i, 0.0) for i in pool])

        # 统一量纲
        s_pop = rank_normalize(raw_pop)
        s_cf = rank_normalize(raw_cf) if raw_cf.max() > 0 else raw_cf
        s_icf = rank_normalize(raw_icf) if raw_icf.max() > 0 else raw_icf
        s_cat = rank_normalize(raw_cat)

        orderings = {
            "pop": [pool[j] for j in np.argsort(-s_pop)],
            "cf": [pool[j] for j in np.argsort(-s_cf)],
        }

        # ---- ours_v4（v4 冷区强化）：统一加权融合 + 冷区定向加分 ----
        # 主路 CF（已 rank_normalize）+ 辅路（0.5*热度 + 0.5*物物相似）+ 品类内排名
        # 冷区商品额外 +COLD_BOOST，引导 top-10 落进冷区 → 拉 ColdHit
        aux = 0.5 * s_pop + 0.5 * s_icf
        coldness = np.array([1.0 if c in cold_cats else 0.0 for c in pc])
        final_score = 0.5 * s_cf + 0.3 * aux + 0.2 * s_cat + COLD_BOOST * coldness
        trio = [(pool[j], pc[j], float(final_score[j])) for j in range(len(pool))]
        orderings["ours_v4"] = rerank(trio, alpha=ALPHA, k=K)

        for m, order in orderings.items():
            top = list(order[:K])
            hits[m].append(1 if truth & set(top) else 0)
            pos = next((r for r, it in enumerate(top, 1) if it in truth), None)
            ndcgs[m].append(1.0 / np.log2(pos + 1) if pos else 0.0)
            tcs = [cat_of.get(i, 0) for i in top]
            divs[m].append(len(set(tcs)) / max(1, len(tcs)))
            truth_cold = {i for i in truth if cat_of.get(i, 0) in cold_cats}
            if truth_cold:
                cold_hits[m].append(1 if truth_cold & set(top) else 0)

    def _mean(v):
        return float(np.mean(v)) if len(v) else 0.0

    report = {
        "protocol": {
            "n_users": len(users), "holdout": meta["holdout"], "seed": meta["seed"],
            "cf_items": CF_ITEMS, "item_cf_items": ITEM_CF_ITEMS,
            "per_cat": PER_CAT, "global_top": GLOBAL_TOP,
            "cold_inject_per_cat": COLD_INJECT_PER_CAT,
            "pool_cap": POOL_CAP, "alpha": ALPHA,
            "n_cold_categories": len(cold_cats),
            "n_cold_items_injected": cold_injected_total,
        },
        "pool": {
            "mean_size": round(_mean(pool_sizes), 1),
            "ceiling_recall": round(_mean(ceiling), 4),
        },
        "metrics": {},
    }
    for m in METHODS:
        d = {f"recall@{K}": round(_mean(hits[m]), 4)}
        d[f"ndcg@{K}"] = round(_mean(ndcgs[m]), 4)
        d["diversity@10"] = round(_mean(divs[m]), 4)
        d["cold_category_hit_rate"] = round(_mean(cold_hits[m]), 4)
        report["metrics"][m] = d

    out = SPLIT_DIR / "eval_v4.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"候选池 平均 {_mean(pool_sizes):.0f}/用户 | ceiling = {_mean(ceiling):.2%}")
    print(f"冷区类目 {len(cold_cats)} | 冷区注入商品（累计） {cold_injected_total}")
    print(f"{'方法':<10} {'R@'+str(K):>8} {'N@'+str(K):>8} {'Div@10':>9} {'ColdHit':>9}")
    for m in METHODS:
        d = report["metrics"][m]
        print(f"{m:<10} {d['recall@'+str(K)]:>8.4f} {d['ndcg@'+str(K)]:>8.4f} {d['diversity@10']:>9.4f} {d['cold_category_hit_rate']:>9.4f}")
    print(f"\n结果: {out}    总耗时 {time.perf_counter() - t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
