"""JDsearch 离线评估 v2：在 v1 基础上补 **协同过滤召回**并修正**分数量纲**。

v1 的两个致命问题（本脚本要解决的）：
1. **候选池 ceiling 仅 1.9%** —— 纯热度召回（全局热门/品类热门）在这份极稀疏数据上
   捞不到"全球稀有但与用户相关"的商品 ⇒ 指标被池子锁死。
   ⇒ 本版加入 **user-based CF** 召回路（`src/eval/user_cf.py`）。
2. **`ours` 退化成 `pop`** —— 全局热度（可达数千）与类目内排名分（≤1）量纲差 3~4 个数量级，
   `fuse` 的结果被热度完全淹没。
   ⇒ 本版所有召回路的分数**先转成排名分（0~1）再融合**（`rank_normalize`）。

协议不变：所有方法在**同一候选池**上排序；输出池内召回率 ceiling，不夸大。
"""
from __future__ import annotations

import argparse
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
from src.models.diversity import rerank  # noqa: E402
from src.models.transfer import fuse  # noqa: E402

logger = logging.getLogger("run_offline_eval_v2")

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"
COLD_LO, COLD_HI = 20, 100


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JDsearch 离线评估 v2（含 CF 召回）")
    p.add_argument("--split-dir", default=None)
    p.add_argument("--per-cat", type=int, default=30)
    p.add_argument("--cf-items", type=int, default=100, help="CF 每用户取多少候选")
    p.add_argument("--global-top", type=int, default=100)
    p.add_argument("--pool-cap", type=int, default=300)
    p.add_argument("--k", type=int, nargs="+", default=[10, 20])
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--max-item-users", type=int, default=2000)
    p.add_argument("--cf-scope", choices=["full", "cohort"], default="full",
                   help="CF 建索引范围：full=全量交互（默认），cohort=仅评估队列")
    p.add_argument("--limit-users", type=int, default=None, help="只跑前 N 个用户（调试）")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()
    split = Path(args.split_dir) if args.split_dir else SPLIT_DIR

    train = pd.read_parquet(split / "train.parquet")
    test = pd.read_parquet(split / "test.parquet")
    meta = json.loads((split / "meta.json").read_text(encoding="utf-8"))

    # ---- 统计缓存（v1 已生成）----
    cache = split / "stats_cache.parquet"
    if not cache.exists():
        raise SystemExit(f"缺少统计缓存 {cache}，请先跑 run_offline_eval.py")
    stats = pd.read_parquet(cache)
    pop = stats[stats["kind"] == "pop"].set_index("item_id")["n"]
    cat_top = {
        int(c): g["item_id"].tolist()
        for c, g in stats[stats["kind"] == "cat_item"].sort_values(["cid", "n"], ascending=[True, False]).groupby("cid")
    }
    cate_n = stats[stats["kind"] == "cat"].set_index("cid")["n"]
    cold_cats = set(cate_n[(cate_n >= COLD_LO) & (cate_n < COLD_HI)].index.tolist())
    global_top = pop.sort_values(ascending=False).index.tolist()
    logger.info("缓存载入：热门商品 %d / 类目 %d / 冷区类目 %d", len(pop), len(cat_top), len(cold_cats))

    # ---- item → 叶子类 ----
    conn = pymysql.connect(
        host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
        password=settings.DB_PASSWORD, database=settings.DB_NAME,
        charset="utf8mb4", connect_timeout=10,
    )
    item2cat = pd.read_sql("SELECT item_id, category_id FROM item WHERE category_id<>0", conn)
    item2cat = item2cat.set_index("item_id")["category_id"]
    cat_of = {int(i): int(c) for i, c in item2cat.items()}

    # ---- UserCF ----
    train_by_user = {int(u): [int(x) for x in g] for u, g in train.groupby("user_id")["item_id"]}
    eval_users = [int(u) for u in test["user_id"].unique()]

    if args.cf_scope == "full":
        # CF 建在全量交互上（挖掉评估用户的全部事件），否则 ceiling 上不去：
        # 用户的后继商品常常只出现在"别人"的历史里
        pairs_cache = split / f"cf_pairs_{args.max_item_users}.parquet"
        if pairs_cache.exists():
            logger.info("载入 CF 交互对缓存 %s", pairs_cache)
            pdf = pd.read_parquet(pairs_cache, columns=["item_id", "user_id"])
        else:
            logger.info("拉取全量交互对用于 CF（挖除 %d 个评估用户）…", len(eval_users))
            with conn.cursor() as c0:
                c0.execute("CREATE TEMPORARY TABLE eval_users (user_id BIGINT PRIMARY KEY)")
                c0.executemany("INSERT INTO eval_users (user_id) VALUES (%s)", [(u,) for u in eval_users])
                conn.commit()
            pdf = pd.read_sql(
                "SELECT b.item_id AS item_id, b.user_id AS user_id FROM behavior b "
                "WHERE b.user_id NOT IN (SELECT user_id FROM eval_users)",
                conn,
            )
            pdf.to_parquet(pairs_cache, index=False)
            logger.info("CF 交互对 %d 已缓存到 %s", len(pdf), pairs_cache)
        cf = UserCF(pdf["item_id"].to_numpy(np.int64), pdf["user_id"].to_numpy(np.int64),
                    max_item_users=args.max_item_users)
        del pdf
    else:
        cf = UserCF.from_user_items(train_by_user, max_item_users=args.max_item_users)
    conn.close()
    logger.info("UserCF 就绪（%.0fs）", time.perf_counter() - t0)

    test_by_user = test.groupby("user_id")["item_id"].apply(set).to_dict()
    users = sorted(test_by_user)
    if args.limit_users:
        users = users[: args.limit_users]

    K = max(args.k)
    METHODS = ("pop", "cat_pop", "cf", "ours")
    hits = {m: [] for m in METHODS}
    ndcgs = {m: [] for m in METHODS}
    divs = {m: [] for m in METHODS}
    cold_hits = {m: [] for m in METHODS}
    ceiling = []
    pool_sizes = []
    cf_recall_at_pool = []  # CF 单独贡献了多少 ceiling

    for n_done, u in enumerate(users, 1):
        if n_done % 2000 == 0:
            logger.info("  %d/%d 用户（%.0fs）", n_done, len(users), time.perf_counter() - t0)
        truth = test_by_user[u]
        hist = train_by_user.get(u, [])
        hist_set = set(hist)
        ucats = {cat_of[i] for i in hist if i in cat_of}

        cf_items, cf_scores = cf.recommend(u, hist, top_items=args.cf_items)
        cf_map = dict(zip(cf_items.tolist(), cf_scores.tolist()))

        # ---- 候选池：CF ∪ 品类内热门 ∪ 全局热门（所有方法共用）----
        pool: list[int] = []
        mark = set()
        for i in cf_items.tolist():
            if i not in hist_set and i not in mark:
                mark.add(i); pool.append(int(i))
        for c in ucats:
            for i in cat_top.get(c, ())[: args.per_cat]:
                if i not in hist_set and i not in mark:
                    mark.add(i); pool.append(int(i))
        for i in global_top[: args.global_top]:
            if i not in hist_set and i not in mark:
                mark.add(i); pool.append(int(i))
        pool = pool[: args.pool_cap]
        pool_sizes.append(len(pool))
        if not pool:
            continue
        pset = set(pool)
        ceiling.append(len(truth & pset) / max(1, len(truth)))
        cf_recall_at_pool.append(len(truth & set(cf_map)) / max(1, len(truth)))

        pc = [cat_of.get(i, 0) for i in pool]
        raw_pop = np.array([float(pop.get(i, 0)) for i in pool])
        raw_cf = np.array([float(cf_map.get(i, 0.0)) for i in pool])
        # 品类内分：同品类内按全局热度的排名（0~1），无类目给 0
        cat_rank: dict[int, list[int]] = {}
        for i, c in zip(pool, pc):
            cat_rank.setdefault(c, []).append(i)
        cat_pos: dict[int, float] = {}
        for c, items in cat_rank.items():
            if c not in cat_top:
                continue
            pos = {it: r for r, it in enumerate(cat_top[c])}
            for it in items:
                if it in pos:
                    cat_pos[it] = 1.0 / (1.0 + pos[it])
        raw_cat = np.array([cat_pos.get(i, 0.0) for i in pool])

        # ---- 统一量纲：全部转排名分 ----
        s_pop = rank_normalize(raw_pop)
        s_cf = rank_normalize(raw_cf) if raw_cf.max() > 0 else raw_cf
        s_cat = rank_normalize(raw_cat)

        orderings = {
            "pop": [pool[j] for j in np.argsort(-s_pop)],
            "cat_pop": [pool[j] for j in np.argsort(-s_cat)],
            "cf": [pool[j] for j in np.argsort(-s_cf)],
        }
        # ours：辅助路(跨域/全局 + CF) 与 目标路(品类内) 双路 → 不确定性加权融合 → α 多样性重排
        aux = 0.5 * s_pop + 0.5 * s_cf
        tgt = s_cat
        coldness = np.array([1.0 if c in cold_cats else 0.0 for c in pc])
        conf = list(0.25 + 0.6 * coldness)   # 冷类目置信度低 → 更依赖辅助路
        fused = np.array(fuse(aux=aux.tolist(), tgt=tgt.tolist(), confs=conf))
        trio = [(pool[j], pc[j], float(fused[j])) for j in range(len(pool))]
        orderings["ours"] = rerank(trio, alpha=args.alpha, k=K)

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
            "cf_items": args.cf_items, "per_cat": args.per_cat,
            "global_top": args.global_top, "pool_cap": args.pool_cap, "alpha": args.alpha,
            "n_cold_categories": len(cold_cats),
        },
        "pool": {
            "mean_size": round(_mean(pool_sizes), 1),
            "ceiling_recall": round(_mean(ceiling), 4),
            "cf_only_recall": round(_mean(cf_recall_at_pool), 4),
        },
        "metrics": {},
    }
    for m in METHODS:
        d = {f"recall@{k}": round(_mean(hits[m]), 4) for k in args.k}
        d.update({f"ndcg@{k}": round(_mean(ndcgs[m]), 4) for k in args.k})
        d["diversity@10"] = round(_mean(divs[m]), 4)
        d["cold_category_hit_rate"] = round(_mean(cold_hits[m]), 4)
        report["metrics"][m] = d

    out = Path(args.out) if args.out else split / "eval_v2.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"候选池 平均 {_mean(pool_sizes):.0f}/用户 | ceiling = {_mean(ceiling):.2%} | CF 单独召回 = {_mean(cf_recall_at_pool):.2%}")
    print(f"目标品类簇（[20,100)）: {len(cold_cats)} 个类目")
    print(f"{'方法':<9}" + "".join(f"{'R@'+str(k):>8}{'N@'+str(k):>8}" for k in args.k) + f"{'Div@10':>9}{'ColdHit':>9}")
    for m in METHODS:
        d = report["metrics"][m]
        row = f"{m:<9}" + "".join(f"{d['recall@'+str(k)]:>8.4f}{d['ndcg@'+str(k)]:>8.4f}" for k in args.k)
        print(row + f"{d['diversity@10']:>9.4f}{d['cold_category_hit_rate']:>9.4f}")
    print(f"\n结果: {out}    总耗时 {time.perf_counter() - t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
