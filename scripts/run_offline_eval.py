"""JDsearch 离线评估 v1：基线 vs 本方案（品类冷启动）。

协议（对所有方法**完全一致**，保证公平）：
- 拟合数据 = ``shoprec_jd`` 全量 behavior **挖掉**评估用户被留出的尾部事件
  （即模型可用其余 2657 万条交互学热度/类目分布，而评估只在 2 万用户上做）
- 评估用户 = ``build_eval_split.py`` 采样出的 2 万用户，leave-last-5-out
- 候选池 = 每个用户**同一份**（品类内热门 ∪ 全局热门），所有方法在同一池上排序
- 指标 = Recall@K / NDCG@K / 品类多样性@K；并单独报告**目标品类簇**（冷区）表现

方法：
- ``pop``        全局热度排序（最朴素基线）
- ``cat_pop``    品类内热度排序（不做多样性控制）
- ``ours``       品类内召回 + 跨品类辅助召回 → ``transfer.fuse`` 不确定性加权融合
                 → ``diversity.rerank`` 可控品类多样性（α 可调）

⚠️ 候选池决定了 Recall 上限，脚本会**如实输出"池内召回率"（ceiling）**，避免夸大。
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
from src.models.diversity import rerank  # noqa: E402
from src.models.transfer import fuse  # noqa: E402

logger = logging.getLogger("run_offline_eval")

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"
# 目标品类簇：叶子类交互量 [20,100)（实测标定，见交接文档）
COLD_LO, COLD_HI = 20, 100


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JDsearch 离线评估 v1")
    p.add_argument("--split-dir", default=None)
    p.add_argument("--per-cat", type=int, default=50, help="每用户每类目取多少热门商品")
    p.add_argument("--pool-cap", type=int, default=300, help="候选池上限")
    p.add_argument("--k", type=int, nargs="+", default=[10, 20])
    p.add_argument("--alpha", type=float, default=0.3, help="可控品类多样性系数")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_item_category() -> pd.Series:
    """item_id -> cate_id_4（叶子类）"""
    conn = pymysql.connect(
        host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
        password=settings.DB_PASSWORD, database=settings.DB_NAME,
        charset="utf8mb4", connect_timeout=10,
    )
    df = pd.read_sql(
        "SELECT item_id, category_id FROM item WHERE category_id <> 0", conn
    )
    conn.close()
    return df.set_index("item_id")["category_id"]


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()
    split = Path(args.split_dir) if args.split_dir else SPLIT_DIR

    train = pd.read_parquet(split / "train.parquet")
    test = pd.read_parquet(split / "test.parquet")
    meta = json.loads((split / "meta.json").read_text(encoding="utf-8"))
    logger.info("评估集: 用户 %d / 训练 %d / 测试 %d", meta["n_users"], meta["train_rows"], meta["test_rows"])

    item2cat = load_item_category()
    logger.info("item→类目映射: %d", len(item2cat))

    # ---- 拟合用统计：全量 behavior 挖掉评估用户的留出事件（带缓存，重跑只付一次代价）----
    cache = split / "stats_cache.parquet"
    if cache.exists():
        logger.info("载入统计缓存 %s", cache)
        stats = pd.read_parquet(cache)
        pop_df = stats[stats["kind"] == "pop"]
        cat_pop_df = stats[stats["kind"] == "cat_item"]
        cate_total = stats[stats["kind"] == "cat"][
            ["cid", "n"]
        ].drop_duplicates("cid")
    else:
        conn = pymysql.connect(
            host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
            password=settings.DB_PASSWORD, database=settings.DB_NAME,
            charset="utf8mb4", connect_timeout=10,
        )
        eval_users = [int(u) for u in test["user_id"].unique()]
        logger.info("统计全量热度（挖除 %d 个评估用户的全部事件）…", len(eval_users))
        with conn.cursor() as c0:  # 临时表做排除（NOT IN 两万字面量会让 MySQL 退化）
            c0.execute("CREATE TEMPORARY TABLE eval_users (user_id BIGINT PRIMARY KEY)")
            c0.executemany("INSERT INTO eval_users (user_id) VALUES (%s)", [(u,) for u in eval_users])
            conn.commit()
        EXCL = "b.user_id NOT IN (SELECT user_id FROM eval_users)"

        pop_df = pd.read_sql(
            f"SELECT b.item_id, COUNT(*) AS n FROM behavior b WHERE {EXCL} GROUP BY b.item_id",
            conn,
        )
        # 每类目的 top-N 热门商品：把 top-N 下推到 SQL（否则要回传上千万行）
        t = time.perf_counter()
        cat_pop_df = pd.read_sql(
            f"""SELECT cid, item_id, n FROM (
                  SELECT i.category_id AS cid, b.item_id AS item_id, COUNT(*) AS n,
                         ROW_NUMBER() OVER (PARTITION BY i.category_id
                                            ORDER BY COUNT(*) DESC) AS rn
                  FROM behavior b JOIN item i ON b.item_id = i.item_id
                  WHERE {EXCL} AND i.category_id <> 0
                  GROUP BY i.category_id, b.item_id
                ) t WHERE rn <= {args.per_cat}""",
            conn,
        )
        logger.info("类目 top-%d 查询完成（%.0fs）", args.per_cat, time.perf_counter() - t)
        cate_total = pd.read_sql(
            f"""SELECT i.category_id AS cid, COUNT(*) AS n FROM behavior b
                JOIN item i ON b.item_id = i.item_id
                WHERE {EXCL} AND i.category_id <> 0 GROUP BY i.category_id""",
            conn,
        )
        conn.close()

        pd.concat([
            pop_df.assign(kind="pop", cid=0),
            cat_pop_df.assign(kind="cat_item"),
            cate_total.assign(kind="cat", item_id=0),
        ], ignore_index=True).to_parquet(cache, index=False)
        logger.info("统计已缓存到 %s", cache)
    logger.info("统计就绪（%.0fs）", time.perf_counter() - t0)

    pop = pop_df.set_index("item_id")["n"]
    # 每个类目内的商品按热度排序
    cat_items = cat_pop_df.sort_values(["cid", "n"], ascending=[True, False])
    cat_top: dict[int, list[int]] = {
        int(c): g["item_id"].tolist()[: args.per_cat] for c, g in cat_items.groupby("cid")
    }
    # 叶子类交互量 -> 是否属于冷区（目标品类簇）
    cate_n = cate_total.set_index("cid")["n"]
    cold_cats = set(cate_n[(cate_n >= COLD_LO) & (cate_n < COLD_HI)].index.tolist())
    logger.info("目标品类簇（叶子类交互量 [%d,%d)）: %d 个", COLD_LO, COLD_HI, len(cold_cats))

    global_top = pop.sort_values(ascending=False).index.tolist()
    # 类目内排名（与用户无关，只算一次）
    cat_rank = {c: r for c, lst in cat_top.items() for r, it in enumerate(lst)}

    # ---- 逐用户构建候选池并排序 ----
    train_by_user = train.groupby("user_id")["item_id"].apply(list).to_dict()
    test_by_user = test.groupby("user_id")["item_id"].apply(set).to_dict()
    user_cats = {
        u: {int(item2cat[i]) for i in items if i in item2cat.index}
        for u, items in train_by_user.items()
    }

    K = max(args.k)
    results: dict[str, dict] = {}
    hits: dict[str, list] = {m: [] for m in ("pop", "cat_pop", "ours")}
    ndcgs: dict[str, list] = {m: [] for m in ("pop", "cat_pop", "ours")}
    divs: dict[str, list] = {m: [] for m in ("pop", "cat_pop", "ours")}
    cold_hits: dict[str, list] = {m: [] for m in ("pop", "cat_pop", "ours")}
    ceiling: list[float] = []
    pool_sizes: list[int] = []

    users = list(test_by_user.keys())
    for n_done, u in enumerate(users, 1):
        if n_done % 5000 == 0:
            logger.info("  已处理 %d/%d 用户（%.0fs）", n_done, len(users), time.perf_counter() - t0)
        truth = test_by_user[u]
        cats = user_cats.get(u, set())

        # 候选池：用户历史类目内热门 + 全局热门（所有方法共用）
        pool: list[int] = []
        for c in cats:
            pool.extend(cat_top.get(c, ()))
        pool.extend(global_top[: args.pool_cap])
        # 去重并排除训练中已交互过的商品
        seen_hist = set(train_by_user.get(u, ()))
        uniq: list[int] = []
        mark = set()
        for i in pool:
            if i not in mark and i not in seen_hist:
                mark.add(i)
                uniq.append(i)
        pool = uniq[: args.pool_cap]
        pool_sizes.append(len(pool))
        if not pool:
            continue

        ceil_hits = len(truth & set(pool))
        ceiling.append(ceil_hits / max(1, len(truth)))

        pc = [int(item2cat.get(i, 0)) for i in pool]
        g_score = [float(pop.get(i, 0)) for i in pool]
        # 品类内分数：同类目内的相对热度排名（避免与全局热度量纲失衡）
        c_score = [1.0 / (1.0 + cat_rank.get(c, 10**6)) for c in pc]

        # 三个方法的排序：统一返回 item_id 列表（rerank 返回的就是 item_id）
        orderings = {
            "pop": [pool[j] for j in sorted(range(len(pool)), key=lambda j: -g_score[j])],
            "cat_pop": [pool[j] for j in sorted(range(len(pool)), key=lambda j: -c_score[j])],
        }
        # ours：品类内(tgt) 与 跨品类辅助(aux) 双路 → fuse → α 多样性重排
        coldness = [1.0 if c in cold_cats else 0.0 for c in pc]
        conf = [0.25 + 0.6 * cd for cd in coldness]  # 冷类目置信度低 → 更依赖辅助路
        fused = fuse(aux=g_score, tgt=c_score, confs=conf)
        trio = [(pool[j], pc[j], fused[j]) for j in range(len(pool))]
        orderings["ours"] = rerank(trio, alpha=args.alpha, k=K)

        for m, order in orderings.items():
            top = order[:K]
            hit = 1 if truth & set(top) else 0
            hits[m].append(hit)
            # NDCG@K（单命中近似：命中位置越靠前越好）
            rank_pos = next((r for r, it in enumerate(top, 1) if it in truth), None)
            ndcgs[m].append(1.0 / np.log2(rank_pos + 1) if rank_pos else 0.0)
            # 品类多样性@K：top-K 内去重类目占比
            top_cats = [int(item2cat.get(i, 0)) for i in top]
            divs[m].append(len(set(top_cats)) / max(1, len(top_cats)))
            # 冷区命中：真值里属于冷类目的商品是否进了 top-K
            truth_cold = {i for i in truth if int(item2cat.get(i, 0)) in cold_cats}
            if truth_cold:
                cold_hits[m].append(1 if truth_cold & set(top) else 0)

    # ---- 汇总 ----
    def _mean(v):
        return float(np.mean(v)) if len(v) else 0.0

    report = {
        "protocol": {
            "n_users": meta["n_users"], "holdout": meta["holdout"],
            "per_cat": args.per_cat, "pool_cap": args.pool_cap, "alpha": args.alpha,
            "cold_category_range": [COLD_LO, COLD_HI], "n_cold_categories": len(cold_cats),
        },
        "pool": {
            "mean_size": round(_mean(pool_sizes), 1),
            "ceiling_recall": round(_mean(ceiling), 4),
        },
        "metrics": {},
    }
    for m in ("pop", "cat_pop", "ours"):
        report["metrics"][m] = {
            f"recall@{k}": round(_mean(hits[m]), 4) for k in args.k
        }
        report["metrics"][m].update({
            f"ndcg@{k}": round(_mean(ndcgs[m]), 4) for k in args.k
        })
        report["metrics"][m]["diversity@10"] = round(_mean(divs[m]), 4)
        report["metrics"][m]["cold_category_hit_rate"] = round(_mean(cold_hits[m]), 4)

    out = Path(args.out) if args.out else split / "eval_v1.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"候选池: 平均 {_mean(pool_sizes):.0f} 个 / 用户，池内召回率 ceiling = {_mean(ceiling):.1%}")
    print(f"目标品类簇（冷区）: {len(cold_cats)} 个类目")
    hdr = f"{'方法':<10}" + "".join(f"{'R@'+str(k):>9}{'N@'+str(k):>9}" for k in args.k) + f"{'Div@10':>9}{'ColdHit':>9}"
    print(hdr)
    for m in ("pop", "cat_pop", "ours"):
        d = report["metrics"][m]
        row = f"{m:<10}" + "".join(f"{d['recall@'+str(k)]:>9.4f}{d['ndcg@'+str(k)]:>9.4f}" for k in args.k)
        print(row + f"{d['diversity@10']:>9.4f}{d['cold_category_hit_rate']:>9.4f}")
    print(f"\n结果已写入: {out}")
    print(f"总耗时: {time.perf_counter() - t0:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
