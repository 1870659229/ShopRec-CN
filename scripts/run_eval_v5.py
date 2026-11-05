"""统一口径离线评估 v5：补「分解基线」+ 统一协议（修复 P0-2 / P0-3）。

背景（两个必须修的问题）
------------------------
P0-2 缺分解基线：``diversity.rerank`` 是可套在任意打分上的通用模块。旧对比表只有
"cf 原始" 与 "pop 原始"，无法回答评委必问的「把你的多样性重排套在 CF 上是多少」
⇒ 创新点①（可控多样性）的增量无法与"rerank 模块本身"分离。
本脚本新增 **cf_rerank@α / pop_rerank@α**：把同一个 rerank 以同样的 α 套在
CF / POP 打分上，与 ours@α 在**同一协议、同一候选池**下逐 α 对比。

P0-3 口径不一致：旧 ``alpha_scan.json`` 的基线是**从 eval_v2_full.json 读入**的
（v2 协议，pool_cap=2000），而 ``eval_v4.json`` 是 v4 协议（pool_cap=3000 +
冷区注入），导致同一 CF 基线出现 0.0272 与 0.0176 两个值。
本脚本**不再跨协议引用基线**，全部方法在同一候选池、同一次运行中现算。

协议
----
与 eval_v4 完全一致（CF_ITEMS=1200 / ITEM_CF_ITEMS=200 / PER_CAT=60 /
GLOBAL_TOP=300 / COLD_INJECT_PER_CAT=12 / COLD_INJECT_CAT_CAP=40 /
POOL_CAP=3000 / COLD_BOOST=0.20），保证与 eval_v4.json 可直接互证。

输出
----
- eval_v5.json：全部方法 × 全部 α 的 R@10 / N@10 / Div@10 / ColdHit
- eval_v5_pareto.pdf / .png：帕累托前沿（x=Div@10，y=R@10），
  ours 与 cf_rerank / pop_rerank 同图对比
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src.eval.user_cf import UserCF, rank_normalize  # noqa: E402
from src.eval.light_item_cf import ItemCF  # noqa: E402
from src.models.diversity import rerank  # noqa: E402

logger = logging.getLogger("run_eval_v5")

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"
COLD_LO, COLD_HI = 20, 100

# α 扫描点：覆盖 v4 采用的工作点 0.05 及其两侧，且含 0（纯分数排序）
ALPHA_VALUES = [0.0, 0.02, 0.05, 0.1, 0.2]

FIG_W_INCH = 5.77
FIG_H_INCH = 4.0
FONT_SIZE = 9
DPI = 300
COLOR_OURS = "#2E86C1"
COLOR_CF = "#E74C3C"
COLOR_POP = "#7F8C8D"


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
        for c, g in stats[stats["kind"] == "cat_item"].sort_values(["cid", "n"], ascending=[True, False]).groupby("cid")
    }
    cate_n = stats[stats["kind"] == "cat"].set_index("cid")["n"]
    cold_cats = set(cate_n[(cate_n >= COLD_LO) & (cate_n < COLD_HI)].index.tolist())
    global_top = pop.sort_values(ascending=False).index.tolist()
    logger.info("缓存载入: 热门 %d / 类目 %d / 冷区 %d", len(pop), len(cat_top), len(cold_cats))

    conn = pymysql.connect(
        host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
        password=settings.DB_PASSWORD, database=settings.DB_NAME,
        charset="utf8mb4", connect_timeout=10,
    )
    item2cat = pd.read_sql("SELECT item_id, category_id FROM item WHERE category_id<>0", conn)
    conn.close()
    cat_of = {int(i): int(c) for i, c in zip(item2cat["item_id"], item2cat["category_id"])}
    del item2cat

    cold_items_by_cat: dict[int, list[int]] = {}
    for c in cold_cats:
        if c in cat_top:
            cold_items_by_cat[c] = cat_top[c][:20]
    logger.info("冷区类目 %d → 可注入类目 %d", len(cold_cats), len(cold_items_by_cat))

    train_by_user = {int(u): [int(x) for x in g] for u, g in train.groupby("user_id")["item_id"]}
    pdf = pd.read_parquet(SPLIT_DIR / "cf_pairs_2000.parquet", columns=["item_id", "user_id"])
    ucf = UserCF(pdf["item_id"].to_numpy(np.int64), pdf["user_id"].to_numpy(np.int64), max_item_users=2000)
    del pdf
    icf = ItemCF(ucf, top_m=15000, top_similar=20)
    logger.info("UserCF/ItemCF 就绪 (%.0fs)", time.perf_counter() - t0)

    test_by_user = test.groupby("user_id")["item_id"].apply(set).to_dict()
    users = sorted(test_by_user)
    # 冒烟测试用：EVAL_V5_MAX_USERS>0 时只跑前 N 个用户（不写正式产物路径）
    max_users = int(os.environ.get("EVAL_V5_MAX_USERS", "0"))
    if max_users:
        users = users[:max_users]
        logger.warning("冒烟模式：仅评估前 %d 个用户", len(users))

    CF_ITEMS = 1200
    ITEM_CF_ITEMS = 200
    PER_CAT = 60
    GLOBAL_TOP = 300
    COLD_INJECT_PER_CAT = 12
    COLD_INJECT_CAT_CAP = 40
    POOL_CAP = 3000
    K = 10
    COLD_BOOST = 0.20

    RAW_METHODS = ("pop", "cf")
    RERANK_METHODS = ("pop_rerank", "cf_rerank", "ours")
    method_names = list(RAW_METHODS) + [
        f"{m}@{a}" for m in RERANK_METHODS for a in ALPHA_VALUES
    ]
    hits: dict[str, list[int]] = {m: [] for m in method_names}
    ndcgs: dict[str, list[float]] = {m: [] for m in method_names}
    divs: dict[str, list[float]] = {m: [] for m in method_names}
    cold_hits: dict[str, list[int]] = {m: [] for m in method_names}
    ceiling: list[float] = []
    pool_sizes: list[int] = []
    cold_injected_total = 0

    for n_done, u in enumerate(users, 1):
        if n_done % 5000 == 0:
            logger.info("  %d/%d 用户 (%.0fs)", n_done, len(users), time.perf_counter() - t0)
        truth = test_by_user[u]
        hist = train_by_user.get(u, [])
        hist_set = set(hist)
        ucats = {cat_of[i] for i in hist if i in cat_of}

        cf_items, cf_scores = ucf.recommend(u, hist, top_items=CF_ITEMS)
        cf_map = dict(zip(cf_items.tolist(), cf_scores.tolist()))
        icf_items, icf_scores = icf.recommend(hist, top_items=ITEM_CF_ITEMS)
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
        n_cold_injected = 0
        user_cold_cats = [c for c in ucats if c in cold_cats] or list(cold_cats)[:10]
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

        s_pop = rank_normalize(raw_pop)
        s_cf = rank_normalize(raw_cf) if raw_cf.max() > 0 else raw_cf
        s_icf = rank_normalize(raw_icf) if raw_icf.max() > 0 else raw_icf
        s_cat = rank_normalize(raw_cat)

        coldness = np.array([1.0 if c in cold_cats else 0.0 for c in pc])
        aux = 0.5 * s_pop + 0.5 * s_icf
        final_score = 0.5 * s_cf + 0.3 * aux + 0.2 * s_cat + COLD_BOOST * coldness

        # 三条打分线共用同一候选池与同一品类标注，保证可比
        score_lines = {
            "pop": s_pop,
            "cf": s_cf,
            "ours": final_score,
        }
        orderings: dict[str, list[int]] = {
            "pop": [pool[j] for j in np.argsort(-s_pop)],
            "cf": [pool[j] for j in np.argsort(-s_cf)],
        }
        for m_key in ("pop", "cf", "ours"):
            trio = [(pool[j], pc[j], float(score_lines[m_key][j])) for j in range(len(pool))]
            for alpha in ALPHA_VALUES:
                orderings[f"{m_key}_rerank@{alpha}" if m_key != "ours" else f"ours@{alpha}"] = rerank(trio, alpha=alpha, k=K)

        truth_cold = {i for i in truth if cat_of.get(i, 0) in cold_cats}
        for m, order in orderings.items():
            top = list(order[:K])
            hits[m].append(1 if truth & set(top) else 0)
            pos = next((r for r, it in enumerate(top, 1) if it in truth), None)
            ndcgs[m].append(1.0 / np.log2(pos + 1) if pos else 0.0)
            tcs = [cat_of.get(i, 0) for i in top]
            divs[m].append(len(set(tcs)) / max(1, len(tcs)))
            if truth_cold:
                cold_hits[m].append(1 if truth_cold & set(top) else 0)

    def _mean(v: list) -> float:
        return float(np.mean(v)) if len(v) else 0.0

    metrics = {}
    for m in method_names:
        metrics[m] = {
            "recall@10": round(_mean(hits[m]), 4),
            "ndcg@10": round(_mean(ndcgs[m]), 4),
            "diversity@10": round(_mean(divs[m]), 4),
            "cold_category_hit_rate": round(_mean(cold_hits[m]), 4),
        }

    report = {
        "protocol": {
            "n_users": len(users), "holdout": meta["holdout"], "seed": meta["seed"],
            "cf_items": CF_ITEMS, "item_cf_items": ITEM_CF_ITEMS,
            "per_cat": PER_CAT, "global_top": GLOBAL_TOP,
            "cold_inject_per_cat": COLD_INJECT_PER_CAT,
            "pool_cap": POOL_CAP, "cold_boost": COLD_BOOST,
            "alpha_values": ALPHA_VALUES,
            "n_cold_categories": len(cold_cats),
            "n_cold_items_injected": cold_injected_total,
            "note": "全部方法在同一候选池、同一协议下现算；不再跨协议引用基线（修复 P0-3）",
        },
        "pool": {
            "mean_size": round(_mean(pool_sizes), 1),
            "ceiling_recall": round(_mean(ceiling), 4),
        },
        "metrics": metrics,
    }
    suffix = "_smoke" if max_users else ""
    out_json = SPLIT_DIR / f"eval_v5{suffix}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("JSON 报告: %s", out_json)

    # ---- 帕累托前沿：同 α 下 ours 与分解基线对比 ----
    fig, ax = plt.subplots(figsize=(FIG_W_INCH, FIG_H_INCH), dpi=DPI)
    for m_key, color, marker, label in (
        ("ours", COLOR_OURS, "o", "ours (Ours)"),
        ("cf_rerank", COLOR_CF, "s", "CF + same rerank"),
        ("pop_rerank", COLOR_POP, "^", "POP + same rerank"),
    ):
        xs = [metrics[f"{m_key}@{a}"]["diversity@10"] for a in ALPHA_VALUES]
        ys = [metrics[f"{m_key}@{a}"]["recall@10"] for a in ALPHA_VALUES]
        ax.plot(xs, ys, marker=marker, linestyle="-", color=color, linewidth=1.5,
                markersize=5, label=label)
        for x, y, a in zip(xs, ys, ALPHA_VALUES):
            ax.annotate(f"a={a}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=FONT_SIZE - 2, color=color)
    for m_key, color, marker, label in (
        ("cf", COLOR_CF, "X", "CF raw"),
        ("pop", COLOR_POP, "X", "POP raw"),
    ):
        ax.plot(metrics[m_key]["diversity@10"], metrics[m_key]["recall@10"],
                marker=marker, color=color, markersize=7, linestyle="none", label=label)

    ax.set_xlabel("Diversity@10", fontsize=FONT_SIZE)
    ax.set_ylabel("Recall@10", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=FONT_SIZE - 1, frameon=False)
    ax.set_title("Pareto frontier: accuracy vs category diversity (unified protocol)",
                 fontsize=FONT_SIZE)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SPLIT_DIR / f"eval_v5_pareto{suffix}.{ext}", dpi=DPI)
    logger.info("帕累托图: %s", SPLIT_DIR / f"eval_v5_pareto{suffix}.pdf")

    print("=" * 78)
    print(f"候选池 平均 {_mean(pool_sizes):.0f}/用户 | ceiling = {_mean(ceiling):.2%} | 冷区类目 {len(cold_cats)}")
    print(f"{'方法':<20} {'R@10':>8} {'N@10':>8} {'Div@10':>9} {'ColdHit':>9}")
    for m in method_names:
        d = metrics[m]
        print(f"{m:<20} {d['recall@10']:>8.4f} {d['ndcg@10']:>8.4f} {d['diversity@10']:>9.4f} {d['cold_category_hit_rate']:>9.4f}")
    print(f"\n结果: {out_json}    总耗时 {time.perf_counter() - t0:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
