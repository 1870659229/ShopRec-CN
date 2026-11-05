"""α 扫描：画"召回–多样性权衡曲线"（创新点① 的正确论证方式）。

对 α ∈ [0.0, 0.1, ..., 1.0] 在**全量 2 万评估用户**上跑 rerank，
统计 Recall@10 / NDCG@10 / Diversity@10，画权衡曲线图。

复用 run_offline_eval_v2 的候选池构建逻辑（CF ∪ 品类热门 ∪ 全局热门），
但只跑 `ours` 方法（含 fuse），对每个 α 值跑一遍 rerank。

输出：
- alpha_scan.json（数值结果）
- alpha_scan_tradeoff.pdf（Recall@10 vs Diversity@10 权衡曲线）
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体设置（避免 STIXGeneral 替换乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src.eval.user_cf import UserCF, rank_normalize  # noqa: E402
from src.models.diversity import rerank  # noqa: E402
from src.models.transfer import fuse  # noqa: E402

logger = logging.getLogger("alpha_scan")

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"
COLD_LO, COLD_HI = 20, 100
# 加密 α 值以展示有效调节范围（0~0.2 区间）
ALPHA_VALUES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

# 学术作图常量
FIG_W_INCH = 5.77  # 学术图宽
FIG_H_INCH = 4.0
FONT_SIZE = 9
DPI = 300

# 低饱和配色
COLOR_OURS = "#2E86C1"   # 蓝
COLOR_CF = "#E74C3C"     # 红（CF 基线）
COLOR_POP = "#7F8C8D"    # 灰


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
    item2cat_df = pd.read_sql("SELECT item_id, category_id FROM item WHERE category_id<>0", conn)
    conn.close()
    cat_of = {int(i): int(c) for i, c in zip(item2cat_df["item_id"], item2cat_df["category_id"])}
    del item2cat_df

    # ---- UserCF ----
    train_by_user = {int(u): [int(x) for x in g] for u, g in train.groupby("user_id")["item_id"]}
    eval_users = [int(u) for u in test["user_id"].unique()]
    pairs_path = SPLIT_DIR / "cf_pairs_2000.parquet"
    pdf = pd.read_parquet(pairs_path, columns=["item_id", "user_id"])
    cf = UserCF(pdf["item_id"].to_numpy(np.int64), pdf["user_id"].to_numpy(np.int64), max_item_users=2000)
    del pdf
    logger.info("UserCF 就绪 (%.0fs)", time.perf_counter() - t0)

    test_by_user = test.groupby("user_id")["item_id"].apply(set).to_dict()
    users = sorted(test_by_user)

    # ---- 参数（与 v2_full 一致）----
    CF_ITEMS = 1200
    PER_CAT = 60
    GLOBAL_TOP = 300
    POOL_CAP = 2000
    K = 10

    # ---- 预构建每个用户的候选池和融合分数（只算一次，可缓存）----
    cache_path = SPLIT_DIR / "alpha_scan_pool_cache.parquet"
    if cache_path.exists():
        logger.info("载入候选池缓存 %s", cache_path)
        cache_df = pd.read_parquet(cache_path)
        # 重建 user_data
        user_data = []
        for uid, group in cache_df.groupby("uid", sort=False):
            truth = set(map(int, group["truth"].iloc[0].split(","))) if group["truth"].iloc[0] else set()
            truth_cold = set(map(int, group["truth_cold"].iloc[0].split(","))) if group["truth_cold"].iloc[0] else set()
            trio = [(int(r["item_id"]), int(r["cate"]), float(r["fused"])) for _, r in group.iterrows()]
            user_data.append({"trio": trio, "truth": truth, "truth_cold": truth_cold})
        logger.info("缓存载入完成: %d 用户 (%.0fs)", len(user_data), time.perf_counter() - t0)
    else:
        logger.info("构建候选池与融合分数（%d 用户）…", len(users))
        user_data: list[dict] = []  # 每个用户: {pool, pc, fused, truth, cold_truth}
    for n_done, u in enumerate(users, 1):
        if n_done % 5000 == 0:
            logger.info("  池构建 %d/%d (%.0fs)", n_done, len(users), time.perf_counter() - t0)
        truth = test_by_user[u]
        hist = train_by_user.get(u, [])
        hist_set = set(hist)
        ucats = {cat_of[i] for i in hist if i in cat_of}

        cf_items, cf_scores = cf.recommend(u, hist, top_items=CF_ITEMS)
        cf_map = dict(zip(cf_items.tolist(), cf_scores.tolist()))

        pool: list[int] = []
        mark = set()
        for i in cf_items.tolist():
            if i not in hist_set and i not in mark:
                mark.add(i); pool.append(int(i))
        for c in ucats:
            for i in cat_top.get(c, ())[:PER_CAT]:
                if i not in hist_set and i not in mark:
                    mark.add(i); pool.append(int(i))
        for i in global_top[:GLOBAL_TOP]:
            if i not in hist_set and i not in mark:
                mark.add(i); pool.append(int(i))
        pool = pool[:POOL_CAP]
        if not pool:
            continue

        pc = [cat_of.get(i, 0) for i in pool]
        raw_pop = np.array([float(pop.get(i, 0)) for i in pool])
        raw_cf = np.array([float(cf_map.get(i, 0.0)) for i in pool])
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

        s_pop = rank_normalize(raw_pop)
        s_cf = rank_normalize(raw_cf) if raw_cf.max() > 0 else raw_cf
        s_cat = rank_normalize(raw_cat)

        aux = 0.5 * s_pop + 0.5 * s_cf
        tgt = s_cat
        coldness = np.array([1.0 if c in cold_cats else 0.0 for c in pc])
        conf = list(0.25 + 0.6 * coldness)
        fused = np.array(fuse(aux=aux.tolist(), tgt=tgt.tolist(), confs=conf))
        trio = [(pool[j], pc[j], float(fused[j])) for j in range(len(pool))]

        truth_cold = {i for i in truth if cat_of.get(i, 0) in cold_cats}
        user_data.append({
            "trio": trio,
            "truth": truth,
            "truth_cold": truth_cold,
        })

    logger.info("候选池构建完成: %d 用户有有效池 (%.0fs)", len(user_data), time.perf_counter() - t0)

    # ---- 缓存候选池到 parquet（下次跑直接加载）----
    cache_rows = []
    for uid, ud in enumerate(user_data):
        truth_str = ",".join(str(i) for i in ud["truth"]) if ud["truth"] else ""
        cold_str = ",".join(str(i) for i in ud["truth_cold"]) if ud["truth_cold"] else ""
        for item_id, cate, fused in ud["trio"]:
            cache_rows.append({"uid": uid, "item_id": item_id, "cate": cate, "fused": fused, "truth": truth_str, "truth_cold": cold_str})
    cache_df = pd.DataFrame(cache_rows)
    cache_df.to_parquet(cache_path, index=False)
    logger.info("候选池已缓存: %s", cache_path)

    # ---- 对每个 α 跑 rerank ----
    results = []
    for alpha in ALPHA_VALUES:
        t_alpha = time.perf_counter()
        hits = []
        ndcgs = []
        divs = []
        cold_hits = []
        for ud in user_data:
            top = rerank(ud["trio"], alpha=alpha, k=K)
            truth = ud["truth"]
            hits.append(1 if truth & set(top) else 0)
            pos = next((r for r, it in enumerate(top, 1) if it in truth), None)
            ndcgs.append(1.0 / np.log2(pos + 1) if pos else 0.0)
            tcs = [cat_of.get(i, 0) for i in top]
            divs.append(len(set(tcs)) / max(1, len(tcs)))
            if ud["truth_cold"]:
                cold_hits.append(1 if ud["truth_cold"] & set(top) else 0)

        r10 = float(np.mean(hits)) if hits else 0.0
        n10 = float(np.mean(ndcgs)) if ndcgs else 0.0
        d10 = float(np.mean(divs)) if divs else 0.0
        ch = float(np.mean(cold_hits)) if cold_hits else 0.0
        elapsed = time.perf_counter() - t_alpha
        logger.info("α=%.1f: R@10=%.4f N@10=%.4f Div@10=%.4f ColdHit=%.4f (%.1fs)", alpha, r10, n10, d10, ch, elapsed)
        results.append({
            "alpha": alpha,
            "recall@10": round(r10, 4),
            "ndcg@10": round(n10, 4),
            "diversity@10": round(d10, 4),
            "cold_hit_rate": round(ch, 4),
        })

    # ---- 基线参照（从 eval_v2_full.json 读）----
    baseline_path = SPLIT_DIR / "eval_v2_full.json"
    baselines = {}
    if baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        for m in ("cf", "pop"):
            if m in baseline_data.get("metrics", {}):
                baselines[m] = baseline_data["metrics"][m]

    # ---- 保存 JSON ----
    report = {
        "protocol": {
            "n_users": len(user_data),
            "k": K,
            "cf_items": CF_ITEMS,
            "per_cat": PER_CAT,
            "global_top": GLOBAL_TOP,
            "pool_cap": POOL_CAP,
            "alpha_values": ALPHA_VALUES,
        },
        "baselines": baselines,
        "alpha_scan": results,
    }
    out_json = SPLIT_DIR / "alpha_scan.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("JSON 报告: %s", out_json)

    # ---- 出图：Recall@10 vs Diversity@10 权衡曲线 ----
    alphas = [r["alpha"] for r in results]
    recalls = [r["recall@10"] for r in results]
    diversities = [r["diversity@10"] for r in results]

    fig, ax = plt.subplots(figsize=(FIG_W_INCH, FIG_H_INCH), dpi=DPI)

    # α 扫描曲线（Recall 用左轴，Diversity 用右轴）
    ax.plot(alphas, recalls, "o-", color=COLOR_OURS, linewidth=1.5, markersize=4, label="Recall@10")
    ax.set_xlabel(r"$\alpha$ (diversity coefficient)", fontsize=FONT_SIZE)
    ax.set_ylabel("Recall@10", fontsize=FONT_SIZE, color=COLOR_OURS)
    ax.tick_params(axis="y", labelcolor=COLOR_OURS, labelsize=FONT_SIZE)
    ax.tick_params(axis="x", labelsize=FONT_SIZE)

    ax2 = ax.twinx()
    ax2.plot(alphas, diversities, "s--", color=COLOR_CF, linewidth=1.5, markersize=4, label="Diversity@10")
    ax2.set_ylabel("Diversity@10", fontsize=FONT_SIZE, color=COLOR_CF)
    ax2.tick_params(axis="y", labelcolor=COLOR_CF, labelsize=FONT_SIZE)

    # 标注 CF 基线水平线
    if "cf" in baselines:
        cf_r = baselines["cf"].get("recall@10", 0)
        cf_d = baselines["cf"].get("diversity@10", 0)
        ax.axhline(y=cf_r, color=COLOR_CF, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(0.02, cf_r + 0.0005, f"CF baseline R@10={cf_r:.4f}", fontsize=FONT_SIZE-1, color=COLOR_CF, transform=ax.get_yaxis_transform())
        ax2.axhline(y=cf_d, color=COLOR_OURS, linestyle=":", linewidth=0.8, alpha=0.7)
        ax2.text(0.02, cf_d + 0.01, f"CF baseline Div@10={cf_d:.4f}", fontsize=FONT_SIZE-1, color=COLOR_OURS, transform=ax2.get_yaxis_transform())

    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(ALPHA_VALUES)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=FONT_SIZE-1, loc="center left")

    plt.tight_layout()
    out_pdf = SPLIT_DIR / "alpha_scan_tradeoff.pdf"
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("权衡曲线图: %s", out_pdf)

    # ---- 第二张图：Pareto 散点（Div 为 x, Recall 为 y）----
    fig2, ax3 = plt.subplots(figsize=(FIG_W_INCH, FIG_H_INCH), dpi=DPI)
    ax3.plot(diversities, recalls, "o-", color=COLOR_OURS, linewidth=1.5, markersize=5)
    for i, a in enumerate(alphas):
        ax3.annotate(f"α={a:.1f}", (diversities[i], recalls[i]), fontsize=FONT_SIZE-2,
                     textcoords="offset points", xytext=(5, 5))
    if "cf" in baselines:
        ax3.scatter([baselines["cf"]["diversity@10"]], [baselines["cf"]["recall@10"]],
                    marker="D", s=60, color=COLOR_CF, zorder=5, label="CF baseline")
    if "pop" in baselines:
        ax3.scatter([baselines["pop"]["diversity@10"]], [baselines["pop"]["recall@10"]],
                    marker="s", s=50, color=COLOR_POP, zorder=5, label="Pop baseline")
    ax3.set_xlabel("Diversity@10", fontsize=FONT_SIZE)
    ax3.set_ylabel("Recall@10", fontsize=FONT_SIZE)
    ax3.tick_params(labelsize=FONT_SIZE)
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    ax3.legend(fontsize=FONT_SIZE-1)
    plt.tight_layout()
    out_pdf2 = SPLIT_DIR / "alpha_scan_pareto.pdf"
    fig2.savefig(out_pdf2, dpi=DPI, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Pareto 图: %s", out_pdf2)

    # ---- 控制台摘要 ----
    print("=" * 72)
    print(f"α 扫描结果（{len(user_data)} 用户）")
    print(f"{'α':>5} {'R@10':>8} {'N@10':>8} {'Div@10':>8} {'ColdHit':>9}")
    for r in results:
        print(f"{r['alpha']:>5.1f} {r['recall@10']:>8.4f} {r['ndcg@10']:>8.4f} {r['diversity@10']:>8.4f} {r['cold_hit_rate']:>9.4f}")
    if "cf" in baselines:
        b = baselines["cf"]
        print(f"\nCF 基线: R@10={b['recall@10']:.4f} Div@10={b['diversity@10']:.4f}")
    print(f"\n图: {out_pdf}")
    print(f"     {out_pdf2}")
    print(f"JSON: {out_json}")
    print(f"总耗时 {time.perf_counter()-t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
