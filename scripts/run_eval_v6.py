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
- eval_v6.json：全部方法 × 全部 α 的 R@10 / N@10 / Div@10 / ColdHit
- eval_v6_pareto.pdf / .png：帕累托前沿（x=Div@10，y=R@10），
  ours 与 cf_rerank / pop_rerank 同图对比

多行为加权（v6 相对 v5 的唯一改动）
----------------------------------
设计规格 §4.4 承诺"JDsearch 原生 4 类真实行为可直接做多行为推荐"，但旧实现
只在 ``cf_pairs`` 里保留了**隐式频次**（同一 (user,item) 的 CLICK/CART/ORD
产生多行，``UserCF.__init__`` 的 ``np.add.at`` 自然累加），**没有利用行为
意图强度的语义差异**。

本脚本把评估用户的查询历史按 ``behavior_type`` 加权后传给
``UserCF.recommend(dict)``：ORD=5 / CART=3 / FLW=2 / CLICK=1。
同一 (user,item) 多行为取 **max**（最强意图信号），与 ``set(history)`` 去重
语义一致；求和等于退回频次加权，会丢失语义增量。

**单变量纪律**：仅改 UserCF 的查询侧历史权重。不动 ``cf_pairs`` 共现计算、
不动 ItemCF（仍吃**未去重**的原始历史，见 ``ours_scorer`` 模块 docstring）、
不动召回池组装、不动 rerank。``EVAL_V6_MB=null`` 提供全 1 锚点模式，
其全部行必须与 ``eval_v5.json`` 逐位一致（Δ=0.00）——已实测通过。
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
from src.eval.user_cf import UserCF  # noqa: E402
from src.eval.light_item_cf import ItemCF  # noqa: E402
from src.models.diversity import (  # noqa: E402
    encode_categories,
    prepare,
    rerank_prepared,
)
# 候选池组装与 ours 打分已抽到公共模块，在线服务与离线评估共用同一实现，
# 保证论文指标与线上行为同源（此前二者各有一套，演示跑不出论文数字）。
from src.models.ours_scorer import (  # noqa: E402
    CF_ITEMS,
    COLD_BOOST,
    COLD_INJECT_CAT_CAP,
    COLD_INJECT_PER_CAT,
    GLOBAL_TOP,
    ITEM_CF_ITEMS,
    OursScorer,
    PER_CAT,
    POOL_CAP,
    user_categories,
)

logger = logging.getLogger("run_eval_v6")

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"
COLD_LO, COLD_HI = 20, 100

# 多行为意图权重：仅作用于**查询侧历史**（单变量，见模块 docstring）。
# EVAL_V6_MB=null → 全 1 锚点模式（应与 eval_v5 逐位一致）；
# EVAL_V6_MB='{"CLICK":1,"FLW":2,"CART":3,"ORD":5}' → 自定义权重。
DEFAULT_MB_WEIGHTS: dict[str, float] | None = {"CLICK": 1.0, "FLW": 2.0, "CART": 3.0, "ORD": 5.0}
_mb_env = os.environ.get("EVAL_V6_MB")
MB_WEIGHTS: dict[str, float] | None = (
    None if _mb_env == "null" else (json.loads(_mb_env) if _mb_env else DEFAULT_MB_WEIGHTS)
)

# 冷区消融开关（P0-1）：隔离「召回层注入」与「排序层加分」的各自贡献。
#   EVAL_COLD_INJECT=0  关闭冷区注入（商品进不了候选池）
#   EVAL_COLD_BOOST=0   关闭冷区加分（进池但不加权）
# 不设时取模块常量 12 / 0.20，与 eval_v5.json 逐位一致。
COLD_INJECT_PER_CAT_EVAL = int(os.environ.get("EVAL_COLD_INJECT", "12"))
COLD_BOOST_EVAL = float(os.environ.get("EVAL_COLD_BOOST", "0.20"))

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

    # 离线检查点：EVAL_USE_CKPT=1 时复用 scripts/train_offline.py 的产物，
    # 跳过「统计 parquet 解析 + MySQL 商品→类目查询(1214 万行, 35s) + CF 重建(11s)」。
    # 默认关闭以保证冻结指标走原路径；检查点带数据源指纹校验，数据换过而索引
    # 没重建时会直接报错，不会静默用旧索引。
    _sources = {
        "cf_pairs": SPLIT_DIR / "cf_pairs_2000.parquet",
        "stats_cache": SPLIT_DIR / "stats_cache.parquet",
        "train": SPLIT_DIR / "train.parquet",
    }
    ckpt = None
    if os.environ.get("EVAL_USE_CKPT") == "1":
        from src.models.checkpoint import load_checkpoint

        ckpt = load_checkpoint(SPLIT_DIR / "model_ckpt.pkl", _sources)
        logger.info("复用离线检查点: 跳过统计载入 + 商品→类目查询 + CF 重建")

    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    meta = json.loads((SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))

    if ckpt is not None:
        pop, cat_top, cat_of = ckpt.pop, ckpt.cat_top, ckpt.cat_of
        cold_cats, global_top = ckpt.cold_cats, ckpt.global_top
    else:
        stats = pd.read_parquet(SPLIT_DIR / "stats_cache.parquet")
        pop = stats[stats["kind"] == "pop"].set_index("item_id")["n"]
        cat_top = {
            int(c): g["item_id"].tolist()
            for c, g in stats[stats["kind"] == "cat_item"].sort_values(["cid", "n"], ascending=[True, False]).groupby("cid")
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
    logger.info("缓存载入: 热门 %d / 类目 %d / 冷区 %d", len(pop), len(cat_top), len(cold_cats))

    # 候选池组装与 ours 打分交给公共模块（在线服务复用同一实现）。
    # 冷区可注入商品的构造也随之移入模块内，此处不再单独维护一份。
    scorer = OursScorer(
        pop=pop, cat_top=cat_top, cat_of=cat_of,
        cold_cats=cold_cats, global_top=global_top,
        cold_inject_per_cat=COLD_INJECT_PER_CAT_EVAL,
        cold_boost=COLD_BOOST_EVAL,
    )
    logger.info(
        "打分模块就绪: 冷区类目 %d | 注入 %d/类目 | 加分 %.2f",
        len(cold_cats), COLD_INJECT_PER_CAT_EVAL, COLD_BOOST_EVAL,
    )

    # 评估用户先定：下面的历史结构只为他们构建，避免为百万级用户建两份历史
    # 顶着 ItemCF 的 1.68 GiB 分配（Windows 地址空间碎片化实测会失败）。
    test_by_user = test.groupby("user_id")["item_id"].apply(set).to_dict()
    users = sorted(test_by_user)
    # 冒烟测试用：EVAL_V5_MAX_USERS>0 时只跑前 N 个用户（不写正式产物路径）
    max_users = int(os.environ.get("EVAL_V5_MAX_USERS", "0"))
    if max_users:
        users = users[:max_users]
        logger.warning("冒烟模式：仅评估前 %d 个用户", len(users))

    tr = train[train["user_id"].isin(set(users))]
    # 原始历史：**保留同一商品的多条行为行（不去重）**。
    # ItemCF.recommend 是 `for it in history: acc[nb] += w`，重复行会被重复累加
    # （等价于隐式频次加权）。eval_v5 传的就是这个原始 list，锚点模式若在此去重，
    # 候选池组成就会漂移——实测 pool 均值 2644.0→2643.7、ours R@10 0.0162→0.0163，
    # 连 pop 的 Div@10 都跟着变。故 ItemCF / hist_set / ucats 一律用原始 list。
    train_by_user = {int(u): [int(x) for x in g] for u, g in tr.groupby("user_id")["item_id"]}
    # 查询侧加权历史（**仅 UserCF 用**）：同一 (user,item) 多行为取 max（最强意图）。
    # 单变量纪律：ItemCF、候选池组装、rerank 全部不动，
    # MB 模式相对锚点模式的唯一变量就是 UserCF 的历史权重。
    if MB_WEIGHTS is None:
        hist_w_by_user: dict[int, dict[int, float]] = {
            u: {i: 1.0 for i in items} for u, items in train_by_user.items()
        }
    else:
        wdf = tr.assign(_w=tr["behavior_type"].map(MB_WEIGHTS).fillna(1.0))
        wdf = wdf.sort_values("_w", ascending=False).drop_duplicates(
            subset=["user_id", "item_id"], keep="first"
        )
        hist_w_by_user = {
            int(u): dict(zip(g["item_id"], g["_w"]))
            for u, g in wdf.groupby("user_id")
        }
        del wdf
    del tr
    logger.info(
        "多行为权重: %s（None=全1锚点）| 原始历史 %d 用户 | 加权历史 %d 用户",
        MB_WEIGHTS, len(train_by_user), len(hist_w_by_user),
    )
    # 离线检查点：EVAL_USE_CKPT=1 时复用已训练好的 UserCF/ItemCF，省去 43s 重建。
    # 默认关闭以保证冻结指标完全走原路径；检查点由 scripts/train_offline.py 产出，
    # 且带数据源指纹校验——数据换过而索引没重建时会直接报错而不是静默用旧索引。
    if ckpt is not None:
        ucf, icf = ckpt.ucf, ckpt.icf
        logger.info("复用检查点内 CF 索引（跳过重建）")
    else:
        pdf = pd.read_parquet(SPLIT_DIR / "cf_pairs_2000.parquet", columns=["item_id", "user_id"])
        ucf = UserCF(pdf["item_id"].to_numpy(np.int64), pdf["user_id"].to_numpy(np.int64), max_item_users=2000)
        del pdf
        icf = ItemCF(ucf, top_m=15000, top_similar=20)
    logger.info("UserCF/ItemCF 就绪 (%.0fs)", time.perf_counter() - t0)

    # CF_ITEMS / ITEM_CF_ITEMS / PER_CAT / GLOBAL_TOP / 冷区注入参数 / POOL_CAP / COLD_BOOST
    # 均已移入 src.models.ours_scorer（与在线服务共用），此处只保留评估特有的 K。
    K = 10

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
        # hist 必须是**未去重**的原始历史（ItemCF 侧隐式频次加权，与 eval_v5 同源）
        hist = train_by_user.get(u, [])
        hist_w = hist_w_by_user.get(u, {})
        hist_set = set(hist)
        ucats = user_categories(hist, cat_of)

        cf_items, cf_scores = ucf.recommend(u, hist_w, top_items=CF_ITEMS)
        cf_map = dict(zip(cf_items.tolist(), cf_scores.tolist()))
        icf_items, icf_scores = icf.recommend(hist, top_items=ITEM_CF_ITEMS)
        icf_map = dict(zip(icf_items.tolist(), icf_scores.tolist()))

        pool, n_cold_injected = scorer.build_pool(
            hist_set, ucats, cf_items.tolist(), icf_items.tolist()
        )
        cold_injected_total += n_cold_injected
        pool_sizes.append(len(pool))
        if not pool:
            continue
        pset = set(pool)
        ceiling.append(len(truth & pset) / max(1, len(truth)))

        # 三条打分线共用同一候选池与同一品类标注，保证可比
        pc, score_lines = scorer.score(pool, cf_map, icf_map)
        s_pop = score_lines["pop"]
        s_cf = score_lines["cf"]
        orderings: dict[str, list[int]] = {
            "pop": [pool[j] for j in np.argsort(-s_pop)],
            "cf": [pool[j] for j in np.argsort(-s_cf)],
        }
        # 类目编码只依赖候选池、与打分无关 → 三个方法共用一次编码；
        # 每个方法只排一次序供 5 个 α 复用（rerank 提速 30.9x，见 diversity 模块 docstring）
        pool_codes, pool_n_cats = encode_categories(pc)
        pool_arr = np.asarray(pool, dtype=np.int64)
        for m_key in ("pop", "cf", "ours"):
            prep = prepare(pool_arr, pool_codes, score_lines[m_key], pool_n_cats)
            for alpha in ALPHA_VALUES:
                name = f"{m_key}_rerank@{alpha}" if m_key != "ours" else f"ours@{alpha}"
                orderings[name] = rerank_prepared(prep, alpha=alpha, k=K)

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
            "cold_inject_per_cat": COLD_INJECT_PER_CAT_EVAL,
            "pool_cap": POOL_CAP, "cold_boost": COLD_BOOST_EVAL,
            "alpha_values": ALPHA_VALUES,
            "n_cold_categories": len(cold_cats),
            "n_cold_items_injected": cold_injected_total,
            "mb_weights": MB_WEIGHTS,
            "note": "全部方法在同一候选池、同一协议下现算；不再跨协议引用基线（修复 P0-3）；"
                    "v6 唯一改动=查询侧多行为意图加权（见模块 docstring）",
        },
        "pool": {
            "mean_size": round(_mean(pool_sizes), 1),
            "ceiling_recall": round(_mean(ceiling), 4),
        },
        "metrics": metrics,
    }
    # 冒烟模式加 _smoke；锚点模式（MB_WEIGHTS=None）加 _anchor，避免覆盖加权版结果；
    # 冷区消融模式再加参数后缀，避免覆盖锚点产物。
    suffix = ("_smoke" if max_users else "") + ("" if MB_WEIGHTS is not None else "_anchor")
    if COLD_INJECT_PER_CAT_EVAL != COLD_INJECT_PER_CAT or COLD_BOOST_EVAL != COLD_BOOST:
        suffix += f"_inj{COLD_INJECT_PER_CAT_EVAL}_boost{COLD_BOOST_EVAL:g}"
    out_json = SPLIT_DIR / f"eval_v6{suffix}.json"
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
        fig.savefig(SPLIT_DIR / f"eval_v6_pareto{suffix}.{ext}", dpi=DPI)
    logger.info("帕累托图: %s", SPLIT_DIR / f"eval_v6_pareto{suffix}.pdf")

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
