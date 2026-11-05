"""重建离线评估所需的两个缓存文件（stats_cache.parquet / cf_pairs_2000.parquet）。

背景
----
2026-10-09 的目录整理为节省空间删除了 splits 下的中间缓存（596MB→20MB），
导致 ``run_eval_v3.py`` / ``alpha_scan.py`` / ``diagnose_*.py`` 全部因缺少
``stats_cache.parquet`` 与 ``cf_pairs_2000.parquet`` 而无法运行。
本脚本按原先的生成口径重建这两个文件，使评估链可再次复现。

口径（与原 run_offline_eval.py 一致）
------------------------------------
- 统计量一律**挖除评估用户的全部事件**：先在 pandas 侧加载全量交互对，再按
  user_id 过滤评估用户（避免 MySQL 侧 26.67M 行 × 两万字面量 NOT IN 退化）。
- ``pop``：全量 behavior 按 item 计数。
- ``cat_item``：每个叶子类目按交互量取前 ``--per-cat`` 个热门商品。
- ``cat``：每个叶子类目的交互总量（用于按 [20,100) 标定冷区类目）。
- ``cf_pairs_{max_item_users}``：挖除评估用户后的 (item_id, user_id) 交互对。

实现说明：类目维度统计在 **pandas 侧**完成。原实现把
``ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY COUNT(*) DESC)`` 下推到
MySQL，在 26.67M 行为 × 12.14M 商品上实测超过 28 分钟仍未返回；改为一次性拉取
交互对后在 pandas 聚合，可在数分钟内完成。

用法
----
    SHOPREC_DB_PASSWORD=<密码> python scripts/rebuild_eval_cache.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402

logger = logging.getLogger("rebuild_eval_cache")
SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"


def _connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
        password=settings.DB_PASSWORD, database=settings.DB_NAME,
        charset="utf8mb4", connect_timeout=10,
    )


def _load_pairs_excluding(eval_users: list[int]) -> pd.DataFrame:
    """一次性拉取全量交互对并在 pandas 侧剔除评估用户。"""
    conn = _connect()
    try:
        t0 = time.perf_counter()
        pairs = pd.read_sql(
            "SELECT b.item_id AS item_id, b.user_id AS user_id FROM behavior b",
            conn,
        )
        logger.info("交互对拉取 %d 行 (%.0fs)", len(pairs), time.perf_counter() - t0)
    finally:
        conn.close()
    excluded = set(eval_users)
    kept = pairs[~pairs["user_id"].isin(excluded)]
    logger.info("剔除 %d 个评估用户后剩余 %d 行", len(excluded), len(kept))
    return kept


def _load_item_category() -> pd.DataFrame:
    conn = _connect()
    try:
        return pd.read_sql("SELECT item_id, category_id FROM item WHERE category_id<>0", conn)
    finally:
        conn.close()


def rebuild_stats(per_cat: int) -> Path:
    """重建统计缓存：pop / cat_item / cat 三类统计（类目维度在 pandas 侧聚合）。"""
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    eval_users = [int(u) for u in test["user_id"].unique()]
    pairs = _load_pairs_excluding(eval_users)

    t0 = time.perf_counter()
    pop_df = pairs.groupby("item_id", as_index=False).size().rename(columns={"size": "n"})
    logger.info("商品热度 %d 行 (%.0fs)", len(pop_df), time.perf_counter() - t0)

    t0 = time.perf_counter()
    item2cat = _load_item_category()
    merged = pairs.merge(item2cat, on="item_id", how="inner")
    cat_all = merged.groupby(["category_id", "item_id"], as_index=False).size().rename(
        columns={"size": "n", "category_id": "cid"}
    )
    # 类目总量必须用**全部商品**的交互数求和（冷区按 [20,100) 标定）；
    # 若只对 top-N 求和会系统性低估，导致冷区类目集合错误。
    cate_total = cat_all.groupby("cid", as_index=False)["n"].sum()
    cat_item = (
        cat_all.sort_values(["cid", "n"], ascending=[True, False])
        .groupby("cid")
        .head(per_cat)
        .reset_index(drop=True)
    )
    logger.info("类目 top-%d %d 行 / 类目 %d 个 (%.0fs)",
                per_cat, len(cat_item), len(cate_total), time.perf_counter() - t0)

    cache = SPLIT_DIR / "stats_cache.parquet"
    pd.concat([
        pop_df.assign(kind="pop", cid=0),
        cat_item.assign(kind="cat_item"),
        cate_total.assign(kind="cat", item_id=0),
    ], ignore_index=True).to_parquet(cache, index=False)
    logger.info("统计缓存已写入 %s", cache)
    return cache


def rebuild_cf_pairs(max_item_users: int) -> Path:
    """重建 UserCF 用的交互对缓存（挖除评估用户）。"""
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    eval_users = [int(u) for u in test["user_id"].unique()]
    pdf = _load_pairs_excluding(eval_users)
    path = SPLIT_DIR / f"cf_pairs_{max_item_users}.parquet"
    pdf.to_parquet(path, index=False)
    logger.info("CF 交互对 %d 行已写入 %s", len(pdf), path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cat", type=int, default=60, help="每类目缓存多少个热门商品")
    parser.add_argument("--max-item-users", type=int, default=2000, help="交互对缓存文件名后缀")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--skip-pairs", action="store_true")
    args = parser.parse_args()

    if not args.skip_stats:
        rebuild_stats(args.per_cat)
    if not args.skip_pairs:
        rebuild_cf_pairs(args.max_item_users)
    logger.info("重建完成；meta=%s", json.loads((SPLIT_DIR / "meta.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
