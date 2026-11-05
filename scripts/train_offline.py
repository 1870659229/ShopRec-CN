"""离线训练：构建 UserCF / ItemCF 与静态统计，落盘为可复用检查点。

用法::

    python scripts/train_offline.py              # 训练并写检查点
    python scripts/train_offline.py --check      # 只检查现有检查点是否过期
    python scripts/train_offline.py --force      # 忽略现有检查点，强制重建

产出
----
``data/splits/jdsearch_eval/model_ckpt.pkl``（+ 同名 ``.meta.json``）

后续评估脚本与在线服务可直接 ``load_checkpoint`` 而不再重建索引——
重建实测 43 秒 / 次，是"每次运行都要等"的主要来源之一。

关于"继续训练"
--------------
协同过滤是无参数模型，没有可继续下降的损失，因此不存在"加载旧权重接着训"。
本脚本的语义是：**数据与参数都没变就复用，变了就从头重建**（见
``src/models/checkpoint.py`` 的说明）。这是 CF 的固有性质，不做增量更新的假承诺。
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
from src.eval.light_item_cf import ItemCF  # noqa: E402
from src.eval.user_cf import UserCF  # noqa: E402
from src.models.checkpoint import (  # noqa: E402
    Bundle,
    is_fresh,
    load_checkpoint,
    save_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_offline")

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"
CKPT_PATH = SPLIT_DIR / "model_ckpt.pkl"
COLD_LO, COLD_HI = 20, 100

# 构建参数：改动任一都会让旧检查点失效（写入 meta.params 供比对）
PARAMS = {
    "max_item_users": 2000,
    "top_m": 15000,
    "top_similar": 20,
    "cold_lo": COLD_LO,
    "cold_hi": COLD_HI,
    "db_name": settings.DB_NAME,
}


def _sources() -> dict[str, Path]:
    return {
        "cf_pairs": SPLIT_DIR / "cf_pairs_2000.parquet",
        "stats_cache": SPLIT_DIR / "stats_cache.parquet",
        "train": SPLIT_DIR / "train.parquet",
    }


def _build_cat_of() -> dict[int, int]:
    """商品 → 叶子类目。来自 MySQL ``item`` 表（无法做文件指纹，见 meta 说明）。"""
    conn = pymysql.connect(
        host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
        password=settings.DB_PASSWORD, database=settings.DB_NAME,
        charset="utf8mb4", connect_timeout=10,
    )
    try:
        item2cat = pd.read_sql(
            "SELECT item_id, category_id FROM item WHERE category_id<>0", conn
        )
    finally:
        conn.close()
    return {int(i): int(c) for i, c in zip(item2cat["item_id"], item2cat["category_id"])}


def train() -> Bundle:
    t0 = time.perf_counter()
    stats = pd.read_parquet(SPLIT_DIR / "stats_cache.parquet")
    pop = stats[stats["kind"] == "pop"].set_index("item_id")["n"]
    cat_top = {
        int(c): g["item_id"].tolist()
        for c, g in stats[stats["kind"] == "cat_item"]
        .sort_values(["cid", "n"], ascending=[True, False])
        .groupby("cid")
    }
    cate_n = stats[stats["kind"] == "cat"].set_index("cid")["n"]
    cold_cats = set(cate_n[(cate_n >= COLD_LO) & (cate_n < COLD_HI)].index.tolist())
    global_top = pop.sort_values(ascending=False).index.tolist()
    logger.info("统计载入: 热门 %d / 类目 %d / 冷区 %d", len(pop), len(cat_top), len(cold_cats))

    cat_of = _build_cat_of()
    logger.info("商品→类目 %d 条 (%.0fs)", len(cat_of), time.perf_counter() - t0)

    pdf = pd.read_parquet(SPLIT_DIR / "cf_pairs_2000.parquet", columns=["item_id", "user_id"])
    ucf = UserCF(
        pdf["item_id"].to_numpy("int64"),
        pdf["user_id"].to_numpy("int64"),
        max_item_users=PARAMS["max_item_users"],
    )
    del pdf
    icf = ItemCF(ucf, top_m=PARAMS["top_m"], top_similar=PARAMS["top_similar"])
    logger.info("索引构建完成 (%.0fs)", time.perf_counter() - t0)

    return Bundle(
        ucf=ucf, icf=icf, pop=pop, cat_top=cat_top, cat_of=cat_of,
        cold_cats=cold_cats, global_top=global_top, meta={},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查检查点是否过期")
    ap.add_argument("--force", action="store_true", help="忽略现有检查点强制重建")
    args = ap.parse_args()

    sources = _sources()

    if args.check:
        fresh = is_fresh(CKPT_PATH, sources)
        print(json.dumps({"path": str(CKPT_PATH), "fresh": fresh}, ensure_ascii=False))
        return

    if not args.force and is_fresh(CKPT_PATH, sources):
        t0 = time.perf_counter()
        bundle = load_checkpoint(CKPT_PATH, sources)
        logger.info(
            "检查点有效，直接复用（无需重建）: %s | 载入 %.1fs | %s",
            CKPT_PATH, time.perf_counter() - t0, bundle.meta,
        )
        return

    bundle = train()
    t0 = time.perf_counter()
    save_checkpoint(CKPT_PATH, bundle, PARAMS, sources)
    size_mb = CKPT_PATH.stat().st_size / 1e6
    logger.info("检查点已写入: %s (%.0f MB, 序列化 %.1fs)", CKPT_PATH, size_mb,
                time.perf_counter() - t0)
    logger.info("下一步: 评估/在线服务可用 load_checkpoint 复用，省去 43s 重建")


if __name__ == "__main__":
    main()
