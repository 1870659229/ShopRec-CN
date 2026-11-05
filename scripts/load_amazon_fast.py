"""Amazon Reviews 2023 · Grocery 极速入库（``LOAD DATA LOCAL INFILE``）。

与 :mod:`scripts.load_jdsearch_fast` 同样思路：流式生成 TSV → ``LOAD DATA`` → 重建索引。
适配 ``schema_amazon.sql``（ID 为 VARCHAR：asin / Amazon user_id）。

映射规则：
- ``item.item_id`` = ``parent_asin``（商品级唯一键）；``category_id`` = 其**最长路径**的叶子类。
- ``category``：把 ``categories``（**实测为扁平字符串列表 = 一条层级路径**，如
  ``["Grocery & Gourmet Food","Beverages","Coffee"]``；同时兼容嵌套的路径列表）
  按**路径前缀**建树，分配**合成整数 id**，``name`` 存**真实类目名**（Amazon 的卖点）；
  root 的 ``parent_id=0``。
- ``behavior``：每条 review 一行，``behavior_type='rate'``，``rating``=评分，``ts`` 由毫秒时间戳转 UTC 朴素 DATETIME；
  ``item_id`` 取 **``parent_asin``**（与 ``item`` 主键一致；用 ``asin`` 会有约 55% 的行挂不上 item）。

库名由 ``SHOPREC_DB_NAME`` 决定（Amazon 用 ``shoprec_amazon``，会自动选 VARCHAR 版 schema）。
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import logging
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src.db import repo  # noqa: E402

logger = logging.getLogger("load_amazon_fast")

SOURCE = "amazon"
DEFERRED_INDEXES: dict[str, tuple[tuple[str, str], ...]] = {
    "behavior": (
        ("idx_bh_user_ts", "(`user_id`, `ts`)"),
        ("idx_bh_item", "(`item_id`)"),
    ),
    "item": (("idx_item_cate", "(`category_id`)"),),
}
TABLES = ("behavior", "item", "category")


def connect(local_infile: bool = True) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        charset="utf8mb4",
        local_infile=local_infile,
        autocommit=True,
        connect_timeout=10,
    )


def iter_jsonl_gz(path: str) -> Iterator[dict]:
    """逐行读 ``.jsonl.gz``，产出 dict（内存有界）。"""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _ms_to_ts(ms: int | float | None) -> str:
    """毫秒时间戳 → UTC 朴素 ``YYYY-MM-DD HH:MM:SS``。"""
    if not ms:
        return "1970-01-01 00:00:00"
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def write_meta_tsvs(
    meta_path: str, work: Path, nrows: int | None
) -> tuple[Path, Path, int, int]:
    """流式把 meta 展成 ``item.tsv`` / ``category.tsv``。

    返回 ``(item_tsv, category_tsv, item 行数, category 行数)``。
    """
    item_path = work / "item.tsv"
    cat_path = work / "category.tsv"
    path_to_id: dict[tuple[str, ...], int] = {}
    ids = itertools.count(1)
    n_item = n_cat = 0
    t0 = time.perf_counter()
    with item_path.open("w", encoding="utf-8", newline="\n") as fi, cat_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as fc:
        for row in iter_jsonl_gz(meta_path):
            asin = row.get("parent_asin")
            if not asin:
                continue
            cats = row.get("categories") or []
            # 实测：**扁平字符串列表 = 一条层级路径**（如
            # ["Grocery & Gourmet Food","Beverages","Coffee"]）；
            # 同时兼容"路径列表"的嵌套形态（[[...],[...]]）。
            if cats and all(isinstance(p, str) for p in cats):
                paths = [cats]
            else:
                paths = [p for p in cats if isinstance(p, list) and p]
            best_leaf_id = 0
            best_depth = -1
            for path in paths:
                if not isinstance(path, list) or not path:
                    continue
                for level in range(1, len(path) + 1):
                    key = tuple(path[:level])
                    if key not in path_to_id:
                        cid = next(ids)
                        path_to_id[key] = cid
                        parent = path_to_id.get(tuple(path[: level - 1]), 0) if level > 1 else 0
                        name = str(path[level - 1]).replace("\t", " ").replace("\n", " ")
                        fc.write(f"{cid}\t{parent}\t{level}\t{name}\n")
                        n_cat += 1
                if len(path) > best_depth:
                    best_depth = len(path)
                    best_leaf_id = path_to_id[tuple(path)]
            fi.write(f"{asin}\t{best_leaf_id}\t{SOURCE}\t0\n")
            n_item += 1
            if nrows and n_item >= nrows:
                break
    logger.info(
        "生成 item.tsv %d 行 / category.tsv %d 行（%.1fs）",
        n_item, n_cat, time.perf_counter() - t0,
    )
    return item_path, cat_path, n_item, n_cat


def write_behavior_tsv(
    review_path: str, work: Path, nrows: int | None
) -> tuple[Path, int]:
    """流式把 review 展成 ``behavior.tsv``（一行一条评分行为）。"""
    beh_path = work / "behavior.tsv"
    n = 0
    t0 = time.perf_counter()
    with beh_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in iter_jsonl_gz(review_path):
            uid = row.get("user_id")
            # ⚠️ 必须用 parent_asin（与 item.item_id 主键一致），不能用 asin：
            # asin 是具体变体、parent_asin 才是商品；用 asin 会让约 55% 的
            # behavior 找不到对应 item（实测悬空 7,958,544 行）。
            asin = row.get("parent_asin")
            if not uid or not asin:
                continue
            rating = row.get("rating")
            rating_s = "\\N" if rating is None else str(int(round(float(rating))))
            ts = _ms_to_ts(row.get("timestamp"))
            f.write(f"{uid}\t{asin}\trate\t{rating_s}\t{ts}\t{SOURCE}\n")
            n += 1
            if nrows and n >= nrows:
                break
    logger.info("生成 behavior.tsv %d 行（%.1fs）", n, time.perf_counter() - t0)
    return beh_path, n


def _index_exists(cur: pymysql.cursors.Cursor, table: str, name: str) -> bool:
    cur.execute(
        "SELECT COUNT(*) FROM information_schema.STATISTICS"
        " WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND INDEX_NAME=%s",
        (settings.DB_NAME, table, name),
    )
    return int(cur.fetchone()[0]) > 0


def drop_deferred_indexes(cur: pymysql.cursors.Cursor) -> list[tuple[str, str, str]]:
    dropped: list[tuple[str, str, str]] = []
    for table, indexes in DEFERRED_INDEXES.items():
        for name, definition in indexes:
            if _index_exists(cur, table, name):
                cur.execute(f"ALTER TABLE `{table}` DROP INDEX `{name}`")
                dropped.append((table, name, definition))
                logger.info("已删索引 %s.%s", table, name)
    return dropped


def rebuild_indexes(cur: pymysql.cursors.Cursor, dropped: list[tuple[str, str, str]]) -> None:
    for table, name, definition in dropped:
        t0 = time.perf_counter()
        cur.execute(f"ALTER TABLE `{table}` ADD INDEX `{name}` {definition}")
        logger.info("重建索引 %s.%s（%.1fs）", table, name, time.perf_counter() - t0)


def load_tsv(
    cur: pymysql.cursors.Cursor, table: str, path: Path, columns: tuple[str, ...]
) -> int:
    posix = str(path).replace("\\", "/")
    col_sql = ", ".join(f"`{c}`" for c in columns)
    sql = (
        f"LOAD DATA LOCAL INFILE '{posix}' INTO TABLE `{table}` "
        f"FIELDS TERMINATED BY '\\t' LINES TERMINATED BY '\\n' "
        f"({col_sql})"
    )
    t0 = time.perf_counter()
    cur.execute(sql)
    rows = cur.rowcount
    logger.info("LOAD %s: %d 行（%.1fs）", table, rows, time.perf_counter() - t0)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amazon Grocery 极速入库（LOAD DATA）")
    parser.add_argument("--meta-path", required=True)
    parser.add_argument("--review-path", required=True)
    parser.add_argument("--sample-rows", type=int, default=None, help="只取前 N 行（调试）")
    parser.add_argument("--work-dir", default=None, help="TSV 目录（默认 data/_load_amazon）")
    parser.add_argument(
        "--no-fast-index", dest="fast_index", action="store_false",
        help="不删/重建二级索引（默认会删，灌完再建）",
    )
    parser.add_argument("--buffer-pool-mb", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    t_all = time.perf_counter()

    work = Path(args.work_dir) if args.work_dir else Path(settings.DATA_DIR) / "_load_amazon"
    work.mkdir(parents=True, exist_ok=True)
    logger.info("目标库=%s / TSV 目录=%s", settings.DB_NAME, work)

    repo.init_db()
    conn = connect(local_infile=True)
    cur = conn.cursor()

    try:
        cur.execute(f"SET GLOBAL innodb_buffer_pool_size = {args.buffer_pool_mb} * 1024 * 1024")
        cur.execute("SELECT @@innodb_buffer_pool_size")
        logger.info("innodb_buffer_pool_size=%s 字节", cur.fetchone()[0])
    except pymysql.err.MySQLError as exc:
        logger.warning("调大 buffer pool 失败（忽略）：%s", exc)
    try:
        cur.execute("SET GLOBAL local_infile = 1")
    except pymysql.err.MySQLError as exc:
        logger.warning("开启 local_infile 失败（若已 ON 可忽略）：%s", exc)

    for table in TABLES:
        cur.execute(f"TRUNCATE TABLE `{table}`")
    cur.execute("SET SESSION unique_checks = 0")
    cur.execute("SET SESSION foreign_key_checks = 0")

    dropped = drop_deferred_indexes(cur) if args.fast_index else []
    try:
        item_tsv, cat_tsv, _ni, _nc = write_meta_tsvs(args.meta_path, work, args.sample_rows)
        load_tsv(cur, "category", cat_tsv, ("category_id", "parent_id", "level", "name"))
        load_tsv(cur, "item", item_tsv, ("item_id", "category_id", "source", "pop_score"))

        beh_tsv, _nb = write_behavior_tsv(args.review_path, work, args.sample_rows)
        load_tsv(
            cur, "behavior", beh_tsv,
            ("user_id", "item_id", "behavior_type", "rating", "ts", "source"),
        )
    finally:
        rebuild_indexes(cur, dropped)

    for table in TABLES:
        cur.execute(f"ANALYZE TABLE `{table}`")
        cur.execute(f"SELECT COUNT(*) FROM `{table}`")
        print(f"  {table:10} COUNT = {cur.fetchone()[0]:,}")

    print("=" * 56)
    print(f"库名      : {settings.DB_NAME}")
    print(f"TSV 目录  : {work}")
    print(f"总耗时    : {time.perf_counter() - t_all:.1f}s")
    print("=" * 56)
    conn.close()


if __name__ == "__main__":
    main()
