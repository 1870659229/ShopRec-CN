"""JDsearch 极速入库（``LOAD DATA LOCAL INFILE``）。

相对 :mod:`scripts.prepare_jdsearch`（逐批 ``INSERT IGNORE``）快约一个数量级：

1. 流式把 meta 展成 ``item.tsv`` / ``category.tsv``、把会话日志展成 ``behavior.tsv``；
2. 用 ``LOAD DATA LOCAL INFILE`` 一次性灌入；
3. 入库期关闭 ``unique_checks`` / ``foreign_key_checks``，灌完 ``ANALYZE TABLE``。

内存有界：复用 ``jdsearch_loader.iter_*_chunks`` 与 ``prepare_jdsearch.iter_flattened_events``。
TSV 落在 ``--work-dir``，灌完可手动删除（本轮默认保留以便复核）。

库名由 ``SHOPREC_DB_NAME`` 决定；**不硬编码库名与密码**。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scripts.prepare_jdsearch import iter_flattened_events  # noqa: E402
from src.data import jdsearch_loader as loader  # noqa: E402
from src.db import repo  # noqa: E402

logger = logging.getLogger("load_jdsearch_fast")

SOURCE = "jdsearch"
TS_PLACEHOLDER = "1970-01-01 00:00:00"
TABLES = ("behavior", "item", "category")

# 批量灌数据前先删、灌完再建的二级索引（LOAD 期间维护索引是主要开销）
DEFERRED_INDEXES: dict[str, tuple[tuple[str, str], ...]] = {
    "behavior": (
        ("idx_bh_user_ts", "(`user_id`, `ts`)"),
        ("idx_bh_item", "(`item_id`)"),
    ),
    "item": (("idx_item_cate", "(`category_id`)"),),
}


def _index_exists(cur: pymysql.cursors.Cursor, table: str, name: str) -> bool:
    cur.execute(
        "SELECT COUNT(*) FROM information_schema.STATISTICS"
        " WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND INDEX_NAME=%s",
        (settings.DB_NAME, table, name),
    )
    return int(cur.fetchone()[0]) > 0


def drop_deferred_indexes(cur: pymysql.cursors.Cursor) -> list[tuple[str, str, str]]:
    """删掉待重建的二级索引，返回 ``[(表, 索引名, 定义), ...]`` 供之后重建。"""
    dropped: list[tuple[str, str, str]] = []
    for table, indexes in DEFERRED_INDEXES.items():
        for name, definition in indexes:
            if _index_exists(cur, table, name):
                cur.execute(f"ALTER TABLE `{table}` DROP INDEX `{name}`")
                dropped.append((table, name, definition))
                logger.info("已删索引 %s.%s", table, name)
    return dropped


def rebuild_indexes(cur: pymysql.cursors.Cursor, dropped: list[tuple[str, str, str]]) -> None:
    """重建之前删掉的索引。"""
    for table, name, definition in dropped:
        t0 = time.perf_counter()
        cur.execute(f"ALTER TABLE `{table}` ADD INDEX `{name}` {definition}")
        logger.info("重建索引 %s.%s（%.1fs）", table, name, time.perf_counter() - t0)


def connect(local_infile: bool = True) -> pymysql.connections.Connection:
    """按 ``settings`` 建立连接（允许 LOCAL INFILE）。"""
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


def write_meta_tsvs(
    meta_path: str, work: Path, chunk_size: int, nrows: int | None
) -> tuple[Path, Path, int, int]:
    """流式把 meta 展成 ``item.tsv`` / ``category.tsv``（跨块按主键去重）。

    返回 ``(item_tsv, category_tsv, item 行数, category 行数)``。
    """
    item_path = work / "item.tsv"
    cat_path = work / "category.tsv"
    seen_item: set[int] = set()
    seen_cat: set[int] = set()
    n_item = n_cat = 0
    t0 = time.perf_counter()
    with item_path.open("w", encoding="utf-8", newline="\n") as fi, cat_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as fc:
        for chunk in loader.iter_meta_chunks(meta_path, chunk_size, nrows):
            for rec in chunk.itertuples(index=False):
                pairs = (
                    (rec.cate_id_1, "0"),
                    (rec.cate_id_2, rec.cate_id_1),
                    (rec.cate_id_3, rec.cate_id_2),
                    (rec.cate_id_4, rec.cate_id_3),
                )
                for level, (cid, pid) in enumerate(pairs, start=1):
                    if not cid or not pid:
                        continue
                    try:
                        c_int, p_int = int(cid), int(pid)
                    except (ValueError, TypeError):
                        continue
                    if c_int in seen_cat:
                        continue
                    seen_cat.add(c_int)
                    fc.write(f"{c_int}\t{p_int}\t{level}\t\n")
                    n_cat += 1
                try:
                    item_int, cate4_int = int(rec.wid), int(rec.cate_id_4)
                except (ValueError, TypeError):
                    continue
                if item_int in seen_item:
                    continue
                seen_item.add(item_int)
                fi.write(f"{item_int}\t{cate4_int}\t{SOURCE}\t0\n")
                n_item += 1
    logger.info(
        "生成 item.tsv %d 行 / category.tsv %d 行（%.1fs）", n_item, n_cat, time.perf_counter() - t0
    )
    return item_path, cat_path, n_item, n_cat


def write_behavior_tsv(
    behavior_path: str, work: Path, chunk_size: int, nrows: int | None
) -> tuple[Path, int]:
    """流式把会话日志展平成逐条 train 事件写入 ``behavior.tsv``。

    只写 train（历史行为）；test 候选不入库。返回 ``(tsv 路径, 行数)``。
    """
    beh_path = work / "behavior.tsv"
    n = 0
    t0 = time.perf_counter()
    with beh_path.open("w", encoding="utf-8", newline="\n") as f:
        for train, _test in iter_flattened_events(behavior_path, chunk_size, nrows):
            for uid, iid, btype in zip(
                train["user_id"].tolist(),
                train["item_id"].tolist(),
                train["behavior_type"].tolist(),
            ):
                try:
                    int(iid)
                except (ValueError, TypeError):
                    continue
                f.write(f"{uid}\t{iid}\t{btype}\t{TS_PLACEHOLDER}\t{SOURCE}\n")
                n += 1
    logger.info("生成 behavior.tsv %d 行（%.1fs）", n, time.perf_counter() - t0)
    return beh_path, n


def load_tsv(
    cur: pymysql.cursors.Cursor, table: str, path: Path, columns: tuple[str, ...]
) -> int:
    """``LOAD DATA LOCAL INFILE`` 灌入 TSV，返回受影响行数。"""
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
    parser = argparse.ArgumentParser(description="JDsearch 极速入库（LOAD DATA）")
    parser.add_argument("--meta-path", required=True)
    parser.add_argument("--behavior-path", required=True)
    parser.add_argument("--sample-rows", type=int, default=None, help="只取前 N 行（调试）")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--work-dir", default=None, help="TSV 目录（默认 data/_load）")
    parser.add_argument("--keep-tsv", action="store_true", help="保留 TSV（默认保留）")
    parser.add_argument(
        "--no-fast-index",
        dest="fast_index",
        action="store_false",
        help="不删/重建二级索引（默认会删，灌完再建，显著提速）",
    )
    parser.add_argument(
        "--buffer-pool-mb", type=int, default=2048, help="临时调大 innodb_buffer_pool_size"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    t_all = time.perf_counter()

    work = Path(args.work_dir) if args.work_dir else Path(settings.DATA_DIR) / "_load"
    work.mkdir(parents=True, exist_ok=True)
    logger.info("目标库=%s / TSV 目录=%s", settings.DB_NAME, work)

    repo.init_db()
    conn = connect(local_infile=True)
    cur = conn.cursor()

    # 尝试临时调大 buffer pool（动态生效，不持久化）
    try:
        cur.execute(f"SET GLOBAL innodb_buffer_pool_size = {args.buffer_pool_mb} * 1024 * 1024")
        cur.execute("SELECT @@innodb_buffer_pool_size")
        logger.info("innodb_buffer_pool_size 临时调至 %s 字节", cur.fetchone()[0])
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
    logger.info("已清空三表并关闭 unique/foreign_key 检查")

    dropped = drop_deferred_indexes(cur) if args.fast_index else []
    try:
        item_tsv, cat_tsv, _n_item, _n_cat = write_meta_tsvs(
            args.meta_path, work, args.chunk_size, args.sample_rows
        )
        load_tsv(cur, "category", cat_tsv, ("category_id", "parent_id", "level", "name"))
        load_tsv(cur, "item", item_tsv, ("item_id", "category_id", "source", "pop_score"))

        beh_tsv, _n_beh = write_behavior_tsv(
            args.behavior_path, work, args.chunk_size, args.sample_rows
        )
        load_tsv(
            cur, "behavior", beh_tsv, ("user_id", "item_id", "behavior_type", "ts", "source")
        )
    finally:
        rebuild_indexes(cur, dropped)

    for table in TABLES:
        cur.execute(f"ANALYZE TABLE `{table}`")

    for table in TABLES:
        cur.execute(f"SELECT COUNT(*) FROM `{table}`")
        print(f"  {table:10} COUNT = {cur.fetchone()[0]:,}")

    print("=" * 56)
    print(f"库名        : {settings.DB_NAME}")
    print(f"TSV 目录    : {work}")
    print(f"总耗时      : {time.perf_counter() - t_all:.1f}s")
    print("=" * 56)
    conn.close()


if __name__ == "__main__":
    main()
