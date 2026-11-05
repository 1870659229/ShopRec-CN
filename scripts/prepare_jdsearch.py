"""JDsearch 数据准备入口（主实验，入库 ``shoprec_jd``）。

流程（**分块边读边入库，内存有界**）：
元数据 → 逐块展开 category/item 并 ``INSERT IGNORE`` → 行为逐块展平 train/test 并入库
→ 入库后走 SQL 聚合做目标品类簇标定。

内存边界：全程使用 ``jdsearch_loader.iter_*_chunks``，内存峰值与 ``--chunk-size``
成正比、与文件总行数**无关**；只在 Python 侧跨块维护 ``category_id`` / ``wid`` 的
全局去重集合（基数小）。**不会**在内存里存全量商品→类目映射表。

库名由 ``SHOPREC_DB_NAME`` 决定（主实验跑时显式传 ``shoprec_jd``）；
本脚本**不硬编码库名与密码**——密码只从会话内联环境变量 ``SHOPREC_DB_PASSWORD`` 读取。

⚠️ 计划原稿的坑（已处理）：原计划 docstring 写「INSERT IGNORE」，但其代码用的是
``chunk.to_sql(..., if_exists="append")``。**pandas 的 ``to_sql`` 不生成 IGNORE**：
一旦遇到主键重复（同一 ``wid`` 出现多次、同一 ``category_id`` 出现在多级/多行），
就会直接抛 ``IntegrityError`` 并中断入库。因此本脚本**不使用 ``to_sql``**，改为：

1. 入库前对 ``item``（按 ``wid``）与 ``category``（按 ``category_id``）用**全局集合去重**；
2. 用 SQLAlchemy Core 的 ``INSERT IGNORE`` 分批写（对重复主键静默跳过，脚本可重跑）。

⚠️ JDsearch **无绝对时间戳**：``behavior.ts`` 用 ``1970-01-01`` 占位，仅满足非空约束，
**不参与任何按时间的切分/排序**（测试集用官方 test query 划分，见 ``extract_test_queries``）。

⚠️ 默认**不**清表：重跑需显式传 ``--truncate``（否则 ``INSERT IGNORE`` 会在旧数据上追加）。
"""
import argparse
import ctypes
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src.data import jdsearch_loader as loader  # noqa: E402
from src.db import repo  # noqa: E402
from src.db.repo import get_engine  # noqa: E402

logger = logging.getLogger("prepare_jdsearch")

TABLES_TO_CLEAR = ("behavior", "item", "category")
ITEM_COLUMNS = ("item_id", "category_id", "source", "pop_score")
CATEGORY_COLUMNS = ("category_id", "parent_id", "level", "name")
BEHAVIOR_COLUMNS = ("user_id", "item_id", "behavior_type", "ts", "source")
TS_PLACEHOLDER = pd.Timestamp("1970-01-01")

# 行为「原始行 → 事件」放大约 150×（一条会话日志含上百条 history）。若按 chunk_size
# 大块展平，单块事件数可达数百万 ⇒ OOM。故展平/入库再按此子批粒度切分，令事件级
# 内存与 chunk_size 解耦（与总行数、chunk_size 均无关）。
FLATTEN_BATCH_ROWS = 1000


def _bulk_insert(
    df: pd.DataFrame, table: str, columns: tuple[str, ...], chunk_size: int = 5000
) -> int:
    """用 ``INSERT IGNORE`` 分批写入（**每批一个事务**），返回提交的行数。

    逐批构造记录 dict（**不**对整块 ``df`` 做 ``to_dict``）并逐批提交：既避免一次性
    复制内存，又把单个事务限制在 ``chunk_size`` 行以内——否则单块数百万行的大事务一旦
    中断，InnoDB 回滚会长时间持锁（实测单块 592 万行回滚即阻塞整个库）。
    ``item`` / ``category`` 的主键来自数据本身，调用方已按主键全局去重；IGNORE 只是
    重跑保护（逐批提交下重跑可能重入，IGNORE 保证幂等）。
    """
    cols = list(columns)
    col_sql = ", ".join(f"`{c}`" for c in cols)
    val_sql = ", ".join(f":{c}" for c in cols)
    stmt = text(f"INSERT IGNORE INTO `{table}` ({col_sql}) VALUES ({val_sql})")
    written = 0
    for start in range(0, len(df), chunk_size):
        batch = df.iloc[start:start + chunk_size][cols].to_dict(orient="records")
        with get_engine().begin() as conn:
            conn.execute(stmt, batch)
        written += len(batch)
    logger.debug("写入表 %s：提交 %d 行", table, written)
    return written


def build_categories(meta: pd.DataFrame) -> pd.DataFrame:
    """由四级类目 id 展开去重成 ``category`` 表。

    ``cate_id_1..4`` 各为一个节点：level=1 的 ``parent_id=0``，level=k 的
    ``parent_id=cate_id_{k-1}``。同一 ``category_id`` 只保留一次（按首次出现）。

    ``category_id`` 与 ``parent_id`` **同时**做 ``pd.to_numeric(errors="coerce")`` 防御：
    任一为空串/非数字的节点整行丢弃并打印计数（与 ``build_behaviors`` 一致）。
    JDsearch 类目名是匿名 token，无法映射为可读名称，故 ``name`` 一律置空串。
    """
    rows: list[tuple[str, str, int]] = []
    for rec in meta.itertuples(index=False):
        c1, c2, c3, c4 = rec.cate_id_1, rec.cate_id_2, rec.cate_id_3, rec.cate_id_4
        rows.append((c1, "0", 1))
        rows.append((c2, c1, 2))
        rows.append((c3, c2, 3))
        rows.append((c4, c3, 4))
    df = pd.DataFrame(rows, columns=["category_id", "parent_id", "level"])
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce")
    df["parent_id"] = pd.to_numeric(df["parent_id"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["category_id", "parent_id"])
    dropped = before - len(df)
    if dropped:
        logger.info("category 丢弃非数字 category_id/parent_id 的节点：%d", dropped)
    before = len(df)
    df = df.drop_duplicates("category_id").reset_index(drop=True)
    logger.info("category 去重：%d -> %d 行", before, len(df))
    df["category_id"] = df["category_id"].astype("int64")
    df["parent_id"] = df["parent_id"].astype("int64")
    df["level"] = df["level"].astype("int64")
    df["name"] = ""
    return df


def build_items(meta: pd.DataFrame) -> pd.DataFrame:
    """由 ``wid`` / ``cate_id_4`` 生成 ``item`` 表（按 ``wid`` 去重）。

    ``source='jdsearch'``、``pop_score=0``。``wid`` 与 ``cate_id_4`` 用
    ``pd.to_numeric(errors="coerce")``：非数字/缺失的行无法建立 item→category
    关系，直接丢弃并打印条数。
    """
    df = meta[["wid", "cate_id_4"]].rename(
        columns={"wid": "item_id", "cate_id_4": "category_id"}
    ).copy()
    item_num = pd.to_numeric(df["item_id"], errors="coerce")
    cate_num = pd.to_numeric(df["category_id"], errors="coerce")
    dropped = int((item_num.isna() | cate_num.isna()).sum())
    if dropped:
        logger.info("item 丢弃非数字/缺 wid|cate_id_4 的行：%d", dropped)
    df = df[item_num.notna() & cate_num.notna()]
    before = len(df)
    df = df.drop_duplicates("item_id").reset_index(drop=True)
    logger.info("item 去重：%d -> %d 行", before, len(df))
    df["item_id"] = df["item_id"].astype("int64")
    df["category_id"] = df["category_id"].astype("int64")
    df["source"] = "jdsearch"
    df["pop_score"] = 0.0
    return df


def build_behaviors(train: pd.DataFrame) -> pd.DataFrame:
    """把展平后的 train 事件整理为 ``behavior`` 表（只写 train）。

    ``item_id`` 非数字的 token 行丢弃并打印；``ts`` 用 ``1970-01-01`` 占位（无绝对时间）。
    """
    df = train.copy()
    item_num = pd.to_numeric(df["item_id"], errors="coerce")
    dropped = int(item_num.isna().sum())
    if dropped:
        logger.info("behavior 丢弃非数字 item_id 行：%d", dropped)
    df = df[item_num.notna()].copy()
    df["user_id"] = df["user_id"].astype("int64")
    df["item_id"] = df["item_id"].astype("int64")
    df["ts"] = TS_PLACEHOLDER
    df["source"] = "jdsearch"
    return df[list(BEHAVIOR_COLUMNS)]


def calibrate_target_cluster(
    meta: pd.DataFrame, train: pd.DataFrame, lower: int = 20, upper: int = 100
) -> pd.Series:
    """目标品类簇标定（内存版，仅小样本/测试用）。

    统计每个 ``cate_id_4`` 叶子类的交互量分布。阈值**不写死**：先打印实测分位数
    （p10/p25/p50/p75/p90）供后续决策，再把 ``[lower, upper)`` 仅作为**候选区间**
    报告其覆盖的叶子类数。返回每个叶子类的交互量 Series（按交互量降序）。

    ⚠️ 该实现需在内存里持有 ``wid → cate_id_4`` 映射；全量入库请用
    :func:`calibrate_target_cluster_sql`（走 SQL 聚合，不持有映射表）。
    """
    pair = meta[["wid", "cate_id_4"]].drop_duplicates("wid")
    mapping = dict(zip(pair["wid"], pair["cate_id_4"]))
    cate = train["item_id"].map(mapping)
    counts = cate.dropna().value_counts()

    covered = int(cate.notna().sum())
    coverage = covered / len(train) if len(train) else 0.0
    quantiles = counts.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    print("-" * 56)
    print("目标品类簇标定 —— cate_id_4 叶子类交互量分位（实测）")
    print(f"  ⚠️ join 覆盖：{covered}/{len(train)} 条 train 事件可映射到 meta（{coverage:.1%}）")
    print("     采样时 meta 与 behavior 各自截前 N 行、并非同一批商品 ⇒ 覆盖率偏低属预期，")
    print("     全量（meta 全量 join）下单再以其分位为准；此处阈值一律不写死。")
    for label, value in zip(("p10", "p25", "p50", "p75", "p90"), quantiles.tolist()):
        print(f"  {label}: {value:.1f}")
    n_candidate = int(((counts >= lower) & (counts < upper)).sum())
    print(f"  叶子类总数: {counts.size}")
    print(f"  候选区间 [{lower},{upper}) 覆盖叶子类数: {n_candidate}（阈值待定，非最终决策）")
    print("-" * 56)
    return counts


def calibrate_target_cluster_sql(lower: int = 20, upper: int = 100) -> pd.Series:
    """目标品类簇标定（**入库后 SQL 聚合版**，全量用）。

    ``behavior JOIN item GROUP BY item.category_id``（叶子类 = cate_id_4）统计交互量，
    只把「每个叶子类一行」的聚合结果读回内存，**不在 Python 里存全量商品映射表**。
    """
    sql = (
        "SELECT i.category_id AS cate_id_4, COUNT(*) AS n "
        "FROM `behavior` b JOIN `item` i ON b.item_id = i.item_id "
        "WHERE b.source = 'jdsearch' GROUP BY i.category_id"
    )
    agg = repo.query_df(sql)
    counts = agg.set_index("cate_id_4")["n"].sort_values(ascending=False)
    total = int(counts.sum())
    quantiles = counts.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    print("-" * 56)
    print("目标品类簇标定 —— 入库后 SQL 聚合（item JOIN behavior，按 cate_id_4）")
    print(f"  可映射交互总数：{total}（聚合后叶子类 {counts.size} 个）")
    for label, value in zip(("p10", "p25", "p50", "p75", "p90"), quantiles.tolist()):
        print(f"  {label}: {value:.1f}")
    n_candidate = int(((counts >= lower) & (counts < upper)).sum())
    print(f"  候选区间 [{lower},{upper}) 覆盖叶子类数: {n_candidate}（阈值待定，非最终决策）")
    print("-" * 56)
    return counts


class _ParquetChunkWriter:
    """把分块 DataFrame 流式追加写成一个 parquet 文件（内存有界）。

    首块确定 schema，后续块 schema 必须一致（本流程各块列/类型固定）。
    """

    def __init__(self, path: Path) -> None:
        import pyarrow.parquet as pq

        self._path = path
        self._pq = pq
        self._writer = None

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        import pyarrow as pa

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._writer = self._pq.ParquetWriter(self._path, table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.info("落盘 %s", self._path)


def _ingest_meta(
    meta_path: str,
    chunk_size: int,
    nrows: int | None,
    seen_cat: set[int],
    seen_item: set[int],
) -> tuple[int, int]:
    """分块读元数据并逐块入库 category/item（跨块用全局集合按主键去重）。"""
    n_cat = n_item = 0
    for chunk in loader.iter_meta_chunks(meta_path, chunk_size, nrows):
        cats = build_categories(chunk)
        new_cats = cats[~cats["category_id"].isin(seen_cat)]
        seen_cat.update(new_cats["category_id"].tolist())
        if not new_cats.empty:
            n_cat += _bulk_insert(new_cats, "category", CATEGORY_COLUMNS)

        items = build_items(chunk)
        new_items = items[~items["item_id"].isin(seen_item)]
        seen_item.update(new_items["item_id"].tolist())
        if not new_items.empty:
            n_item += _bulk_insert(new_items, "item", ITEM_COLUMNS)
    logger.info("meta 入库：category 提交 %d 行 / item 提交 %d 行", n_cat, n_item)
    return n_cat, n_item


def iter_flattened_events(
    behavior_path: str, chunk_size: int, nrows: int | None = None
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """分块 + 子批展平行为日志，产出 ``(train, test)`` 事件块（内存有界）。

    ``user_id`` 跨块、跨子批**全局连续**：子批用 ``iloc`` 切分且保留块内原始索引，
    ``extract_*`` 再叠加本块之前的全局行数 ``offset``，故 ``user_id = offset + 行索引``，
    等价于对整表调用 ``extract_train_behaviors`` / ``extract_test_queries``。
    """
    offset = 0
    for chunk in loader.iter_behavior_chunks(behavior_path, chunk_size, nrows):
        for start in range(0, len(chunk), FLATTEN_BATCH_ROWS):
            # sub 保留 chunk 内索引（= start + 批内位移），extract_* 内部再加 row_idx，
            # 故此处只能给「本块之前的全局行数」offset，不能再加 start（否则 start 被算两次）
            sub = chunk.iloc[start:start + FLATTEN_BATCH_ROWS]
            row_offset = offset
            yield (
                loader.extract_train_behaviors(sub, row_offset=row_offset),
                loader.extract_test_queries(sub, row_offset=row_offset),
            )
        offset += len(chunk)


def _ingest_behaviors(
    behavior_path: str,
    chunk_size: int,
    nrows: int | None,
    train_writer: _ParquetChunkWriter | None = None,
    test_writer: _ParquetChunkWriter | None = None,
) -> tuple[int, int, int]:
    """逐块展平行为并入库 behavior（事件级内存与总行数、chunk_size 均无关）。

    展平走 :func:`iter_flattened_events`（内含子批切分，避免行为行→事件约 150×
    放大导致的 OOM）。返回 ``(behavior 提交行数, train 事件数, test 候选数)``。
    """
    n_beh = n_train = n_test = 0
    for train, test in iter_flattened_events(behavior_path, chunk_size, nrows):
        n_train += len(train)
        n_test += len(test)
        if train_writer is not None:
            train_writer.write(train)
        if test_writer is not None:
            test_writer.write(test)
        beh = build_behaviors(train)
        if not beh.empty:
            n_beh += _bulk_insert(beh, "behavior", BEHAVIOR_COLUMNS)
    logger.info("behavior 入库：提交 %d 行（train=%d / test=%d）", n_beh, n_train, n_test)
    return n_beh, n_train, n_test


def _counts() -> dict[str, int]:
    """回读三表 COUNT(*) 作为入库证据。"""
    out: dict[str, int] = {}
    for table in TABLES_TO_CLEAR:
        df = repo.query_df(f"SELECT COUNT(*) AS c FROM `{table}`")
        out[table] = int(df.iloc[0]["c"])
    return out


def _peak_rss_mb() -> float | None:
    """读取本进程**峰值常驻内存**（Windows: ``PeakWorkingSetSize``）。

    刻意不用 ``tracemalloc``：它按每次分配记录调用栈，在全量（上千万事件、数千万次
    分配）下会带来数十倍的时间开销与数 GB 的登记表内存，反而制造 OOM。故改用操作
    系统计数器，零 tracing 开销。非 Windows 返回 ``None``。
    """
    if sys.platform != "win32":
        return None

    class _MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(_MemoryCounters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    return counters.PeakWorkingSetSize / 1e6 if ok else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JDsearch 数据准备（入库主实验库）")
    parser.add_argument("--meta-path", required=True, help="product_meta_data.txt 路径")
    parser.add_argument("--behavior-path", required=True, help="user_behavior_data.txt 路径")
    parser.add_argument("--sample-rows", type=int, default=None, help="只取前 N 行（调试/压测）")
    parser.add_argument("--chunk-size", type=int, default=50000, help="每块行数（内存上界）")
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="入库前清空 behavior/item/category 三表（重跑用；默认关闭以免误删）",
    )
    parser.add_argument("--save-parquet", action="store_true", help="额外落盘 parquet 到 data/splits/jdsearch/")
    return parser.parse_args()


def main() -> None:
    """命令行入口：分块加载 / 展平 / 标定 / 入库。"""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info("目标库：%s（分库铁律：主实验=shoprec_jd，测试=shoprec_test）", settings.DB_NAME)
    logger.info(
        "元数据：%s / 行为：%s / sample_rows=%s / chunk_size=%s / truncate=%s",
        args.meta_path, args.behavior_path, args.sample_rows, args.chunk_size, args.truncate,
    )

    t0 = time.perf_counter()
    repo.init_db()
    if args.truncate:
        for table in TABLES_TO_CLEAR:
            repo.execute(f"TRUNCATE TABLE `{table}`")
        logger.info("已清空三表（--truncate）")
    else:
        logger.info("未传 --truncate，保留现有数据（INSERT IGNORE 在旧数据上追加）")

    seen_cat: set[int] = set()
    seen_item: set[int] = set()

    t = time.perf_counter()
    _ingest_meta(args.meta_path, args.chunk_size, args.sample_rows, seen_cat, seen_item)
    t_meta = time.perf_counter() - t

    train_writer = test_writer = None
    if args.save_parquet:
        out_dir = Path(settings.DATA_DIR) / "splits" / "jdsearch"
        out_dir.mkdir(parents=True, exist_ok=True)
        train_writer = _ParquetChunkWriter(out_dir / "train.parquet")
        test_writer = _ParquetChunkWriter(out_dir / "test.parquet")

    t = time.perf_counter()
    _ingest_behaviors(
        args.behavior_path, args.chunk_size, args.sample_rows, train_writer, test_writer
    )
    t_beh = time.perf_counter() - t
    for writer in (train_writer, test_writer):
        if writer is not None:
            writer.close()

    t = time.perf_counter()
    calibrate_target_cluster_sql()
    t_cal = time.perf_counter() - t

    counts = _counts()
    peak = _peak_rss_mb()

    print("=" * 56)
    print("JDsearch 数据准备汇总（分块入库）")
    print(f"  库名           : {settings.DB_NAME}")
    print(f"  chunk_size     : {args.chunk_size}（内存上界，与总行数无关）")
    print(f"  category COUNT : {counts['category']}")
    print(f"  item COUNT     : {counts['item']}")
    print(f"  behavior COUNT : {counts['behavior']}")
    print(f"  元数据入库耗时 : {t_meta:.1f}s")
    print(f"  行为展平+入库  : {t_beh:.1f}s")
    print(f"  SQL 标定       : {t_cal:.1f}s")
    peak_txt = f"{peak:.1f} MB" if peak is not None else "N/A（非 Windows）"
    print(f"  进程峰值常驻内存: {peak_txt}")
    print(f"  总耗时         : {time.perf_counter() - t0:.1f}s")
    print("=" * 56)


if __name__ == "__main__":
    main()
