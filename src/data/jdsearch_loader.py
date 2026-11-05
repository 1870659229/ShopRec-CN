"""JDsearch 加载器与展平函数。

内存边界（务必先读）：

- ``iter_meta_chunks`` / ``iter_behavior_chunks``：**真流式**——边读边累积到
  ``chunk_size`` 条即 ``yield`` 一个 DataFrame，内存峰值与 ``chunk_size`` 成正比、
  与文件总行数**无关**。全量（4.51 GB / 1214 万行）入库请用它们。
- ``load_meta`` / ``load_behaviors``：**会把全部解析结果物化**成一个 DataFrame
  （内部 ``list(...)``），因此内存峰值≈行数×每行开销，**仅适用于小样本/单元测试**；
  对 4.51 GB 全量使用会 OOM。

展平函数：

- ``extract_train_behaviors`` / ``extract_test_queries``：把会话日志的
  ``history_*`` / ``candidate_*`` 列展平成逐条事件；``row_offset`` 用于分块场景——
  传入「已处理行数」即可让 ``user_id`` 保持全局连续（默认 0，不影响整表调用）。

⚠️ JDsearch **无绝对时间戳**：``history_time_list`` 是「相邻行为间隔」，首元素恒为 0，
且长度比其他 history 列多 1。因此 ``interval`` 取 ``history_time_list[i]``（第 0 条为 0），
**禁止**用它伪造绝对时间做切分。
"""
import logging
from collections.abc import Iterator
from typing import Callable

import pandas as pd

from src.data import jdsearch_parser
from src.data.jdsearch_parser import BEHAVIOR_COLUMNS, META_COLUMNS

logger = logging.getLogger(__name__)

TRAIN_COLUMNS: tuple[str, ...] = (
    "user_id",
    "item_id",
    "behavior_type",
    "query",
    "interval",
)
TEST_COLUMNS: tuple[str, ...] = ("user_id", "query", "item_id", "label")


def _iter_parsed(
    path: str,
    parser: Callable[[str], dict[str, object]],
    nrows: int | None,
) -> Iterator[dict[str, object]]:
    """逐行读取（第一行视作表头跳过）并解析，最多产出 nrows 条。

    单行解析抛 ``ValueError`` / ``IndexError`` / ``TypeError`` / ``AttributeError``
    等具体异常时，记录**物理行号 + 该行前 200 字符**后跳过该行继续；结束时汇总
    跳过行数。绝不因一行脏数据中断全量。
    """
    skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            return
        for i, line in enumerate(fh):
            if nrows is not None and i >= nrows:
                break
            try:
                rec = parser(line)
            except (ValueError, IndexError, TypeError, AttributeError) as exc:
                skipped += 1
                logger.warning(
                    "跳过第 %d 行（%s：%s）：%r", i + 2, type(exc).__name__, exc, line[:200]
                )
                continue
            yield rec
    if skipped:
        logger.warning("解析共跳过 %d 行（行号以文件物理行为准，含表头）", skipped)


def load_meta(path: str, nrows: int | None = None) -> pd.DataFrame:
    """加载商品元数据为 DataFrame（列顺序 ``META_COLUMNS``）。

    ⚠️ **非流式**：内部会把全部解析结果物化（``list``），内存随行数线性增长。
    大文件请改用 :func:`iter_meta_chunks`；本函数仅适合小样本/测试。
    """
    records = list(_iter_parsed(path, jdsearch_parser.parse_meta_line, nrows))
    df = pd.DataFrame(records, columns=list(META_COLUMNS))
    logger.info("加载元数据 %d 行（nrows=%s，已物化）", len(df), nrows)
    return df


def load_behaviors(path: str, nrows: int | None = None) -> pd.DataFrame:
    """加载搜索会话日志为 DataFrame（列顺序 ``BEHAVIOR_COLUMNS``）。

    ⚠️ **非流式**：内部会把全部解析结果物化（``list``），内存随行数线性增长。
    大文件请改用 :func:`iter_behavior_chunks`；本函数仅适合小样本/测试。
    """
    records = list(_iter_parsed(path, jdsearch_parser.parse_behavior_line, nrows))
    df = pd.DataFrame(records, columns=list(BEHAVIOR_COLUMNS))
    logger.info("加载行为 %d 行（nrows=%s，已物化）", len(df), nrows)
    return df


def iter_meta_chunks(
    path: str, chunk_size: int, nrows: int | None = None
) -> Iterator[pd.DataFrame]:
    """**真流式**分块产出商品元数据，每块至多 ``chunk_size`` 行。

    内存峰值与 ``chunk_size`` 成正比、与文件行数无关；``nrows`` 限制总行数。
    """
    buf: list[dict[str, object]] = []
    for rec in _iter_parsed(path, jdsearch_parser.parse_meta_line, nrows):
        buf.append(rec)
        if len(buf) >= chunk_size:
            yield pd.DataFrame(buf, columns=list(META_COLUMNS))
            buf = []
    if buf:
        yield pd.DataFrame(buf, columns=list(META_COLUMNS))


def iter_behavior_chunks(
    path: str, chunk_size: int, nrows: int | None = None
) -> Iterator[pd.DataFrame]:
    """**真流式**分块产出搜索会话日志，每块至多 ``chunk_size`` 行。

    内存峰值与 ``chunk_size`` 成正比、与文件行数无关；``nrows`` 限制总行数。
    """
    buf: list[dict[str, object]] = []
    for rec in _iter_parsed(path, jdsearch_parser.parse_behavior_line, nrows):
        buf.append(rec)
        if len(buf) >= chunk_size:
            yield pd.DataFrame(buf, columns=list(BEHAVIOR_COLUMNS))
            buf = []
    if buf:
        yield pd.DataFrame(buf, columns=list(BEHAVIOR_COLUMNS))


def extract_train_behaviors(df: pd.DataFrame, row_offset: int = 0) -> pd.DataFrame:
    """把 ``history_*`` 展平为逐条训练事件表。

    输出列 ``user_id``（取 ``df`` 行索引 **+ row_offset**）/ ``item_id`` /
    ``behavior_type`` / ``query`` / ``interval``。``history_wid_list`` /
    ``history_type_list`` / ``history_qry_list`` **三列按最短长度取 min 对齐**
    （残缺行直接截断）；``interval`` 取 ``history_time_list[i]``，缺失时补 0。

    ``row_offset`` 供分块处理使用：累加「已处理行数」即可让各块的 ``user_id``
    与整表加载一致（默认 0，整表调用行为不变）。
    """
    rows: list[dict[str, object]] = []
    for row_idx, rec in df.iterrows():
        user_id = row_idx + row_offset
        wids = rec["history_wid_list"]
        types = rec["history_type_list"]
        qrys = rec["history_qry_list"]
        times = rec["history_time_list"]
        n = min(len(wids), len(types), len(qrys))
        for i in range(n):
            rows.append({
                "user_id": user_id,
                "item_id": wids[i],
                "behavior_type": types[i],
                "query": qrys[i],
                "interval": float(times[i]) if i < len(times) else 0.0,
            })
    return pd.DataFrame(rows, columns=list(TRAIN_COLUMNS))


def extract_test_queries(df: pd.DataFrame, row_offset: int = 0) -> pd.DataFrame:
    """把 ``candidate_*`` 展平为逐条测试候选表。

    输出列 ``user_id``（取 ``df`` 行索引 **+ row_offset**）/ ``query``（该行的
    ``query`` 字段）/ ``item_id`` / ``label``。``candidate_wid_list`` 与
    ``candidate_label_list`` 按最短长度取 min 对齐。``row_offset`` 语义同
    :func:`extract_train_behaviors`。
    """
    rows: list[dict[str, object]] = []
    for row_idx, rec in df.iterrows():
        user_id = row_idx + row_offset
        wids = rec["candidate_wid_list"]
        labels = rec["candidate_label_list"]
        n = min(len(wids), len(labels))
        for i in range(n):
            rows.append({
                "user_id": user_id,
                "query": rec["query"],
                "item_id": wids[i],
                "label": float(labels[i]),
            })
    return pd.DataFrame(rows, columns=list(TEST_COLUMNS))
