import logging
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import settings

logger = logging.getLogger(__name__)

# 建表 SQL 变体：默认 BIGINT 版；Amazon 侧 ID 是字符串，需 VARCHAR 版
SCHEMA_VARIANTS = {"default": "schema.sql", "amazon": "schema_amazon.sql"}


def schema_file() -> Path:
    """按库名/环境变量选择建表 SQL。

    优先级：``SHOPREC_SCHEMA``（显式指定，取值 ``default`` / ``amazon``）
    → 库名以 ``_amazon`` 结尾则用 Amazon VARCHAR 版 → 否则用默认版。
    """
    variant = os.environ.get("SHOPREC_SCHEMA", "").strip().lower()
    if not variant:
        variant = "amazon" if settings.DB_NAME.endswith("_amazon") else "default"
    if variant not in SCHEMA_VARIANTS:
        raise ValueError(f"未知的 SHOPREC_SCHEMA={variant!r}，可选 {sorted(SCHEMA_VARIANTS)}")
    return Path(__file__).with_name(SCHEMA_VARIANTS[variant])


@lru_cache(maxsize=1)
def _build_engine() -> Engine:
    """构建并缓存数据库引擎。

    结果为进程内单例（@lru_cache(maxsize=1)），从而让连接池真正生效。代价是
    引擎被永久缓存：进程启动后再修改 SHOPREC_DB_* 环境变量（含库名）不会生效。
    确需在同一进程内切换连接，请先调用 ``_build_engine.cache_clear()``。
    """
    return create_engine(settings.url, pool_pre_ping=True, future=True)


def get_engine() -> Engine:
    return _build_engine()


def init_db() -> None:
    """建库（若不存在）并执行对应建表 SQL 的建表语句。

    建表 SQL 由 :func:`schema_file` 选择：默认 ``schema.sql``（BIGINT 版 ID）；
    当库名以 ``_amazon`` 结尾（或显式设 ``SHOPREC_SCHEMA=amazon``）时用
    ``schema_amazon.sql``（VARCHAR 版 ID，适配 Amazon 的字符串 asin / user_id）。

    库名取自 settings.DB_NAME，不写死在 SQL 里；建库时用不带库名的
    连接，建表时用带库名的连接。建表语句需先剥掉注释行，再按分号切分。

    注意：库名仅支持启动期配置。本函数每次都即时读取 settings，而 get_engine()
    走 _build_engine() 的进程内单例缓存；若在进程内热改库名后重新调用本函数，
    会出现 "init_db 建到新库、query_df 仍读旧库" 的不一致。确需切换请先调用
    ``_build_engine.cache_clear()``。
    """
    admin_engine = create_engine(settings.url_no_db, future=True)
    try:
        with admin_engine.begin() as conn:
            conn.execute(
                text(f"CREATE DATABASE IF NOT EXISTS `{settings.DB_NAME}` DEFAULT CHARSET utf8mb4")
            )
    finally:
        admin_engine.dispose()

    schema_path = schema_file()
    ddl = schema_path.read_text(encoding="utf-8")
    without_comments = "\n".join(
        line for line in ddl.splitlines() if not line.strip().startswith("--")
    )
    stmts = [s.strip() for s in without_comments.split(";") if s.strip()]

    schema_engine = create_engine(settings.url, future=True)
    try:
        with schema_engine.begin() as conn:
            for stmt in stmts:
                conn.execute(text(stmt))
    finally:
        schema_engine.dispose()
    logger.info(
        "数据库 %s 初始化完成，使用 %s，执行 %d 条建表语句",
        settings.DB_NAME,
        schema_path.name,
        len(stmts),
    )


def execute(sql: str, params: dict | None = None) -> None:
    with get_engine().begin() as conn:
        conn.execute(text(sql), params or {})


def execute_many(sql: str, params_list: Sequence[Mapping[str, Any]]) -> int:
    """同一条 SQL 在**单个事务**内批量执行，返回影响行数。

    性能意义
    --------
    ``execute`` 每次调用都 ``begin()`` 一次，即一条语句一个事务（一次往返 + 一次
    提交）。推荐接口原先对 k 个结果逐条 INSERT ``recommend_log``，实测 k=40 时
    这部分占端到端 249ms 中的约 234ms（94%）。改为批量后网络往返与事务提交从
    k 次降为 1 次。

    底层驱动（PyMySQL）会把同构 INSERT 合并为多值 ``INSERT ... VALUES (..),(..)``，
    因此这里不需要手工拼多值 SQL。

    注意
    ----
    空列表直接返回 0，不开事务。批量执行是**原子**的：任一行失败则整批回滚，
    这与逐条执行（部分成功）的语义不同——对 recommend_log 这类"整体才有意义"
    的日志是期望行为。
    """
    rows = list(params_list)
    if not rows:
        return 0
    with get_engine().begin() as conn:
        result = conn.execute(text(sql), rows)
        return int(result.rowcount or 0)


def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame → list[dict]，可空列的 NaN 统一转 None 后再交付 JSON。

    背景：pandas 把 SQL NULL 读成 NaN，``json.dumps`` 会输出 ``NaN`` 字面量
    （非法 JSON），前端 ``JSON.parse`` 静默失败——项目踩坑记录第 2 条。
    LEFT JOIN 的可空列（如 category_name）与可空数值列（如 demo_price）
    走本函数出口，杜绝该失败模式。
    """
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")
