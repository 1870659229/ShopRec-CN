import logging

import pandas as pd
from sqlalchemy import text

from src.db.repo import get_engine

logger = logging.getLogger(__name__)

# 各表要写入的列（与 src/db/schema.sql 的列名一致）
TABLE_COLUMNS: dict[str, list[str]] = {
    "behavior": ["user_id", "item_id", "behavior_type", "ts"],
    "item": ["item_id", "category_id"],
    "category": ["category_id"],
}
# 这两张表的主键来自数据本身，重复写入需忽略以便脚本可重跑
_IGNORE_TABLES = frozenset({"item", "category"})


def time_split(df: pd.DataFrame, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间排序后切分，禁止随机切分。"""
    ordered = df.sort_values("ts").reset_index(drop=True)
    cut = int(len(ordered) * ratio)
    train, test = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    logger.info("时间切分：train=%d test=%d", len(train), len(test))
    return train, test


def to_db(df: pd.DataFrame, table: str, chunk_size: int = 5000) -> int:
    """把 DataFrame 的对应列分批写入 MySQL 表，返回本次提交给数据库的行数。

    返回值语义：返回值是"本次提交给数据库的行数"，**不等于**"实际新增行数"。
    - behavior 使用普通 INSERT，返回值即实际插入行数；
    - item / category 使用 INSERT IGNORE，写入前按全部列去重，返回值等于去重后
      提交的行数；其中因主键已存在而被 IGNORE 丢弃的行同样计入该值，故它
      大于等于实际新增行数——脚本重跑时该返回值不变，但实际新增行数为 0。

    仅取 TABLE_COLUMNS[table] 指定的列，因此会丢掉 CSV 的 timestamp（epoch 秒）
    列——behavior 表只有 DATETIME ts 列。item / category 的主键来自数据本身，
    写入前先去重并使用 INSERT IGNORE，保证脚本可重复运行；behavior 为自增主键，
    用普通 INSERT。

    注意：behavior.ts 统一按 UTC 朴素时间写入（与 cleaner 的 ts 约定一致），
    而 recommend_log.created_at 用的是服务器本地 CURRENT_TIMESTAMP，二者语义不同。
    """
    if table not in TABLE_COLUMNS:
        raise ValueError(f"未知表名: {table}")

    cols = TABLE_COLUMNS[table]
    subset = df[cols]
    verb = "INSERT INTO"
    if table in _IGNORE_TABLES:
        before = len(subset)
        subset = subset.drop_duplicates()
        verb = "INSERT IGNORE INTO"
        logger.info("表 %s 去重：%d -> %d 行", table, before, len(subset))

    col_sql = ", ".join(f"`{c}`" for c in cols)
    val_sql = ", ".join(f":{c}" for c in cols)
    stmt = text(f"{verb} `{table}` ({col_sql}) VALUES ({val_sql})")

    records = subset.to_dict(orient="records")
    written = 0
    with get_engine().begin() as conn:
        for start in range(0, len(records), chunk_size):
            batch = records[start:start + chunk_size]
            conn.execute(stmt, batch)
            written += len(batch)
    logger.info("写入表 %s：%d 行", table, written)
    return written
