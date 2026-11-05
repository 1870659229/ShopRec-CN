"""真实数据准备入口（数据准备脚本，非测试）。

两遍扫描 3.42GB 的 UserBehavior.csv：
1) 第一遍只读 user_id，收集全量 distinct 用户；
2) 按 seed 抽样 n_users 个用户，第二遍读全列过滤并累加。
随后清洗、按时间切分、落盘 parquet，并把 train 写入 MySQL 的 behavior 表，
由全量数据派生 item / category 表。脚本可重复运行（先清空三张表）。
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src.data import cleaner, loader, sampler, splitter  # noqa: E402
from src.db import repo  # noqa: E402

logger = logging.getLogger("prepare_data")

CHUNK_SIZE = 2_000_000
TABLES_TO_CLEAR = ("behavior", "item", "category")


def collect_distinct_users(path: str, chunk_size: int) -> pd.Series:
    """第一遍：逐 chunk 只读 user_id 并累加去重，返回 distinct 用户。"""
    parts: list[pd.Series] = []
    for chunk in pd.read_csv(
        path,
        header=None,
        names=loader.COLUMNS,
        usecols=["user_id"],
        chunksize=chunk_size,
    ):
        parts.append(chunk["user_id"].drop_duplicates())
    users = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    logger.info("第一遍扫描完成：distinct 用户 %d", len(users))
    return users


def load_selected_rows(path: str, users: pd.Series, chunk_size: int) -> pd.DataFrame:
    """第二遍：逐 chunk 读全列，保留抽中用户的行为行。"""
    selected = set(users.tolist())
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path, header=None, names=loader.COLUMNS, chunksize=chunk_size
    ):
        parts.append(chunk[chunk["user_id"].isin(selected)])
    df = pd.concat(parts, ignore_index=True)
    logger.info("第二遍扫描完成：抽中用户行为行 %d", len(df))
    return df


def count_exclusions(df: pd.DataFrame) -> tuple[int, int, int]:
    """按 cleaner.clean() 的同一口径统计三项剔除数：非法类型 / 重复 / 时间戳越界。

    与 cleaner.clean() 使用同一组常量（VALID_TYPES / TS_VALID_MIN / TS_VALID_MAX）
    与同一判定顺序，仅用于汇总打印，不改变清洗结果。
    """
    before = len(df)
    typed = df[df["behavior_type"].isin(cleaner.VALID_TYPES)]
    n_bad_type = before - len(typed)
    deduped = typed.drop_duplicates(
        subset=["user_id", "item_id", "behavior_type", "timestamp"]
    )
    n_dup = before - n_bad_type - len(deduped)
    ts = (
        pd.to_datetime(deduped["timestamp"], unit="s")
        if "ts" not in deduped.columns
        else deduped["ts"]
    )
    in_window = (ts >= pd.Timestamp(cleaner.TS_VALID_MIN, unit="s")) & (
        ts < pd.Timestamp(cleaner.TS_VALID_MAX, unit="s")
    )
    n_oob = int((~in_window).sum())
    return n_bad_type, n_dup, n_oob


def derive_item_category(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """由全量 df 派生 item(item_id→category_id) 与 category(distinct) 表。

    先统计同一 item_id 对应多个 category_id 的冲突条数；若有冲突，取该 item
    出现次数最多的 category 作为其归属。当同一 item 的多个 category 出现次数
    相同时，显式规定 tie-break：取 category_id 较小者（见下方排序的第三键）。
    """
    pairs = df[["item_id", "category_id"]].drop_duplicates()
    nunique = pairs.groupby("item_id")["category_id"].nunique()
    n_conflict = int((nunique > 1).sum())
    logger.info("item->category 冲突 item 数：%d", n_conflict)

    if n_conflict > 0:
        counts = (
            df.groupby(["item_id", "category_id"])
            .size()
            .reset_index(name="n")
            # 主键 item_id 升序；次键频次 n 降序（取最常出现的 category）；
            # 末键 category_id 升序（同频冲突时显式取较小 category_id，不依赖分组隐含顺序）
            .sort_values(
                ["item_id", "n", "category_id"], ascending=[True, False, True]
            )
        )
        item_df = (
            counts.drop_duplicates("item_id")[["item_id", "category_id"]]
            .reset_index(drop=True)
        )
        logger.info(
            "冲突处理策略：取出现次数最多的 category（同频取较小 category_id），涉及 item %d 个",
            n_conflict,
        )
    else:
        item_df = pairs.reset_index(drop=True)

    category_df = df[["category_id"]].drop_duplicates().reset_index(drop=True)
    return item_df, category_df


def _print_summary(
    distinct_users: int,
    sampled_users: int,
    raw_rows: int,
    n_bad_type: int,
    n_dup: int,
    n_oob: int,
    clean_rows: int,
    train_rows: int,
    test_rows: int,
    item_rows: int,
    category_rows: int,
    times: dict[str, float],
) -> None:
    """打印最终汇总表。"""
    print("=" * 56)
    print("数据准备汇总")
    print("-" * 56)
    print(f"distinct 用户数        : {distinct_users}")
    print(f"抽样用户数             : {sampled_users}")
    print(f"清洗前行数             : {raw_rows}")
    print(f"  非法类型剔除数       : {n_bad_type}")
    print(f"  重复剔除数           : {n_dup}")
    print(f"  时间戳越界剔除数     : {n_oob}")
    print(f"清洗后总行数           : {clean_rows}")
    print(f"train 行数             : {train_rows}")
    print(f"test  行数             : {test_rows}")
    print(f"item 行数              : {item_rows}")
    print(f"category 行数          : {category_rows}")
    print("-" * 56)
    for label, seconds in times.items():
        print(f"{label:<22}: {seconds:.1f}s")
    print("=" * 56)


def main() -> None:
    """命令行入口：采样 / 清洗 / 切分 / 落盘 / 入库。"""
    parser = argparse.ArgumentParser(description="真实数据准备：采样/清洗/切分/入库")
    parser.add_argument(
        "--data-path", default=Path(settings.DATA_DIR) / "UserBehavior.csv"
    )
    parser.add_argument("--n-users", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_path = Path(args.data_path)
    logger.info("数据文件：%s", data_path.resolve())

    t_init = time.perf_counter()
    repo.init_db()
    t_init = time.perf_counter() - t_init
    logger.info("init_db 完成，耗时 %.1fs", t_init)

    for table in TABLES_TO_CLEAR:
        repo.execute(f"TRUNCATE TABLE `{table}`")
        logger.info("已清空表 %s", table)

    t = time.perf_counter()
    all_users = collect_distinct_users(str(data_path), args.chunk_size)
    t_pass1 = time.perf_counter() - t

    users_df = pd.DataFrame({"user_id": all_users})
    sampled_users = sampler.sample_users(
        users_df, n_users=args.n_users, seed=args.seed
    )["user_id"]
    logger.info("抽样用户数 %d（seed=%d）", len(sampled_users), args.seed)

    t = time.perf_counter()
    raw = load_selected_rows(str(data_path), sampled_users, args.chunk_size)
    t_pass2 = time.perf_counter() - t

    n_bad_type, n_dup, n_oob = count_exclusions(raw)

    t = time.perf_counter()
    df = cleaner.clean(raw)
    t_clean = time.perf_counter() - t

    t = time.perf_counter()
    train, test = splitter.time_split(df, args.ratio)
    t_split = time.perf_counter() - t

    out_dir = Path(settings.DATA_DIR) / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("切分输出目录：%s", out_dir.resolve())
    train_path, test_path = out_dir / "train.parquet", out_dir / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    logger.info("落盘 %s：%d 行，%d 字节", train_path, len(train), train_path.stat().st_size)
    logger.info("落盘 %s：%d 行，%d 字节", test_path, len(test), test_path.stat().st_size)

    item_df, category_df = derive_item_category(df)

    t = time.perf_counter()
    # 只把 train 写入 behavior：把测试期行为灌进在线库会造成时间泄露
    n_behavior = splitter.to_db(train, "behavior")
    n_item = splitter.to_db(item_df, "item")
    n_category = splitter.to_db(category_df, "category")
    t_db = time.perf_counter() - t
    logger.info(
        "入库完成：behavior=%d item=%d category=%d", n_behavior, n_item, n_category
    )

    _print_summary(
        distinct_users=len(all_users),
        sampled_users=len(sampled_users),
        raw_rows=len(raw),
        n_bad_type=n_bad_type,
        n_dup=n_dup,
        n_oob=n_oob,
        clean_rows=len(df),
        train_rows=len(train),
        test_rows=len(test),
        item_rows=n_item,
        category_rows=n_category,
        times={
            "init_db": t_init,
            "第一遍扫描": t_pass1,
            "第二遍扫描": t_pass2,
            "清洗": t_clean,
            "切分": t_split,
            "入库": t_db,
        },
    )


if __name__ == "__main__":
    main()
