"""构建 JDsearch 离线评估数据集（时序 leave-last-out）。

**为什么不用随机切分**：JDsearch 无绝对时间戳，随机切分会泄露未来。
但已实测验证（`tests` 外的核验脚本）：`behavior.id` 的顺序**完全等于**原始
`history_*` 的发生顺序（抽样用户逐条比对 121/121、8/8、35/35 全一致），
因此可按 `id` 升序做**合法的时序留出**：每个用户最后 N 条 → 测试集，其余 → 训练集。

产出（默认 ``data/splits/jdsearch_eval/``）：
- ``train.parquet``：训练交互 (user_id, item_id, behavior_type, seq)
- ``test.parquet`` ：测试正样本 (user_id, item_id, seq)，每人最多 ``--holdout`` 条
- ``meta.json``    ：采样参数与统计摘要

库名/密码均来自环境变量，不硬编码。
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

logger = logging.getLogger("build_eval_split")

DEFAULT_OUT = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 JDsearch 离线评估集（leave-last-out）")
    p.add_argument("--n-users", type=int, default=20000, help="采样用户数")
    p.add_argument("--min-events", type=int, default=20, help="用户最少事件数")
    p.add_argument("--holdout", type=int, default=5, help="每人留作测试的尾部事件数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = pymysql.connect(
        host=settings.DB_HOST, port=settings.DB_PORT, user=settings.DB_USER,
        password=settings.DB_PASSWORD, database=settings.DB_NAME,
        charset="utf8mb4", connect_timeout=10,
    )
    cur = conn.cursor()

    # 1) 取满足最小事件数的用户，按用户抽样（可复现）
    logger.info("统计用户事件数分布…")
    cur.execute(
        "SELECT user_id, COUNT(*) AS n FROM behavior GROUP BY user_id HAVING n >= %s",
        (args.min_events,),
    )
    eligible = cur.fetchall()
    logger.info("合格用户（事件 >= %d）: %d", args.min_events, len(eligible))

    import random

    rng = random.Random(args.seed)
    picked = rng.sample([u for u, _ in eligible], min(args.n_users, len(eligible)))
    picked.sort()
    logger.info("采样用户: %d（seed=%d）", len(picked), args.seed)

    # 2) 按 id 升序取这些用户的事件，切分成 train / test
    placeholders = ",".join(["%s"] * len(picked))
    logger.info("拉取事件（按 user_id, id 升序）…")
    cur.execute(
        f"SELECT user_id, item_id, behavior_type, id FROM behavior "
        f"WHERE user_id IN ({placeholders}) ORDER BY user_id, id",
        picked,
    )
    rows = cur.fetchall()
    conn.close()
    logger.info("拉取事件 %d 条（%.0fs）", len(rows), time.perf_counter() - t0)

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "behavior_type", "id"])
    # 每人按出现次序编号
    df["seq"] = df.groupby("user_id").cumcount()

    total_events = df.groupby("user_id").size()
    # 尾部 holdout 条为测试；不足 holdout+1 的用户丢弃（无法同时有训练与测试）
    keep = total_events[total_events > args.holdout].index
    df = df[df["user_id"].isin(keep)]
    last_seq = df.groupby("user_id")["seq"].transform("max")
    is_test = last_seq - df["seq"] < args.holdout

    train = df[~is_test][["user_id", "item_id", "behavior_type", "seq"]].reset_index(drop=True)
    test = df[is_test][["user_id", "item_id", "seq"]].reset_index(drop=True)

    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    summary = {
        "n_users": int(train["user_id"].nunique()),
        "min_events": args.min_events,
        "holdout": args.holdout,
        "seed": args.seed,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_items": int(train["item_id"].nunique()),
        "test_items": int(test["item_id"].nunique()),
        "eligible_users": int(len(eligible)),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 56)
    for k, v in summary.items():
        print(f"  {k:<16}: {v:,}" if isinstance(v, int) else f"  {k:<16}: {v}")
    print(f"  输出目录          : {out_dir}")
    print("=" * 56)


if __name__ == "__main__":
    main()
