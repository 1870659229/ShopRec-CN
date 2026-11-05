import logging

import pandas as pd

logger = logging.getLogger(__name__)


def sample_users(df: pd.DataFrame, n_users: int = 10000, seed: int = 42) -> pd.DataFrame:
    """按 user_id 抽样并保留这些用户的**全部**行为（不按行抽，避免切断行为序列）。

    用 seed 保证可复现；若用户总数 <= n_users 则原样返回（reset_index 后）。
    """
    unique_users = df["user_id"].drop_duplicates()
    total_users = len(unique_users)
    if total_users <= n_users:
        picked = unique_users
    else:
        picked = unique_users.sample(n=n_users, random_state=seed)

    out = df[df["user_id"].isin(picked)].reset_index(drop=True)
    logger.info(
        "按用户采样：用户 %d -> %d，行 %d -> %d",
        total_users,
        len(picked),
        len(df),
        len(out),
    )
    return out
