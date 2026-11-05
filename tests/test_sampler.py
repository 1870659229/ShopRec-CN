import pandas as pd

from src.data.sampler import sample_users


def _df() -> pd.DataFrame:
    """5 个用户，行数分别为 2 / 3 / 1 / 4 / 2（共 12 行）。"""
    rows: list[dict[str, object]] = []
    for user_id, n_rows in [(1, 2), (2, 3), (3, 1), (4, 4), (5, 2)]:
        for j in range(n_rows):
            rows.append({
                "user_id": user_id,
                "item_id": j,
                "ts": pd.Timestamp("2017-11-25") + pd.Timedelta(hours=j),
            })
    return pd.DataFrame(rows)


def test_sample_returns_only_requested_number_of_users():
    out = sample_users(_df(), n_users=2, seed=42)
    assert out["user_id"].nunique() == 2


def test_sample_keeps_all_rows_of_selected_users():
    """抽中的用户必须保留其全部行为行（不被按行切断）。"""
    df = _df()
    out = sample_users(df, n_users=2, seed=42)
    for user_id in out["user_id"].unique():
        assert (out["user_id"] == user_id).sum() == (df["user_id"] == user_id).sum()


def test_sample_is_reproducible_with_same_seed():
    a = sample_users(_df(), n_users=2, seed=42)
    b = sample_users(_df(), n_users=2, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_sample_returns_all_when_n_exceeds_total():
    df = _df()
    out = sample_users(df, n_users=100, seed=42)
    assert len(out) == len(df)
    assert out["user_id"].nunique() == df["user_id"].nunique()
