"""``scripts/prepare_jdsearch.py`` 纯变换函数测试。

只用**内存合成 DataFrame**，既不连数据库也不读真实大文件（CI 友好）。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_jdsearch import (  # noqa: E402
    BEHAVIOR_COLUMNS,
    TS_PLACEHOLDER,
    build_behaviors,
    build_categories,
    build_items,
    calibrate_target_cluster,
)


def _meta() -> pd.DataFrame:
    """两行元数据：共用 cate_id_1=1，另有一个重复 wid。"""
    return pd.DataFrame({
        "wid": ["100", "100", "200"],
        "cate_id_1": ["1", "1", "1"],
        "cate_id_2": ["2", "2", "5"],
        "cate_id_3": ["3", "3", "6"],
        "cate_id_4": ["4", "4", "7"],
    })


def test_build_categories_expands_four_levels_and_dedups():
    cats = build_categories(_meta())
    # {1,2,3,4,5,6,7} 共 7 个去重节点（两行共用 1,2,3,4）
    assert len(cats) == 7
    assert list(cats["category_id"]) == [1, 2, 3, 4, 5, 6, 7]
    level_of = dict(zip(cats["category_id"], cats["level"]))
    parent_of = dict(zip(cats["category_id"], cats["parent_id"]))
    assert level_of[1] == 1 and parent_of[1] == 0
    assert level_of[4] == 4 and parent_of[4] == 3
    assert level_of[7] == 4 and parent_of[7] == 6


def test_build_categories_name_is_empty_string():
    cats = build_categories(_meta())
    # JDsearch 类目名是匿名 token，无法还原可读名 → name 一律空串
    assert (cats["name"] == "").all()


def test_build_items_dedups_by_wid_and_sets_source():
    items = build_items(_meta())
    assert len(items) == 2                    # 100 重复 → 去重
    assert items["item_id"].tolist() == [100, 200]
    assert items["category_id"].tolist() == [4, 7]
    assert (items["source"] == "jdsearch").all()
    assert (items["pop_score"] == 0.0).all()


def test_build_items_drops_rows_missing_cate_id_4():
    meta = pd.DataFrame({
        "wid": ["1", "2"], "cate_id_4": ["9", ""],
        "cate_id_1": ["1", "1"], "cate_id_2": ["2", "2"], "cate_id_3": ["3", "3"],
    })
    items = build_items(meta)
    assert items["item_id"].tolist() == [1]   # 缺 cate_id_4 的 wid=2 被丢弃


def test_build_behaviors_drops_non_numeric_item_and_sets_placeholder_ts():
    train = pd.DataFrame({
        "user_id": [0, 0, 1],
        "item_id": ["10", "abc", "11"],        # abc 非数字 → 丢弃
        "behavior_type": ["CLICK", "ORD", "CART"],
        "query": ["-1", "-1", "q"],
        "interval": [0.0, 5.0, 0.0],
    })
    bh = build_behaviors(train)
    assert list(bh.columns) == list(BEHAVIOR_COLUMNS)
    assert len(bh) == 2
    assert bh["item_id"].tolist() == [10, 11]
    assert (bh["ts"] == TS_PLACEHOLDER).all()  # 无绝对时间 → 占位
    assert (bh["source"] == "jdsearch").all()


def test_calibrate_target_cluster_prints_quantiles(capsys):
    meta = pd.DataFrame({"wid": ["1", "2", "3"], "cate_id_4": ["90", "91", "92"]})
    # 叶子 90 有 3 条交互，91 有 2 条，92 有 1 条
    train = pd.DataFrame({"item_id": ["1", "1", "1", "2", "2", "3"]})
    counts = calibrate_target_cluster(meta, train, lower=20, upper=100)
    out = capsys.readouterr().out
    assert counts.loc["90"] == 3
    for label in ("p10", "p25", "p50", "p75", "p90"):
        assert label in out
    assert "[20,100)" in out


# --- astype 防御（Important 2）---

def test_build_categories_drops_non_numeric_category_and_parent() -> None:
    # cate_id_2=""：既作 level2 的 category_id（丢），也作 level3 的 parent_id（丢）
    meta = pd.DataFrame({
        "wid": ["1"],
        "cate_id_1": ["1"], "cate_id_2": [""], "cate_id_3": ["3"], "cate_id_4": ["4"],
    })
    cats = build_categories(meta)   # 不得抛 ValueError
    assert set(cats["category_id"]) == {1, 4}


def test_build_categories_drops_non_numeric_leaf() -> None:
    meta = pd.DataFrame({
        "wid": ["1"],
        "cate_id_1": ["1"], "cate_id_2": ["2"], "cate_id_3": ["3"], "cate_id_4": ["abc"],
    })
    cats = build_categories(meta)   # 非法叶子丢弃，其余三级保留
    assert set(cats["category_id"]) == {1, 2, 3}


def test_build_items_drops_non_numeric_wid_and_cate_id_4() -> None:
    meta = pd.DataFrame({
        "wid": ["1", "abc", "3"],
        "cate_id_4": ["4", "5", ""],
        "cate_id_1": ["1", "1", "1"], "cate_id_2": ["2", "2", "2"], "cate_id_3": ["3", "3", "3"],
    })
    items = build_items(meta)       # 不得抛 ValueError
    assert items["item_id"].tolist() == [1]


# --- 分块 + 子批展平一致性（row_offset 全局连续）---

def _write_behavior_file(path: Path) -> None:
    from src.data.jdsearch_parser import BEHAVIOR_COLUMNS as RAW_COLS

    token, lst = "\x18", "_"
    lines = ["\t".join(RAW_COLS)]
    for i in range(5):
        lines.append("\t".join([
            token.join([f"q{i}", "91"]),
            lst.join(["c1", "c2"]),
            lst.join(["1.0", "0.0"]),
            lst.join(["-1", "9"]),
            lst.join([f"1{i}", f"2{i}"]),
            lst.join(["CLICK", "CART"]),
            lst.join(["0", "5", "6"]),
        ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_iter_flattened_events_matches_full_flatten(tmp_path, monkeypatch) -> None:
    import scripts.prepare_jdsearch as prep
    from src.data.jdsearch_loader import (
        extract_test_queries,
        extract_train_behaviors,
        load_behaviors,
    )

    # 强制子批 = 1 行，检验「读块 × 子批」两层偏移算术拼出全局连续 user_id
    monkeypatch.setattr(prep, "FLATTEN_BATCH_ROWS", 1)
    p = tmp_path / "beh.txt"
    _write_behavior_file(p)

    trains, tests = [], []
    for train, test in prep.iter_flattened_events(str(p), chunk_size=2):
        trains.append(train)
        tests.append(test)
    full = load_behaviors(str(p))
    pd.testing.assert_frame_equal(
        pd.concat(trains, ignore_index=True), extract_train_behaviors(full)
    )
    pd.testing.assert_frame_equal(
        pd.concat(tests, ignore_index=True), extract_test_queries(full)
    )
