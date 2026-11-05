"""JDsearch 流式加载器与展平函数单元测试。

展平函数用**内存构造的小 DataFrame**；load_* 用 tmp_path 下的**合成小文件**。
全部不读取 4.51GB / 912.8MB 真实数据文件（CI 友好）。
"""
from pathlib import Path

import pandas as pd

from src.data.jdsearch_loader import (
    extract_test_queries,
    extract_train_behaviors,
    load_behaviors,
    load_meta,
)
from src.data.jdsearch_parser import BEHAVIOR_COLUMNS, META_COLUMNS

# 夹具用**字面量**分隔符（独立于被测模块常量），保证「改常量」的突变能被捕获
_TOKEN = "\x18"
_LIST = "_"

_TRAIN_COLUMNS = ["user_id", "item_id", "behavior_type", "query", "interval"]
_TEST_COLUMNS = ["user_id", "query", "item_id", "label"]


def _behavior_df() -> pd.DataFrame:
    """两行会话日志：行 0 正常，行 1 的 history_qry 故意短 1（测 min 对齐）。"""
    return pd.DataFrame({
        "query": ["q0", "q1"],
        "candidate_wid_list": [["c1", "c2"], ["c3", "c4", "c5"]],
        "candidate_label_list": [[1.0, 0.0], [2.0, 3.0]],
        "history_qry_list": [["-1", "qa"], ["qb"]],
        "history_wid_list": [["10", "11"], ["20", "21", "22"]],
        "history_type_list": [["CLICK", "ORD"], ["CART", "ORD", "CLICK"]],
        "history_time_list": [[0.0, 3600.0, 7200.0], [0.0, 5.0, 6.0, 7.0]],
    })


# --- extract_train_behaviors ---

def test_extract_train_behaviors_flattens_with_row_index_as_user_id():
    train = extract_train_behaviors(_behavior_df())
    assert list(train.columns) == _TRAIN_COLUMNS
    # 行 0 展平出 2 条；行 1 因 history_qry 只有 1 个 → min(3,3,1)=1 条
    assert train["user_id"].tolist() == [0, 0, 1]
    assert train["item_id"].tolist() == ["10", "11", "20"]


def test_extract_train_behaviors_aligns_by_min_length():
    train = extract_train_behaviors(_behavior_df())
    # 行 1 的三列长度 3/3/1，必须按 min=1 截断，若去掉 min 对齐会 IndexError / 错行
    assert len(train) == 3
    assert train[train["user_id"] == 1]["item_id"].tolist() == ["20"]
    assert train[train["user_id"] == 1]["behavior_type"].tolist() == ["CART"]


def test_extract_train_behaviors_interval_from_time_first_is_zero():
    train = extract_train_behaviors(_behavior_df())
    assert train["interval"].tolist() == [0.0, 3600.0, 0.0]
    assert train.loc[0, "query"] == "-1"
    assert train.loc[1, "behavior_type"] == "ORD"


# --- extract_test_queries ---

def test_extract_test_queries_flattens_candidates():
    test = extract_test_queries(_behavior_df())
    assert list(test.columns) == _TEST_COLUMNS
    # 行 0：2 候选；行 1：2 标签（与 3 个 wid 取 min=2）
    assert test["user_id"].tolist() == [0, 0, 1, 1]
    assert test["query"].tolist() == ["q0", "q0", "q1", "q1"]
    assert test["item_id"].tolist() == ["c1", "c2", "c3", "c4"]


def test_extract_test_queries_label_is_float():
    test = extract_test_queries(_behavior_df())
    assert test["label"].tolist() == [1.0, 0.0, 2.0, 3.0]
    assert str(test["label"].dtype).startswith("float")


# --- load_meta / load_behaviors（合成小文件，非真实大文件）---

def _write_meta_file(path: Path) -> None:
    lines = ["\t".join(META_COLUMNS)]
    for wid in ("1", "2", "3"):
        lines.append("\t".join([
            wid,
            _TOKEN.join([f"{wid}a", f"{wid}b"]),
            "9", "9",
            "100", "n1", "200", "n2", "300", "n3",
            "400", _TOKEN.join(["n4", "n4b"]),
            "7",
        ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_behavior_file(path: Path) -> None:
    lines = ["\t".join(BEHAVIOR_COLUMNS)]
    for i in range(3):
        lines.append("\t".join([
            _TOKEN.join([f"q{i}", "91"]),
            _LIST.join(["c1", "c2"]),
            _LIST.join(["1.0", "0.0"]),
            _LIST.join(["-1", "9"]),
            _LIST.join(["10", "11"]),
            _LIST.join(["CLICK", "CART"]),
            _LIST.join(["0", "5", "6"]),
        ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_meta_streams_skips_header_and_parses(tmp_path: Path):
    p = tmp_path / "meta.txt"
    _write_meta_file(p)
    df = load_meta(str(p))
    assert len(df) == 3                       # 表头被跳过
    assert list(df.columns) == list(META_COLUMNS)
    assert df.iloc[0]["name"] == ["1a", "1b"]  # 多 token 字段为 list
    assert df.iloc[2]["wid"] == "3"


def test_load_meta_respects_nrows(tmp_path: Path):
    p = tmp_path / "meta.txt"
    _write_meta_file(p)
    df = load_meta(str(p), nrows=2)
    assert len(df) == 2
    assert df["wid"].tolist() == ["1", "2"]


def test_load_behaviors_streams_and_parses(tmp_path: Path):
    p = tmp_path / "beh.txt"
    _write_behavior_file(p)
    df = load_behaviors(str(p), nrows=2)
    assert len(df) == 2
    assert list(df.columns) == list(BEHAVIOR_COLUMNS)
    assert df.iloc[0]["history_time_list"] == [0.0, 5.0, 6.0]
    assert df.iloc[1]["candidate_label_list"] == [1.0, 0.0]
    # 端到端：加载后可展平
    train = extract_train_behaviors(df)
    assert len(train) == 4
