"""JDsearch 行解析器单元测试。

全部使用**合成夹具**（手工拼字符串），绝不读取 4.51GB / 912.8MB 真实数据文件。

⚠️ 夹具统一用**字面量**分隔符 ``_TOKEN`` / ``_LIST``（而非被测模块的常量），并额外
用守卫断言锁定常量取值；否则「把常量改掉」的突变会同时改到夹具，测试无法捕获。
"""
import logging

from src.data.jdsearch_parser import (
    BEHAVIOR_COLUMNS,
    LIST_SEP,
    META_COLUMNS,
    TOKEN_SEP,
    _split_floats,
    parse_behavior_line,
    parse_meta_line,
)

_TOKEN = "\x18"   # 文件格式里含 term id 字段的固定内部分隔符
_LIST = "_"       # 文件格式里 *list 字段的固定元素分隔符

# 元数据表 13 列的确定性取值（多 token 字段用字面量分隔符拼）
_META_FIELDS: dict[str, str] = {
    "wid": "84303384",
    "name": _TOKEN.join(["111", "222", "333"]),
    "brand_id": "81493954",
    "brand_name": _TOKEN.join(["56009272", "60257419"]),
    "cate_id_1": "22259905",
    "cate_name_1": "44386912",
    "cate_id_2": "80083037",
    "cate_name_2": _TOKEN.join(["27877021", "49701337"]),
    "cate_id_3": "13599099",
    "cate_name_3": "19027665",
    "cate_id_4": "50485737",
    "cate_name_4": _TOKEN.join(["47460757", "88309887", "86521100"]),
    "shop_id": "1462363",
}


def _meta_line() -> str:
    return "\t".join(_META_FIELDS[c] for c in META_COLUMNS)


def _behavior_line() -> str:
    """7 列行为行：history_time_list 比其他 history 列多 1（首元素 0）。"""
    fields: dict[str, str] = {
        "query": _TOKEN.join(["63995226", "60257419"]),
        "candidate_wid_list": _LIST.join(["36098226", "17729172", "34921664"]),
        "candidate_label_list": _LIST.join(["1.0", "0.0", "2.0"]),
        "history_qry_list": _LIST.join(["-1", "60257419", "3663289"]),
        "history_wid_list": _LIST.join(["10", "11", "12"]),
        "history_type_list": _LIST.join(["CLICK", "CART", "ORD"]),
        "history_time_list": _LIST.join(["0", "22967659", "82448", "432"]),
    }
    return "\t".join(fields[c] for c in BEHAVIOR_COLUMNS)


# --- parse_meta_line ---

def test_meta_columns_and_behavior_columns_length():
    assert len(META_COLUMNS) == 13
    assert len(BEHAVIOR_COLUMNS) == 7


def test_separator_constants_match_file_format():
    # 守卫：把分隔符常量改掉必须让解析测试失败，故此断言锁定文件格式
    assert TOKEN_SEP == "\x18"
    assert LIST_SEP == "_"


def test_parse_meta_line_returns_all_13_columns():
    out = parse_meta_line(_meta_line())
    assert list(out.keys()) == list(META_COLUMNS)
    assert out["wid"] == "84303384"
    assert out["shop_id"] == "1462363"


def test_parse_meta_line_multi_token_fields_are_lists():
    out = parse_meta_line(_meta_line())
    assert out["name"] == ["111", "222", "333"]
    assert out["brand_name"] == ["56009272", "60257419"]
    assert out["cate_name_4"] == ["47460757", "88309887", "86521100"]


def test_parse_meta_line_single_token_fields_are_str():
    out = parse_meta_line(_meta_line())
    for col in ("wid", "brand_id", "cate_id_1", "cate_id_2", "cate_id_3", "cate_id_4"):
        assert isinstance(out[col], str), col


def test_parse_meta_line_missing_columns_tolerated():
    # 只有 5 列：缺失的标量列应为 ""，缺失的多 token 列应为 []
    out = parse_meta_line("\t".join(["1", "111" + _TOKEN + "222", "2", "3", "4"]))
    assert list(out.keys()) == list(META_COLUMNS)
    assert out["cate_id_1"] == "4"
    assert out["cate_name_1"] == []      # 缺失的多 token 列
    assert out["shop_id"] == ""          # 缺失的标量列
    assert out["name"] == ["111", "222"]


# --- parse_behavior_line ---

def test_parse_behavior_line_returns_all_7_columns():
    out = parse_behavior_line(_behavior_line())
    assert list(out.keys()) == list(BEHAVIOR_COLUMNS)
    assert out["query"] == _TOKEN.join(["63995226", "60257419"])


def test_parse_behavior_line_list_columns_are_lists():
    out = parse_behavior_line(_behavior_line())
    assert out["candidate_wid_list"] == ["36098226", "17729172", "34921664"]
    assert out["history_wid_list"] == ["10", "11", "12"]
    assert out["history_type_list"] == ["CLICK", "CART", "ORD"]
    assert out["history_qry_list"] == ["-1", "60257419", "3663289"]


def test_parse_behavior_line_float_columns_are_floats():
    out = parse_behavior_line(_behavior_line())
    assert out["candidate_label_list"] == [1.0, 0.0, 2.0]
    assert out["history_time_list"] == [0.0, 22967659.0, 82448.0, 432.0]
    assert all(isinstance(v, float) for v in out["history_time_list"])


def test_parse_behavior_line_time_has_one_more_than_wid_first_is_zero():
    out = parse_behavior_line(_behavior_line())
    assert len(out["history_time_list"]) == len(out["history_wid_list"]) + 1
    assert out["history_time_list"][0] == 0.0


def test_parse_behavior_line_preserves_negative_one_query():
    out = parse_behavior_line(_behavior_line())
    assert "-1" in out["history_qry_list"]
    assert isinstance(out["history_qry_list"][0], str)


def test_parse_behavior_line_empty_field_gives_empty_list():
    line = "\t".join(["", "", "", "", "", "", ""])
    out = parse_behavior_line(line)
    assert out["candidate_wid_list"] == []
    assert out["candidate_label_list"] == []
    assert out["history_time_list"] == []


# --- 畸形行容错（Important 1）---

def test_parse_behavior_line_short_line_does_not_raise() -> None:
    # 只有 3 列：不得把普通字段 "c" 当 float 列表而抛 ValueError
    out = parse_behavior_line("a\tb\tc")
    assert list(out.keys()) == list(BEHAVIOR_COLUMNS)
    assert out["candidate_wid_list"] == ["b"]
    assert out["candidate_label_list"] == []   # 非数字 token 被跳过，而非抛异常
    assert out["history_time_list"] == []


def test_parse_behavior_line_row_with_non_numeric_float_token_is_tolerated(caplog) -> None:
    # candidate_label_list 混入非数字 token：跳过后仍产出合法 float 列表
    fields = {
        "query": _TOKEN.join(["q", "91"]),
        "candidate_wid_list": _LIST.join(["c1", "c2"]),
        "candidate_label_list": _LIST.join(["1.0", "abc"]),
        "history_qry_list": _LIST.join(["-1"]),
        "history_wid_list": _LIST.join(["10"]),
        "history_type_list": _LIST.join(["CLICK"]),
        "history_time_list": _LIST.join(["0", "5"]),
    }
    line = "\t".join(fields[c] for c in BEHAVIOR_COLUMNS)
    with caplog.at_level(logging.WARNING):
        out = parse_behavior_line(line)
    assert out["candidate_label_list"] == [1.0]
    assert "跳过" in caplog.text


def test_split_floats_skips_non_numeric_and_logs_count(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        got = _split_floats(_LIST.join(["1.0", "abc", "2.0"]))
    assert got == [1.0, 2.0]
    assert "跳过 1 个" in caplog.text
