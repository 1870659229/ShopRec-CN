"""JDsearch 真流式分块加载器一致性测试。

用 ``tmp_path`` 下的**合成小文件**，断言：

1. 「分块迭代拼接」与「``load_*`` 一次性加载」逐行相等（分块不改变结果）；
2. 跨块传入 ``row_offset`` 后 ``user_id`` 连续（第 2 块首行 == 第 1 块行数）；
3. ``_iter_parsed`` 遇解析异常能跳过并记录行号（不中断全量）。

全部不读 4.51GB / 912.8MB 真实数据文件（CI 友好）。
"""
import logging
from pathlib import Path

import pandas as pd

from src.data.jdsearch_loader import (
    _iter_parsed,
    extract_train_behaviors,
    iter_behavior_chunks,
    iter_meta_chunks,
    load_behaviors,
    load_meta,
)
from src.data.jdsearch_parser import BEHAVIOR_COLUMNS, META_COLUMNS

# 夹具用**字面量**分隔符（独立于被测模块常量），保证「改常量」的突变能被捕获
_TOKEN = "\x18"
_LIST = "_"


def _write_meta_file(path: Path) -> None:
    lines = ["\t".join(META_COLUMNS)]
    for wid in ("1", "2", "3", "4", "5"):
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
    for i in range(5):
        lines.append("\t".join([
            _TOKEN.join([f"q{i}", "91"]),
            _LIST.join(["c1", "c2"]),
            _LIST.join(["1.0", "0.0"]),
            _LIST.join(["-1", "9"]),
            _LIST.join([f"1{i}", f"2{i}"]),
            _LIST.join(["CLICK", "CART"]),
            _LIST.join(["0", "5", "6"]),
        ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- 一致性：分块拼接 == 一次性加载 ---

def test_iter_meta_chunks_matches_load_meta(tmp_path: Path) -> None:
    p = tmp_path / "meta.txt"
    _write_meta_file(p)
    chunks = list(iter_meta_chunks(str(p), chunk_size=2))
    assert len(chunks) == 3                       # 5 行 → 2/2/1
    joined = pd.concat(chunks, ignore_index=True)
    full = load_meta(str(p))
    pd.testing.assert_frame_equal(joined, full)


def test_iter_behavior_chunks_matches_load_behaviors(tmp_path: Path) -> None:
    p = tmp_path / "beh.txt"
    _write_behavior_file(p)
    chunks = list(iter_behavior_chunks(str(p), chunk_size=2))
    assert len(chunks) == 3
    joined = pd.concat(chunks, ignore_index=True)
    full = load_behaviors(str(p))
    pd.testing.assert_frame_equal(joined, full)


def test_iter_meta_chunks_respects_nrows(tmp_path: Path) -> None:
    p = tmp_path / "meta.txt"
    _write_meta_file(p)
    joined = pd.concat(list(iter_meta_chunks(str(p), chunk_size=2, nrows=3)), ignore_index=True)
    assert joined["wid"].tolist() == ["1", "2", "3"]


def test_iter_behavior_chunks_respects_nrows(tmp_path: Path) -> None:
    p = tmp_path / "beh.txt"
    _write_behavior_file(p)
    joined = pd.concat(list(iter_behavior_chunks(str(p), chunk_size=2, nrows=3)), ignore_index=True)
    assert len(joined) == 3


# --- row_offset：跨块 user_id 连续 ---

def test_chunk_row_offset_makes_user_id_continuous(tmp_path: Path) -> None:
    p = tmp_path / "beh.txt"
    _write_behavior_file(p)
    chunks = list(iter_behavior_chunks(str(p), chunk_size=2))
    parts: list[pd.DataFrame] = []
    offset = 0
    for chunk in chunks:
        parts.append(extract_train_behaviors(chunk, row_offset=offset))
        offset += len(chunk)
    got = pd.concat(parts, ignore_index=True)
    want = extract_train_behaviors(load_behaviors(str(p)))
    pd.testing.assert_frame_equal(got, want)
    # 第 2 块首行 user_id 必须 == 第 1 块行数（若每块从 0 重数则会被破坏）
    assert parts[1]["user_id"].iloc[0] == len(chunks[0])


# --- _iter_parsed 容错：跳过异常行并记录行号 ---

def test_iter_parsed_skips_raising_line_and_logs(tmp_path: Path, caplog) -> None:
    p = tmp_path / "raw.txt"
    p.write_text("header\nok0\nBOOM\nok2\n", encoding="utf-8")

    def _parser(line: str) -> dict[str, object]:
        if "BOOM" in line:
            raise ValueError("坏行")
        return {"v": line.strip()}

    with caplog.at_level(logging.WARNING):
        rows = list(_iter_parsed(str(p), _parser, nrows=None))

    assert [r["v"] for r in rows] == ["ok0", "ok2"]      # BOOM 行被跳过
    assert "跳过" in caplog.text                          # 有跳过计数日志
    assert "坏行" in caplog.text                          # 记录具体异常信息
