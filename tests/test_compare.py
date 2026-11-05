"""Task 14：对比实验脚本测试（重排 × 品类多样性，α 越大多样性不降）。

run_all 会把结果写到模块级 OUT；写文件用例通过 monkeypatch 把 OUT
指向 tmp_path，避免污染真实 data/。
"""

import json
from pathlib import Path

import pytest

from src.eval import compare

# 两个用户：alpha=0 时两人 top-2 全为同品类（多样性 0），
# alpha=1 时贪心被迫探索新品类（多样性 > 0）
_SAMPLES: list[list[tuple[int, int, float]]] = [
    [(1, 101, 0.9), (2, 101, 0.8), (3, 102, 0.7), (4, 103, 0.6)],
    [(5, 201, 0.95), (6, 201, 0.85), (7, 202, 0.75), (8, 203, 0.65)],
]
_ALPHAS: list[float] = [0.0, 0.5, 1.0]


def test_run_all_diversity_nondecreasing(tmp_path, monkeypatch) -> None:
    """α 越大，品类多样性单调不降。"""
    monkeypatch.setattr(compare, "OUT", tmp_path / "unused.json")
    results = compare.run_all(_ALPHAS, _SAMPLES, k=2)
    divs = [r["category_diversity"] for r in results]
    assert all(divs[i] <= divs[i + 1] for i in range(len(divs) - 1))
    # k=2 且 alpha=0 时 top-2 全为同品类 → 去重占比 1/2；alpha=1 时品类互异 → 1.0
    assert divs[0] == pytest.approx(0.5)
    assert divs[-1] == pytest.approx(1.0)


def test_run_all_result_structure(tmp_path, monkeypatch) -> None:
    """输出为 list[dict]，每项含 alpha 与 category_diversity，与输入 α 一一对应。"""
    monkeypatch.setattr(compare, "OUT", tmp_path / "unused.json")
    results = compare.run_all(_ALPHAS, _SAMPLES, k=2)
    assert isinstance(results, list)
    assert len(results) == len(_ALPHAS)
    for r, alpha in zip(results, _ALPHAS):
        assert set(r.keys()) == {"alpha", "category_diversity"}
        assert r["alpha"] == pytest.approx(alpha)
        assert 0.0 <= r["category_diversity"] <= 1.0


def test_run_all_writes_out(tmp_path: Path, monkeypatch) -> None:
    """结果按 JSON 数组写入 OUT（monkeypatch 到 tmp_path）。"""
    out = tmp_path / "compare_result.json"
    monkeypatch.setattr(compare, "OUT", out)
    results = compare.run_all(_ALPHAS, _SAMPLES, k=2)

    assert out.is_file()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == results
