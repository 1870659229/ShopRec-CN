"""Task 14：评估结果接口测试（读 eval_v5 / alpha_scan，缺失 404）。

用 monkeypatch 把路由的数据目录指向 tmp_path，造 mini JSON，
避免依赖 data/ 下的真实运行时产物。

eval_v5.json 的 ours 按 α 多档展开，接口须归一化并选中稿件引用的
ours@0.05 档——test_eval_results_picks_cited_alpha 守护该行为。
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routers import eval_results as eval_router

client = TestClient(app)


def _metric(recall: float) -> dict:
    return {
        "recall@10": recall,
        "ndcg@10": recall / 2,
        "diversity@10": 0.9,
        "cold_category_hit_rate": 0.005,
    }


_MINI_EVAL_V5: dict = {
    "protocol": {"n_users": 3, "holdout": 5, "seed": 42},
    "metrics": {
        "pop": _metric(0.1),
        "cf": _metric(0.2),
        "ours@0.0": _metric(0.16),
        "ours@0.05": _metric(0.137),
        "ours@0.2": _metric(0.134),
    },
}

_MINI_ALPHA_SCAN: dict = {
    "protocol": {"n_users": 3, "holdout": 5, "seed": 42},
    "baselines": {
        "cf": {"recall@10": 0.2, "diversity@10": 0.55},
        "pop": {"recall@10": 0.1, "diversity@10": 0.37},
    },
    "alpha_scan": [
        {"alpha": 0.0, "recall@10": 0.2, "ndcg@10": 0.1, "diversity@10": 0.8, "cold_hit_rate": 0.0},
        {"alpha": 1.0, "recall@10": 0.1, "ndcg@10": 0.05, "diversity@10": 1.0, "cold_hit_rate": 0.3},
    ],
}


def _make_eval_dir(tmp_path: Path, eval_v5: dict | None = None) -> Path:
    d = tmp_path / "splits" / "jdsearch_eval"
    d.mkdir(parents=True)
    (d / "eval_v5.json").write_text(
        json.dumps(eval_v5 if eval_v5 is not None else _MINI_EVAL_V5, ensure_ascii=False),
        encoding="utf-8",
    )
    (d / "alpha_scan.json").write_text(
        json.dumps(_MINI_ALPHA_SCAN, ensure_ascii=False), encoding="utf-8"
    )
    return d


@pytest.fixture()
def patched_dir(tmp_path: Path, monkeypatch) -> Path:
    """把路由的数据目录替换为 tmp_path 下的 mini 数据。"""
    d = _make_eval_dir(tmp_path)
    monkeypatch.setattr(eval_router, "_eval_dir", lambda: d)
    return d


def test_eval_results_ok_structure(patched_dir: Path) -> None:
    """文件齐全时返回 200，且 eval_v5 / alpha_scan 顶层结构完整。"""
    r = client.get("/api/eval/results")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"eval_v5", "alpha_scan"}
    # eval_v5：归一化后仅 pop / cf / ours 三方法 × 四指标
    metrics = body["eval_v5"]["metrics"]
    assert set(metrics.keys()) == {"pop", "cf", "ours"}
    for m in metrics.values():
        assert set(m.keys()) == {
            "recall@10",
            "ndcg@10",
            "diversity@10",
            "cold_category_hit_rate",
        }
    # alpha_scan：数组 + 基线
    assert isinstance(body["alpha_scan"]["alpha_scan"], list)
    assert len(body["alpha_scan"]["alpha_scan"]) == 2
    assert body["alpha_scan"]["baselines"]["cf"]["recall@10"] == 0.2
    # protocol 透传（看板数据来源脚注用）
    assert body["eval_v5"]["protocol"]["n_users"] == 3


def test_eval_results_picks_cited_alpha(patched_dir: Path) -> None:
    """ours 必须取稿件引用的 ours@0.05 档，而不是字典序首档 ours@0.0。"""
    r = client.get("/api/eval/results")
    assert r.status_code == 200
    ours = r.json()["eval_v5"]["metrics"]["ours"]
    # ours@0.05 的 recall 为 0.137；若误取 ours@0.0 则会是 0.16
    assert ours["recall@10"] == 0.137


def test_eval_results_falls_back_when_cited_alpha_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """冻结档缺失时回退首个 ours@ 档并仍返回 200（不 500）。"""
    metrics = {
        "pop": _metric(0.1),
        "cf": _metric(0.2),
        "ours@0.1": _metric(0.111),
    }
    d = _make_eval_dir(tmp_path, {"protocol": {"n_users": 3}, "metrics": metrics})
    monkeypatch.setattr(eval_router, "_eval_dir", lambda: d)

    r = client.get("/api/eval/results")
    assert r.status_code == 200
    assert r.json()["eval_v5"]["metrics"]["ours"]["recall@10"] == 0.111


def test_eval_results_missing_file(tmp_path: Path, monkeypatch) -> None:
    """任一数据文件缺失时返回 404 且带 detail，不允许 500。"""
    d = tmp_path / "splits" / "jdsearch_eval"
    d.mkdir(parents=True)
    # 只写 eval_v5，缺 alpha_scan
    (d / "eval_v5.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(eval_router, "_eval_dir", lambda: d)

    r = client.get("/api/eval/results")
    assert r.status_code == 404
    assert "detail" in r.json()


def test_eval_results_dir_absent(tmp_path: Path, monkeypatch) -> None:
    """整个评估目录不存在（新环境未跑离线评估）也应 404。"""
    d = tmp_path / "nonexistent"
    monkeypatch.setattr(eval_router, "_eval_dir", lambda: d)

    r = client.get("/api/eval/results")
    assert r.status_code == 404
    assert "detail" in r.json()
