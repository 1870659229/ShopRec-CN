"""评估结果接口：只读返回离线评估 JSON（eval_v5 / alpha_scan）。

数据文件是 gitignore 的运行时产物（data/splits/jdsearch_eval/），
接口不写不缓存；缺失时 404 带 detail，不抛 500。

metrics 归一化说明：eval_v5.json 的 ours 按 α 展开为多个键
（ours@0.0 ~ ours@0.2），直接透出会令前端按字典序误取 ours@0.0。
本接口显式选取稿件/B3 引用的冻结档 ours@0.05，归一为单一 "ours" 键。
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["eval"])

# 稿件/B3 引用的冻结配置：eval_v5.json 中 ours@0.05
# （R@10 0.0137 / N@10 0.0070 / Div@10 0.9909 / ColdHit 0.0055）。
# 看板 KPI 与对比图统一取这一档，确保与稿件数字逐位一致。
_CITED_OURS_KEY = "ours@0.05"


def _eval_dir() -> Path:
    """评估产物目录（独立函数便于测试 monkeypatch）。"""
    return Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"


def _load_json(path: Path) -> dict:
    """读取单个 JSON；缺失时抛 404。"""
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail=f"评估结果文件缺失：{path}，请先运行离线评估"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_metrics(metrics: dict) -> dict:
    """归一化 metrics 为 {pop, cf, ours}；ours 取稿件引用的 α=0.05 档。

    Args:
        metrics: eval_v5.json 原始 metrics（含 pop/cf 与 ours@<α> 多档）。

    Returns:
        仅含 pop / cf / ours 三键的字典；ours 缺失时回退首个 ours@ 档并告警，
        全部缺失则不返回 ours 键（前端据此隐藏 KPI 与本方案系列）。
    """
    normalized: dict = {}
    for key in ("pop", "cf"):
        if key in metrics:
            normalized[key] = metrics[key]
    ours = metrics.get(_CITED_OURS_KEY)
    if ours is None:
        ours = next((v for k, v in metrics.items() if k.startswith("ours@")), None)
        if ours is not None:
            logger.warning(
                "eval_v5.json 缺少冻结档 %s，已回退首个 ours@ 档", _CITED_OURS_KEY
            )
    if ours is not None:
        normalized["ours"] = ours
    return normalized


@router.get("/eval/results")
def eval_results() -> dict:
    """返回 eval_v5.json（归一化 metrics）与 alpha_scan.json，供后台看板渲染。"""
    d = _eval_dir()
    ev = _load_json(d / "eval_v5.json")
    return {
        "eval_v5": {
            "protocol": ev.get("protocol", {}),
            "metrics": _normalize_metrics(ev.get("metrics", {})),
        },
        "alpha_scan": _load_json(d / "alpha_scan.json"),
    }
