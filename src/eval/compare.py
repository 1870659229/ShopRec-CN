"""对比实验脚本：对每个 alpha 跑重排并统计品类多样性。

纯函数 + 模块级 OUT 常量，便于单测（测试用 monkeypatch 重定向 OUT）。
"""

import json
import logging
from pathlib import Path

from src.eval.metrics import category_diversity
from src.models.diversity import rerank

logger = logging.getLogger(__name__)
OUT = Path("data/processed/compare_result.json")


def run_all(
    alpha_values: list[float],
    samples: list[list[tuple[int, int, float]]],
    k: int = 10,
) -> list[dict]:
    """对每个 alpha 跑一遍重排，统计品类多样性。

    samples: 每个元素是一个用户的候选列表 [(item_id, category_id, score), ...]
    """
    cate_of: dict[int, int] = {item: cate for s_list in samples for item, cate, _ in s_list}
    results: list[dict] = []
    for alpha in alpha_values:
        rec_lists: list[list[str]] = []
        for s_list in samples:
            picked = rerank(s_list, alpha=alpha, k=k)
            rec_lists.append([str(cate_of[i]) for i in picked])
        div = category_diversity(rec_lists)
        results.append({"alpha": alpha, "category_diversity": round(div, 4)})
        logger.info("alpha=%.1f diversity=%.4f", alpha, div)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
