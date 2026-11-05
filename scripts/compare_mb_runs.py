"""多行为加权实验：新旧基线对比 + 稿件数字核验。

用法::

    python scripts/compare_mb_runs.py

对比两份 eval_v6 系产物（旧=去重基线，新=未去重基线），并逐项核验
``docs/作品稿件.md`` 中引用的多行为数字是否需要刷新。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402

SPLIT_DIR = Path(settings.DATA_DIR) / "splits" / "jdsearch_eval"

# 稿件 §多行为 引用的 (方法名, 指标) —— 值来自 eval_v6.json
MANUSCRIPT_MB = {
    "cf": "recall@10",
    "ours@0.05": "recall@10",
    "ours@0.0": "recall@10",
    "pop": "recall@10",
}
# 稿件基线列（来自 eval_v5.json，未去重语义）
MANUSCRIPT_BASE = {"cf": 0.0176, "ours@0.05": 0.0137, "ours@0.0": 0.0162, "pop": 0.0100}


def _load(name: str) -> dict:
    p = SPLIT_DIR / name
    if not p.exists():
        raise SystemExit(f"缺少产物: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    old = _load("eval_v6_dedupbase_20261012.json")
    new = _load("eval_v6.json")
    anchor = _load("eval_v5.json")  # 未加权基线 = 稿件冻结值

    print("=" * 78)
    print("候选池")
    print("=" * 78)
    for tag, r in (("旧(去重基线)", old), ("新(未去重)", new), ("v5 基线", anchor)):
        print(f"{tag:>14}: mean={r['pool']['mean_size']:<8} ceiling={r['pool']['ceiling_recall']}")

    print()
    print("=" * 78)
    print("稿件 §多行为 引用的 4 个数字")
    print("=" * 78)
    print(f"{'方法':<12}{'基线(v5)':>10}{'旧加权':>10}{'新加权':>10}"
          f"{'旧Δ%':>9}{'新Δ%':>9}{'稿件需改?':>11}")
    need_edit = False
    for m, metric in MANUSCRIPT_MB.items():
        base = MANUSCRIPT_BASE[m]
        o = old["metrics"][m][metric]
        n = new["metrics"][m][metric]
        o_pct = (o - base) / base * 100
        n_pct = (n - base) / base * 100
        changed = abs(o - n) > 5e-5
        need_edit |= changed
        print(f"{m:<12}{base:>10.4f}{o:>10.4f}{n:>10.4f}"
              f"{o_pct:>8.1f}%{n_pct:>8.1f}%{'是' if changed else '否':>11}")

    print()
    print("=" * 78)
    print("其余方法是否漂移（全量）")
    print("=" * 78)
    drift = 0
    for m in old["metrics"]:
        for k in old["metrics"][m]:
            o, n = old["metrics"][m][k], new["metrics"][m][k]
            if abs(o - n) > 5e-5:
                print(f"  {m:<16}{k:<14}{o:.4f} → {n:.4f}")
                drift += 1
    print(f"  漂移项合计: {drift}")
    print()
    print("结论：稿件多行为段落" + ("【需要刷新】" if need_edit else "【无需改动】"))


if __name__ == "__main__":
    main()
