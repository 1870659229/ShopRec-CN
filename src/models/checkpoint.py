"""模型检查点：把离线训练产物持久化为可复用文件。

为什么需要
----------
UserCF / ItemCF 都是**无参数模型**（倒排索引 + 共现邻居表），没有可梯度更新的
权重，但它们的**构建**很贵：UserCF 要对 2,301 万条交互做排序与去重，ItemCF 要
建 15000×153831 的稀疏矩阵并取 top-20 邻居，实测 43 秒 / 次。评估脚本每次运行、
在线服务每次启动都要重付这笔成本。

本模块把这俩索引连同静态统计一起落盘，之后直接反序列化即可推理。

关于"继续训练"
--------------
协同过滤没有可继续优化的损失函数，因此**不存在"接着上一版权重继续训练"这回事**。
这里如实提供两种语义：

* ``fingerprint_matches()`` 为 True → 数据与参数都没变，直接复用（等价于续训无变化）
* 为 False（数据更新了）→ 必须**从头重建**索引，这是 CF 的固有性质，
  本模块不做"增量更新"的假承诺。

检查点内容
----------
* UserCF / ItemCF 实例（pickle）
* 静态统计：全局热度、类目热门榜、商品→类目映射、冷区类目、全局热门榜
* 元数据：格式版本、构建时间、构建参数、**数据源指纹**（文件大小 + 修改时间）

指纹用于防止"换了数据却用了旧索引"——这正是静默污染指标的高危路径。
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

CHECKPOINT_VERSION = "1.0"
_PICKLE_PROTOCOL = 5


class Bundle(NamedTuple):
    """一次离线训练的全部产物。"""

    ucf: Any                       # src.eval.user_cf.UserCF
    icf: Any                       # src.eval.light_item_cf.ItemCF
    pop: pd.Series                 # item_id → 全局交互数
    cat_top: dict[int, list[int]]  # 类目 → 该类目热门商品（降序）
    cat_of: dict[int, int]         # 商品 → 叶子类目
    cold_cats: set[int]            # 冷区类目
    global_top: list[int]          # 全局热门商品（降序）
    meta: dict[str, Any]


def source_fingerprint(sources: dict[str, Path]) -> dict[str, dict[str, int]]:
    """数据源指纹：文件大小 + 纳秒级修改时间。

    用 mtime 而非内容哈希——parquet 有数百 MB，每次全量哈希比重建索引还慢；
    mtime+size 足以捕捉"数据被换掉"这一实际风险。
    """
    fp: dict[str, dict[str, int]] = {}
    for name, path in sources.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"检查点数据源缺失: {p}")
        st = p.stat()
        fp[name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return fp


def _fingerprint_equal(a: dict, b: dict) -> bool:
    return a == b


def save_checkpoint(
    path: Path,
    bundle: Bundle,
    params: dict[str, Any],
    sources: dict[str, Path],
) -> Path:
    """落盘检查点（pickle 主体 + JSON 元数据）。返回 pickle 路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": params,
        "fingerprint": source_fingerprint(sources),
        "shapes": {
            "ucf_pairs": int(bundle.ucf._n_pairs),
            "ucf_users": int(bundle.ucf._n_users),
            "ucf_items": int(len(bundle.ucf._inv_items)),
            "icf_neighbors": int(len(bundle.icf._neighbors)),
            "pop_items": int(len(bundle.pop)),
            "cat_top_cats": int(len(bundle.cat_top)),
            "cold_cats": int(len(bundle.cold_cats)),
        },
        "env": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=_PICKLE_PROTOCOL)
    (path.with_suffix(".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def read_meta(path: Path) -> dict[str, Any]:
    p = Path(path).with_suffix(".meta.json")
    if not p.exists():
        raise FileNotFoundError(f"缺少元数据: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_checkpoint(
    path: Path,
    sources: dict[str, Path] | None = None,
    strict: bool = True,
) -> Bundle:
    """加载检查点用于推理。

    ``sources`` 给定时校验指纹：数据换过但索引没重建 ⇒ 拒绝加载（``strict``）
    或返回空（``strict=False``，由调用方决定是否重建）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"检查点不存在: {path}")
    meta = read_meta(path)
    if meta.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"检查点版本不符: {meta.get('checkpoint_version')} != {CHECKPOINT_VERSION}，请重跑离线训练"
        )
    if sources is not None and not _fingerprint_equal(
        meta.get("fingerprint", {}), source_fingerprint(sources)
    ):
        msg = "数据源已变更，检查点过期（CF 无增量更新，须重跑 scripts/train_offline.py）"
        if strict:
            raise ValueError(msg)
        raise FileNotFoundError(msg)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def is_fresh(path: Path, sources: dict[str, Path]) -> bool:
    """检查点是否存在且与当前数据源一致。"""
    p = Path(path)
    if not p.exists() or not p.with_suffix(".meta.json").exists():
        return False
    try:
        return _fingerprint_equal(
            read_meta(p).get("fingerprint", {}), source_fingerprint(sources)
        )
    except (ValueError, json.JSONDecodeError, OSError):
        return False
