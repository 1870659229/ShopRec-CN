"""ours 打分模块：候选池组装 + 三条打分线。

**定位**：离线评估与在线服务的共同实现（单一事实源）。

当前接入状态（**引用本模块前务必核对**）
----------------------------------------
- 离线评估 ``scripts/run_eval_v6.py``：**已接入**，本模块是其 ours 打分的唯一实现。
- 在线服务 ``src/api/services/rank.py``：**尚未接入**。在线目前走
  ``transfer.fuse``（目标域=品类内名次，辅助域=全局热度，置信度 0.7）
  加 ``diversity.rerank``，与离线 ours 的加权公式**不同源**。

  因此"论文指标 == 线上实际行为"在当前代码状态下**还不成立**，
  只有在线改调本模块之后才成立。这也是先锁死离线锚点、再迁移在线侧的原因：
  迁移后无需重跑评估即可断言同源。

搬运纪律
--------
本模块的代码由离线评估脚本**原样搬移**而来，不得"顺手优化"。原因是候选池的组装
顺序会影响最终排序：``ucats`` 与 ``cold_cats`` 均为 ``set``，其迭代顺序参与决定
候选进入池子的先后；改为排序后的 list 或其他容器都会静默改变结果。任何改动都必须
以冒烟回归（150 用户、锚点模式）与 ``eval_v5.json`` 逐位一致为准。

⚠️ 已踩过的坑（改动时极易复发）
------------------------------
**ItemCF 的历史必须保留重复行**。``ItemCF.recommend`` 是
``for it in history: acc[nb] += w``，同一商品的多条行为行会被重复累加（隐式频次
加权）。若把历史先经 ``dict`` 去重再传给 ItemCF，候选池组成就会漂移：实测 150 用户
冒烟下 pool 均值 2644.0→2643.7，全量下 ours R@10 0.0162→0.0163、pop 的 Div@10
0.3929→0.3930——连与 ItemCF 无直接关系的 ``pop`` 行都会跟着变。
**UserCF 侧才用去重后的加权字典**（同一商品多行为取 max），两侧输入不是一个东西。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.eval.user_cf import rank_normalize

# ---- 候选池组装参数（离线与在线共用，改这里两边同时生效）----
CF_ITEMS = 1200          # UserCF 召回条数
ITEM_CF_ITEMS = 200      # ItemCF 召回条数
PER_CAT = 60             # 用户历史品类内，每类目补充的热门条数
GLOBAL_TOP = 300         # 全局热门补充条数
COLD_INJECT_PER_CAT = 12  # 每个冷区类目注入条数
COLD_INJECT_CAT_CAP = 40  # 单用户最多注入的冷区类目数
POOL_CAP = 3000          # 候选池上限

# ---- 打分权重 ----
COLD_BOOST = 0.20        # 冷区商品的分数加成


def user_categories(hist: list[int], cat_of: dict[int, int]) -> set[int]:
    """用户历史商品覆盖到的叶子类目集合。

    返回 ``set`` 而非 list：候选池组装依赖其迭代顺序，见模块 docstring。
    """
    return {cat_of[i] for i in hist if i in cat_of}


class OursScorer:
    """候选池组装与 ours 打分。

    持有静态统计（热度、类目热门榜、冷区类目、商品→类目映射），
    用户相关的部分（历史、CF 召回结果）由调用方传入。
    """

    def __init__(
        self,
        pop: Mapping[int, float],
        cat_top: Mapping[int, list[int]],
        cat_of: dict[int, int],
        cold_cats: set[int],
        global_top: list[int],
        cold_inject_per_cat: int = COLD_INJECT_PER_CAT,
        cold_boost: float = COLD_BOOST,
    ) -> None:
        self._pop = pop
        self._cat_top = cat_top
        self._cat_of = cat_of
        self._cold_cats = cold_cats
        self._global_top = global_top
        # 冷区注入条数与冷区加分做成实例参数：消融实验需要单独关掉其中之一，
        # 以区分「召回层注入」与「排序层加分」各自的贡献。默认取模块常量，
        # 因此不传参时行为与改造前逐位一致。
        self.cold_inject_per_cat = cold_inject_per_cat
        self.cold_boost = cold_boost
        # 冷区类目 → 该类目可注入的商品（取类目热门榜前 20）
        self._cold_items_by_cat: dict[int, list[int]] = {
            c: cat_top[c][:20] for c in cold_cats if c in cat_top
        }

    @property
    def cold_cats(self) -> set[int]:
        """冷区类目集合（用于计算冷区命中与冷区加分）。"""
        return self._cold_cats

    def build_pool(
        self,
        hist_set: set[int],
        ucats: set[int],
        cf_items: list[int],
        icf_items: list[int],
    ) -> tuple[list[int], int]:
        """组装候选池，返回 ``(pool, n_cold_injected)``。

        来源顺序：UserCF → ItemCF → 用户历史品类热门 → 全局热门 → 冷区注入。
        已在历史中或已入池的商品跳过。
        """
        pool: list[int] = []
        mark: set[int] = set()
        for i in cf_items:
            if i not in hist_set and i not in mark:
                mark.add(i)
                pool.append(int(i))
        for i in icf_items:
            if i not in hist_set and i not in mark:
                mark.add(i)
                pool.append(int(i))
        for c in ucats:
            for i in self._cat_top.get(c, ())[:PER_CAT]:
                if i not in hist_set and i not in mark:
                    mark.add(i)
                    pool.append(int(i))
        for i in self._global_top[:GLOBAL_TOP]:
            if i not in hist_set and i not in mark:
                mark.add(i)
                pool.append(int(i))

        n_cold_injected = 0
        user_cold_cats = [c for c in ucats if c in self._cold_cats] or list(self._cold_cats)[:10]
        for c in user_cold_cats[:COLD_INJECT_CAT_CAP]:
            for i in self._cold_items_by_cat.get(c, [])[: self.cold_inject_per_cat]:
                if i not in hist_set and i not in mark:
                    mark.add(i)
                    pool.append(int(i))
                    n_cold_injected += 1
        return pool[:POOL_CAP], n_cold_injected

    def score(
        self,
        pool: list[int],
        cf_map: dict[int, float],
        icf_map: dict[int, float],
    ) -> tuple[list[int], dict[str, np.ndarray]]:
        """对候选池打分，返回 ``(品类列表, {打分线: 分数数组})``。

        三条打分线：``pop``（全局热度）、``cf``（UserCF）、``ours``（融合分）。
        在线服务只用 ``ours``；离线评估三条都要，用于同池同协议下的分解对比。

        ours = 0.5·CF + 0.3·(0.5·热度 + 0.5·ItemCF) + 0.2·品类内名次 + 冷区加成
        """
        cat_of = self._cat_of
        pc = [cat_of.get(i, 0) for i in pool]
        raw_pop = np.array([float(self._pop.get(i, 0)) for i in pool])
        raw_cf = np.array([float(cf_map.get(i, 0.0)) for i in pool])
        raw_icf = np.array([float(icf_map.get(i, 0.0)) for i in pool])

        # 品类内名次分 1/(1+名次)：同一商品在「全局热度」与「品类内名次」上是
        # 两个不同信号，故融合会真实改变排序。
        cat_rank: dict[int, list[int]] = {}
        for i, c in zip(pool, pc):
            cat_rank.setdefault(c, []).append(i)
        cat_pos: dict[int, float] = {}
        for c, items in cat_rank.items():
            if c not in self._cat_top:
                continue
            pos = {it: r for r, it in enumerate(self._cat_top[c])}
            for it in items:
                if it in pos:
                    cat_pos[it] = 1.0 / (1.0 + pos[it])
        raw_cat = np.array([cat_pos.get(i, 0.0) for i in pool])

        s_pop = rank_normalize(raw_pop)
        s_cf = rank_normalize(raw_cf) if raw_cf.max() > 0 else raw_cf
        s_icf = rank_normalize(raw_icf) if raw_icf.max() > 0 else raw_icf
        s_cat = rank_normalize(raw_cat)

        coldness = np.array([1.0 if c in self._cold_cats else 0.0 for c in pc])
        aux = 0.5 * s_pop + 0.5 * s_icf
        final_score = 0.5 * s_cf + 0.3 * aux + 0.2 * s_cat + self.cold_boost * coldness

        return pc, {
            "pop": s_pop,
            "cf": s_cf,
            "ours": final_score,
        }
