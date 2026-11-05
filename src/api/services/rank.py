"""排序层：召回候选融合（transfer.fuse）+ α 可控多样性重排（diversity.rerank）。

管线契约：召回层产出两路候选后，本模块为并集中每个商品分别计算辅助域分（全局
热度）与目标域分（品类内排名分），逐元素按置信度融合为单一分数，再交给
diversity.rerank 做 alpha 可控的品类多样性重排。融合语义与离线评估一致。
"""

from src.models.diversity import rerank
from src.models.transfer import fuse

DEFAULT_ALPHA = 0.3
DEFAULT_CONFIDENCE = 0.7

_Candidate = tuple[int, int, float]  # (item_id, category_id, score)


def _aligned_scores(
    candidates_tgt: list[_Candidate], candidates_aux: list[_Candidate], conf: float
) -> tuple[list[_Candidate], dict[int, str]]:
    """对齐两路候选并按置信度融合为单一分数列表，同时给出每个 item 的召回来源。

    融合语义**与离线评估一致**（参见 ``scripts/run_eval_v3.py``）：对候选并集中的
    **每一个**商品分别计算辅助域分与目标域分，再逐元素插值——而不是只对"两路同时
    召回的商品"做融合。

    两路分数定义：
    - **辅助域分 aux**：商品的全局热度分。并集中每个商品都有值（仅被目标路召回的
      商品用其自身热度分补齐），对应离线侧的 ``0.5*热度 + 0.5*物物相似`` 辅助信号。
    - **目标域分 tgt**：品类内排名分 ``1/(1+品类内热度名次)``，**仅**画像品类内的
      候选非零（其余为 0），对应离线侧的 ``rank_normalize(品类内排名分)``。

    逐商品置信度 c：目标域候选取 ``conf``（偏品类内信号），仅被辅助路召回的候选取
    ``0.0``（纯辅助域分，与融合前行为一致）。由此 ``fuse`` 对两类候选都真实执行：
    ``out = aux*(1-c) + tgt*c``。

    ⚠️ 历史缺陷：早期实现只对两路交集调用 ``fuse``。由于目标路取"画像品类内"、
    辅助路取"画像品类外 NOT IN"，两路**结构性互斥**、交集恒为空，``fuse`` 与
    ``DEFAULT_CONFIDENCE`` 在主链路从未执行（创新点②无真实证据）。本实现改为全并集
    逐商品融合后，融合对目标域候选真实生效。

    来源语义：目标域候选→"fused"（aux 与 tgt 真实融合）、仅辅助路→"aux"，
    供前端打推荐原因标签与 recommend_log 记录真实召回路。
    """
    tgt_scores = {item: (cate, score) for item, cate, score in candidates_tgt}
    aux_scores = {item: (cate, score) for item, cate, score in candidates_aux}
    # 并集顺序：目标域在前、辅助路补入，保证结果可复现
    union: list[int] = list(tgt_scores)
    union.extend(item for item in aux_scores if item not in tgt_scores)

    # 目标域分：品类内热度名次的位置衰减分（仅目标域候选非零）。
    # 按**各类目内**分别排名（对齐离线侧 cat_top[c] 的语义）：同一商品在
    # 「全局热度」与「品类内名次」上是两个不同信号，故融合会真实改变排序；
    # 若改为对全部候选统一排名，则该分与热度分同序，融合退化为无效果。
    by_cate: dict[int, list[int]] = {}
    for item, (cate, _score) in tgt_scores.items():
        by_cate.setdefault(cate, []).append(item)
    cat_score: dict[int, float] = {}
    for cate, items in by_cate.items():
        ordered_items = sorted(items, key=lambda i: tgt_scores[i][1], reverse=True)
        for rank, item in enumerate(ordered_items):
            cat_score[item] = 1.0 / (1.0 + rank)

    # 注意：不可用 dict.get(i, tgt_scores[i]) —— 默认值会被提前求值，
    # 仅辅助路候选在 tgt_scores 中不存在，会抛 KeyError。
    aux_list = [
        aux_scores[i][1] if i in aux_scores else tgt_scores[i][1] for i in union
    ]
    tgt_list = [cat_score.get(i, 0.0) for i in union]
    conf_list = [conf if i in tgt_scores else 0.0 for i in union]

    fused = fuse(aux=aux_list, tgt=tgt_list, confs=conf_list)
    merged: dict[int, tuple[int, float]] = {}
    sources: dict[int, str] = {}
    for item, score in zip(union, fused):
        cate = tgt_scores[item][0] if item in tgt_scores else aux_scores[item][0]
        merged[item] = (cate, score)
        # 目标域候选经 aux/tgt 两路分数真实融合，故标记为 fused（不再有"仅目标路"）
        sources[item] = "fused" if item in tgt_scores else "aux"
    return (
        [(item, cate, score) for item, (cate, score) in merged.items()],
        sources,
    )


def rank_with_sources(
    candidates_tgt: list[_Candidate],
    candidates_aux: list[_Candidate],
    k: int,
    alpha: float = DEFAULT_ALPHA,
    conf: float = DEFAULT_CONFIDENCE,
) -> list[tuple[int, str]]:
    """同 ``rank``，但额外透出每个 item 的召回来源 (item_id, source)。

    source ∈ {"fused", "aux"}：目标域候选经两路分数真实融合故为 "fused"，
    仅被辅助路召回的候选为 "aux"；无画像时全部候选均为辅助路兜底，语义
    由调用方（recommend 路由）改写为 "hot"——本层只看得到两路候选。
    """
    if not candidates_tgt and not candidates_aux:
        return []
    merged, sources = _aligned_scores(candidates_tgt, candidates_aux, conf)
    ordered = rerank(merged, alpha=alpha, k=k)
    return [(item, sources.get(item, "tgt")) for item in ordered]


def rank(
    candidates_tgt: list[_Candidate],
    candidates_aux: list[_Candidate],
    k: int,
    alpha: float = DEFAULT_ALPHA,
    conf: float = DEFAULT_CONFIDENCE,
) -> list[int]:
    """融合两路候选分数并做 α 多样性重排，返回前 k 个 item_id。

    candidates_tgt / candidates_aux: [(item_id, category_id, score)]，
    分别来自品类内召回（目标域）与跨品类迁移召回（辅助域）。

    conf 为目标域候选拿到目标域分数的信任度，即 ``fuse`` 中的逐元素 c；
    仅被辅助路召回的候选强制 c=0（纯辅助域分）。
    默认 0.7 的取值依据：当前仅有热度信号、无历史点击反馈可校准置信度，
    迁移信号（辅助路）弱于品类内信号（目标路），故信任度略偏目标域
    （0.7 > 0.5）以抑制迁移噪声；但不取 1.0，保留辅助域对重叠候选的
    分数修正与探索信息。

    alpha 与 k 的语义透传 ``diversity.rerank``：alpha=0 纯分数排序，
    alpha=1 最大品类多样性。
    """
    if not candidates_tgt and not candidates_aux:
        return []
    return [item for item, _ in rank_with_sources(
        candidates_tgt, candidates_aux, k, alpha=alpha, conf=conf
    )]
