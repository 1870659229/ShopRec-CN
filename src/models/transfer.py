"""双迁移 + 不确定性加权融合（创新点②）。

融合辅助品类域与目标品类域两路分数：逐元素以置信度作为目标域信任度，
置信度越高越偏向目标域，越低越偏向辅助域。
"""

import logging

# 与其他模块保持一致：定义模块级 logger；当前无日志输出，保留备用。
logger = logging.getLogger(__name__)


def uncertainty_weight(confs: list[float]) -> list[float]:
    """置信度越高权重越大；归一化后总和为 1。

    说明：这是**跨来源**的相对权重（总和恒为 1），适用于把多个来源
    聚合成一个分布；它不等于「逐元素信任度」，后者见 fuse。
    """
    if not confs:
        return []
    shifted = [max(0.0, c) for c in confs]
    total = sum(shifted)
    if total == 0:
        return [1.0 / len(shifted)] * len(shifted)
    return [c / total for c in shifted]


def fuse(aux: list[float], tgt: list[float], confs: list[float]) -> list[float]:
    """按不确定性加权融合辅助品类与目标品类两路分数。

    逐元素以 clamp 到 [0,1] 的置信度 c 作为目标域权重：
    ``out = aux * (1 - c) + tgt * c`` —— c=1 全取目标域，c=0 全取辅助域。

    注：此处**不**复用 uncertainty_weight —— 后者把权重归一化到总和为 1，
    用于逐元素插值时会丢失幅度信息（例如两个相等置信度恒得 0.5/0.5），
    无法体现「高置信度偏好目标域」的语义。
    """
    return [
        a * (1 - c) + t * c
        for a, t, c in zip(aux, tgt, (max(0.0, min(1.0, x)) for x in confs))
    ]
