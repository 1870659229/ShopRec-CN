"""绘制「品智推」四层架构与推荐闭环原理图（学术风，300dpi PNG）。

输出：docs/原理结构图.png（4:3）

设计约束（与项目绘图规则一致）：
- 学术风：宋体/Times New Roman、低饱和配色、无 emoji、简洁边框
- 浅色底 + 深色字（适配浅色主题与论文排版）
- 图形全部由代码绘制，可复现；不做像素级修补
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "docs" / "原理结构图.png"

FIG_W, FIG_H = 8.0, 6.0
DPI = 300
FS_TITLE = 11
FS_LAYER = 9.0
FS_BOX = 7.5
FS_ARROW = 7.0

# 低饱和配色：填充浅、边框深
LAYERS = [
    ("演示层", "#F4ECF7", "#7D3C98",
     ["Vue 3 + Vite + Element Plus", "推荐流 / 商品详情 / 后台看板"]),
    ("服务层", "#FDEBD0", "#CA6F1E",
     ["FastAPI", "/api/recommend      /api/feedback"]),
    ("模型层", "#D5F5E3", "#1E8449",
     ["召回：UserCF / ItemCF / 热度 / 冷区注入",
      "融合：双域置信度加权      重排：$\\alpha$ 可控多样性"]),
    ("数据层", "#D6EAF8", "#2E86C1",
     ["MySQL 8.0.44    item / behavior / user_profile",
      "recommend_log / model_version"]),
]

LOOP = [
    ("推荐请求\n/api/recommend", "#FDEBD0", "#CA6F1E"),
    ("召回 + 融合\n+ $\\alpha$ 重排", "#D5F5E3", "#1E8449"),
    ("推荐结果\nrecommend_log", "#D6EAF8", "#2E86C1"),
    ("用户行为\nbehavior", "#F4ECF7", "#7D3C98"),
    ("反馈达标\n触发重训", "#FADBD8", "#943126"),
]

EDGE_TXT = "#1C2833"
ARROW_CLR = "#566573"


def _box(ax, x: float, y: float, w: float, h: float, label: str,
         face: str, edge: str, fs: float = FS_BOX, bold: bool = False) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.010",
        linewidth=1.0, edgecolor=edge, facecolor=face,
    ))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=EDGE_TXT,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def _arrow(ax, p1: tuple[float, float], p2: tuple[float, float],
           color: str = ARROW_CLR, lw: float = 1.1, ls: str = "-") -> None:
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ============ 上：四层架构（自顶向下堆叠） ============
    ax.text(0.5, 0.985, "品智推：四层系统架构", ha="center", va="top",
            fontsize=FS_TITLE, fontweight="bold", color=EDGE_TXT)

    layer_h, gap = 0.125, 0.030
    top0 = 0.930
    label_w = 0.082
    body_x = label_w + 0.014
    body_w = 1.0 - body_x - 0.014

    tops: list[float] = []
    for i, (name, face, edge, items) in enumerate(LAYERS):
        top = top0 - i * (layer_h + gap)
        bot = top - layer_h
        tops.append(top)
        _box(ax, 0.012, bot, label_w, layer_h, name, face, edge,
             fs=FS_LAYER, bold=True)
        n = len(items)
        pad = 0.012
        bh = (layer_h - pad * (n + 1)) / n
        for j, it in enumerate(items):
            sub_top = top - pad - j * (bh + pad)
            _box(ax, body_x, sub_top - bh, body_w, bh, it, face, edge)

    # 层间契约：箭头 + 右侧小字
    contracts = ["HTTP / JSON", "召回 → 融合 → 重排", "SQL / ORM"]
    for i, c in enumerate(contracts):
        y_from = tops[i] - layer_h          # 第 i 层底部
        y_to = tops[i + 1]                  # 第 i+1 层顶部
        _arrow(ax, (0.42, y_from), (0.42, y_to))
        ax.text(0.435, (y_from + y_to) / 2, c, ha="left", va="center",
                fontsize=FS_ARROW, color=ARROW_CLR, style="italic")

    # ============ 下：推荐闭环 ============
    ax.text(0.5, 0.335, "推荐闭环：反馈驱动模型更新", ha="center", va="top",
            fontsize=FS_TITLE, fontweight="bold", color=EDGE_TXT)

    n = len(LOOP)
    bw, bgap = 0.152, 0.036
    x0 = (1.0 - (n * bw + (n - 1) * bgap)) / 2
    bh2 = 0.088
    by2 = 0.150
    boxes = []
    for i, (label, face, edge) in enumerate(LOOP):
        x = x0 + i * (bw + bgap)
        _box(ax, x, by2, bw, bh2, label, face, edge)
        boxes.append((x, x + bw))

    # 正向流（左→右）
    for i in range(n - 1):
        _arrow(ax, (boxes[i][1], by2 + bh2 / 2), (boxes[i + 1][0], by2 + bh2 / 2))

    # 回流：末盒底部 → 下行 → 左行 → 上行回首盒底部
    x_last = (boxes[-1][0] + boxes[-1][1]) / 2
    x_first = (boxes[0][0] + boxes[0][1]) / 2
    y_back = 0.062
    ax.plot([x_last, x_last], [by2, y_back], color="#943126",
            linewidth=1.1, linestyle="--")
    ax.plot([x_last, x_first], [y_back, y_back], color="#943126",
            linewidth=1.1, linestyle="--")
    _arrow(ax, (x_first, y_back), (x_first, by2), color="#943126", ls="--")
    ax.text(0.5, y_back - 0.028,
            "model_version 记录迭代 · 新反馈达标即触发重训",
            ha="center", va="center", fontsize=FS_ARROW, color="#943126")

    fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white",
                pad_inches=0.12)
    print(f"已生成 {OUT}")


if __name__ == "__main__":
    main()
