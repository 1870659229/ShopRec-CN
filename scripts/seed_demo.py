"""演示数据 seed 脚本：向 *_test 库灌入 9300 段位演示商品、类目与种子行为。

用途：新人按 README 建库建表后，item / category 表为空，端到端 7 步清单
第 2 步（推荐流出卡片）会失败。本脚本幂等地写入 120 个 item（9300001-9300120）、
8 个 category（9301-9308）与约 246 条种子行为（user3 钟表/眼镜纯净画像 +
其余用户对新商品的覆盖式曝光），供演示与联调使用。

编号布局：旧 40 个（9300001-9300040）按类目连续编号保持不变；
新增 80 个（9300041-9300120）按类目追加段分配：
  9301 数码家电 +8（9300041-48）
  9302 服饰鞋包 +32（9300049-80：上装 15 / 鞋 13 / 裤 4）
  9303 美妆个护 +3（9300081-83）
  9304 食品饮料 +1（9300084）
  9306 运动户外 +8（9300085-92）
  9307 钟表 +14（9300093-106，新类目）
  9308 眼镜配饰 +14（9300107-120，新类目）
  9305 家居日用 不变（图池无新增可用素材）

幂等策略：先 DELETE 本段位旧行（item_id 9300001-9300120、category_id
9301-9308、behavior.source='seed'），再 INSERT，重复执行结果一致。

安全护栏：目标库名必须以 ``_test`` 结尾（取 ``SHOPREC_DB_NAME``），
否则拒绝执行——防止误灌生产库或真实演示库。

用法（密码必填）::

    SHOPREC_DB_PASSWORD=<你的密码> SHOPREC_DB_NAME=shoprec_test \
        python scripts/seed_demo.py
"""

import os
import random
import sys
from datetime import datetime, timedelta
from typing import Final

import pymysql

# 演示数据段位：item_id 9300001-9300120，category_id 9301-9308
ITEM_ID_RANGE: Final[tuple[int, int]] = (9300001, 9300120)
CATEGORY_IDS: Final[tuple[int, ...]] = (9301, 9302, 9303, 9304, 9305, 9306, 9307, 9308)

# (category_id, name)：演示用类目名，覆盖 8 个常见电商品类
CATEGORIES: Final[list[tuple[int, str]]] = [
    (9301, "数码家电"),
    (9302, "服饰鞋包"),
    (9303, "美妆个护"),
    (9304, "食品饮料"),
    (9305, "家居日用"),
    (9306, "运动户外"),
    (9307, "钟表"),
    (9308, "眼镜配饰"),
]

# 旧 40 个（9300001-9300040）各类目数量（合计 40），按 9301 → 9306 依次连续编号
_CATEGORY_SIZES_OLD: Final[tuple[int, ...]] = (8, 8, 6, 6, 6, 6)

# 新 80 个（9300041-9300120）按类目追加段：(category_id, 数量)。
# 顺序即编号顺序，pop_score 沿该顺序单调递减（与"9301 类整体最高"叙事一致）。
_NEW_SEGMENTS: Final[list[tuple[int, int]]] = [
    (9301, 8),   # 9300041-9300048
    (9302, 32),  # 9300049-9300080（上装 15 / 鞋 13 / 裤 4）
    (9303, 3),   # 9300081-9300083
    (9304, 1),   # 9300084
    (9306, 8),   # 9300085-9300092
    (9307, 14),  # 9300093-9300106
    (9308, 14),  # 9300107-9300120
]

# 演示标题（旧 40 个，按 item_id 顺序，与 _CATEGORY_SIZES_OLD 展开顺序一致）。
# 红线：JDsearch 数据无标题字段，此列表是演示用文案（与 public/items/ 示意图
# 一一对应），真实数据行为空串；报告与页面标注"演示标题，非数据集内容"。
TITLES_OLD: Final[list[str]] = [
    # 9301 数码家电
    "小米 全面屏智能手机",
    "华为 全面屏智能手机",
    "小米 4G智能手机",
    "华为 大屏双摄手机",
    "小米 4K智能电视 55英寸",
    "OPPO 全面屏智能手机",
    "魅族 智能手机（四色可选）",
    "小米 智能电视（旗舰款）",
    # 9302 服饰鞋包
    "PLAYBOY 直筒牛仔裤（男）",
    "高腰修身牛仔裤（女）",
    "纯棉圆领短袖T恤（男）",
    "经典低帮帆布鞋（米白）",
    "尖头浅口中跟鞋（女）",
    "高帮帆布鞋（情侣款）",
    "低帮休闲帆布鞋（蓝）",
    "长绒棉免烫衬衫",
    # 9303 美妆个护
    "IPSA 流金水乳礼盒",
    "IPSA 光透恒美粉底霜",
    "雅诗兰黛 倾慕唇膏",
    "去屑洗发水洗护套装 3瓶装",
    "IPSA 轻盈防晒乳",
    "CLARINS 水乳焕颜套装",
    # 9304 食品饮料
    "法国原瓶干红葡萄酒 双支装",
    "每日坚果礼盒 8袋装",
    "香辣卤味 熟食零食",
    "波尔多干红葡萄酒 双支装",
    "慕醍甜红葡萄酒",
    "原瓶进口赤霞珠干红",
    # 9305 家居日用
    "扫拖一体扫地机器人",
    "智能扫拖地机器人",
    "激光导航扫地机器人",
    "简约白色收纳架",
    "实木衣架 10支装",
    "纯棉纱布方巾 6条装",
    # 9306 运动户外
    "速干透气运动T恤 男女款",
    "运动休闲套装（女）",
    "速干运动套装（女）",
    "多功能智能运动手表",
    "SUUNTO 松拓运动手表",
    "复古通勤自行车",
]

# 演示价格（旧 40 个，与 TITLES_OLD 一一对应，单位：元）。
# 红线：JDsearch 数据无价格字段，此列表是演示文案（与 title 同理），
# 仅 demo 段位（9300 段）有值；真实数据行为 NULL，不参与指标计算。
PRICES_OLD: Final[list[float]] = [
    # 9301 数码家电
    1999, 2499, 1599, 2699, 2999, 2099, 1799, 3299,
    # 9302 服饰鞋包
    299, 269, 99, 159, 329, 189, 149, 219,
    # 9303 美妆个护
    899, 399, 329, 129, 289, 759,
    # 9304 食品饮料
    298, 139, 49, 268, 189, 328,
    # 9305 家居日用
    2699, 2199, 1899, 89, 59, 39,
    # 9306 运动户外
    129, 299, 259, 699, 1299, 899,
]

# 新 80 个标题（9300041-9300120，与 _NEW_SEGMENTS 展开顺序一致）。
# 红线同上：演示文案与 public/items/9300041-9300120.jpg 一一对应。
TITLES_NEW: Final[list[str]] = [
    # 9301 数码家电（9300041-9300048）
    "魅族 全面屏智能手机（银白）",
    "OPPO 全面屏拍照手机（樱语粉）",
    "魅族 全面屏智能手机（曜石黑）",
    "小米 全面屏智能手机（浅蓝灰）",
    "魅族 旗舰智能手机（深海黑）",
    "OPPO 梦镜全面屏手机（热力红）",
    "小米 全高清智能电视 32英寸",
    "液晶智能平板电视 43英寸",
    # 9302 服饰鞋包·上装（9300049-9300063）
    "PLAYBOY 商务休闲polo衫（白）",
    "潮流枫叶印花短袖T恤（黑）",
    "商务休闲长袖衬衫（蓝）",
    "优雅气质连衣裙（黑）",
    "PLAYBOY 休闲长袖T恤（黑）",
    "纯棉圆领短袖T恤（浅绿）",
    "韩版修身长袖衬衫（女·白）",
    "清新短袖T恤（女·粉）",
    "宽松圆领短袖T恤（军绿）",
    "枫叶印花短袖T恤（白）",
    "条纹短袖polo衫（白）",
    "商务免烫长袖衬衫（白）",
    "卡通情侣款短袖T恤（白）",
    "字母印花短袖T恤（女·白）",
    "碎花雪纺连衣裙（裸粉）",
    # 9302 服饰鞋包·鞋靴（9300064-9300076）
    "轻便缓震跑步鞋（灰）",
    "简约百搭小白鞋（白）",
    "透气网面运动鞋（白）",
    "经典高帮帆布鞋（女·藏蓝）",
    "尖头中跟鞋（白）",
    "涂鸦印花高帮帆布鞋（黑白）",
    "经典低帮帆布鞋（黑）",
    "绑带芭蕾平底鞋（裸粉）",
    "清新高帮帆布鞋（女·浅蓝）",
    "简约休闲板鞋（白）",
    "印花高帮帆布鞋（白）",
    "牛仔布高帮帆布鞋（深蓝）",
    "牛仔布低帮帆布鞋（深蓝）",
    # 9302 服饰鞋包·裤装（9300077-9300080）
    "修身弹力牛仔裤（女）",
    "浅色直筒牛仔裤（女）",
    "休闲运动短裤（男·黑）",
    "高腰小脚牛仔裤（女）",
    # 9303 美妆个护（9300081-9300083）
    "IPSA 茵芙纱 温和洁面乳",
    "IPSA 茵芙纱 清透洁面乳",
    "IPSA 流金岁月水乳礼盒",
    # 9304 食品饮料（9300084）
    "梦陇 法国原瓶进口干红葡萄酒",
    # 9306 运动户外（9300085-9300092）
    "运动织带腰带（军绿）",
    "速干透气运动T恤（女·荧光绿）",
    "运动休闲高帮帆布鞋（女·浅蓝）",
    "复古通勤自行车（青柠绿）",
    "轻质缓震训练鞋（白）",
    "透气休闲运动板鞋（白）",
    "轻弹慢跑鞋（灰）",
    "运动休闲短袖T恤（男·军绿）",
    # 9307 钟表（9300093-9300106）
    "Daniel Wellington 经典石英腕表（黑）",
    "天梭 力洛克机械男表（黑盘·钢带）",
    "卡西欧 G-SHOCK 运动电子表",
    "全自动镂空机械表（黑带）",
    "商务休闲机械表（白盘）",
    "满天星情侣对表（银）",
    "经典罗马刻度石英表（银）",
    "Daniel Wellington 玫瑰金腕表",
    "天梭 罗马刻度石英男表",
    "玫瑰金镂空机械表（黑带）",
    "Daniel Wellington 黑盘金框腕表",
    "天梭 经典机械男表（黑盘）",
    "卡西欧 G-SHOCK 金属带电子表",
    "卡西欧 G-SHOCK 经典款（黑金）",
    # 9308 眼镜配饰（9300107-9300120）
    "暴龙 偏光飞行员墨镜（蓝）",
    "暴龙 高清偏光太阳镜（黑）",
    "暴龙 偏光驾驶镜（蓝）",
    "暴龙 女士渐变墨镜（紫粉）",
    "暴龙 男士偏光墨镜（军绿）",
    "暴龙 经典飞行员墨镜（黑）",
    "暴龙 复古方框太阳镜（墨绿）",
    "暴龙 优雅圆框墨镜（金）",
    "暴龙 商务偏光墨镜（灰）",
    "施华洛世奇 恶魔眼项链",
    "施华洛世奇 蝴蝶结手链（粉）",
    "施华洛世奇 恶魔眼锁骨链",
    "施华洛世奇 恶魔眼项链套装",
    "PANDORA 蝴蝶结银手链",
]

# 新 80 个价格（与 TITLES_NEW 一一对应，单位：元；红线同上）。
PRICES_NEW: Final[list[float]] = [
    # 9301 数码家电
    1299, 1799, 1499, 1699, 1999, 2499, 899, 1199,
    # 9302 上装
    129, 89, 159, 599, 119, 69, 139, 79, 99, 85, 135, 169, 99, 75, 199,
    # 9302 鞋靴
    269, 159, 199, 139, 399, 149, 129, 219, 135, 169, 145, 155, 139,
    # 9302 裤装
    189, 179, 69, 195,
    # 9303 美妆个护
    259, 239, 799,
    # 9304 食品饮料
    168,
    # 9306 运动户外
    59, 119, 129, 899, 289, 159, 249, 89,
    # 9307 钟表
    1399, 3299, 899, 599, 699, 1099, 799, 1299, 2899, 899, 1199, 3599, 1099, 949,
    # 9308 眼镜配饰
    499, 459, 469, 519, 489, 479, 449, 529, 509, 899, 799, 859, 999, 699,
]

# 价格与标题必须逐位对齐：错位 = 商品挂错价（品牌错配同类事故），导入即校验
if len(PRICES_OLD) != len(TITLES_OLD):
    raise ValueError(
        f"PRICES_OLD({len(PRICES_OLD)}) 与 TITLES_OLD({len(TITLES_OLD)}) 数量不一致，"
        "请保持两个列表按 item_id 顺序一一对应"
    )
if len(PRICES_NEW) != len(TITLES_NEW):
    raise ValueError(
        f"PRICES_NEW({len(PRICES_NEW)}) 与 TITLES_NEW({len(TITLES_NEW)}) 数量不一致，"
        "请保持两个列表按 item_id 顺序一一对应"
    )


def _ensure_title_column(conn: "pymysql.connections.Connection") -> None:
    """存量库迁移：item 表若无 title 列则补上（新库由 schema.sql 直接建出）。

    MySQL 8.0 不支持 ADD COLUMN IF NOT EXISTS，故查 information_schema 判断。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.COLUMNS"
            " WHERE TABLE_SCHEMA = DATABASE()"
            " AND TABLE_NAME = 'item' AND COLUMN_NAME = 'title'"
        )
        if cur.fetchone()[0] == 0:
            cur.execute(
                "ALTER TABLE item ADD COLUMN title VARCHAR(128) DEFAULT ''"
                " AFTER source"
            )
            print("[seed_demo] 已迁移：item 表新增 title 列")


def _ensure_demo_price_column(conn: "pymysql.connections.Connection") -> None:
    """存量库迁移：item 表若无 demo_price 列则补上（新库由 schema.sql 直接建出）。

    MySQL 8.0 不支持 ADD COLUMN IF NOT EXISTS，故查 information_schema 判断。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.COLUMNS"
            " WHERE TABLE_SCHEMA = DATABASE()"
            " AND TABLE_NAME = 'item' AND COLUMN_NAME = 'demo_price'"
        )
        if cur.fetchone()[0] == 0:
            cur.execute(
                "ALTER TABLE item ADD COLUMN demo_price DECIMAL(10,2) DEFAULT NULL"
                " AFTER title"
            )
            print("[seed_demo] 已迁移：item 表新增 demo_price 列")


def _build_items() -> list[tuple[int, int, str, float, float]]:
    """生成 (item_id, category_id, title, pop_score, demo_price) 列表。

    编号布局：旧 40 个按 _CATEGORY_SIZES_OLD 连续编号，pop_score 全局单调
    递减（0.98 起、步长 0.022）——与历史演示数据完全一致，不做任何改动；
    新 80 个按 _NEW_SEGMENTS 类目追加编号，pop_score 自 0.680 起步长 0.0042
    递减（最低 0.348），定位"当季热销"量纲——高于旧段尾部（0.122）但低于
    旧爆款（0.98），符合真实电商"新品带流量、经典爆款居首"的热度结构。
    量纲依据：在线排序辅助路取 pop_score 原值、目标域分 1/(1+名次) 与
    conf=0.7 融合，新商品 pop 过低（<0.13）会被旧爆款完全压制，个性化
    召回在 top40 中仅剩零星名额；抬到 0.35+ 后"类目偏好"能在推荐流中
    获得合理曝光（详见 2026-10-15 扩容记录）。
    """
    items: list[tuple[int, int, str, float, float]] = []
    nid = ITEM_ID_RANGE[0] - 1
    for cid, size in zip(CATEGORY_IDS[:6], _CATEGORY_SIZES_OLD):
        for _ in range(size):
            nid += 1
            items.append(
                (
                    nid,
                    cid,
                    TITLES_OLD[len(items)],
                    round(0.98 - 0.022 * len(items), 3),
                    PRICES_OLD[len(items)],
                )
            )
    # 新 80 个：9300041-9300120
    for i, (cid, size) in enumerate(_NEW_SEGMENTS):
        for _ in range(size):
            nid += 1
            idx_new = len(items) - len(TITLES_OLD)
            items.append(
                (
                    nid,
                    cid,
                    TITLES_NEW[idx_new],
                    round(0.680 - 0.0042 * idx_new, 3),
                    PRICES_NEW[idx_new],
                )
            )
    return items


# (item_id, category_id, title, pop_score, demo_price)
ITEMS: Final[list[tuple[int, int, str, float, float]]] = _build_items()


def _build_behaviors() -> list[tuple[int, int, str, float | None]]:
    """生成种子行为 (user_id, item_id, behavior_type, rating) 列表。

    设计目标：扩容 120 商品后协同过滤信号不被稀释，且让演示叙事更清晰。
      - user3（原 0 行为、画像纯净）：钟表 + 眼镜配饰全类目
        EXPOSE×2 + CLICK×1，另 ORD 两件（一块表 + 一副墨镜）——
        前端以 user3 登录即可演示"浏览钟表 → 推荐全是钟表/配饰"。
      - user1/2/4/5/6：对全部 80 个新商品轮转覆盖 2 条 EXPOSE，
        保证冷启动曝光覆盖（不稀释各自既有画像权重）。
    合计 28×3 + 2 + 80×2 = 246 条，全部以 source='seed' 写入（幂等键）。
    """
    rows: list[tuple[int, int, str, float | None]] = []
    clock_items = list(range(9300093, 9300107))
    glass_items = list(range(9300107, 9300121))
    for iid in clock_items + glass_items:
        rows.append((3, iid, "EXPOSE", None))
        rows.append((3, iid, "EXPOSE", None))
        rows.append((3, iid, "CLICK", None))
    rows.append((3, 9300093, "ORD", 5))
    rows.append((3, 9300107, "ORD", 5))
    others = [1, 2, 4, 5, 6]
    for k, iid in enumerate(range(9300041, 9300121)):
        rows.append((others[k % 5], iid, "EXPOSE", None))
        rows.append((others[(k + 2) % 5], iid, "EXPOSE", None))
    return rows


# (user_id, item_id, behavior_type, rating)
BEHAVIORS: Final[list[tuple[int, int, str, float | None]]] = _build_behaviors()


def resolve_test_db_name() -> str:
    """读取 SHOPREC_DB_NAME，校验必须以 _test 结尾，否则退出。

    Returns:
        合法的目标库名。

    Raises:
        SystemExit: 库名缺失或不以 _test 结尾时，打印原因并退出。
    """
    db_name = os.environ.get("SHOPREC_DB_NAME", "").strip()
    if not db_name:
        print("[seed_demo] 拒绝执行：未设置 SHOPREC_DB_NAME。", file=sys.stderr)
        print("请显式指定一个以 _test 结尾的库名，例如：", file=sys.stderr)
        print("  SHOPREC_DB_NAME=shoprec_test python scripts/seed_demo.py", file=sys.stderr)
        raise SystemExit(1)
    if not db_name.endswith("_test"):
        print(f"[seed_demo] 拒绝执行：库名 {db_name!r} 不以 _test 结尾。", file=sys.stderr)
        print("护栏目的：防止误灌生产库 / 真实演示库。", file=sys.stderr)
        print("请改用测试库名，例如 SHOPREC_DB_NAME=shoprec_test。", file=sys.stderr)
        raise SystemExit(1)
    return db_name


def _seed_ts(i: int) -> datetime:
    """确定性生成种子行为时间戳：最近 10 天内均匀散布（固定种子可重现）。"""
    rng = random.Random(20261015 + i)
    delta = timedelta(
        days=rng.uniform(0, 10),
        hours=rng.uniform(0, 24),
        minutes=rng.uniform(0, 60),
        seconds=rng.uniform(0, 60),
    )
    return datetime.now() - delta


def _refresh_profiles() -> None:
    """seed 后刷新受影响用户的实时画像，让推荐流立即反映种子行为。

    复用 src.api.services.profile.refresh_profile（与线上同一折叠逻辑：
    recent_items 最近 50 条、interest_tags 最近 10 个类目）。失败不阻塞
    seed 本身——用户登录互动后画像会自动补算。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from src.api.services.profile import refresh_profile

        user_ids = sorted({b[0] for b in BEHAVIORS})
        for u in user_ids:
            refresh_profile(u)
        print(f"[seed_demo] 已刷新 {len(user_ids)} 个用户的实时画像: {user_ids}")
    except Exception as exc:  # noqa: BLE001 - 刷新失败不应阻塞 seed 结果
        print(f"[seed_demo] 画像刷新跳过（用户登录互动后会自动补算）：{exc}", file=sys.stderr)


def seed(db_name: str) -> None:
    """向目标库幂等写入演示 item、category 与种子 behavior，并打印 COUNT。

    Args:
        db_name: 已通过 _test 护栏校验的目标库名。
    """
    conn = pymysql.connect(
        host=os.environ.get("SHOPREC_DB_HOST", "localhost"),
        port=int(os.environ.get("SHOPREC_DB_PORT", "3306")),
        user=os.environ.get("SHOPREC_DB_USER", "root"),
        password=os.environ.get("SHOPREC_DB_PASSWORD", ""),
        database=db_name,
        charset="utf8mb4",
    )
    cat_ph = ", ".join(["%s"] * len(CATEGORY_IDS))
    lo, hi = ITEM_ID_RANGE
    try:
        with conn:
            with conn.cursor() as cur:
                # 存量库迁移：补 title 列（新库由 schema.sql 直接建出）
                _ensure_title_column(conn)
                # 存量库迁移：补 demo_price 列（新库由 schema.sql 直接建出）
                _ensure_demo_price_column(conn)
                # 幂等：先清本段位旧行，再插入
                cur.execute(
                    "DELETE FROM item WHERE item_id BETWEEN %s AND %s",
                    (lo, hi),
                )
                cur.execute(
                    f"DELETE FROM category WHERE category_id IN ({cat_ph})",
                    CATEGORY_IDS,
                )
                # 种子行为幂等：只清 source='seed' 的行，保留真实演示积累
                cur.execute("DELETE FROM behavior WHERE source = 'seed'")
                cur.executemany(
                    "INSERT INTO item (item_id, category_id, source, title, pop_score, demo_price) "
                    "VALUES (%s, %s, 'demo', %s, %s, %s)",
                    ITEMS,
                )
                cur.executemany(
                    "INSERT INTO category (category_id, parent_id, level, name) "
                    "VALUES (%s, 0, 1, %s)",
                    CATEGORIES,
                )
                cur.executemany(
                    "INSERT INTO behavior (user_id, item_id, behavior_type, rating, ts, source) "
                    "VALUES (%s, %s, %s, %s, %s, 'seed')",
                    [
                        (u, iid, btype, rating, _seed_ts(i).strftime("%Y-%m-%d %H:%M:%S"))
                        for i, (u, iid, btype, rating) in enumerate(BEHAVIORS)
                    ],
                )
                conn.commit()

                # 打印 COUNT 供核对
                cur.execute("SELECT COUNT(*) FROM item WHERE item_id BETWEEN %s AND %s", (lo, hi))
                item_count = cur.fetchone()[0]
                cur.execute(
                    "SELECT category_id, COUNT(*) FROM item "
                    "WHERE item_id BETWEEN %s AND %s GROUP BY category_id ORDER BY category_id",
                    (lo, hi),
                )
                per_category = cur.fetchall()
                cur.execute(
                    f"SELECT COUNT(*) FROM category WHERE category_id IN ({cat_ph})",
                    CATEGORY_IDS,
                )
                category_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM behavior WHERE source = 'seed'")
                behavior_count = cur.fetchone()[0]

        print(f"[seed_demo] 库 {db_name} 写入完成：")
        print(f"  item（{lo}-{hi}）: {item_count} 行")
        for cate_id, cnt in per_category:
            print(f"    category {cate_id}: {cnt} 行")
        print(f"  category（{'/'.join(str(c) for c in CATEGORY_IDS)}）: {category_count} 行")
        print(f"  behavior（source='seed'）: {behavior_count} 行")
    except pymysql.err.OperationalError as exc:
        print(f"[seed_demo] 连接或查询失败：{exc}", file=sys.stderr)
        print("请确认 MySQL 已启动、表已建好（先按 README 执行 init_db），且密码正确。", file=sys.stderr)
        raise SystemExit(2) from exc


def main() -> None:
    """入口：校验库名护栏后执行 seed。"""
    db_name = resolve_test_db_name()
    seed(db_name)
    _refresh_profiles()


if __name__ == "__main__":
    main()
