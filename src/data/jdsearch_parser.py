"""JDsearch 原始行解析器。

JDsearch（京东，SIGIR 2023）两个原始文件均为**制表符分隔 + 有表头**的纯文本：

- ``product_meta_data.txt``（4.51 GB，13 列）：商品元数据；
- ``user_behavior_data.txt``（912.8 MB，7 列）：搜索会话日志。

分隔符约定（实测核验）：

- 含 term id 的字段内部 token 用 ``\\x18``（chr 24）分隔（``TOKEN_SEP``）；
- ``*list`` 字段的元素用 ``_`` 分隔（``LIST_SEP``）；
- ``cate_id_1`` ~ ``cate_id_4`` 各为**单 token**（4 级类目层级），
  而 ``name`` / ``brand_name`` / ``cate_name_1`` ~ ``cate_name_4`` **可能是多 token**。

本模块只做「一行字符串 → dict」，不读文件、不依赖 pandas，便于单测。

容错约定（Important 1）：脏数据**不抛异常中断全量**——``*list[float]`` 字段里无法转
float 的 token 被跳过并计数（WARNING 日志汇总），列数不足的行按列语义补 ``""``/``[]``。
"""
import logging

logger = logging.getLogger(__name__)

TOKEN_SEP: str = "\x18"
LIST_SEP: str = "_"

META_COLUMNS: tuple[str, ...] = (
    "wid",
    "name",
    "brand_id",
    "brand_name",
    "cate_id_1",
    "cate_name_1",
    "cate_id_2",
    "cate_name_2",
    "cate_id_3",
    "cate_name_3",
    "cate_id_4",
    "cate_name_4",
    "shop_id",
)

BEHAVIOR_COLUMNS: tuple[str, ...] = (
    "query",
    "candidate_wid_list",
    "candidate_label_list",
    "history_qry_list",
    "history_wid_list",
    "history_type_list",
    "history_time_list",
)

# 元数据中按 TOKEN_SEP 拆成多 token 的字段（其余为单 token 字符串）
META_MULTI_COLUMNS: frozenset[str] = frozenset(
    {"name", "brand_name", "cate_name_1", "cate_name_2", "cate_name_3", "cate_name_4"}
)

# 行为表中按 LIST_SEP 拆成 list[str] 的字段
BEHAVIOR_LIST_COLUMNS: tuple[str, ...] = (
    "candidate_wid_list",
    "history_qry_list",
    "history_wid_list",
    "history_type_list",
)

# 行为表中按 LIST_SEP 拆成 list[float] 的字段
BEHAVIOR_FLOAT_COLUMNS: tuple[str, ...] = (
    "candidate_label_list",
    "history_time_list",
)


def _pad(cells: list[str], width: int) -> list[str]:
    """把切分结果补齐到 width 列（多出的列忽略），容忍残缺行。"""
    if len(cells) < width:
        return cells + [""] * (width - len(cells))
    return cells[:width]


def _split_tokens(raw: str) -> list[str]:
    """空串 → ``[]``；否则按 ``TOKEN_SEP`` 切分。"""
    return raw.split(TOKEN_SEP) if raw else []


def _split_list(raw: str) -> list[str]:
    """空串 → ``[]``；否则按 ``LIST_SEP`` 切分（``-1`` 等原样保留）。"""
    return raw.split(LIST_SEP) if raw else []


def _split_floats(raw: str) -> list[float]:
    """空串 → ``[]``；否则按 ``LIST_SEP`` 切分并逐个转 float。

    无法转 float 的 token（脏数据）**跳过并计数、绝不抛异常**；本行存在跳过时以
    WARNING 记录计数与样例 token，便于全量入库后汇总定位脏数据。
    """
    if not raw:
        return []
    out: list[float] = []
    skipped: list[str] = []
    for tok in raw.split(LIST_SEP):
        if tok == "":
            continue
        try:
            out.append(float(tok))
        except ValueError:
            skipped.append(tok)
    if skipped:
        logger.warning("_split_floats 跳过 %d 个非数字 token：%s", len(skipped), skipped[:5])
    return out


def parse_meta_line(line: str) -> dict[str, object]:
    """解析一行商品元数据。

    返回 dict，键顺序与 ``META_COLUMNS`` 一致：多 token 字段返回 ``list[str]``，
    单 token 字段返回 ``str``。列数不足时，缺失的标量字段补 ``""``、多 token
    字段补 ``[]``；列数过多则忽略多余列，保证输出恒为 13 键。
    """
    cells = _pad(line.rstrip("\r\n").split("\t"), len(META_COLUMNS))
    out: dict[str, object] = {}
    for name, raw in zip(META_COLUMNS, cells):
        out[name] = _split_tokens(raw) if name in META_MULTI_COLUMNS else raw
    return out


def parse_behavior_line(line: str) -> dict[str, object]:
    """解析一行搜索会话日志。

    返回 dict，键顺序与 ``BEHAVIOR_COLUMNS`` 一致：

    - ``query`` 原样返回 ``str``（其内部 token 仍以 ``TOKEN_SEP`` 连接）；
    - ``candidate_wid_list`` / ``history_qry_list`` / ``history_wid_list`` /
      ``history_type_list`` 返回 ``list[str]``；
    - ``candidate_label_list`` / ``history_time_list`` 返回 ``list[float]``：无法转
      float 的 token 被跳过并计数（见 ``_split_floats``）。

    列数不足时按列语义安全补齐 7 键：标量/字符串列补 ``""``，``*list`` 列补 ``[]``，
    避免残缺行把普通字段误当作 float 列表而抛 ``ValueError``。``history_qry_list``
    中的 ``-1``（表示「无查询的交互」）作为普通字符串原样保留。
    """
    cells = _pad(line.rstrip("\r\n").split("\t"), len(BEHAVIOR_COLUMNS))
    out: dict[str, object] = {}
    for name, raw in zip(BEHAVIOR_COLUMNS, cells):
        if name in BEHAVIOR_LIST_COLUMNS:
            out[name] = _split_list(raw)
        elif name in BEHAVIOR_FLOAT_COLUMNS:
            out[name] = _split_floats(raw)
        else:
            out[name] = raw
    return out
