"""pytest 全局配置：把测试隔离到独立库，避免误清演示库。"""

import os

# 必须在任何 config / src.* 导入之前设置：config.Settings 是 frozen dataclass，
# 默认值在导入期求值，运行期再改环境变量无效。
# 这里用硬赋值（而非 setdefault），防止开发者 shell 里恰好导出 SHOPREC_DB_NAME=shoprec 时误连演示库。
#
# 库名为 shoprec_ci_test（而非演示库 shoprec_test）：test_repo.py 有 DROP DATABASE
# 级联的用例（test_init_db_creates_database），若与演示库同名，全量跑一次 pytest
# 就会把演示数据（item/user/behavior）全部清掉——2026-10-10 实测踩坑，故真分库。
os.environ["SHOPREC_DB_NAME"] = "shoprec_ci_test"

import pytest  # noqa: E402

from config import settings  # noqa: E402
from src.db import repo  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_db() -> None:
    """确保测试库存在；并断言库名以 _test 结尾，杜绝误连演示库。"""
    assert settings.DB_NAME.endswith("_test"), f"测试库名必须以 _test 结尾，当前为 {settings.DB_NAME}"
    repo.init_db()
