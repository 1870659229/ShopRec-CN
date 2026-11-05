from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from config import settings
from src.db import repo


def test_schema_file_default_is_bigint_variant(monkeypatch):
    """库名不以 _amazon 结尾时用默认 BIGINT 版 schema。"""
    monkeypatch.delenv("SHOPREC_SCHEMA", raising=False)
    monkeypatch.setattr(repo, "settings", SimpleNamespace(DB_NAME="shoprec_jd"))
    assert repo.schema_file().name == "schema.sql"


def test_schema_file_amazon_by_db_name(monkeypatch):
    """库名以 _amazon 结尾时自动切到 VARCHAR 版 schema。"""
    monkeypatch.delenv("SHOPREC_SCHEMA", raising=False)
    monkeypatch.setattr(repo, "settings", SimpleNamespace(DB_NAME="shoprec_amazon"))
    assert repo.schema_file().name == "schema_amazon.sql"


def test_schema_file_env_overrides_db_name(monkeypatch):
    """SHOPREC_SCHEMA 显式指定优先于库名推断。"""
    monkeypatch.setenv("SHOPREC_SCHEMA", "amazon")
    monkeypatch.setattr(repo, "settings", SimpleNamespace(DB_NAME="shoprec_jd"))
    assert repo.schema_file().name == "schema_amazon.sql"


def test_schema_file_rejects_unknown_variant(monkeypatch):
    monkeypatch.setenv("SHOPREC_SCHEMA", "nope")
    monkeypatch.setattr(repo, "settings", SimpleNamespace(DB_NAME="shoprec_jd"))
    with pytest.raises(ValueError, match="未知的 SHOPREC_SCHEMA"):
        repo.schema_file()


def test_amazon_schema_uses_varchar_ids():
    """Amazon 版 schema 的 ID 列必须是 VARCHAR（asin / user_id 是字符串）。"""
    schema_dir = Path(repo.__file__).parent
    sql = (schema_dir / "schema_amazon.sql").read_text(encoding="utf-8")
    assert "item_id     VARCHAR(20) PRIMARY KEY" in sql
    assert "user_id       VARCHAR(64) NOT NULL" in sql


def test_engine_connects():
    engine = repo.get_engine()
    assert engine is not None


def test_query_df_returns_dataframe():
    df = repo.query_df("SELECT 1 AS v")
    assert int(df.iloc[0]["v"]) == 1


def test_init_db_creates_database():
    # 库名取 settings.DB_NAME，避免写死导致 SHOPREC_DB_NAME 生效时删错库
    with create_engine(settings.url_no_db, future=True).begin() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS `{settings.DB_NAME}`"))
    repo.init_db()
    tables = repo.query_df(f"SHOW TABLES FROM `{settings.DB_NAME}`")
    assert len(tables) == 7
