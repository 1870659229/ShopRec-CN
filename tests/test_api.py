"""API 用户服务的 TDD 测试（沿用 conftest 的 shoprec_ci_test 隔离库）。"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.db import repo

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_test_users() -> None:
    """每条测试前清理历史同名用户，保证测试可重复运行。"""
    repo.execute("DELETE FROM user WHERE username IN ('u_test', 'u_test2')")


def test_register_then_login():
    r = client.post("/api/user/register", json={"username": "u_test", "password": "pw_test_9"})
    assert r.status_code == 200
    r2 = client.post("/api/user/login", json={"username": "u_test", "password": "pw_test_9"})
    assert r2.status_code == 200
    assert "token" in r2.json()


def test_register_duplicate_username_returns_409():
    r1 = client.post("/api/user/register", json={"username": "u_test", "password": "pw_test_9"})
    assert r1.status_code == 200
    r2 = client.post("/api/user/register", json={"username": "u_test", "password": "pw_test_9"})
    assert r2.status_code == 409


def test_login_rejects_wrong_password():
    r = client.post("/api/user/register", json={"username": "u_test2", "password": "pw_test_9"})
    assert r.status_code == 200
    r2 = client.post("/api/user/login", json={"username": "u_test2", "password": "wrong_pw_9"})
    assert r2.status_code == 401
