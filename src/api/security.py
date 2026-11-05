"""JWT 认证与密码哈希工具。"""
import os
from datetime import datetime, timedelta, timezone

from jose import jwt
from passlib.context import CryptContext

# 优先读环境变量，开发环境回退固定值（模块级常量，非可变状态）
SECRET = os.environ.get("SHOPREC_API_SECRET", "shoprec-dev-secret")
ALGO = "HS256"
TOKEN_TTL_DAYS = 7
# passlib 1.7.4 与已装的 bcrypt 5.x 不兼容（探测 bcrypt.__about__ 失败），
# 故用 passlib 内置的 pbkdf2_sha256（纯 Python，无外部后端依赖）
_pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(raw: str) -> str:
    return _pwd.hash(raw)


def verify_password(raw: str, hashed: str) -> bool:
    return _pwd.verify(raw, hashed)


def create_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(days=TOKEN_TTL_DAYS),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGO)


def decode_token(token: str) -> int | None:
    """返回 user_id 或 None（token 无效/过期）。"""
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        return int(payload["sub"])
    except (jwt.exceptions.JWTError, KeyError, ValueError):
        return None
