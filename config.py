import os
from dataclasses import dataclass

from sqlalchemy.engine import URL


@dataclass(frozen=True)
class Settings:
    DB_HOST: str = os.getenv("SHOPREC_DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("SHOPREC_DB_PORT", "3306"))
    DB_USER: str = os.getenv("SHOPREC_DB_USER", "root")
    DB_PASSWORD: str = os.getenv("SHOPREC_DB_PASSWORD", "")
    DB_NAME: str = os.getenv("SHOPREC_DB_NAME", "shoprec")
    DATA_DIR: str = os.getenv("SHOPREC_DATA_DIR", "data")

    @property
    def url(self) -> URL:
        """带库名的连接 URL（自动转义用户名/密码中的特殊字符）。"""
        return URL.create(
            "mysql+pymysql",
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.DB_NAME,
            query={"charset": "utf8mb4"},
        )

    @property
    def url_no_db(self) -> URL:
        """不带库名的连接 URL，用于建库与执行 schema.sql。"""
        return URL.create(
            "mysql+pymysql",
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            query={"charset": "utf8mb4"},
        )

    @property
    def dsn(self) -> str:
        return self.url.render_as_string(hide_password=False)


settings = Settings()
