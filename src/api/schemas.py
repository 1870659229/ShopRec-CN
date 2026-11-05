"""API 请求体模型。"""
from pydantic import BaseModel, Field


class RegisterIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    age_bucket: str = Field(default="unknown", max_length=16)
    gender: int = 0
    city_level: int = 0


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=6, max_length=128)
