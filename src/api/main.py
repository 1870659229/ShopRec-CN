"""ShopRec API 入口。"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import eval_results, feedback, item, recommend, user

app = FastAPI(title="ShopRec API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.include_router(user.router)
app.include_router(recommend.router)
app.include_router(feedback.router)
app.include_router(eval_results.router)
app.include_router(item.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
