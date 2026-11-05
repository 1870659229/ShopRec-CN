"""答辩演示用静态托管：serve dist/ 并把 /api 反代到 FastAPI（零第三方依赖）。

用法（两条命令，两个终端）：
    cd frontend && npm run build          # 产出 dist/
    cd frontend && python serve_dist.py   # 本脚本，默认 127.0.0.1:4173
    # 另一终端照常起后端：uvicorn src.api.main:app --port 8000
浏览器打开 http://127.0.0.1:4173

为什么不用 vite preview：preview 只托管静态文件、不转发 /api。
为什么不需要 SPA fallback：路由是 createWebHashHistory（hash 模式），
页面路径恒为 /，子路由全在 # 后，静态服务器只服务 index.html 即可。

注意：本脚本不缓存（每次请求读盘），改完重新 build + 重启即可生效。
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 4173
BACKEND = "http://127.0.0.1:8000"
API_PREFIX = "/api"
BACKEND_TIMEOUT_S = 30
DIST = Path(__file__).resolve().parent / "dist"

BACKEND_DOWN_MSG = '{"detail":"后端 8000 未启动：请先运行 uvicorn src.api.main:app --port 8000"}'

# 环境变量里的 http_proxy 会劫持对 127.0.0.1:8000 的反代（实测返回代理自己的
# 502 "upstream connect failed"，错误信息完全对不上后端）——必须显式直连
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class DemoHandler(SimpleHTTPRequestHandler):
    """静态托管 dist/；/api/* 按原 method/headers/body 反代到 BACKEND。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(DIST), **kwargs)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        if self.path.startswith(API_PREFIX):
            self._proxy(method)
        elif method == "POST":
            self.send_error(405, "静态资源只接受 GET/HEAD")
        else:
            super().do_GET() if method == "GET" else super().do_HEAD()

    def _proxy(self, method: str) -> None:
        """原样转发（含 4xx/5xx 响应体，前端 422 归一化依赖 detail 数组）。"""
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(BACKEND + self.path, data=body, method=method)
        if self.headers.get("Content-Type"):
            req.add_header("Content-Type", self.headers["Content-Type"])
        try:
            with _OPENER.open(req, timeout=BACKEND_TIMEOUT_S) as resp:
                self._respond(resp.status, resp.headers.get("Content-Type"), resp.read(), method)
        except urllib.error.HTTPError as exc:  # 后端 4xx/5xx 照实转发
            self._respond(exc.code, exc.headers.get("Content-Type"), exc.read(), method)
        except urllib.error.URLError:
            self._respond(502, "application/json; charset=utf-8",
                          BACKEND_DOWN_MSG.encode("utf-8"), method)

    def _respond(self, status: int, ctype: str | None, payload: bytes, method: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002 - 标准库签名
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    if not (DIST / "index.html").exists():
        sys.exit("dist/ 不存在：请先在 frontend/ 下执行 npm run build")
    server = ThreadingHTTPServer((HOST, PORT), DemoHandler)
    print(f"演示服务已启动: http://{HOST}:{PORT}")
    print(f"  静态资源: {DIST}")
    print(f"  /api 反代: {BACKEND}")
    print("Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print("\n已停止")


if __name__ == "__main__":
    main()
