# ShopRec 推荐系统

ShopRec：双品类迁移 + 实时画像 + 可控多样性重排的推荐系统（FastAPI + Vue3 + MySQL）。

## 一、环境准备

| 依赖 | 要求 |
| --- | --- |
| Python | 3.13（其他 3.10+ 理论可用） |
| Node.js | 20+（前端构建） |
| MySQL | 8.x（需 `JSON` 类型与 `JSON_OBJECT` 函数支持） |

```bash
# 后端依赖（pip 走清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 前端依赖（npm 走 npmmirror 镜像）
cd frontend
npm install --registry=https://registry.npmmirror.com
```

## 二、建库与建表

数据库名由环境变量 `SHOPREC_DB_NAME` 控制（默认 `shoprec`），**不写死在任何 SQL 中**。
首次启动前执行一次初始化（建库 + 执行 `src/db/schema.sql` 建 7 张表）：

```bash
# bash / Git Bash 写法
SHOPREC_DB_PASSWORD=<你的密码> SHOPREC_DB_NAME=shoprec python -c "from src.db import repo; repo.init_db()"
```

```powershell
# PowerShell 写法
$env:SHOPREC_DB_PASSWORD = "<你的密码>"
$env:SHOPREC_DB_NAME = "shoprec"
python -c "from src.db import repo; repo.init_db()"
```

常用环境变量（均有默认值，密码必须提供）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `SHOPREC_DB_HOST` | `localhost` | MySQL 地址 |
| `SHOPREC_DB_PORT` | `3306` | MySQL 端口 |
| `SHOPREC_DB_USER` | `root` | 用户名 |
| `SHOPREC_DB_PASSWORD` | （空） | 密码，**必填** |
| `SHOPREC_DB_NAME` | `shoprec` | 库名；以 `_test` 结尾时 pytest 自动隔离使用 |
| `SHOPREC_API_SECRET` | `shoprec-dev-secret` | JWT 签名密钥；生产必须设真值，缺省回退 dev 值仅限本机演示 |

建库建表后，可运行演示 seed 脚本灌入 10 个演示商品与 3 个类目（幂等，可重复执行；库名必须以 `_test` 结尾，否则拒绝执行）：

```bash
SHOPREC_DB_PASSWORD=<你的密码> SHOPREC_DB_NAME=shoprec_test python scripts/seed_demo.py
```

### 数据来源与仓库范围（2026-11-04 更新）

`data/` 已被 `.gitignore` 忽略，因此本仓库**不包含** JDsearch / Amazon 原始数据集，也不包含 `data/splits/` 下的训练切分与冻结指标。按第一、二节建库并执行 seed 即可运行演示和下面的端到端验证，不需要原始数据。

若需从真实数据重建 `shoprec_jd` / `shoprec_amazon`，需先下载 4 个原始文件，再运行 `scripts/prepare_jdsearch.py`（需 `--meta-path`、`--behavior-path`）与 `scripts/load_amazon_fast.py`。下载命令见 `docs/superpowers/specs/2026-10-08-推荐系统-design.md`：

| 数据集 | 文件 | 来源 |
| --- | --- | --- |
| JDsearch | `product_meta_data.txt`、`user_behavior_data.txt` | HuggingFace 镜像 `hf-mirror.com/datasets/lswhim/JDsearch` |
| Amazon Reviews 2023 · Grocery | `Grocery_and_Gourmet_Food.jsonl.gz`、`meta_Grocery_and_Gourmet_Food.jsonl.gz` | UCSD McAuley Lab |

本地开发副本的原始文件已于 2026-11-04 移入回收站，当前两个 MySQL 库是完整数据副本；清空回收站前请先做整库备份。离线评估只读 `data/splits/jdsearch_eval/` 下的 `*.parquet` 与 `eval_v5*.json`，不依赖原始文件。

## 三、启动

```bash
# 后端（bash 前缀注入环境变量；PowerShell 请改用 $env: 逐条设置后再运行）
SHOPREC_DB_PASSWORD=<你的密码> SHOPREC_DB_NAME=shoprec python -m uvicorn src.api.main:app --port 8000

# 前端（另开终端，frontend/ 目录下）
npm run dev
```

## 四、访问地址

| 入口 | 地址 |
| --- | --- |
| 前端页面 | http://localhost:5173 （hash 路由，首页 `#/login`） |
| 后端健康检查 | http://localhost:8000/api/health |
| Swagger 文档 | http://localhost:8000/docs |

## 五、端到端验证清单（7 步）

> 行为枚举与 JDsearch 对齐：`CLICK`（点击，评分也以 CLICK 落库）/ `CART`（加购）/ `ORD`（下单）/ `FLW`（收藏）；评分 1–5 整数。
> 每步的 SQL 在 MySQL 客户端中对 `SHOPREC_DB_NAME` 指定的库执行。

1. **注册新用户并登录**
   打开 `#/login`，输入用户名/密码，点「注册」→ 提示成功 → 点「登录」→ 跳转 `#/feed`。
   核对：`SELECT user_id, username FROM user ORDER BY user_id DESC LIMIT 1;`

2. **推荐流出现商品卡片（冷启动走全局热门）**
   `#/feed` 应显示 8 张商品卡片（无画像用户走 `_hot_global` 全局热门兜底）。
   核对：`SELECT item_id, position FROM recommend_log ORDER BY id DESC LIMIT 8;`（position 1–8，记下第一屏顺序）

3. **点击商品 → behavior 新增 CLICK**
   点任意卡片进入详情页。
   核对：`SELECT id, user_id, item_id, behavior_type, rating, source FROM behavior ORDER BY id DESC LIMIT 1;`（behavior_type 应为 `CLICK`）

4. **打 5 星 → 新增 CLICK+rating=5，画像刷新**
   在详情页点第 5 颗星（评分以 CLICK 行为落库）。
   核对：
   ```sql
   SELECT id, item_id, behavior_type, rating FROM behavior ORDER BY id DESC LIMIT 2;
   SELECT recent_items, interest_tags FROM user_profile WHERE user_id = <上一步的 user_id>;
   ```
   （`recent_items` 含该商品，`interest_tags` 含其品类）

5. **回推荐流刷新 → 推荐结果变化（画像生效）**
   浏览器回退到 `#/feed` 并刷新页面，对比第 2 步记录的卡片顺序。
   核对：对比 `recommend_log` 中最近两个 `request_id` 的内容（MySQL 不支持 `LIMIT` 与 `IN` 子查询同用，分两条执行）：
   ```sql
   -- 第 1 条：取最近 16 行对应的 request_id（两次请求各 8 行），认出最近两个不同的 request_id
   SELECT id, request_id FROM recommend_log ORDER BY id DESC LIMIT 16;
   ```
   ```sql
   -- 第 2 条：把 <request_id> 换成上一步实际查到的两个值（逗号分隔，带引号）
   SELECT request_id, position, item_id FROM recommend_log
   WHERE request_id IN (<贴入实际值>)
   ORDER BY request_id, position;
   ```
   机制：点击后画像 `interest_tags` 更新，推荐从「全局热门兜底」切换为「画像品类内 + 品类外」两路对齐合并（当前两路候选无重叠，置信度融合在此场景不触发）；画像生效的可见效果是候选池扩大与多样性重排的边界翻转，尾部推荐位变化。

6. **看板点「检查是否触发重训」→ 返回计数**
   打开 `#/dashboard`，点按钮，显示「未达阈值，当前本地行为 N 条」（默认阈值 500）。
   核对：`SELECT COUNT(*) FROM behavior WHERE source='local';`（应等于页面显示的 N）

7. **手动触发重训 → model_version 新增记录（幂等）**
   把阈值调成当前本地计数，用 curl 调用（`<N>` 替换为第 6 步的计数）：
   ```bash
   curl -X POST "http://localhost:8000/api/update/check?threshold=<N>"
   ```
   返回 `retrain_triggered=true` 且 `already_registered=false`；**再调一次**应返回 `already_registered=true`（同一 `data_range` 只登记一次，不重复插入）。
   核对：
   ```sql
   SELECT version_id, model_type, data_range, is_active FROM model_version ORDER BY version_id DESC LIMIT 2;
   ```
   （`model_type='retrain'`、`data_range='local_<N>'`、`is_active=0`，且重复调用后行数不再增加）
