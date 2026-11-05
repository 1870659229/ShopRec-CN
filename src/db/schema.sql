-- src/db/schema.sql
-- 库名不在此写死：由 repo.init_db() 依据 settings.DB_NAME 建库后，再执行本文件的建表语句。

CREATE TABLE IF NOT EXISTS user (
  user_id      BIGINT PRIMARY KEY AUTO_INCREMENT,
  username     VARCHAR(64)  NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  age_bucket   VARCHAR(16)  DEFAULT 'unknown',
  gender       TINYINT      DEFAULT 0,
  city_level   TINYINT      DEFAULT 0,
  register_time DATETIME    DEFAULT CURRENT_TIMESTAMP,
  cold_start_done TINYINT   DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS item (
  item_id     BIGINT PRIMARY KEY,
  category_id BIGINT,
  source      VARCHAR(16) DEFAULT 'tianchi',
  -- 演示标题：JDsearch 数据无标题字段，仅 demo 段位（9300 段）有值，
  -- 供前端卡片展示与演示搜索使用；真实数据行为空串，不参与指标计算。
  title       VARCHAR(128) DEFAULT '',
  -- 演示价格：JDsearch 数据无价格字段，仅 demo 段位（9300 段）有值，
  -- 供前端商品卡/详情页展示（电商卡片三大必选要素之一）；
  -- 真实数据行为 NULL，不参与指标计算。
  demo_price  DECIMAL(10,2) DEFAULT NULL,
  pop_score   DOUBLE DEFAULT 0,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_item_cate (category_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS category (
  category_id BIGINT PRIMARY KEY,
  parent_id   BIGINT DEFAULT 0,
  level       TINYINT DEFAULT 1,
  name        VARCHAR(64) DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS behavior (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id       BIGINT NOT NULL,
  item_id       BIGINT NOT NULL,
  behavior_type VARCHAR(8) NOT NULL,
  rating        TINYINT DEFAULT NULL,
  ts            DATETIME NOT NULL,
  source        VARCHAR(16) DEFAULT 'local',
  INDEX idx_bh_user_ts (user_id, ts),
  INDEX idx_bh_item (item_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS recommend_log (
  id           BIGINT PRIMARY KEY AUTO_INCREMENT,
  request_id   VARCHAR(36) NOT NULL,
  user_id      BIGINT NOT NULL,
  item_id      BIGINT NOT NULL,
  recall_source VARCHAR(32),
  rank_score   DOUBLE,
  position     INT,
  is_clicked   TINYINT DEFAULT 0,
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_log_req (request_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS user_profile (
  user_id       BIGINT PRIMARY KEY,
  recent_items  JSON,
  interest_tags JSON,
  updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_version (
  version_id  BIGINT PRIMARY KEY AUTO_INCREMENT,
  model_type  VARCHAR(32) NOT NULL,
  train_time  DATETIME DEFAULT CURRENT_TIMESTAMP,
  data_range  VARCHAR(64),
  metrics     JSON,
  path        VARCHAR(255),
  is_active   TINYINT DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
