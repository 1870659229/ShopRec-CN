-- src/db/schema_amazon.sql
-- Amazon Reviews 2023 · Grocery 专用建表语句（VARCHAR 版 ID）。
--
-- 与 schema.sql（默认 BIGINT 版）的唯一区别：**数据集侧的 ID 列改为字符串**，
-- 因为 Amazon 的 parent_asin（如 B00CM36GAQ）与 user_id（如 AGKHLEW2SW...）
-- 都是字符串，无法放进 BIGINT。差异清单：
--   item.item_id          BIGINT      -> VARCHAR(20)    （parent_asin）
--   behavior.user_id      BIGINT      -> VARCHAR(64)    （Amazon user_id）
--   behavior.item_id      BIGINT      -> VARCHAR(20)    （asin）
--   recommend_log.item_id BIGINT      -> VARCHAR(20)    （asin）
--   category.name         VARCHAR(64) -> VARCHAR(255)   （Amazon 类目名较长）
--
-- 应用侧表（user / user_profile / model_version）与 recommend_log.user_id
-- 仍为 BIGINT：它们承载的是**本站注册用户**，与数据集用户是两回事。
--
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
  item_id     VARCHAR(20) PRIMARY KEY,
  category_id BIGINT,
  source      VARCHAR(16) DEFAULT 'amazon',
  pop_score   DOUBLE DEFAULT 0,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_item_cate (category_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS category (
  category_id BIGINT PRIMARY KEY,
  parent_id   BIGINT DEFAULT 0,
  level       TINYINT DEFAULT 1,
  name        VARCHAR(255) DEFAULT ''
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS behavior (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id       VARCHAR(64) NOT NULL,
  item_id       VARCHAR(20) NOT NULL,
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
  item_id      VARCHAR(20) NOT NULL,
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
