-- migrate_notifications_schema.sql
-- Purpose:
-- 1) Bring notification-related tables in line with current ORM expectations.
-- 2) Backfill legacy notification_emails rows into site-scoped rows.
-- 3) Insert one sample notification_email and one sample notification row.
--
-- Run:
--   mysql -h <host> -P <port> -u<user> -p <database> < Backend/scripts/migrate_notifications_schema.sql

SET NAMES utf8mb4;
SET @db_name := DATABASE();

-- ---------------------------------------------------------------------------
-- A) Ensure notification_emails table exists (legacy-safe baseline)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS notification_emails (
  id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  site_uuid BINARY(16) NULL,
  email VARCHAR(255) NOT NULL,
  is_enabled TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (id),
  KEY ix_notification_emails_user_id (user_id),
  KEY ix_notification_emails_site_uuid (site_uuid)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Add missing columns if table already existed in older shape.
SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification_emails' AND column_name = 'site_uuid';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification_emails ADD COLUMN site_uuid BINARY(16) NULL AFTER user_id',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification_emails' AND column_name = 'is_enabled';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification_emails ADD COLUMN is_enabled TINYINT(1) NOT NULL DEFAULT 1 AFTER email',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification_emails' AND column_name = 'updated_at';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification_emails ADD COLUMN updated_at DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6) AFTER created_at',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- If table had global emails (no site_uuid), explode them to all sites for that user.
INSERT INTO notification_emails (user_id, site_uuid, email, is_enabled, created_at, updated_at)
SELECT DISTINCT
  ne.user_id,
  s.site_uuid,
  ne.email,
  COALESCE(ne.is_enabled, 1),
  COALESCE(ne.created_at, UTC_TIMESTAMP(6)),
  UTC_TIMESTAMP(6)
FROM notification_emails ne
JOIN sites s ON s.user_id = ne.user_id
WHERE ne.site_uuid IS NULL
  AND NOT EXISTS (
    SELECT 1
    FROM notification_emails x
    WHERE x.user_id = ne.user_id
      AND x.email = ne.email
      AND x.site_uuid = s.site_uuid
  );

-- Remove legacy rows that still have NULL site_uuid (no usable site scope).
DELETE FROM notification_emails
WHERE site_uuid IS NULL;

-- Remove duplicate rows before adding unique constraint.
DELETE ne1
FROM notification_emails ne1
JOIN notification_emails ne2
  ON ne1.user_id = ne2.user_id
 AND ne1.email = ne2.email
 AND ne1.site_uuid = ne2.site_uuid
 AND ne1.id > ne2.id;

-- Ensure indexes exist.
SELECT COUNT(*) INTO @idx_exists
FROM information_schema.statistics
WHERE table_schema = @db_name AND table_name = 'notification_emails' AND index_name = 'ix_notification_emails_user_id';
SET @sql := IF(
  @idx_exists = 0,
  'ALTER TABLE notification_emails ADD INDEX ix_notification_emails_user_id (user_id)',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @idx_exists
FROM information_schema.statistics
WHERE table_schema = @db_name AND table_name = 'notification_emails' AND index_name = 'ix_notification_emails_site_uuid';
SET @sql := IF(
  @idx_exists = 0,
  'ALTER TABLE notification_emails ADD INDEX ix_notification_emails_site_uuid (site_uuid)',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- Ensure unique constraint exists.
SELECT COUNT(*) INTO @uq_exists
FROM information_schema.table_constraints
WHERE table_schema = @db_name
  AND table_name = 'notification_emails'
  AND constraint_name = 'uq_notif_email_user_site_email'
  AND constraint_type = 'UNIQUE';
SET @sql := IF(
  @uq_exists = 0,
  'ALTER TABLE notification_emails ADD CONSTRAINT uq_notif_email_user_site_email UNIQUE (user_id, site_uuid, email)',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- Ensure FKs exist.
SELECT COUNT(*) INTO @fk_exists
FROM information_schema.table_constraints
WHERE table_schema = @db_name
  AND table_name = 'notification_emails'
  AND constraint_name = 'fk_notification_emails_user_id'
  AND constraint_type = 'FOREIGN KEY';
SET @sql := IF(
  @fk_exists = 0,
  'ALTER TABLE notification_emails ADD CONSTRAINT fk_notification_emails_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @fk_exists
FROM information_schema.table_constraints
WHERE table_schema = @db_name
  AND table_name = 'notification_emails'
  AND constraint_name = 'fk_notification_emails_site_uuid'
  AND constraint_type = 'FOREIGN KEY';
SET @sql := IF(
  @fk_exists = 0,
  'ALTER TABLE notification_emails ADD CONSTRAINT fk_notification_emails_site_uuid FOREIGN KEY (site_uuid) REFERENCES sites(site_uuid) ON DELETE CASCADE',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ---------------------------------------------------------------------------
-- B) Ensure notification table exists and has expected columns
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS notification (
  id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  site_uuid BINARY(16) NOT NULL,
  camera_uuid BINARY(16) NULL,
  device_uuid BINARY(16) NULL,
  event_type VARCHAR(64) NOT NULL DEFAULT 'detection',
  title VARCHAR(255) NULL,
  message TEXT NULL,
  payload JSON NULL,
  detected_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  read_at DATETIME(6) NULL,
  sent_at DATETIME(6) NULL,
  status VARCHAR(32) NOT NULL DEFAULT 'created',
  PRIMARY KEY (id),
  KEY ix_notification_user_id (user_id),
  KEY ix_notification_site_uuid (site_uuid),
  KEY ix_notification_camera_uuid (camera_uuid),
  KEY ix_notification_device_uuid (device_uuid),
  KEY ix_notification_detected_at (detected_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Add missing columns (if table already exists but is older).
SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification' AND column_name = 'site_uuid';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification ADD COLUMN site_uuid BINARY(16) NOT NULL AFTER user_id',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification' AND column_name = 'camera_uuid';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification ADD COLUMN camera_uuid BINARY(16) NULL AFTER site_uuid',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification' AND column_name = 'device_uuid';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification ADD COLUMN device_uuid BINARY(16) NULL AFTER camera_uuid',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification' AND column_name = 'payload';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification ADD COLUMN payload JSON NULL AFTER message',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification' AND column_name = 'read_at';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification ADD COLUMN read_at DATETIME(6) NULL AFTER created_at',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification' AND column_name = 'sent_at';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification ADD COLUMN sent_at DATETIME(6) NULL AFTER read_at',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @col_exists
FROM information_schema.columns
WHERE table_schema = @db_name AND table_name = 'notification' AND column_name = 'status';
SET @sql := IF(
  @col_exists = 0,
  'ALTER TABLE notification ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT ''created'' AFTER sent_at',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- Ensure notification indexes.
SELECT COUNT(*) INTO @idx_exists
FROM information_schema.statistics
WHERE table_schema = @db_name AND table_name = 'notification' AND index_name = 'ix_notification_user_id';
SET @sql := IF(
  @idx_exists = 0,
  'ALTER TABLE notification ADD INDEX ix_notification_user_id (user_id)',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @idx_exists
FROM information_schema.statistics
WHERE table_schema = @db_name AND table_name = 'notification' AND index_name = 'ix_notification_site_uuid';
SET @sql := IF(
  @idx_exists = 0,
  'ALTER TABLE notification ADD INDEX ix_notification_site_uuid (site_uuid)',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- Ensure notification foreign keys.
SELECT COUNT(*) INTO @fk_exists
FROM information_schema.table_constraints
WHERE table_schema = @db_name
  AND table_name = 'notification'
  AND constraint_name = 'fk_notification_user_id'
  AND constraint_type = 'FOREIGN KEY';
SET @sql := IF(
  @fk_exists = 0,
  'ALTER TABLE notification ADD CONSTRAINT fk_notification_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SELECT COUNT(*) INTO @fk_exists
FROM information_schema.table_constraints
WHERE table_schema = @db_name
  AND table_name = 'notification'
  AND constraint_name = 'fk_notification_site_uuid'
  AND constraint_type = 'FOREIGN KEY';
SET @sql := IF(
  @fk_exists = 0,
  'ALTER TABLE notification ADD CONSTRAINT fk_notification_site_uuid FOREIGN KEY (site_uuid) REFERENCES sites(site_uuid) ON DELETE CASCADE',
  'DO 1'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ---------------------------------------------------------------------------
-- C) Insert one sample row into notification_emails and notification
-- ---------------------------------------------------------------------------
-- Change these two values if you want.
SET @target_user_id := 1;
SET @target_email := 'alerts@example.com';

-- Pick one site for this user.
SET @target_site_uuid := NULL;
SELECT s.site_uuid INTO @target_site_uuid
FROM sites s
WHERE s.user_id = @target_user_id
ORDER BY s.site_uuid
LIMIT 1;

-- Add one site-scoped notification email row (idempotent because of UNIQUE key).
INSERT INTO notification_emails (user_id, site_uuid, email, is_enabled, created_at, updated_at)
SELECT
  @target_user_id,
  @target_site_uuid,
  @target_email,
  1,
  UTC_TIMESTAMP(6),
  UTC_TIMESTAMP(6)
WHERE @target_site_uuid IS NOT NULL
ON DUPLICATE KEY UPDATE
  is_enabled = VALUES(is_enabled),
  updated_at = VALUES(updated_at);

-- Add one sample notification row.
INSERT INTO notification (
  user_id, site_uuid, camera_uuid, device_uuid,
  event_type, title, message, payload,
  detected_at, created_at, status
)
SELECT
  @target_user_id,
  @target_site_uuid,
  NULL,
  NULL,
  'manual_test',
  'Manual Notification Row',
  'Inserted by migrate_notifications_schema.sql',
  JSON_OBJECT('source', 'manual_migration'),
  UTC_TIMESTAMP(6),
  UTC_TIMESTAMP(6),
  'created'
WHERE @target_site_uuid IS NOT NULL;

-- ---------------------------------------------------------------------------
-- D) Verification queries
-- ---------------------------------------------------------------------------
SELECT * FROM notification_emails LIMIT 100;
SELECT * FROM notification ORDER BY id DESC LIMIT 100;
