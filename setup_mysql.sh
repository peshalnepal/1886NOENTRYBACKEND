#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y mysql-server

# Create DB + user (edit password if you want)
DB="appdb"
USER="appuser"
PASS="AppUser@2025!"

sudo mysql <<SQL
-- ✅ Remove existing database (drops all tables inside)
DROP DATABASE IF EXISTS \`${DB}\`;

-- ✅ Recreate fresh database
CREATE DATABASE \`${DB}\`;

-- ✅ Ensure user exists + has privileges
CREATE USER IF NOT EXISTS '${USER}'@'%' IDENTIFIED BY '${PASS}';
GRANT ALL PRIVILEGES ON \`${DB}\`.* TO '${USER}'@'%';
FLUSH PRIVILEGES;

-- (Optional) show confirmation
SELECT 'Database reset complete' AS status, '${DB}' AS db, '${USER}' AS user;
SQL

echo "✅ MySQL ready (fresh): db=${DB}, user=${USER}"
