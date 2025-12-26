#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y mysql-server

# Create DB + user (edit password if you want)
DB="appdb"
USER="appuser"
PASS="AppUser@2025!"

sudo mysql <<SQL
CREATE DATABASE IF NOT EXISTS \`${DB}\`;
CREATE USER IF NOT EXISTS '${USER}'@'%' IDENTIFIED BY '${PASS}';
GRANT ALL PRIVILEGES ON \`${DB}\`.* TO '${USER}'@'%';
FLUSH PRIVILEGES;
SQL

echo "✅ MySQL ready: db=${DB}, user=${USER}"