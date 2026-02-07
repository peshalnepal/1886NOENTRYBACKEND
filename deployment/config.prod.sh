#!/bin/bash
# 1886NOENTRY - Production config (non-secrets + temporary passwords for now)

ENVIRONMENT_NAME="Production"

# Bicep
BICEP_FILE="./main-prod.bicep"

# App / naming
NAME_PREFIX="noentry"
APP_NAME_MAIN="noentry-api-prod"

# Set to 'Multiple' for blue/green style revisions, 'Single' for normal
REVISION_MODE="Single"

# Uvicorn module
APP_MODULE="main:app"

# --------------------------
# MySQL (TEMP - for now no secrets)
# --------------------------
MYSQL_ADMIN_USER="mysqladmin"
MYSQL_ADMIN_PASSWORD="ChangeThis_AdminPassword_2026!"   # TEMP
MYSQL_DB_NAME="appdb"
APP_DB_USER="appuser"
APP_DB_PASSWORD="ChangeThis_AppPassword_2026!"          # TEMP

# --------------------------
# SMTP (optional)
# --------------------------
ENABLE_SMTP="false"
SMTP_USERNAME=""
SMTP_PASSWORD=""            # TEMP if ENABLE_SMTP=true
SMTP_FROM="no-reply@1886noentry.com"

# --------------------------
# WebRTC / MediaMTX settings (non-secret)
# --------------------------
WEBRTC_ADMIN_API_URL="http://127.0.0.1:9997"
WEBRTC_PUBLIC_BASE_URL="http://localhost:8889"
WEBRTC_ADMIN_API_KEY=""

WEBRTC_ADMIN_UPSERT_PATH=""
WEBRTC_ADMIN_UPDATE_PATH=""
WEBRTC_ADMIN_DELETE_PATH=""

# Optional: ACI MediaMTX (kept for later; off by default)
DEPLOY_MEDIA_MTX="false"
MEDIAMTX_API_USER="api"
MEDIAMTX_API_PASS=""        # TEMP if DEPLOY_MEDIA_MTX=true
