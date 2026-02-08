#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------
# Deploy 1886NOENTRY backend to Azure Container Apps (prod/dev)
#
# Usage:
#   ./deploy.sh --environment <dev|prod>
#
# Workflow provides ONLY:
#   AZURE_SUBSCRIPTION
#   AZURE_RESOURCE_GROUP
#   AZURE_ACR_NAME
#   AZURE_ENVIRONMENT_NAME
#   AZURE_LOCATION
#   AZURE_KEY_VAULT_NAME
#
# Everything else comes from: config.<env>.sh
# For now: NO KeyVault secrets (passwords come from config file).
# ------------------------------------------------------------

# --- Argument Parsing ---
if [[ $# -eq 0 ]] ; then
  echo "Usage: ./deploy.sh --environment <dev|prod>" >&2
  exit 1
fi

ENVIRONMENT=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --environment)
      ENVIRONMENT="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1" >&2
      exit 1
      ;;
  esac
done

# --- Load Configuration ---
CONFIG_FILE="./config.${ENVIRONMENT}.sh"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# --- Workflow-provided env (minimal) ---
: "${AZURE_RESOURCE_GROUP:?AZURE_RESOURCE_GROUP is required}"
: "${AZURE_ACR_NAME:?AZURE_ACR_NAME is required}"
: "${AZURE_ENVIRONMENT_NAME:?AZURE_ENVIRONMENT_NAME is required}"
: "${AZURE_LOCATION:?AZURE_LOCATION is required}"
: "${AZURE_KEY_VAULT_NAME:?AZURE_KEY_VAULT_NAME is required}"
: "${AZURE_SUBSCRIPTION:=}"

# --- Global Variables ---
ACR_SERVER="${AZURE_ACR_NAME}.azurecr.io"
BUILD_TIMESTAMP=$(date +%Y%m%d%H%M%S)
GIT_SHA=${GITHUB_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo "manual")}
GIT_SHA_SHORT=$(echo "${GIT_SHA}" | cut -c1-7)
IMAGE_TAG="${GIT_SHA_SHORT}-${BUILD_TIMESTAMP}"
REVISION_SUFFIX="${GIT_SHA_SHORT}-${BUILD_TIMESTAMP}"

# --- Logging ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
write_info() { echo -e "${YELLOW}[INFO] $1${NC}" >&2; }
write_success() { echo -e "${GREEN}[SUCCESS] $1${NC}" >&2; }
write_error() { echo -e "${RED}[ERROR] $1${NC}" >&2; }

# --- Error Handling & Rollback (kept similar) ---
function rollback_deployment() {
  write_info "Attempting rollback..."
  if [[ "${REVISION_MODE}" == "Multiple" ]]; then
    local previous_revision
    previous_revision=$(az containerapp revision list \
      -n "${APP_NAME_MAIN}" \
      -g "${AZURE_RESOURCE_GROUP}" \
      --query "[?properties.trafficWeight > 0 && !contains(name, '${REVISION_SUFFIX}')].name | [0]" -o tsv 2>/dev/null || true)

    if [[ -n "${previous_revision}" ]]; then
      write_info "Shifting traffic back to stable revision: ${previous_revision}"
      az containerapp ingress traffic set \
        -n "${APP_NAME_MAIN}" \
        -g "${AZURE_RESOURCE_GROUP}" \
        --revision-weight "${previous_revision}=100" >/dev/null
      write_success "Rollback successful."
    else
      write_error "No previous stable revision found."
    fi
  else
    local failed_revision
    failed_revision=$(az containerapp revision list \
      -n "${APP_NAME_MAIN}" \
      -g "${AZURE_RESOURCE_GROUP}" \
      --query "[?contains(name, '${REVISION_SUFFIX}')].name | [0]" -o tsv 2>/dev/null || true)

    if [[ -n "${failed_revision}" ]]; then
      write_info "Deactivating failed revision: ${failed_revision}"
      az containerapp revision deactivate \
        -n "${APP_NAME_MAIN}" \
        -g "${AZURE_RESOURCE_GROUP}" \
        --revision "${failed_revision}" >/dev/null || true
    fi
  fi
}

function cleanup_on_error() {
  write_error "${ENVIRONMENT_NAME} deployment failed. Initiating rollback..."
  rollback_deployment || true
  exit 1
}
trap cleanup_on_error ERR

# --- Helpers ---
function validate_prerequisites() {
  command -v az >/dev/null 2>&1 || { write_error "Azure CLI (az) not found"; exit 1; }
  command -v docker >/dev/null 2>&1 || { write_error "Docker not found"; exit 1; }
  command -v curl >/dev/null 2>&1 || { write_error "curl not found"; exit 1; }

  # best-effort: ensure we can call az
  az account show >/dev/null 2>&1 || { write_error "Not logged in to Azure (azure/login missing?)"; exit 1; }
  write_success "Prerequisites validated."
}

function ensure_subscription_context() {
  if [[ -n "${AZURE_SUBSCRIPTION}" ]]; then
    write_info "Setting subscription: ${AZURE_SUBSCRIPTION}"
    az account set --subscription "${AZURE_SUBSCRIPTION}" >/dev/null 2>&1 || true
  fi
}

function ensure_resource_group() {
  write_info "Ensuring Resource Group: ${AZURE_RESOURCE_GROUP} (${AZURE_LOCATION})"
  az group create -n "${AZURE_RESOURCE_GROUP}" -l "${AZURE_LOCATION}" >/dev/null
}

function ensure_acr_exists() {
  write_info "Ensuring ACR exists: ${AZURE_ACR_NAME}"
  if ! az acr show -n "${AZURE_ACR_NAME}" -g "${AZURE_RESOURCE_GROUP}" >/dev/null 2>&1; then
    write_info "ACR not found. Creating: ${AZURE_ACR_NAME}"
    az acr create -n "${AZURE_ACR_NAME}" -g "${AZURE_RESOURCE_GROUP}" --sku Basic >/dev/null
  fi
  write_success "ACR ready: ${AZURE_ACR_NAME}"
}

function ensure_keyvault_exists() {
  # Not used for secrets for now; just ensuring it exists since workflow includes it.
  write_info "Ensuring Key Vault exists: ${AZURE_KEY_VAULT_NAME}"
  if ! az keyvault show -n "${AZURE_KEY_VAULT_NAME}" -g "${AZURE_RESOURCE_GROUP}" >/dev/null 2>&1; then
    write_info "Key Vault not found. Creating: ${AZURE_KEY_VAULT_NAME}"
    az keyvault create -n "${AZURE_KEY_VAULT_NAME}" -g "${AZURE_RESOURCE_GROUP}" -l "${AZURE_LOCATION}" >/dev/null
  fi
  write_success "Key Vault ready: ${AZURE_KEY_VAULT_NAME}"
}

function build_and_push_image() {
  local dockerfile_path=$1
  local context_path=$2

  write_info "Logging in to ACR: ${AZURE_ACR_NAME}"
  az acr login --name "${AZURE_ACR_NAME}" >/dev/null

  write_info "Building image: ${ACR_SERVER}/${APP_NAME_MAIN}:${IMAGE_TAG}"
  docker build \
    --build-arg APP_MODULE="${APP_MODULE}" \
    -t "${ACR_SERVER}/${APP_NAME_MAIN}:${IMAGE_TAG}" \
    -f "${dockerfile_path}" \
    "${context_path}" >&2

  write_info "Pushing image..."
  docker push "${ACR_SERVER}/${APP_NAME_MAIN}:${IMAGE_TAG}" >&2

  write_success "Image pushed: ${ACR_SERVER}/${APP_NAME_MAIN}:${IMAGE_TAG}"
}

function bootstrap_mysql_db_user() {
  # Requires: rdbms-connect extension
  if ! az extension show -n rdbms-connect >/dev/null 2>&1; then
    write_info "Installing Azure CLI extension: rdbms-connect"
    az extension add -n rdbms-connect >/dev/null
  fi

  local mysql_server_name=$1
  write_info "Bootstrapping MySQL DB/user on server: ${mysql_server_name}"

  # Escape single quotes in password for SQL
  local app_pass_escaped
  app_pass_escaped="$(printf "%s" "${APP_DB_PASSWORD}" | sed "s/'/''/g")"

  az mysql flexible-server execute \
    -g "${AZURE_RESOURCE_GROUP}" \
    -n "${mysql_server_name}" \
    --admin-user "${MYSQL_ADMIN_USER}" \
    --admin-password "${MYSQL_ADMIN_PASSWORD}" \
    --database-name "${MYSQL_DB_NAME}" \
    --querytext "CREATE DATABASE IF NOT EXISTS \`${MYSQL_DB_NAME}\`;
CREATE USER IF NOT EXISTS '${APP_DB_USER}'@'%' IDENTIFIED BY '${app_pass_escaped}';
ALTER USER '${APP_DB_USER}'@'%' IDENTIFIED BY '${app_pass_escaped}';
GRANT ALL PRIVILEGES ON \`${MYSQL_DB_NAME}\`.* TO '${APP_DB_USER}'@'%';
FLUSH PRIVILEGES;" \
    >/dev/null

  write_success "MySQL bootstrap complete."
}

function deploy_infrastructure() {
  write_info "Fetching ACR credentials for registry auth..."
  local ACR_USER ACR_PASS
  ACR_USER="$(az acr credential show -n "${AZURE_ACR_NAME}" -g "${AZURE_RESOURCE_GROUP}" --query username -o tsv)"
  ACR_PASS="$(az acr credential show -n "${AZURE_ACR_NAME}" -g "${AZURE_RESOURCE_GROUP}" --query "passwords[0].value" -o tsv)"
  write_info "Starting Bicep deployment for ${ENVIRONMENT_NAME} environment..."

  local app_fqdn
  app_fqdn=$(az deployment group create \
    --resource-group "${AZURE_RESOURCE_GROUP}" \
    --template-file "${BICEP_FILE}" \
    --parameters \
      location="${AZURE_LOCATION}" \
      environmentName="${AZURE_ENVIRONMENT_NAME}" \
      keyVaultName="${AZURE_KEY_VAULT_NAME}" \
      acrName="${AZURE_ACR_NAME}" \
      appImageTag="${IMAGE_TAG}" \
      revisionSuffix="${REV_SUFFIX}" \
      acrUsername="${ACR_USER}" \
      acrPassword="${ACR_PASS}" \
      mysqlLocation="${MYSQL_LOCATION:-canadacentral}" \
      revisionMode="${REVISION_MODE}" \
      namePrefix="${NAME_PREFIX}" \
      appName="${APP_NAME_MAIN}" \
      mysqlAdminUser="${MYSQL_ADMIN_USER}" \
      mysqlAdminPassword="${MYSQL_ADMIN_PASSWORD}" \
      mysqlDatabaseName="${MYSQL_DB_NAME}" \
      appDbUser="${APP_DB_USER}" \
      appDbPassword="${APP_DB_PASSWORD}" \
      enableSmtp="${ENABLE_SMTP}" \
      smtpUsername="${SMTP_USERNAME}" \
      smtpPassword="${SMTP_PASSWORD}" \
      smtpFrom="${SMTP_FROM}" \
      webrtcAdminApiUrl="${WEBRTC_ADMIN_API_URL}" \
      webrtcPublicBaseUrl="${WEBRTC_PUBLIC_BASE_URL}" \
      webrtcAdminApiKey="${WEBRTC_ADMIN_API_KEY}" \
      webrtcAdminUpsertPath="${WEBRTC_ADMIN_UPSERT_PATH}" \
      webrtcAdminUpdatePath="${WEBRTC_ADMIN_UPDATE_PATH}" \
      webrtcAdminDeletePath="${WEBRTC_ADMIN_DELETE_PATH}" \
      deployMediaMtx="${DEPLOY_MEDIA_MTX}" \
      mediamtxApiUser="${MEDIAMTX_API_USER}" \
      mediamtxApiPass="${MEDIAMTX_API_PASS}" \
    --query "properties.outputs.appUrl.value" \
    -o tsv)

  echo "${app_fqdn}"
}

function health_check() {
  local app_fqdn=$1
  local health_endpoint="https://${app_fqdn}/"
  write_info "Performing health check on ${health_endpoint}..."

  for i in {1..20}; do
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "${health_endpoint}" || true)
    if [[ "${http_code}" -ge 200 && "${http_code}" -lt 400 ]]; then
      write_success "Health check passed with status ${http_code}!"
      return 0
    fi
    write_info "Attempt ${i}/20 failed with status ${http_code}, retrying in 5s..."
    sleep 5
  done

  write_error "Health check failed for ${health_endpoint}"
  return 1
}
SHORT_SHA="$(echo "${GITHUB_SHA:-latest}" | cut -c1-12)"
REV_SUFFIX="sha-${SHORT_SHA}"   # always starts with a letter

# --- Main Execution ---
function main() {
  write_success "Starting ${ENVIRONMENT_NAME} deployment..."
  write_info "Deployment ID: ${REVISION_SUFFIX}"

  validate_prerequisites
  ensure_subscription_context
  ensure_resource_group
  ensure_acr_exists
  ensure_keyvault_exists

  # Build + push image
  build_and_push_image "../Dockerfile" ".."

  # Deploy using Bicep
  local app_fqdn
  app_fqdn=$(deploy_infrastructure)
  if [[ -z "${app_fqdn}" ]]; then
    write_error "Failed to get App FQDN from Bicep deployment output."
    exit 1
  fi

  # Read mysql outputs to bootstrap DB user (optional but matches your local script)
  local mysql_server_name
  mysql_server_name=$(az deployment group show \
    -g "${AZURE_RESOURCE_GROUP}" \
    --name "$(az deployment group list -g "${AZURE_RESOURCE_GROUP}" --query "[-1].name" -o tsv)" \
    --query "properties.outputs.mysqlServerName.value" -o tsv 2>/dev/null || true)

  # If we couldn't infer it from last deployment name, query using the known naming pattern isn't reliable.
  # So instead, read it from the deployment we just ran by using its name from deploy_infrastructure call is not available.
  # We’ll do a safer approach: find the latest deployment that has mysqlServerName output.
  if [[ -z "${mysql_server_name}" ]]; then
    mysql_server_name=$(az deployment group list -g "${AZURE_RESOURCE_GROUP}" \
      --query "[?properties.outputs.mysqlServerName.value != null] | [-1].properties.outputs.mysqlServerName.value" -o tsv 2>/dev/null || true)
  fi

  if [[ -n "${mysql_server_name}" ]]; then
    bootstrap_mysql_db_user "${mysql_server_name}"
  else
    write_info "Skipping MySQL bootstrap (could not read mysqlServerName output)."
  fi

  # Final health check
  health_check "${app_fqdn}"

  echo "" >&2
  write_success "=== ${ENVIRONMENT_NAME} DEPLOYMENT COMPLETED ==="
  write_success "Application URL: https://${app_fqdn}"
  write_success "Image Tag: ${IMAGE_TAG}"

  # IMPORTANT: print ONLY the URL to stdout (workflow captures it)
  echo "https://${app_fqdn}"
}

main
