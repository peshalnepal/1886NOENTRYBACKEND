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
    # Single revision mode: there is no previous revision to restore.
    # Deactivating the new revision would leave the app with zero active replicas.
    # Leave it running and let the operator investigate logs.
    write_info "Single revision mode — no previous revision to restore. Leaving current revision active."
    write_info "Inspect logs: az containerapp logs show -n ${APP_NAME_MAIN} -g ${AZURE_RESOURCE_GROUP} --follow"
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

  az acr update -n "${AZURE_ACR_NAME}" -g "${AZURE_RESOURCE_GROUP}" --admin-enabled true >/dev/null

  write_success "ACR ready: ${AZURE_ACR_NAME}"
}
function ensure_keyvault_exists() {
  write_info "Ensuring Key Vault exists: ${AZURE_KEY_VAULT_NAME}"
  if az keyvault show -n "${AZURE_KEY_VAULT_NAME}" -g "${AZURE_RESOURCE_GROUP}" >/dev/null 2>&1; then
    write_success "Key Vault ready: ${AZURE_KEY_VAULT_NAME}"
    return 0
  fi

  # Check soft-deleted state (Key Vault retains names for 90 days after deletion).
  local soft_deleted
  soft_deleted=$(az keyvault list-deleted \
    --query "[?name=='${AZURE_KEY_VAULT_NAME}'].name | [0]" -o tsv 2>/dev/null || true)

  if [[ -n "${soft_deleted}" ]]; then
    write_info "Key Vault is soft-deleted. Recovering: ${AZURE_KEY_VAULT_NAME}"
    az keyvault recover -n "${AZURE_KEY_VAULT_NAME}" -l "${AZURE_LOCATION}" >/dev/null
  else
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

function deploy_infrastructure() {
  write_info "Fetching ACR credentials for registry auth..."
  local ACR_USER ACR_PASS
  ACR_USER="$(az acr credential show -n "${AZURE_ACR_NAME}" -g "${AZURE_RESOURCE_GROUP}" --query username -o tsv)"
  ACR_PASS="$(az acr credential show -n "${AZURE_ACR_NAME}" -g "${AZURE_RESOURCE_GROUP}" --query "passwords[0].value" -o tsv)"
  write_info "Starting Bicep deployment for ${ENVIRONMENT_NAME} environment..."

  local app_fqdn
  app_fqdn=$(DOTNET_SYSTEM_GLOBALIZATION_INVARIANT="${DOTNET_SYSTEM_GLOBALIZATION_INVARIANT:-1}" az deployment group create \
    --resource-group "${AZURE_RESOURCE_GROUP}" \
    --template-file "${BICEP_FILE}" \
    --parameters \
      location="${AZURE_LOCATION}" \
      environmentName="${AZURE_ENVIRONMENT_NAME}" \
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

function ensure_mysql_firewall_for_containerapp() {
  local mysql_server
  mysql_server=$(az mysql flexible-server list \
    -g "${AZURE_RESOURCE_GROUP}" \
    --query "[?starts_with(name, '${NAME_PREFIX}-mysql-')].name | [0]" \
    -o tsv 2>/dev/null || true)

  if [[ -z "${mysql_server}" || "${mysql_server}" == "null" ]]; then
    write_info "No MySQL flexible server found for prefix ${NAME_PREFIX}; skipping firewall sync."
    return 0
  fi

  local outbound_ips
  outbound_ips=$(az containerapp show \
    -n "${APP_NAME_MAIN}" \
    -g "${AZURE_RESOURCE_GROUP}" \
    --query "properties.outboundIpAddresses" \
    -o tsv 2>/dev/null || true)

  if [[ -z "${outbound_ips}" || "${outbound_ips}" == "null" ]]; then
    write_info "Container App outbound IPs not available yet; skipping MySQL firewall sync."
    return 0
  fi

  local added_any=false
  local ip rule_name
  for ip in ${outbound_ips}; do
    [[ -z "${ip}" || "${ip}" == "null" ]] && continue
    rule_name="allow-containerapp-egress-${ip//./-}"

    if az mysql flexible-server firewall-rule show \
      -g "${AZURE_RESOURCE_GROUP}" \
      -n "${mysql_server}" \
      --rule-name "${rule_name}" >/dev/null 2>&1; then
      continue
    fi

    write_info "Allowing Container App outbound IP ${ip} on MySQL server ${mysql_server}"
    az mysql flexible-server firewall-rule create \
      -g "${AZURE_RESOURCE_GROUP}" \
      -n "${mysql_server}" \
      --rule-name "${rule_name}" \
      --start-ip-address "${ip}" \
      --end-ip-address "${ip}" >/dev/null
    added_any=true
  done

  if [[ "${added_any}" == true ]]; then
    write_success "MySQL firewall updated for Container App outbound IPs."
  else
    write_info "MySQL firewall already allows current Container App outbound IPs."
  fi
}

function dump_startup_diagnostics() {
  write_info "Container App diagnostics for ${APP_NAME_MAIN}..."
  az containerapp revision list \
    -n "${APP_NAME_MAIN}" \
    -g "${AZURE_RESOURCE_GROUP}" \
    -o table >&2 || true

  local latest_revision
  latest_revision=$(az containerapp show \
    -n "${APP_NAME_MAIN}" \
    -g "${AZURE_RESOURCE_GROUP}" \
    --query "properties.latestRevisionName" \
    -o tsv 2>/dev/null || true)

  if [[ -z "${latest_revision}" ]]; then
    write_info "Latest revision name is not available yet."
    return 0
  fi

  write_info "System logs for revision ${latest_revision}:"
  az containerapp logs show \
    -n "${APP_NAME_MAIN}" \
    -g "${AZURE_RESOURCE_GROUP}" \
    --revision "${latest_revision}" \
    --type system \
    --tail 100 \
    --format text >&2 || true

  write_info "Console logs for revision ${latest_revision}:"
  az containerapp logs show \
    -n "${APP_NAME_MAIN}" \
    -g "${AZURE_RESOURCE_GROUP}" \
    --revision "${latest_revision}" \
    --tail 100 \
    --format text >&2 || true
}

function route_traffic_to_latest() {
  if [[ "${REVISION_MODE}" != "Multiple" ]]; then
    write_info "Single revision mode — Azure routes traffic automatically. Skipping manual traffic shift."
    return 0
  fi

  write_info "Routing 100% traffic to latest revision of ${APP_NAME_MAIN}..."

  local latest_revision
  latest_revision=$(az containerapp show \
    -n "${APP_NAME_MAIN}" \
    -g "${AZURE_RESOURCE_GROUP}" \
    --query "properties.latestRevisionName" \
    -o tsv 2>/dev/null || true)

  if [[ -z "${latest_revision}" ]]; then
    write_error "Could not determine latest revision name — skipping traffic shift."
    return 1
  fi

  write_info "Latest revision: ${latest_revision}"

  az containerapp ingress traffic set \
    -n "${APP_NAME_MAIN}" \
    -g "${AZURE_RESOURCE_GROUP}" \
    --revision-weight "${latest_revision}=100" >/dev/null

  write_success "All traffic now routed to: ${latest_revision}"

  # Deactivate any other revisions that have 0% traffic to keep things clean.
  local old_revisions
  old_revisions=$(az containerapp revision list \
    -n "${APP_NAME_MAIN}" \
    -g "${AZURE_RESOURCE_GROUP}" \
    --query "[?name != '${latest_revision}' && properties.active == true].name" \
    -o tsv 2>/dev/null || true)

  if [[ -n "${old_revisions}" ]]; then
    while IFS= read -r rev; do
      write_info "Deactivating old revision: ${rev}"
      az containerapp revision deactivate \
        -n "${APP_NAME_MAIN}" \
        -g "${AZURE_RESOURCE_GROUP}" \
        --revision "${rev}" >/dev/null 2>&1 || true
    done <<< "${old_revisions}"
    write_success "Old revisions deactivated."
  fi
}

function health_check() {
  local app_fqdn=$1
  local health_endpoint="https://${app_fqdn}/"
  write_info "Performing health check on ${health_endpoint}..."
  write_info "Waiting 30s for container to initialize before first probe..."
  sleep 30

  for i in {1..36}; do
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${health_endpoint}" || true)
    if [[ "${http_code}" -ge 200 && "${http_code}" -lt 400 ]]; then
      write_success "Health check passed (attempt ${i}) with status ${http_code}!"
      return 0
    fi
    if (( i % 6 == 0 )); then
      write_info "Health check still failing after ${i} attempt(s); collecting current revision state..."
      az containerapp revision list \
        -n "${APP_NAME_MAIN}" \
        -g "${AZURE_RESOURCE_GROUP}" \
        -o table >&2 || true
    fi
    write_info "Attempt ${i}/36 — status ${http_code}, retrying in 10s..."
    sleep 10
  done

  write_error "Health check failed after ~6 minutes: ${health_endpoint}"
  dump_startup_diagnostics
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
  ensure_mysql_firewall_for_containerapp

  # Build + push image
  build_and_push_image "../Dockerfile" ".."

  # Deploy using Bicep
  local app_fqdn
  app_fqdn=$(deploy_infrastructure)
  if [[ -z "${app_fqdn}" ]]; then
    write_error "Failed to get App FQDN from Bicep deployment output."
    exit 1
  fi

  ensure_mysql_firewall_for_containerapp

  # NOTE: MySQL bootstrap via `az mysql flexible-server execute` has been REMOVED.
  # You said you will run migrations later (recommended).

  # Explicitly route all traffic to the new revision.
  # Bicep with Single mode should do this automatically, but if the Container App
  # was ever in Multiple mode, Azure leaves old revisions with 100% traffic and
  # the new revision at 0%. This step is always safe to run.
  route_traffic_to_latest

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
