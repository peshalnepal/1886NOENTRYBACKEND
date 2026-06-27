// Azure Container Apps deployment using Bicep for Production Environment (1886NOENTRY)

param location string = resourceGroup().location
param environmentName string
param acrName string

param appName string
param namePrefix string = 'noentry'

param appImageTag string = 'latest'
param revisionSuffix string = ''
param deployStamp string = utcNow()

var stamp = toLower(replace(replace(replace(deployStamp, ':', ''), 't', '-'), 'z', ''))
var rsBase = empty(revisionSuffix) ? 'rev' : toLower(revisionSuffix)
var rsRaw = '${rsBase}-${stamp}'
var revisionSuffixFinal = substring(rsRaw, 0, min(length(rsRaw), 40))

param revisionMode string = 'Single' // 'Multiple' for blue/green
param mysqlLocation string = location

// Feature toggles
param deployMediaMtx bool = true
param createMysqlDatabase bool = false
param videoClipStorageAccountName string = '1886noentry'
param videoClipContainerName string = 'event-clips'

param mediamtxImageRepo string = 'mediamtx'
param mediamtxImageTag string = '1.16.1'
param acrUsername string
param caddyImageRepo string = 'caddy'
param caddyImageTag string = '2.8.4'

@secure()
param acrPassword string

// ----------------------------
// App settings (non-secrets)
// ----------------------------
param webrtcAdminApiUrl string = ''
param webrtcPublicBaseUrl string = ''
param webrtcAdminApiKey string = ''

param webrtcAdminUpsertPath string = ''
param webrtcAdminUpdatePath string = ''
param webrtcAdminDeletePath string = ''

param smtpUsername string = ''

@secure()
param smtpPassword string = '' // TEMP (no secrets system for now)

param smtpFrom string = 'no-reply@1886noentry.com'

// ----------------------------
// MySQL Flexible Server
// ----------------------------
param mysqlAdminUser string = 'mysqladmin'

@secure()
param mysqlAdminPassword string // TEMP (no secrets system for now)

param mysqlDatabaseName string = 'appdb'
param appDbUser string = 'appuser'

@secure()
param appDbPassword string // TEMP (no secrets system for now)

// ----------------------------
// Optional MediaMTX (kept for later)
// ----------------------------
param mediamtxApiUser string = 'api'

@secure()
param mediamtxApiPass string = '' // TEMP (only if deployMediaMtx=true)

// ----------------------------
// Reference existing ACR + Key Vault (not used for secrets yet)
// ----------------------------
resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: acrName
}

// ----------------------------
// Log Analytics Workspace
// ----------------------------
var suffix = toLower(substring(uniqueString(resourceGroup().id, namePrefix, acrName, environmentName), 0, 6))

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${namePrefix}-prod-logs-${suffix}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ----------------------------
// Container Apps Environment
// ----------------------------
resource environment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: environmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

// ----------------------------
// MySQL Flexible Server + DB
// ----------------------------
var mysqlSuffix = toLower(substring(uniqueString(resourceGroup().id, namePrefix, environmentName, mysqlLocation), 0, 6))
var mysqlServerName = toLower('${namePrefix}-mysql-${mysqlSuffix}')
var mysqlFqdn = '${mysqlServerName}.mysql.database.azure.com'
var clipStorageName = videoClipStorageAccountName != ''
  ? toLower(videoClipStorageAccountName)
  : toLower(substring(replace('${namePrefix}clips${suffix}', '-', ''), 0, 24))
var clipStorageKey = clipStorage.listKeys().keys[0].value
var clipStorageConnectionString = 'DefaultEndpointsProtocol=https;AccountName=${clipStorage.name};AccountKey=${clipStorageKey};EndpointSuffix=core.windows.net'

var databaseUrl = 'Driver={MySQL ODBC 8.0 Unicode Driver};Server=${mysqlFqdn};Port=3306;Database=${mysqlDatabaseName};User=${appDbUser};Password=${appDbPassword};Option=3;'

resource mysql 'Microsoft.DBforMySQL/flexibleServers@2024-12-30' = {
  name: mysqlServerName
  location: mysqlLocation
  sku: {
    name: 'Standard_D2ads_v5'
    tier: 'GeneralPurpose'
  }
  properties: {
    administratorLogin: mysqlAdminUser
    administratorLoginPassword: mysqlAdminPassword
    version: '8.0.21'
    storage: {
      storageSizeGB: 32
      autoGrow: 'Enabled'
      autoIoScaling: 'Enabled'
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    network: {
      publicNetworkAccess: 'Enabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
  }
}

resource mysqlFwAzure 'Microsoft.DBforMySQL/flexibleServers/firewallRules@2024-12-30' = {
  parent: mysql
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

resource mysqlRequireSecureTransport 'Microsoft.DBforMySQL/flexibleServers/configurations@2024-12-30' = {
  parent: mysql
  name: 'require_secure_transport'
  properties: {
    value: 'OFF'
    source: 'user-override'
  }
}

resource mysqlDb 'Microsoft.DBforMySQL/flexibleServers/databases@2024-12-30' = if (createMysqlDatabase) {
  parent: mysql
  name: mysqlDatabaseName
  properties: {
    charset: 'utf8mb4'
    collation: 'utf8mb4_unicode_ci'
  }
}

resource clipStorage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: clipStorageName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource clipStorageBlobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: clipStorage
  name: 'default'
}

resource clipStorageContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: clipStorageBlobService
  name: videoClipContainerName
  properties: {
    publicAccess: 'None'
  }
}
resource caddyDataFileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = if (deployMediaMtx) {
  name: '${clipStorage.name}/default/caddy-tls'
  properties: {
    shareQuota: 1
  }
}
resource caddyConfigFileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = if (deployMediaMtx) {
  name: '${clipStorage.name}/default/caddy-config'
  properties: {
    shareQuota: 1
  }
}
// ----------------------------
// Backend API (Container App)
// ----------------------------
resource app 'Microsoft.App/containerApps@2023-05-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: revisionMode
      ingress: {
        external: true
        targetPort: 8080
        transport: 'auto'
        allowInsecure: false
        corsPolicy: {
          allowedOrigins: [
            '*'
          ]
          allowedMethods: [
            'GET'
            'POST'
            'PUT'
            'PATCH'
            'DELETE'
            'OPTIONS'
          ]
          allowedHeaders: [
            '*'
          ]
          allowCredentials: true
        }
      }
      secrets: [
        {
          name: 'acr-password'
          value: acrPassword
        }
        {
          name: 'video-clip-blob-connection-string'
          value: clipStorageConnectionString
        }
      ]
      registries: [
        {
          server: acr.properties.loginServer
          username: acrUsername
          passwordSecretRef: 'acr-password'
        }
      ]
    }

    template: {
      revisionSuffix: revisionSuffixFinal
      containers: [
        {
          name: 'notentapi'
          image: '${acr.properties.loginServer}/${appName}:${appImageTag}'
          resources: {
            cpu: json('2.0')
            memory: '4.0Gi'
          }
          env: [
            { name: 'ENVIRONMENT', value: 'production' }
            { name: 'DEBUG', value: 'false' }
            { name: 'PORT', value: '8080' }
            { name: 'DB_INIT_MAX_ATTEMPTS', value: '40' }
            { name: 'DB_INIT_RETRY_DELAY_S', value: '5' }
            { name: 'DB_CONNECT_TIMEOUT_S', value: '5' }
            { name: 'DB_IO_TIMEOUT_S', value: '15' }

            { name: 'DATABASE_URL', value: databaseUrl }

            { name: 'SMTP_USERNAME', value: smtpUsername }
            { name: 'SMTP_PASSWORD', value: smtpPassword }
            { name: 'SMTP_FROM', value: smtpFrom }
            { name: 'DEFAULT_SAMPLE_FPS', value: '3.0' }
            { name: 'INFER_QUEUE_MAX', value: '8' }
            { name: 'CHANNEL_OUT_Q_MAX', value: '4' }
            { name: 'PIPELINE_OUT_QUEUE_MAX', value: '500' }
            { name: 'PENDING_KEY_MAX', value: '1000' }
            { name: 'PIPELINE_LOG_EVERY_N_FRAMES', value: '0' }
            { name: 'WEBRTC_ADMIN_API_URL', value: webrtcAdminApiUrl }
            { name: 'WEBRTC_PUBLIC_BASE_URL', value: webrtcPublicBaseUrl }
            { name: 'WEBRTC_ADMIN_API_KEY', value: webrtcAdminApiKey }

            { name: 'WEBRTC_ADMIN_UPSERT_PATH', value: webrtcAdminUpsertPath }
            { name: 'WEBRTC_ADMIN_UPDATE_PATH', value: webrtcAdminUpdatePath }
            { name: 'WEBRTC_ADMIN_DELETE_PATH', value: webrtcAdminDeletePath }

            // MediaMTX admin API credentials (must match mediamtx.yml authInternalUsers)
            { name: 'MTX_API_USER', value: mediamtxApiUser }
            { name: 'MTX_API_PASS', value: mediamtxApiPass }

            // Video clip capture (pre-record on detection)
            { name: 'VIDEO_CLIP_CAPTURE_ENABLED', value: 'true' }
            // Clip layout: 90s pre-roll + 30s post-roll = 120s total, event anchored at 90s.
            { name: 'VIDEO_CLIP_PRE_EVENT_S', value: '90' }
            { name: 'VIDEO_CLIP_POST_EVENT_S', value: '30' }
            { name: 'VIDEO_CLIP_DURATION_S', value: '120' }
            { name: 'VIDEO_CLIP_COOLDOWN_S', value: '120' }
            { name: 'VIDEO_CLIP_MIN_DURATION_S', value: '10' }
            { name: 'VIDEO_CLIP_SAS_TTL_HOURS', value: '168' }
            { name: 'ALERT_RETENTION_DAYS', value: '7' }
            { name: 'VIDEO_CLIP_HTTP_TIMEOUT_S', value: '180' }
            { name: 'VIDEO_CLIP_BLOB_CONTAINER', value: videoClipContainerName }
            { name: 'MEDIAMTX_PLAYBACK_BASE_URL', value: deployMediaMtx ? 'https://${proxyHost}/playback' : '' }
            { name: 'VIDEO_CLIP_BLOB_CONNECTION_STRING', secretRef: 'video-clip-blob-connection-string' }
          ]
          probes: [
            {
              type: 'Startup'
              httpGet: {
                path: '/'
                port: 8080
              }
              initialDelaySeconds: 15
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/'
                port: 8080
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 6
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/'
                port: 8080
              }
              initialDelaySeconds: 30
              periodSeconds: 30
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }

  dependsOn: createMysqlDatabase
    ? [
        clipStorageContainer
        mysqlDb
        mysqlFwAzure
        mysqlRequireSecureTransport
      ]
    : [
        clipStorageContainer
        mysqlFwAzure
        mysqlRequireSecureTransport
      ]
}

@description('Optional host override for public URL advertised by MediaMTX (custom domain). Leave blank to use ACI FQDN.')
param mediamtxHostOverride string = ''

param caddyEmail string = 'peshalnepal3@gmail.com'

var mediamtxName = '${namePrefix}-mtx-${suffix}'
var mediamtxDns = '${namePrefix}mtx${suffix}'

var regionForHost = toLower(replace(location, ' ', ''))
var mediamtxPublicHost = '${mediamtxDns}.${regionForHost}.azurecontainer.io'

var proxyHost = (mediamtxHostOverride != '') ? mediamtxHostOverride : mediamtxPublicHost
var mediamtxYaml = $'''
logLevel: info
logDestinations: [stdout]
writeQueueSize: 1024
udpReadBufferSize: 0

readTimeout: 60s
writeTimeout: 60s

authMethod: internal
authInternalUsers:
  - user: any
    pass: ""
    ips: []
    permissions:
      - action: publish
        path:
      - action: read
        path:
      - action: playback
        path:
  - user: "${mediamtxApiUser}"
    pass: "${mediamtxApiPass}"
    ips: []
    permissions:
      - action: api

api: yes
apiAddress: :9997
apiAllowOrigins: ["*"]

playback: yes
playbackAddress: :9996
playbackAllowOrigins: ["*"]

rtsp: yes
rtspTransports: [tcp]
rtspAddress: :8554

webrtc: yes
webrtcAddress: :8889
webrtcLocalUDPAddress: :8189
webrtcLocalTCPAddress: ""
webrtcAllowOrigins: ["*"]
webrtcIPsFromInterfaces: no
webrtcAdditionalHosts: ["${proxyHost}"]

webrtcICEServers2:
  - url: stun:stun.l.google.com:19302

pathDefaults:
  record: yes
  recordPath: /recordings/%path/%Y-%m-%d_%H-%M-%S-%f
  recordFormat: fmp4
  recordPartDuration: 1s
  recordSegmentDuration: 15m
  recordDeleteAfter: 7d

# Source/ingest protocols. A camera's source_url can be any scheme — MediaMTX
# PULLS it (path "source" set via the admin API) and re-broadcasts to browsers
# over WebRTC/WHEP. Enabling these makes the rtmp/srt/http(HLS) source readers
# available (rtsp + webrtc were already on). The listeners bind internally only;
# they are NOT exposed on the public ACI IP below, so this stays pull-only (no
# anonymous push). To also accept PUSH ingest, publish ports 1935 (rtmp, TCP)
# and 8890 (srt, UDP) on the container group ipAddress.
hls: yes
hlsAddress: :8888
hlsAllowOrigins: ["*"]

rtmp: yes
rtmpAddress: :1935

srt: yes
srtAddress: :8890

paths:
  all_others: {}
'''

var caddyfile = $'''
{
  email ${caddyEmail}
}

${proxyHost} {
  encode gzip

  handle_path /playback/* {
    reverse_proxy 127.0.0.1:9996
  }

  @api path /v3/*
  reverse_proxy @api 127.0.0.1:9997

  reverse_proxy 127.0.0.1:8889
}
'''

resource mediamtx 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = if (deployMediaMtx) {
  name: mediamtxName
  location: location
  properties: {
    osType: 'Linux'
    restartPolicy: 'Always'

    imageRegistryCredentials: [
      {
        server: acr.properties.loginServer
        username: acrUsername
        password: acrPassword
      }
    ]

    ipAddress: {
      type: 'Public'
      dnsNameLabel: mediamtxDns
      ports: [
        { port: 80, protocol: 'TCP' }
        { port: 443, protocol: 'TCP' }
        { port: 8554, protocol: 'TCP' } // RTSP ingest (TCP only — ACI dedupes by port number)
        { port: 8189, protocol: 'UDP' }
      ]
    }

    volumes: [
      {
        name: 'cfg'
        secret: {
          'mediamtx.yml': base64(mediamtxYaml)
        }
      }
      {
        name: 'caddy'
        secret: {
          Caddyfile: base64(caddyfile)
        }
      }
      {
        name: 'caddy-data'
        azureFile: {
          shareName: 'caddy-tls'
          storageAccountName: clipStorage.name
          storageAccountKey: clipStorageKey
          readOnly: false
        }
      }
      {
        name: 'caddy-config'
        azureFile: {
          shareName: 'caddy-config'
          storageAccountName: clipStorage.name
          storageAccountKey: clipStorageKey
          readOnly: false
        }
      }
    ]

    containers: [
      {
        name: 'mediamtx'
        properties: {
          image: '${acr.properties.loginServer}/${mediamtxImageRepo}:${mediamtxImageTag}'
          ports: [
            { port: 8554, protocol: 'TCP' } // RTSP ingest (TCP only — ACI dedupes by port number)
            { port: 9996, protocol: 'TCP' }
            { port: 8889, protocol: 'TCP' }
            { port: 9997, protocol: 'TCP' }
            { port: 8189, protocol: 'UDP' }
            { port: 8888, protocol: 'TCP' } // HLS muxer (internal; source-reader support)
            { port: 1935, protocol: 'TCP' } // RTMP server (internal; expose on ipAddress for push)
            { port: 8890, protocol: 'UDP' } // SRT server (internal; expose on ipAddress for push)
          ]
          resources: {
            requests: {
              cpu: 2
              memoryInGB: json('4')
            }
          }
          volumeMounts: [
            { name: 'cfg', mountPath: '/cfg', readOnly: true }
          ]
          command: [
            '/mediamtx'
            '/cfg/mediamtx.yml'
          ]
        }
      }

      {
        name: 'caddy'
        properties: {
          image: '${acr.properties.loginServer}/${caddyImageRepo}:${caddyImageTag}'
          ports: [
            { port: 80, protocol: 'TCP' }
            { port: 443, protocol: 'TCP' }
          ]
          resources: {
            requests: {
              cpu: json('0.5')
              memoryInGB: json('1')
            }
          }
          volumeMounts: [
            { name: 'caddy', mountPath: '/etc/caddy', readOnly: true }
            { name: 'caddy-data', mountPath: '/data', readOnly: false }
            { name: 'caddy-config', mountPath: '/config', readOnly: false }
          ]
          command: [
            'caddy'
            'run'
            '--config'
            '/etc/caddy/Caddyfile'
            '--adapter'
            'caddyfile'
            '--resume'
          ]
        }
      }
    ]
  }
  dependsOn: [
    caddyDataFileShare
    caddyConfigFileShare
  ]
}

// ----------------------------
// Outputs (matches your deploy.sh query)
// ----------------------------
output appUrl string = app.properties.configuration.ingress.fqdn
output mysqlServerName string = mysql.name
output mysqlHost string = mysqlFqdn
output mysqlDatabase string = mysqlDatabaseName
output mediamtxFqdn string = deployMediaMtx ? proxyHost : ''
output mediamtxHttpsBase string = deployMediaMtx ? 'https://${proxyHost}' : ''

output mediamtxPlaybackUrl string = deployMediaMtx ? 'https://${proxyHost}/playback' : ''
output mediamtxWebrtcBase string = deployMediaMtx ? 'https://${proxyHost}' : ''
output mediamtxRecordingPath string = '/recordings/<stream_key>/<YYYY-MM-DD_HH-MM-SS>'
output clipStorageAccount string = clipStorage.name
output clipBlobContainer string = videoClipContainerName
