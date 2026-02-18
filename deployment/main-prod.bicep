// Azure Container Apps deployment using Bicep for Production Environment (1886NOENTRY)

param location string = resourceGroup().location
param environmentName string
param keyVaultName string
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

param enableSmtp bool = false
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

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' existing = {
  name: keyVaultName
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

// This is the exact format you are using in .env today
var databaseUrl = 'Driver={MySQL ODBC 8.0 Unicode Driver};Server=${mysqlFqdn};Port=3306;Database=${mysqlDatabaseName};User=${appDbUser};Password=${appDbPassword};Option=3;'

resource mysql 'Microsoft.DBforMySQL/flexibleServers@2024-12-30' = {
  name: mysqlServerName
  location: mysqlLocation
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    administratorLogin: mysqlAdminUser
    administratorLoginPassword: mysqlAdminPassword
    version: '8.0.21'
    storage: {
      storageSizeGB: 32
      autoGrow: 'Enabled'
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
  name: '${mysql.name}/AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

resource mysqlRequireSecureTransport 'Microsoft.DBforMySQL/flexibleServers/configurations@2024-12-30' = {
  name: '${mysql.name}/require_secure_transport'
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
            cpu: json('0.5')
            memory: '1.0Gi'
          }
          env: [
            { name: 'ENVIRONMENT', value: 'production' }
            { name: 'DEBUG', value: 'false' }
            { name: 'PORT', value: '8080' }

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
        mysqlDb
        mysqlFwAzure
        mysqlRequireSecureTransport
      ]
    : [
        mysqlFwAzure
        mysqlRequireSecureTransport
      ]
}

@description('Optional host override for public URL advertised by MediaMTX (custom domain). Leave blank to use ACI FQDN.')
param mediamtxHostOverride string = ''

param caddyEmail string = 'peshalnepal3@gmail.com'

var mediamtxName = '${namePrefix}-mtx-${suffix}'
var mediamtxDns  = '${namePrefix}mtx${suffix}'

var regionForHost     = toLower(replace(location, ' ', ''))
var mediamtxPublicHost = '${mediamtxDns}.${regionForHost}.azurecontainer.io'

var proxyHost = (mediamtxHostOverride != '') ? mediamtxHostOverride : mediamtxPublicHost

var mediamtxYaml = $'''
logLevel: info
logDestinations: [stdout]

# helps with throughput when many streams are active (uses a bit more RAM)
writeQueueSize: 2048

# reduce UDP packet loss when networks are busy (0=OS default)
udpReadBufferSize: 1048576

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
apiAllowOrigins: ['*']

webrtc: yes
webrtcAddress: :8889
webrtcAllowOrigins: ['*']

webrtcTrustedProxies: ['127.0.0.1', '::1']

webrtcLocalTCPAddress: :8189

webrtcIPsFromInterfaces: no
webrtcAdditionalHosts: ['${proxyHost}']

# Optional: try WITHOUT STUN first (often faster and simpler when server has a public IP+port)
# If some clients fail to connect, re-enable STUN.
webrtcICEServers2: []
# webrtcICEServers2:
#   - url: stun:stun.l.google.com:19302

# Tighten these only if you want failures to happen quicker; defaults are fine
# webrtcHandshakeTimeout: 10s
# webrtcTrackGatherTimeout: 2s
# webrtcSTUNGatherTimeout: 5s

# If you don't use these, turn them off to save resources
hls: no
rtmp: no
srt: no

# Default path behavior: good for 50 cams
pathDefaults:
  # If your "RTSP -> HTTPS view" means MediaMTX PULLS RTSP sources,
  # forcing TCP avoids UDP/NAT retries and usually reduces startup delay.
  rtspTransport: tcp

  # Scale-friendly: don't pull all 50 cameras 24/7 unless needed.
  # First viewer may still wait for the next keyframe, but this avoids constant bandwidth.
  sourceOnDemand: true
  sourceOnDemandStartTimeout: 10s
  sourceOnDemandCloseAfter: 60s
'''

var caddyfile = $'''
{
  email ${caddyEmail}
}

${proxyHost} {
  encode gzip

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
        { port: 80,  protocol: 'TCP' }
        { port: 443, protocol: 'TCP' }
        { port: 8189, protocol: 'TCP' }
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
          'Caddyfile': base64(caddyfile)
        }
      }
    ]

    containers: [
      {
        name: 'mediamtx'
        properties: {
          image: '${acr.properties.loginServer}/${mediamtxImageRepo}:${mediamtxImageTag}'
          ports: [
            { port: 8889, protocol: 'TCP' }
            { port: 9997, protocol: 'TCP' }
            { port: 8189, protocol: 'TCP' }
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
          ]
          command: [
            'caddy'
            'run'
            '--config'
            '/etc/caddy/Caddyfile'
            '--adapter'
            'caddyfile'
          ]
        }
      }
    ]
  }
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
