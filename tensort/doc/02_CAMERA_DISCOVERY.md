# 02 — Camera discovery and Hikvision connections

## Start with the physical topology

**Direct-camera tower:** cameras and the Jetson connect to a PoE switch. Each camera has a LAN address reachable by the Jetson. Discovery identifies each camera and opens its RTSP stream.

**NVR tower:** the Jetson reaches the NVR's LAN interface. Cameras may live on the NVR's private PoE network, which the Jetson cannot route to. The Jetson asks the recorder for its channel list and opens each stream through the recorder address. Eight channels can legitimately share one host and one username.

**Mixed tower:** static declarations, configured NVRs, locally discovered NVRs, and direct cameras can all be present. The current priority order can fill the selection limit before local cameras are considered. This is a policy decision to understand before deploying a mixed tower.

```text
Direct: Jetson 192.168.50.10 --> camera 192.168.50.21:554
NVR:    Jetson 192.168.50.10 --> NVR 192.168.50.200:554 --> input channel 2
        The RTSP URL uses .200, not a private PoE address such as 192.168.254.2.
```

## Protocols are separate tests

| Protocol | Code's use | Example | What success proves |
| --- | --- | --- | --- |
| WS-Discovery / ONVIF probe | Find responder IPs | UDP multicast `239.255.255.250:3702` | A responder advertised an address |
| HTTP ISAPI | Identify camera/recorder and enumerate inputs | `/ISAPI/System/deviceInfo` | HTTP endpoint and credentials work |
| RTSP | Establish video stream | TCP port 554 and `/Streaming/Channels/102` | Only negotiation, until frames decode |
| RTP video payload | Carry compressed H.264/H.265 video | Usually interleaved over the RTSP TCP session here | Encoded video is arriving |
| GStreamer decode | Turn compressed video into image arrays | BGR `numpy.ndarray` | Video is usable by the edge pipeline |

An ONVIF responder is not automatically supported by this discovery implementation. It still uses Hikvision ISAPI and constructs Hikvision stream paths. There is no complete ONVIF `GetProfiles`/`GetStreamUri` integration here.

## The complete sweep

Read [`scan_network`](../discovery.py), [`DiscoveryService._sweep`](../service.py), and [`PipelineRuntime.verify_source`](../runtime.py).

```mermaid
flowchart TD
    A[Scheduler or request] --> B[Build static NVR candidates]
    B --> C[Probe configured NVRs]
    C --> D[Optional WS-Discovery and subnet ISAPI probes]
    D --> E[Merge identities and endpoints]
    E --> F[Take first MAX_CAMERAS candidates]
    F --> G[Reserve selected URLs and restore matching saved configurations]
    G --> H[Verify selected candidates sequentially]
    H --> I[mark_seen in SQLite]
    I --> J[Link or adopt camera UUID]
    J --> K[Apply collected source updates]
    K --> L[Age missing records and publish report]
```

The scan returns its full candidate list before selection/verification starts. Static candidates are constructed early, but a long local subnet sweep can still delay their first verification because `scan_network()` waits for local scanning to finish.

Priority is **static NVR → configured/dynamic NVR → local NVR → direct local camera**. Dynamic details replace matching static declarations when the configured-NVR scan supplies the same endpoint. Candidate deduplication uses identity plus `(host, RTSP port, channel, is_nvr)` endpoint bookkeeping.

`devices = devices[:camera_limit()]` happens before frame verification. With `MAX_CAMERAS=8`, eight declared but unavailable NVR channels can prevent a ninth, working direct camera from being verified or reported. Increasing a camera limit is not proof that the hardware can sustain the extra workload.

## LAN discovery, step by step

`wsdiscover()` sends the SOAP probe three times, then listens until its default three-second deadline. It extracts responder addresses and IPv4 addresses advertised in `XAddrs`.

`_candidate_hosts()` merges those addresses with `DISCOVERY_SUBNETS`. If that setting is empty, it derives a `/24` from the default-route IPv4 address. This is a convenience, not interface discovery: a tower's default route might be on cellular/Wi-Fi while cameras are attached to Ethernet on another subnet.

Example:

```dotenv
DISCOVERY_LOCAL_ENABLED=true
DISCOVERY_SUBNETS=192.168.50.0/24
DISCOVERY_PROBE_WORKERS=32
DISCOVERY_HTTP_TIMEOUT_S=2
DISCOVERY_WSD_TIMEOUT_S=3
```

A `/24` has 254 usable addresses before excluding the Jetson/configured recorder hosts. In a simplified timeout-heavy case, `ceil(254 / 32) × 2 = 16` seconds of HTTP waves plus the three-second discovery window is about 19 seconds. This is an illustration, not a hard upper bound: authentication adds exchanges, recorder inputs need another request, and socket/request behavior varies.

The parser permits `/16` through `/32`. A `/16` means approximately 65,534 targets, so a misplaced subnet mask can create a very long scan. For a known recorder or fixed camera address use a narrow range or `/32` when appropriate.

`DISCOVERY_SWEEP_ONLY_IF_WSD_EMPTY=false` means a partial multicast response does not suppress the subnet sweep. Setting it to true can reduce probing but can also hide devices that failed to answer multicast.

## Hikvision authentication for a junior developer

`_build_opener()` creates a fresh HTTP opener per probe with Basic and Digest handlers. Credentials are registered for the origin, such as `http://192.168.50.200:80`, so both device information and NVR channel-list requests can authenticate. Device HTTP requests bypass ambient HTTP proxy settings using `ProxyHandler({})`.

With Digest authentication, a typical exchange is:

```text
1. Client GET /ISAPI/System/deviceInfo
2. Device replies 401 with an authentication challenge (realm, nonce, etc.)
3. Client computes the challenge response using username/password and request details
4. Client repeats GET with an Authorization header
5. Device replies with DeviceInfo XML if authentication succeeds
```

The first 401 can be the expected challenge. Repeated authentication failure is different. Do not write your own Digest implementation for this HTTP path; the standard-library handler already implements the exchange. Basic and Digest are authentication mechanisms, not a replacement for a protected network/TLS design.

The service probes `/ISAPI/System/deviceInfo`, strips XML namespaces, and requires a `DeviceInfo` root with a serial number. A generic router web page responding HTTP 200 is rejected.

Illustrative XML:

```xml
<DeviceInfo xmlns="http://www.hikvision.com/ver20/XMLSchema">
  <deviceName>Gate East</deviceName>
  <model>DS-2CD-EXAMPLE</model>
  <serialNumber>CAM-SERIAL-A</serialNumber>
  <macAddress>02:00:00:00:00:21</macAddress>
</DeviceInfo>
```

The resulting `HikDevice` contains `ip='192.168.50.21'`, `serial_number='CAM-SERIAL-A'`, `channel=1`, `rtsp_port=554`, and `is_nvr=False`.

Credentials are global to the discovery configuration: `HIK_*` for cameras and `NVR_*` for recorders. Missing NVR credentials fall back to `HIK_*`. There is no per-host credential map in `DiscoveryConfig`; heterogeneous sites with different camera accounts need a future extension or deliberately managed configurations.

For locally discovered devices, a 401 may trigger a retry with the distinct NVR account. The fallback is accepted only for a recorder. Explicitly configured NVR credentials are used as supplied, not silently replaced by camera credentials.

## NVR enumeration and stream addressing

A recorder is recognized from its device type containing NVR/DVR or common recorder model prefixes. `_probe_nvr_channels()` reads `/ISAPI/ContentMgmt/InputProxy/channels`, filters disabled inputs and configured channel IDs, and retains the recorder host as the source address.

```dotenv
DISCOVERY_NVRS=192.168.50.200:554:1-4:80
NVR_USERNAME=viewer
NVR_PASSWORD=REPLACE_WITH_SITE_SECRET
HIK_RTSP_STREAM=2
NVR_CHANNEL_OFFSET=0
```

The format is `host:rtsp_port[:channels[:http_port]]`. The two ports have different jobs. `192.168.50.200:8554:1,3,5-7:8081` means RTSP port 8554, ISAPI HTTP port 8081, and inputs 1, 3, 5, 6, and 7. Multiple recorder declarations are separated by semicolons. The parser accepts IPv4 or hostnames, not bracketed IPv6.

When ISAPI cannot be used, declare candidates explicitly:

```dotenv
DISCOVERY_LOCAL_ENABLED=false
STATIC_NVRS=192.168.50.200:554:1-4
```

Static declarations bypass ISAPI enumeration, **not frame verification**. A configured channel is not automatically online.

The code calculates:

```python
rtsp_channel = device.channel + (NVR_CHANNEL_OFFSET if device.is_nvr else 0)
stream_id = rtsp_channel * 100 + HIK_RTSP_STREAM
```

| Input channel | Stream | Offset | Suffix |
| --- | --- | --- | --- |
| 1 | Main, 1 | 0 | `101` |
| 1 | Sub, 2 | 0 | `102` |
| 2 | Sub, 2 | 0 | `202` |
| 10 | Sub, 2 | 0 | `1002` |
| NVR input 1 | Sub, 2 | 16 | `1702` |

The normal URL is `rtsp://viewer:PLACEHOLDER@192.168.50.200:554/Streaming/Channels/202`. Hikvision documents channel 1 main/substream as 101/102. See [Hikvision's RTSP reference](https://supportusa.hikvision.com/support/solutions/articles/17000129064-how-do-i-get-my-rtsp-stream-).

The code percent-encodes username/password using `quote(..., safe='')`. An illustrative password `p@ss/word` becomes `p%40ss%2Fword`; otherwise `@` could be mistaken for the boundary between credentials and hostname. HTTP credentials are supplied separately to the HTTP opener and do not need URL encoding there.

Use an offset only after checking recorder RTSP paths. Changing the offset changes the endpoint, while the discovery identity retains the original input channel. Main/substream choice also changes resolution, bitrate, small-object visibility, and possibly codec.

## Verification, identity, and adoption

For an already running, enabled camera with the exact same URL, `verify_source()` uses capture freshness. It accepts an age no greater than `max(15000, 3000 / sample_fps)` milliseconds. At 3 FPS this is 15 seconds. “Verified” therefore does not mean a frame was captured this millisecond.

For a new URL, verification waits for the global connection gate, opens a temporary `VideoChannel` capture, and waits for a decoded frame. It releases that temporary capture. Adoption then starts a separate persistent capture, requiring another connection. This is a correctness check with a real startup/session cost.

Identity examples:

```text
Direct camera with serial:       serial:CAM-SERIAL-A
Direct camera with MAC fallback: mac:02:00:00:00:00:21
Static recorder channel 2:       ip:192.168.50.200#2
NVR channel with serial:         serial:CAM-SERIAL-B#2
```

Serial/MAC identity can survive an IP change. An IP-only identity cannot automatically do that. Two NVR channels must never be matched only by host. Edge matching checks active roster linkage, saved discovery identity/serial with channel, credential-free stream endpoint, and only then direct-camera host fallback.

`mark_seen()` stores presence/history. `_adopt()` links an existing active camera when possible; otherwise it creates a UUID, calls `runtime.add_camera()`, and attaches that UUID to the roster. Despite the error text in `add_camera()` saying the backend must provide IDs, the edge discovery service itself also generates IDs.

The intended source-repointing behavior has a defect in discovery-managed selection: an old URL can be retired before its linked identity is inspected, leaving no active camera to patch. See **B1** in document 08; do not teach automatic DHCP recovery as guaranteed behavior.

## Timing and reports

| Setting | Current default | Effect |
| --- | --- | --- |
| `DISCOVERY_INTERVAL_S` | 150, minimum 60 | Scheduler delay after a sweep; also the `/sync` quiet period measured from completed sweep |
| `CAMERA_START_STAGGER_S` | 1.5 | Global spacing between top-level capture opens |
| `CAPTURE_PROBE_S` | 12 | First GStreamer candidate's frame-probe budget |
| `DISCOVERY_MISS_THRESHOLD` | 2 | Intended absent-sweep threshold |

Normal scheduler start-to-start time is approximately `sweep duration + max(interval, quiet period)`. A 40-second sweep with defaults means roughly 190 seconds between starts, not 60. Cloud `/sync` calls during the quiet period return current information without forcing another scan.

`POST /sync` schedules/coalesces work and returns cameras plus `current_report()`. During a sweep this can be a **partial** roster of selected, active linked rows; it must not be treated as an exhaustive absence report. `GET /discovery/report` returns the **last completed** report, or 404 before one finishes. A completed report contains selected rows, not every historical database row. `POST /discovery/scan` blocks for a sweep and bypasses the ordinary request-scan quiet-period check. Even `force=True` cannot make `scan_network()` scan when `cfg.enabled` is false.

Missing-camera aging currently has a reproduced defect: a dynamic camera absent from selection for a second sweep can be classified as historical/excluded and stop accumulating misses. The current tests catch missing/recovery transitions failing. Static candidates that remain in selection but fail verification follow a different path and can still accumulate misses. See **B2**.
