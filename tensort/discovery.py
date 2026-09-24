"""Discover Hikvision LAN cameras and NVR inputs using Python 3.6 stdlib.

Run from a worker thread: network calls are synchronous. ISAPI checks device
access, not video playback. See DISCOVERY.md for setup examples."""

import ipaddress
import logging
import re
import socket
import time
import uuid as uuid_mod
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPDigestAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    ProxyHandler,
    build_opener,
)
from xml.etree import ElementTree

if __package__:
    from .env_utils import env_bool, env_float, env_int, env_str
else:
    from env_utils import env_bool, env_float, env_int, env_str

logger = logging.getLogger("jetson-discovery")

ISAPI_PATH = "/ISAPI/System/deviceInfo"
CHANNELS_PATH = "/ISAPI/ContentMgmt/InputProxy/channels"

# WS-Discovery multicast group (ONVIF Core spec).
WSD_ADDR = ("239.255.255.250", 3702)

# Repeat the UDP probe to tolerate lost packets.
_WSD_SENDS = 3

_TRUTHY = ("true", "yes", "1")

_WSD_PROBE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"'
    ' xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"'
    ' xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"'
    ' xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
    '<e:Header>'
    '<w:MessageID>uuid:{msg_id}</w:MessageID>'
    '<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
    '<w:Action e:mustUnderstand="true">'
    'http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
    '</e:Header>'
    '<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>'
    '</e:Envelope>'
)

# Only XAddrs contains advertised device addresses.
_XADDRS_RE = re.compile(r"<[^>]*XAddrs[^>]*>(.*?)</[^>]*XAddrs>", re.S | re.I)
_IPV4_RE = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class DiscoveryConfig(object):
    """Everything discovery reads from the environment, resolved once."""

    def __init__(self):
        self.enabled = env_bool("DISCOVERY_ENABLED", True)
        self.local_enabled = env_bool("DISCOVERY_LOCAL_ENABLED", True)

        self.username = env_str("HIK_USERNAME", "admin")
        self.password = env_str("HIK_PASSWORD", "")

        # NVR streams use the recorder's account.
        self.nvr_username = env_str("NVR_USERNAME", self.username)
        self.nvr_password = env_str("NVR_PASSWORD", self.password)

        self.discovery_nvrs = env_str("DISCOVERY_NVRS", "")
        self.static_nvrs = env_str("STATIC_NVRS", "")

        subnet_entries = env_str("DISCOVERY_SUBNETS", "").split(",")
        self.subnets = [subnet.strip() for subnet in subnet_entries if subnet.strip()]

        self.wsd_timeout_s = env_float("DISCOVERY_WSD_TIMEOUT_S", 3.0, minimum=0.5)
        self.http_timeout_s = env_float("DISCOVERY_HTTP_TIMEOUT_S", 2.0, minimum=0.2)
        self.probe_workers = env_int("DISCOVERY_PROBE_WORKERS", 32, minimum=1, maximum=128)

        # A partial ONVIF response must not hide other cameras by default.
        self.sweep_only_if_wsd_empty = env_bool("DISCOVERY_SWEEP_ONLY_IF_WSD_EMPTY", False)

        self.rtsp_port = env_int("HIK_RTSP_PORT", 554, minimum=1, maximum=65535)
        # NVR inputs use their ISAPI channel ID instead.
        self.rtsp_channel = env_int("HIK_RTSP_CHANNEL", 1, minimum=1, maximum=64)
        self.rtsp_stream = env_int("HIK_RTSP_STREAM", 2, minimum=1, maximum=3)
        self.http_port = env_int("HIK_HTTP_PORT", 80, minimum=1, maximum=65535)

        # Some recorders map ISAPI inputs to a different RTSP channel range.
        self.nvr_channel_offset = env_int("NVR_CHANNEL_OFFSET", 0, minimum=0, maximum=64)

    def rtsp_url_for(self, dev, force_credentials=None):
        """Build the RTSP URL for a device, optionally overriding credentials."""
        if force_credentials is not None:
            username, password = force_credentials
        elif dev.is_nvr:
            username, password = self.nvr_username, self.nvr_password
        else:
            username, password = self.username, self.password

        auth = ""
        if username:
            auth = "{}:{}@".format(quote(username, safe=""), quote(password, safe=""))

        # Keep the original channel ID unchanged for identity tracking.
        channel = dev.channel
        if getattr(dev, "is_nvr", False):
            channel += self.nvr_channel_offset

        # Example: channel 1, substream 2 -> 102.
        stream_id = channel * 100 + self.rtsp_stream
        return "rtsp://{auth}{ip}:{port}/Streaming/Channels/{stream_id}".format(
            auth=auth,
            ip=dev.ip,
            port=dev.rtsp_port,
            stream_id=stream_id,
        )


# ---------------------------------------------------------------------------
# The device record
# ---------------------------------------------------------------------------

HikDevice = namedtuple(
    "HikDevice",
    "ip serial_number model firmware device_name mac channel rtsp_port is_nvr",
)

HikDevice.__new__.__defaults__ = (1, 554, False)


def identity_of(dev):
    """Return a camera key using serial number, MAC address, or host.

    NVR keys include the channel because inputs may share the recorder serial."""
    channel = getattr(dev, "channel", 1) or 1
    suffix = "#{}".format(channel) if getattr(dev, "is_nvr", False) else ""

    if dev.serial_number:
        return "serial:{}{}".format(dev.serial_number, suffix)
    if dev.mac:
        return "mac:{}{}".format(str(dev.mac).lower(), suffix)
    return "ip:{}{}".format(dev.ip, suffix)


def _to_dict(dev):
    out = dict(dev._asdict())
    out["identity"] = identity_of(dev)
    return out


# Preserve the device methods used by the service and repository.
HikDevice.identity = identity_of
HikDevice.to_dict = _to_dict


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def primary_ipv4():
    """Return the default-route IPv4 address without sending any packets."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 53))
        return sock.getsockname()[0]
    except Exception:
        return None
    finally:
        sock.close()


def _hosts_in_cidr(cidr):
    """Return usable IPv4 hosts in a /16 through /32 subnet."""
    cidr = str(cidr or "").strip()
    if not cidr:
        return []
    if "/" not in cidr:
        cidr += "/32"

    try:
        network = ipaddress.IPv4Network(cidr, strict=False)
    except ValueError:
        logger.warning("Ignoring malformed DISCOVERY_SUBNETS entry: %s", cidr)
        return []

    if not 16 <= network.prefixlen <= 32:
        logger.warning("Ignoring subnet %s: only /16 through /32 are supported", cidr)
        return []

    return [str(host) for host in network.hosts()]


def _default_subnet_from_host(own_ip):
    if not own_ip or own_ip.count(".") != 3:
        return []
    return ["{}.0/24".format(own_ip.rsplit(".", 1)[0])]


# ---------------------------------------------------------------------------
# ONVIF WS-Discovery
# ---------------------------------------------------------------------------

def wsdiscover(timeout_s=3.0):
    """Return ONVIF responder IPv4 addresses; tolerate unavailable multicast."""
    found = set()
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.bind(("", 0))

        payload = _WSD_PROBE.format(msg_id=uuid_mod.uuid4()).encode("utf-8")
        for _ in range(_WSD_SENDS):
            sock.sendto(payload, WSD_ADDR)

        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, addr = sock.recvfrom(65535)
            except Exception:
                break

            if addr and addr[0]:
                found.add(addr[0])
            # A device with multiple interfaces may advertise another address.
            text = data.decode("utf-8", "ignore")
            for chunk in _XADDRS_RE.findall(text):
                found.update(_IPV4_RE.findall(chunk))
    except Exception:
        logger.debug("WS-Discovery unavailable on this host", exc_info=True)
    finally:
        if sock is not None:
            sock.close()

    return found


# ---------------------------------------------------------------------------
# Hikvision ISAPI identification
# ---------------------------------------------------------------------------

def _origin(host, http_port):
    """Build the HTTP base URL shared by a device's ISAPI endpoints."""
    return "http://{host}:{port}".format(host=host, port=http_port)


def _build_opener(base_url, username, password):
    """Create a Basic/Digest HTTP client for one device.

    Register credentials at the origin so both deviceInfo and channel-list
    requests can authenticate. Do not share this client between threads."""
    password_manager = HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, base_url, username, password)
    # Device HTTP traffic must not be routed through an ambient HTTP_PROXY.
    return build_opener(
        ProxyHandler({}),
        HTTPBasicAuthHandler(password_manager),
        HTTPDigestAuthHandler(password_manager),
    )


def _strip_ns(tag):
    return tag.split("}", 1)[1] if "}" in tag else tag


def _child_text(node):
    """Return direct XML child values, ignoring firmware-specific namespaces."""
    return {_strip_ns(child.tag): (child.text or "").strip() for child in node}


def _find_child(node, tag):
    for child in node:
        if _strip_ns(child.tag) == tag:
            return child
    return None


def _get_xml(opener, url, timeout_s, what, raise_auth_error=False):
    """Read an XML response, or return None on failure.

    Optionally raise HTTP 401 so the caller can retry with NVR credentials."""
    try:
        with opener.open(url, timeout=timeout_s) as response:
            return ElementTree.fromstring(response.read())
    except HTTPError as e:
        e.close()
        if e.code == 401 and raise_auth_error:
            raise
        if e.code == 401:
            logger.warning(
                "%s: authentication failed (HTTP 401) at %s — check the "
                "configured username/password", what, url,
            )
        else:
            logger.debug("%s: HTTP %s from %s", what, e.code, url)
    except (URLError, socket.timeout):
        logger.debug("%s: no response from %s", what, url)
    except Exception:
        logger.debug("%s: unreadable response from %s", what, url, exc_info=True)
    return None


def _looks_like_a_recorder(device_type, model):
    """Recognize NVR/DVR device types and common Hikvision recorder models."""
    return (
        "NVR" in device_type
        or "DVR" in device_type
        or model.startswith(("DS-7", "DS-8", "DS-9"))
    )


def _probe_nvr_channels(host, cfg, opener, http_port, rtsp_port, channels_override=None):
    """Read NVR inputs and address each stream through the recorder."""
    root = _get_xml(opener, _origin(host, http_port) + CHANNELS_PATH,
                    cfg.http_timeout_s, "NVR channel list")
    if root is None:
        return []

    cameras = []
    skipped = 0
    for channel_node in root:
        if _strip_ns(channel_node.tag) != "InputProxyChannel":
            continue

        channel_fields = _child_text(channel_node)
        channel_id = channel_fields.get("id", "")
        try:
            channel = int(channel_id)
            if not 1 <= channel <= 65535:
                raise ValueError
        except ValueError:
            logger.warning("NVR %s: skipping invalid channel id %r", host, channel_id)
            continue

        if channels_override is not None and channel not in channels_override:
            continue

        if channel_fields.get("enabled", "true").lower() not in _TRUTHY:
            skipped += 1
            continue

        # Camera details are nested inside the NVR input entry.
        descriptor = _find_child(channel_node, "sourceInputPortDescriptor")
        camera_fields = _child_text(descriptor) if descriptor is not None else {}
        serial = camera_fields.get("serialNumber")
        mac = camera_fields.get("macAddress")

        if not serial and not mac:
            logger.debug(
                "NVR %s channel %s reports neither serial nor MAC; "
                "identifying it by host+channel", host, channel,
            )

        cameras.append(HikDevice(
            # Stream through the NVR, not the camera's private PoE address.
            ip=host,
            serial_number=serial or None,
            model=camera_fields.get("model") or "NVR-Proxied-Camera",
            firmware=camera_fields.get("firmwareVersion") or None,
            device_name=channel_fields.get("name") or "Camera {}".format(channel_id),
            mac=mac or None,
            channel=channel,
            rtsp_port=rtsp_port,
            is_nvr=True,
        ))

    if skipped:
        logger.info("NVR %s: skipped %d disabled channel(s)", host, skipped)
    logger.info("NVR %s: %d proxied camera(s) enumerated", host, len(cameras))
    return cameras


def probe_device(host, cfg, http_port=None, rtsp_port=None,
                 credentials=None, channels=None):
    """Return one direct camera, multiple NVR inputs, or an empty list.

    Explicit credentials are used unchanged. Otherwise, an HTTP 401 can retry
    with the configured NVR account; only a recorder is accepted on that retry."""
    http_port = cfg.http_port if http_port is None else http_port
    rtsp_port = cfg.rtsp_port if rtsp_port is None else rtsp_port
    camera_credentials = (cfg.username, cfg.password)
    nvr_credentials = (cfg.nvr_username, cfg.nvr_password)
    selected_credentials = camera_credentials if credentials is None else credentials
    retry_nvr = credentials is None and nvr_credentials != camera_credentials
    used_nvr_fallback = False

    base_url = _origin(host, http_port)
    opener = _build_opener(base_url, *selected_credentials)
    try:
        root = _get_xml(opener, base_url + ISAPI_PATH, cfg.http_timeout_s,
                        "ISAPI probe of {}".format(host), raise_auth_error=retry_nvr)
    except HTTPError:
        # _get_xml raises only HTTP 401 when retry_nvr is enabled.
        selected_credentials = nvr_credentials
        used_nvr_fallback = True
        opener = _build_opener(base_url, *selected_credentials)
        root = _get_xml(opener, base_url + ISAPI_PATH, cfg.http_timeout_s,
                        "NVR ISAPI probe of {}".format(host))
    if root is None or _strip_ns(root.tag) != "DeviceInfo":
        return []

    fields = _child_text(root)
    serial = fields.get("serialNumber")
    if not serial:
        return []

    device_type = fields.get("deviceType", "").upper()
    model = fields.get("model", "").upper()

    if _looks_like_a_recorder(device_type, model):
        logger.info("Found Hikvision recorder at %s (%s)", host, model or device_type)
        if credentials is None and selected_credentials != nvr_credentials:
            # Verify channel access with the same account used in the RTSP URL.
            opener = _build_opener(base_url, *nvr_credentials)
        return _probe_nvr_channels(
            host, cfg, opener,
            http_port=http_port, rtsp_port=rtsp_port, channels_override=channels,
        )

    if used_nvr_fallback:
        logger.warning("Camera at %s requires working HIK_USERNAME/HIK_PASSWORD", host)
        return []

    logger.info("Found Hikvision camera at %s (%s)", host, model or "unknown model")
    return [HikDevice(
        ip=host,
        serial_number=serial,
        model=model or None,
        firmware=fields.get("firmwareVersion") or None,
        device_name=fields.get("deviceName") or None,
        mac=fields.get("macAddress") or None,
        channel=cfg.rtsp_channel,
        rtsp_port=rtsp_port,
        is_nvr=False,
    )]


# ---------------------------------------------------------------------------
# NVR specs (DISCOVERY_NVRS / STATIC_NVRS)
# ---------------------------------------------------------------------------

RemoteNvr = namedtuple("RemoteNvr", "host rtsp_port channels http_port")


def _parse_channel_spec(spec):
    """Expand "1-15" or "1,3,5-7" into [1, 3, 5, 6, 7]. Empty means "all"."""
    spec = str(spec or "").strip()
    if not spec:
        return None
    channels = set()
    for part in spec.split(","):
        part = part.strip()
        try:
            low, _, high = part.partition("-")
            first = int(low)
            last = int(high) if "-" in part else first
            if not 1 <= first <= last <= 65535:
                raise ValueError
        except ValueError:
            # An invalid explicit filter must never turn into "all channels".
            logger.warning("Ignoring malformed channel specification: %r", spec)
            return []
        channels.update(range(first, last + 1))
    return sorted(channels)


def _parse_port(raw, default, host, what):
    try:
        port = int(raw)
        if not 1 <= port <= 65535:
            raise ValueError
        return port
    except ValueError:
        logger.warning(
            "NVR %s: %r is not a valid %s port, using %d", host, raw, what, default,
        )
        return default


def parse_remote_nvrs(spec, default_rtsp_port=554, default_http_port=80):
    """Parse semicolon-separated host:rtsp_port[:channels[:http_port]] entries.

    Channels accept ranges (1-8) or lists (1,3,5). Omitted channels mean all.
    HTTP defaults to default_http_port; hosts accept IPv4 addresses or names."""
    entries = []
    for raw in str(spec or "").split(";"):
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split(":")
        if (not 2 <= len(parts) <= 4 or "/" in raw
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[0].strip())):
            logger.warning(
                "Ignoring malformed NVR entry %r "
                "(expected host:rtsp_port[:channels[:http_port]])", raw,
            )
            continue

        host = parts[0].strip()
        rtsp_port = _parse_port(parts[1], default_rtsp_port, host, "RTSP")
        channels = _parse_channel_spec(parts[2]) if len(parts) >= 3 else None
        if channels == []:
            continue

        http_port = default_http_port
        if len(parts) >= 4 and parts[3].strip():
            http_port = _parse_port(parts[3], default_http_port, host, "HTTP")

        entries.append(RemoteNvr(host, rtsp_port, channels, http_port))
    return entries


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def _candidate_hosts(cfg):
    """Merge ONVIF responders and subnet hosts according to configuration."""
    candidates = wsdiscover(timeout_s=cfg.wsd_timeout_s)
    if candidates:
        logger.info(
            "WS-Discovery found %d ONVIF responder(s): %s",
            len(candidates), sorted(candidates),
        )

    if not candidates or not cfg.sweep_only_if_wsd_empty:
        subnets = cfg.subnets or _default_subnet_from_host(primary_ipv4())
        if not subnets:
            logger.warning(
                "No DISCOVERY_SUBNETS set and this host's subnet could not be "
                "determined; discovery is limited to WS-Discovery responders."
            )
        for cidr in subnets:
            hosts = _hosts_in_cidr(cidr)
            if hosts:
                logger.info("Sweeping %s (%d hosts) for Hikvision cameras", cidr, len(hosts))
                candidates.update(hosts)

    candidates.discard(primary_ipv4())  # never probe ourselves
    return sorted(candidates)


def _scan_local_network(cfg):
    """Probe every candidate address on this LAN, in parallel."""
    # Avoid overriding explicit NVR ports, accounts and channel filters.
    configured_nvrs = parse_remote_nvrs(cfg.discovery_nvrs, cfg.rtsp_port, cfg.http_port)
    configured_nvrs.extend(parse_remote_nvrs(cfg.static_nvrs, cfg.rtsp_port, cfg.http_port))
    configured_hosts = {nvr.host for nvr in configured_nvrs}
    targets = [host for host in _candidate_hosts(cfg) if host not in configured_hosts]
    if not targets:
        logger.info("No local candidates to probe")
        return []

    logger.info("Probing %d candidate host(s) over ISAPI", len(targets))

    def probe(host):
        try:
            return probe_device(host, cfg)
        except Exception:
            logger.debug("ISAPI probe raised for %s", host, exc_info=True)
            return []

    devices = []
    workers = min(cfg.probe_workers, len(targets))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for found in pool.map(probe, targets):
            devices.extend(found or [])
    return devices


def _scan_remote_nvrs(cfg):
    """Probe configured LAN or remote NVRs using their own ports and account."""
    devices = []
    for nvr in parse_remote_nvrs(cfg.discovery_nvrs, cfg.rtsp_port, cfg.http_port):
        logger.info(
            "Probing configured NVR %s (ISAPI port %d, RTSP port %d, channels %s)",
            nvr.host, nvr.http_port, nvr.rtsp_port, nvr.channels or "all",
        )
        try:
            found = probe_device(
                nvr.host, cfg,
                http_port=nvr.http_port,
                rtsp_port=nvr.rtsp_port,
                credentials=(cfg.nvr_username, cfg.nvr_password),
                channels=nvr.channels,
            )
        except Exception:
            logger.exception("Probe of configured NVR %s failed", nvr.host)
            continue

        if not found:
            logger.warning(
                "Configured NVR %s returned no cameras. Check NVR_USERNAME/"
                "NVR_PASSWORD and that %d is its ISAPI (HTTP) port, not its "
                "RTSP port.", nvr.host, nvr.http_port,
            )
        devices.extend(found)
    return devices


def _static_nvr_cameras(cfg):
    """Build declared NVR inputs without checking availability or credentials."""
    devices = []
    for nvr in parse_remote_nvrs(cfg.static_nvrs, cfg.rtsp_port, cfg.http_port):
        if not nvr.channels:
            logger.warning(
                "STATIC_NVRS entry for %s lists no channels, so nothing can be "
                "added. Give it an explicit range, e.g. %s:%d:1-16",
                nvr.host, nvr.host, nvr.rtsp_port,
            )
            continue

        for channel in nvr.channels:
            devices.append(HikDevice(
                ip=nvr.host,
                serial_number=None,
                model="NVR-Static-Channel",
                firmware=None,
                device_name="{} ch{}".format(nvr.host.split(".")[0], channel),
                mac=None,
                channel=channel,
                rtsp_port=nvr.rtsp_port,
                is_nvr=True,
            ))

        logger.info(
            "Static NVR %s: %d declared channel(s) on RTSP port %d (not probed)",
            nvr.host, len(nvr.channels), nvr.rtsp_port,
        )
    return devices


def scan_network(cfg=None):
    """Merge discovered and declared cameras, removing duplicate identities and streams."""
    cfg = cfg or DiscoveryConfig()
    if not cfg.enabled:
        return []

    if not cfg.password and not cfg.discovery_nvrs and not cfg.static_nvrs:
        logger.warning(
            "HIK_PASSWORD is empty. Check the camera credentials and .env loading."
        )

    # Configuration order breaks ties within each priority group.
    static = _static_nvr_cameras(cfg)
    dynamic = _scan_remote_nvrs(cfg)
    # Keep static priority while retaining discovered serial/model information.
    details = {(dev.ip, dev.rtsp_port, dev.channel, dev.is_nvr): dev for dev in dynamic}
    devices = [details.get((dev.ip, dev.rtsp_port, dev.channel, dev.is_nvr), dev) for dev in static]
    devices.extend(dynamic)
    if cfg.local_enabled:
        local = _scan_local_network(cfg)
        devices.extend(dev for dev in local if dev.is_nvr)
        devices.extend(dev for dev in local if not dev.is_nvr)

    by_identity = {}
    endpoints = set()
    for dev in devices:
        endpoint = (dev.ip, dev.rtsp_port, dev.channel, dev.is_nvr)
        if endpoint in endpoints:
            continue
        by_identity.setdefault(identity_of(dev), dev)
        endpoints.add(endpoint)

    found = list(by_identity.values())
    logger.info("Discovery scan complete: %d candidate(s); frame verification follows", len(found))
    return found
