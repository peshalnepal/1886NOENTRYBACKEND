# discovery.py  (Python 3.6, stdlib only)
"""
Hikvision camera discovery for the Jetson edge.

Answers exactly one question: which IPs on this network are Hikvision cameras
we can authenticate to? Returns a list of HikDevice. All pipeline logic lives
in service.py.

Constraints that shape this module:
  - Python 3.6, stdlib only. requirements.txt ships Flask + SQLAlchemy +
    aiosqlite and nothing else; pip-installing extras on a Jetson risks
    shadowing the JetPack cv2/numpy.
  - Nothing here may block the pipeline loop. Every function is synchronous and
    is called from the scheduler's own worker thread.

A sweep is four stages:

  1. WS-Discovery (UDP multicast) builds a candidate list without credentials.
     Managed switches block multicast routinely, so an empty result is normal.
  2. If it found nothing, expand the configured CIDRs into every host address.
  3. Probe each candidate over Hikvision ISAPI. This is the authoritative
     check: only a Hikvision box answers `/ISAPI/System/deviceInfo` with a
     <DeviceInfo> document carrying a <serialNumber>, and a successful answer
     also proves the credentials work. That last part matters — an RTSP URL
     built with a wrong password connects and never decodes a frame, which is
     worse than not discovering the camera at all.
  4. Dedupe by identity, since a multi-homed camera answers on two addresses.
"""

import logging
import re
import socket
import struct
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
    build_opener,
)
from xml.etree import ElementTree

try:
    # Script mode (python main.py from Backend/tensort)
    from env_utils import env_bool, env_float, env_int, env_str
except ModuleNotFoundError:
    # Package mode (python -m Backend.tensort.main)
    from .env_utils import env_bool, env_float, env_int, env_str

logger = logging.getLogger("jetson-discovery")

ISAPI_PATH = "/ISAPI/System/deviceInfo"

# WS-Discovery multicast group (ONVIF Core spec).
WSD_ADDR = ("239.255.255.250", 3702)

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

# IPs are harvested only from <XAddrs> elements, never the whole document, so
# scope strings carrying unrelated addresses do not add phantom hosts.
_XADDRS_RE = re.compile(r"<[^>]*XAddrs[^>]*>(.*?)</[^>]*XAddrs>", re.S | re.I)
_IPV4_RE = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class DiscoveryConfig(object):
    """Everything discovery reads from the environment, resolved once."""

    def __init__(self):
        self.enabled = env_bool("DISCOVERY_ENABLED", True)

        # Credentials applied to every discovered camera.
        self.username = env_str("HIK_USERNAME", "admin")
        self.password = env_str("HIK_PASSWORD", "")

        # Explicit subnets to sweep, e.g. "192.168.1.0/24,10.0.0.0/24".
        # Empty means "derive from this host's own primary IPv4 address".
        self.subnets = [s for s in env_str("DISCOVERY_SUBNETS", "").split(",") if s.strip()]

        self.wsd_timeout_s = env_float("DISCOVERY_WSD_TIMEOUT_S", 3.0, minimum=0.5)
        self.http_timeout_s = env_float("DISCOVERY_HTTP_TIMEOUT_S", 2.0, minimum=0.2)
        self.probe_workers = env_int("DISCOVERY_PROBE_WORKERS", 32, minimum=1, maximum=128)

        # Skip the subnet sweep entirely when WS-Discovery already answered.
        # Turn this off on networks where some cameras have ONVIF disabled.
        self.sweep_only_if_wsd_empty = env_bool("DISCOVERY_SWEEP_ONLY_IF_WSD_EMPTY", True)

        # RTSP URL construction. Hikvision's canonical path is
        # /Streaming/Channels/<channel><stream>, e.g. 101 = channel 1 main
        # stream, 102 = channel 1 sub stream. Sub stream is the sane default
        # for detection: lower resolution, far less decode cost, and the Jetson
        # resizes to 640x480 anyway.
        self.rtsp_port = env_int("HIK_RTSP_PORT", 554, minimum=1, maximum=65535)
        self.rtsp_channel = env_int("HIK_RTSP_CHANNEL", 1, minimum=1, maximum=64)
        self.rtsp_stream = env_int("HIK_RTSP_STREAM", 2, minimum=1, maximum=3)
        self.http_port = env_int("HIK_HTTP_PORT", 80, minimum=1, maximum=65535)

    def rtsp_url_for(self, ip):
        """Build the RTSP URL for a discovered camera.

        Credentials are embedded because the GStreamer/OpenCV decode path in
        channels/channel.py takes a single URL string and has nowhere else to
        put them.
        """
        cred = ""
        if self.username:
            cred = "{u}:{p}@".format(
                u=quote(self.username, safe=""), p=quote(self.password, safe="")
            )
        # Hikvision encodes the path segment as channel*100 + stream, so
        # channel 1 main = 101 and channel 1 sub = 102. Concatenating the two
        # numbers as strings only coincidentally matches for channels 1-9.
        return "rtsp://{cred}{ip}:{port}/Streaming/Channels/{ch}".format(
            cred=cred,
            ip=ip,
            port=self.rtsp_port,
            ch=self.rtsp_channel * 100 + self.rtsp_stream,
        )


# ---------------------------------------------------------------------------
# The device record
# ---------------------------------------------------------------------------

HikDevice = namedtuple(
    "HikDevice", "ip serial_number model firmware device_name mac"
)


def identity_of(dev):
    """Stable identity for a physical camera.

    The serial number is what makes "the camera that was here yesterday"
    recognisable after a DHCP lease change. MAC is the fallback; IP is the last
    resort and is explicitly weak — a camera that moves IP with no serial is
    reported as one gone plus one new.
    """
    if dev.serial_number:
        return "serial:{}".format(dev.serial_number)
    if dev.mac:
        return "mac:{}".format(str(dev.mac).lower())
    return "ip:{}".format(dev.ip)


# `identity()` and `to_dict()` stay on the instance so service.py and the
# roster keep calling devices the same way they always have.
HikDevice.identity = identity_of


def _to_dict(dev):
    out = dev._asdict()
    out["identity"] = identity_of(dev)
    return dict(out)


HikDevice.to_dict = _to_dict


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def primary_ipv4():
    """This host's outward-facing IPv4, without needing a route to the internet.

    Connecting a UDP socket sends no packets; it just makes the kernel pick a
    source address from the routing table, which is the interface the cameras
    are on.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 53))
        return sock.getsockname()[0]
    except Exception:
        return None
    finally:
        sock.close()


def _hosts_in_cidr(cidr):
    """Expand a CIDR into its usable host addresses.

    Only /16 .. /32 are accepted; anything wider would mean tens of thousands
    of HTTP probes per minute, which is not a sweep, it is a port scan.
    """
    cidr = str(cidr or "").strip()
    if not cidr:
        return []
    if "/" not in cidr:
        cidr += "/32"

    net_str, _, bits_str = cidr.partition("/")
    try:
        bits = int(bits_str)
        base = struct.unpack("!I", socket.inet_aton(net_str))[0]
    except Exception:
        logger.warning("Ignoring malformed DISCOVERY_SUBNETS entry: %s", cidr)
        return []

    if not 16 <= bits <= 32:
        logger.warning("Ignoring subnet %s: only /16 through /32 are supported", cidr)
        return []

    size = 1 << (32 - bits)
    network = base & ((0xFFFFFFFF << (32 - bits)) & 0xFFFFFFFF)

    # /31 and /32 have no network/broadcast pair to skip.
    span = range(size) if size <= 2 else range(1, size - 1)
    return [socket.inet_ntoa(struct.pack("!I", network + i)) for i in span]


def _default_subnet_from_host(own_ip):
    if not own_ip or own_ip.count(".") != 3:
        return []
    return ["{}.0/24".format(own_ip.rsplit(".", 1)[0])]


# ---------------------------------------------------------------------------
# Stage 1 — ONVIF WS-Discovery
# ---------------------------------------------------------------------------

def wsdiscover(timeout_s=3.0):
    """Multicast an ONVIF Probe and collect the responding IPv4 addresses.

    Returns a set of IP strings. Never raises — a blocked multicast group is a
    normal condition on managed switches, not an error worth failing a sweep
    over.
    """
    found = set()
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.bind(("", 0))

        payload = _WSD_PROBE.format(msg_id=uuid_mod.uuid4()).encode("utf-8")
        # Sent three times: WS-Discovery is UDP, and a single lost datagram
        # would silently hide every camera on the segment.
        for _ in range(3):
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
            # A multi-homed camera can answer from one address and advertise a
            # different one in <XAddrs>; harvest both.
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
# Stage 3 — Hikvision ISAPI identification
# ---------------------------------------------------------------------------

def _opener_for(url, cfg):
    """An opener that answers both Basic and Digest challenges for `url`.

    Hikvision firmware defaults to digest; urllib handles the 401-then-retry
    dance and the RFC 2617 hashing internally.

    The password manager matches on the URI, so credentials must be registered
    against this camera's own URL — a hostless prefix like "http://" silently
    matches nothing and every probe would fail auth. That is also why the
    opener is built per probe rather than shared across the pool: it is cheap
    (two handler objects, no I/O) and keeps threads off shared mutable state.
    """
    mgr = HTTPPasswordMgrWithDefaultRealm()
    # A realm of None means "any realm this host challenges with", which is
    # what we want — realms vary across firmware versions.
    mgr.add_password(None, url, cfg.username, cfg.password)
    return build_opener(HTTPBasicAuthHandler(mgr), HTTPDigestAuthHandler(mgr))


def _strip_ns(tag):
    return tag.split("}", 1)[1] if "}" in tag else tag


def probe_hikvision(ip, cfg):
    """Identify `ip` as a Hikvision camera via ISAPI, or return None.

    Only a <DeviceInfo> document carrying a serial number counts. A generic web
    server answering 200 on port 80 does not.
    """
    url = "http://{ip}:{port}{path}".format(ip=ip, port=cfg.http_port, path=ISAPI_PATH)
    #this will user url and create a opener 
    # (opener is urlopen but will call the url twice so that first time you get 401 error from camera and send you code you comebine that code with user and password to auth)
    opener = _opener_for(url, cfg)

    try:
        body = opener.open(url, timeout=cfg.http_timeout_s).read()
        #this body is xml which looks something like this 
        # <DeviceInfo xmlns="http://www.hikvision.com/ver20/XMLSchema">
            # <deviceName>Front Gate</deviceName>
            # <serialNumber>DS-2CD2043G0...</serialNumber>
            # ...
        # </DeviceInfo>
        # here this is name space so every element below will look like 
        # {namespace}deviceName (to distinguish from other camera send their response)
        
    except (HTTPError, URLError, socket.timeout):
        return None
    except Exception:
        logger.debug("ISAPI probe failed for %s", ip, exc_info=True)
        return None

    try:
        root = ElementTree.fromstring(body)
    except Exception:
        return None
    if _strip_ns(root.tag) != "DeviceInfo":
        return None

    fields = {_strip_ns(c.tag): (c.text or "").strip() for c in root}
    #cleaning the namespace and just keeping name and value in them like
    # this is converted by strp_ns {http://www.hikvision.com/ver20/XMLSchema}DeviceInfo to just DeviceInfo
    # so the dictonary will look like fields = {"deviceName": "Front Gate", "serialNumber": "DS-2CD...", "model": "", ...}
    serial = fields.get("serialNumber")
    if not serial:
        # Without a serial we cannot track this camera across IP changes, and a
        # device that omits it is very unlikely to be a Hikvision camera.
        return None

    return HikDevice(
        ip=ip,
        serial_number=serial,
        model=fields.get("model") or None,
        firmware=fields.get("firmwareVersion") or None,
        device_name=fields.get("deviceName") or None,
        mac=fields.get("macAddress") or None,
    )


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def scan_network(cfg=None):
    """Run one full discovery sweep and return the confirmed Hikvision cameras.

    Returns a list of HikDevice, deduplicated by identity (a camera answering
    on two addresses is still one camera).
    """
    cfg = cfg or DiscoveryConfig()
    own_ip = primary_ipv4()

    candidates = wsdiscover(timeout_s=cfg.wsd_timeout_s)
    if candidates:
        logger.info("WS-Discovery found %d ONVIF responder(s)", len(candidates))

    if not candidates or not cfg.sweep_only_if_wsd_empty:
        subnets = cfg.subnets or _default_subnet_from_host(own_ip)
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

    candidates.discard(own_ip)  # never probe ourselves
    if not candidates:
        return []

    targets = sorted(candidates)
    workers = min(cfg.probe_workers, len(targets))

    def probe(ip):
        try:
            return probe_hikvision(ip, cfg)
        except Exception:
            logger.debug("Hikvision probe raised for %s", ip, exc_info=True)
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        devices = [d for d in pool.map(probe, targets) if d is not None]

    by_identity = {}
    for dev in devices:
        by_identity.setdefault(identity_of(dev), dev)

    found = list(by_identity.values())
    logger.info(
        "Discovery sweep complete: %d candidate host(s) probed, %d Hikvision camera(s) confirmed",
        len(targets), len(found),
    )
    return found
