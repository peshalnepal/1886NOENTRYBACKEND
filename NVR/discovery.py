"""Hikvision discovery: WS-Discovery, ISAPI probes and a bounded subnet sweep.

Everything here is blocking; the adapter runs `scan_network()` on a worker
thread. Discovery only finds routes; the frame verifier proves playback.
"""

import ipaddress
import logging
import re
import socket
import time
import uuid
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.request import (HTTPBasicAuthHandler, HTTPDigestAuthHandler,
                            HTTPPasswordMgrWithDefaultRealm, ProxyHandler, build_opener)
from xml.etree import ElementTree as ET

from .config import Settings

logger = logging.getLogger(__name__)

ISAPI_PATH = "/ISAPI/System/deviceInfo"
CHANNELS_PATH = "/ISAPI/ContentMgmt/InputProxy/channels"
# Model of an NVR_CHANNELS placeholder: a channel ISAPI did not (or could not) describe.
STATIC_CHANNEL_MODEL = "NVR-Static-Channel"
WSD_TIMEOUT_S = 3.0
HTTP_TIMEOUT_S = 2.0
PROBE_WORKERS = 32

HikDevice = namedtuple("HikDevice", "ip serial_number model firmware device_name mac channel rtsp_port is_nvr")


def identity_of(device: HikDevice) -> str:
    if device.serial_number:
        key = "serial:" + device.serial_number
    elif device.mac:
        key = "mac:" + device.mac.lower()
    else:
        key = "ip:" + device.ip
    return "{}#{}".format(key, device.channel) if device.is_nvr else key


def rtsp_url(device: HikDevice, settings: Settings) -> str:
    """Credential-free Hikvision stream URL for one route."""
    channel = device.channel + (settings.nvr_channel_offset if device.is_nvr else 0)
    return "rtsp://{}:{}/Streaming/Channels/{}".format(
        device.ip, device.rtsp_port, channel * 100 + settings.hik_rtsp_stream)


def scan_network(settings: Settings) -> list:
    """Every route found this sweep: the configured NVR's channels plus LAN devices."""
    devices = {}

    def add(device, replace=False):
        key = (device.ip, device.rtsp_port, device.channel, device.is_nvr)
        if replace or key not in devices:
            devices[key] = device

    if settings.nvr_host:
        channels = settings.nvr_channel_list
        for channel in channels or []:
            add(HikDevice(settings.nvr_host, None, STATIC_CHANNEL_MODEL, None,
                          "{} ch{}".format(settings.nvr_host, channel), None,
                          channel, settings.nvr_rtsp_port, True))
        for device in probe_device(settings.nvr_host, settings.nvr_http_port, settings.nvr_rtsp_port,
                                   settings.nvr_account, channels):
            add(device, replace=True)

    if settings.discovery_local_enabled:
        own_ip = primary_ipv4()
        hosts = wsdiscover(WSD_TIMEOUT_S)
        for subnet in settings.subnet_list or (["{}/24".format(own_ip)] if own_ip else []):
            hosts.update(str(ip) for ip in ipaddress.IPv4Network(subnet, strict=False).hosts())
        hosts -= {settings.nvr_host, own_ip}
        if hosts:
            with ThreadPoolExecutor(max_workers=min(PROBE_WORKERS, len(hosts))) as pool:
                for found in pool.map(lambda host: probe_device(
                        host, settings.hik_http_port, settings.hik_rtsp_port, settings.camera_account),
                        sorted(hosts)):
                    for device in found:
                        add(device)
    return list(devices.values())


def probe_device(host, http_port, rtsp_port, account, channels=None) -> list:
    """ISAPI-probe one host: a camera yields one route, a recorder one per channel."""
    try:
        root = _xml(host, http_port, ISAPI_PATH, account)
        if _tag(root) != "DeviceInfo":
            return []
        info = _fields(root)
        model = info.get("model", "").upper()
        kind = info.get("deviceType", "").upper()
        if not ("NVR" in kind or "DVR" in kind or model.startswith(("DS-7", "DS-8", "DS-9"))):
            if not info.get("serialNumber"):
                return []
            return [HikDevice(host, info["serialNumber"], model, info.get("firmwareVersion"),
                              info.get("deviceName"), info.get("macAddress"), 1, rtsp_port, False)]

        found = []
        for node in _xml(host, http_port, CHANNELS_PATH, account):
            if _tag(node) != "InputProxyChannel":
                continue
            info = _fields(node)
            try:
                channel = int(info.get("id", ""))
            except ValueError:
                continue
            if channel < 1 or (channels is not None and channel not in channels):
                continue
            if info.get("enabled", "true").lower() not in {"true", "yes", "1"}:
                continue
            camera = next((_fields(c) for c in node if _tag(c) == "sourceInputPortDescriptor"), {})
            found.append(HikDevice(host, camera.get("serialNumber") or None,
                                   camera.get("model") or "NVR-Proxied-Camera", camera.get("firmwareVersion"),
                                   info.get("name") or "Camera {}".format(channel),
                                   camera.get("macAddress") or None, channel, rtsp_port, True))
        return found
    except HTTPError as exc:
        exc.close()
        logger.debug("ISAPI HTTP %s at %s", exc.code, host)
    except (URLError, OSError, ET.ParseError):
        logger.debug("ISAPI unavailable at %s", host)
    return []


def wsdiscover(timeout_s: float) -> set:
    """Hosts that answer an ONVIF WS-Discovery probe (plus addresses they advertise)."""
    message = """<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
      xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
      xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
      xmlns:n="http://www.onvif.org/ver10/network/wsdl"><s:Header>
      <a:MessageID>uuid:{}</a:MessageID>
      <a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>
      <a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</a:Action>
      </s:Header><s:Body><d:Probe><d:Types>n:NetworkVideoTransmitter</d:Types>
      </d:Probe></s:Body></s:Envelope>""".format(uuid.uuid4()).encode()
    hosts = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.bind(("", 0))
            for _ in range(3):
                sock.sendto(message, ("239.255.255.250", 3702))
            deadline = time.monotonic() + timeout_s
            while (remaining := deadline - time.monotonic()) > 0:
                sock.settimeout(max(0.001, remaining))
                payload, sender = sock.recvfrom(65535)
                hosts.add(sender[0])
                try:
                    for node in ET.fromstring(payload).iter():
                        if _tag(node) == "XAddrs":
                            for address in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", node.text or ""):
                                hosts.add(str(ipaddress.IPv4Address(address)))
                except (ET.ParseError, ValueError):
                    continue
    except OSError:
        logger.debug("WS-Discovery finished or unavailable")
    return hosts


def primary_ipv4():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 53))
            return sock.getsockname()[0]
        except OSError:
            return None


def _tag(node) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _fields(node) -> dict:
    return {_tag(child): (child.text or "").strip() for child in node}


def _build_opener(origin, username, password):
    passwords = HTTPPasswordMgrWithDefaultRealm()
    passwords.add_password(None, origin, username, password)
    return build_opener(ProxyHandler({}), HTTPDigestAuthHandler(passwords), HTTPBasicAuthHandler(passwords))


def _xml(host, port, path, account):
    origin = "http://{}:{}".format(host, port)
    with _build_opener(origin, *account).open(origin + path, timeout=HTTP_TIMEOUT_S) as response:
        return ET.fromstring(response.read())
