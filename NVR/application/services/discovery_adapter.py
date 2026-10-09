"""Turn raw `scan_network()` devices into per-camera candidates.

* runs the blocking scan on a dedicated worker thread, never the event loop;
* keeps only the configured NVR; any other recorder is reported and ignored;
* merges routes that reach the same physical camera. `identity_of()` suffixes
  NVR routes with `#<channel>`, so a camera seen directly (`serial:X`) and via
  the NVR (`serial:X#3`) arrives twice; routes are joined by serial, then MAC.
  A serial repeated across NVR channels is the recorder's own and never joins;
* orders each camera's routes by SOURCE_PREFERENCE;
* keeps URLs credential-free; `render()` adds the account only at use.
"""

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional
from urllib.parse import quote, urlsplit, urlunsplit

from ... import discovery as hik
from ...config import Settings
from ..dtos import CandidateDTO, NvrDTO, ScanResultDTO, SourceDTO


def normalize_serial(value: Optional[str]) -> Optional[str]:
    return str(value or "").strip().upper() or None


def normalize_mac(value: Optional[str]) -> Optional[str]:
    """'AA-BB-CC-DD-EE-FF' -> 'aa:bb:cc:dd:ee:ff'; malformed or all-zero -> None."""
    value = "".join(ch for ch in str(value or "").lower() if ch in "0123456789abcdef")
    if len(value) != 12 or value == "0" * 12:
        return None
    return ":".join(value[i:i + 2] for i in range(0, 12, 2))


def with_credentials(url: str, username: str, password: str) -> str:
    """Insert percent-encoded userinfo into a credential-free URL."""
    if not username:
        return url
    parts = urlsplit(url)
    netloc = "{}:{}@{}".format(quote(username, safe=""), quote(password or "", safe=""), parts.hostname or "")
    if parts.port:
        netloc += ":{}".format(parts.port)
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def redact(url: str) -> str:
    """Strip userinfo so a URL is safe to log or report."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    if "@" not in parts.netloc:
        return url
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[1], parts.path, parts.query, parts.fragment))


class DiscoveryAdapter:
    def __init__(self, settings: Settings, scan_fn: Optional[Callable[[Settings], list]] = None):
        self._settings = settings
        self._scan_fn = scan_fn or hik.scan_network
        # One worker: sweeps are serialized by the sweep service, and
        # scan_network() fans its probes out on its own pool.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvr-discovery")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def scan(self) -> ScanResultDTO:
        loop = asyncio.get_running_loop()
        devices = await loop.run_in_executor(self._executor, self._scan_fn, self._settings)
        return self.build(devices)

    def render(self, kind: str, url: str) -> str:
        """The stream URL with the right account: NVR routes use NVR_*, direct HIK_*."""
        account = self._settings.nvr_account if kind == "nvr" else self._settings.camera_account
        return with_credentials(url, *account)

    def build(self, devices: List["hik.HikDevice"]) -> ScanResultDTO:
        """Drop unexpected recorders, then group routes into candidates."""
        host = self._settings.nvr_host
        kept, unexpected = [], Counter()
        for dev in devices:
            if dev.is_nvr and dev.ip != host:
                unexpected[(dev.ip, dev.rtsp_port)] += 1
            else:
                kept.append(dev)
        errors = ["Unexpected recorder at {}:{} ({} channel(s)) ignored: {}".format(
                      ip, port, count, "only {} is the NVR".format(host) if host else "the MiniPC is the NVR")
                  for (ip, port), count in sorted(unexpected.items())]

        nvr = None
        if host:
            nvr = NvrDTO(
                host=host,
                rtsp_port=self._settings.nvr_rtsp_port,
                http_port=self._settings.nvr_http_port,
                origin="static" if self._settings.nvr_channels else "isapi",
                channels_spec=self._settings.nvr_channels or None,
                responded=any(dev.is_nvr and dev.model != hik.STATIC_CHANNEL_MODEL for dev in kept),
            )
        return ScanResultDTO(nvr=nvr, candidates=self._group(kept), errors=errors)

    def _group(self, devices: List["hik.HikDevice"]) -> List[CandidateDTO]:
        nvr_serials = Counter(normalize_serial(d.serial_number) for d in devices if d.is_nvr)
        recorder_serials = {serial for serial, count in nvr_serials.items() if serial and count > 1}
        sources = [self._source(dev, recorder_serials) for dev in devices]

        # Union-find over route indexes; any shared serial/MAC key joins two routes.
        parent = list(range(len(sources)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        owner: Dict[str, int] = {}
        for index, source in enumerate(sources):
            for key in _join_keys(source):
                if key in owner:
                    a, b = find(owner[key]), find(index)
                    parent[max(a, b)] = min(a, b)
                else:
                    owner[key] = index

        groups: Dict[int, List[SourceDTO]] = {}
        for index, source in enumerate(sources):
            groups.setdefault(find(index), []).append(source)

        preferred = self._settings.source_preference
        candidates = []
        for members in groups.values():
            # Stable within a kind: discovery order is configuration order.
            members.sort(key=lambda s: s.kind != preferred)
            candidates.append(CandidateDTO(sources=members))
        return candidates

    def _source(self, dev, recorder_serials) -> SourceDTO:
        serial = normalize_serial(dev.serial_number)
        return SourceDTO(
            source_identity=hik.identity_of(dev),
            kind="nvr" if dev.is_nvr else "direct",
            host=dev.ip,
            rtsp_port=dev.rtsp_port,
            channel_no=dev.channel if dev.is_nvr else None,
            source_url=hik.rtsp_url(dev, self._settings),
            serial_number=None if serial in recorder_serials else serial,
            mac_address=normalize_mac(dev.mac),
            model=dev.model,
            firmware=dev.firmware,
            device_name=dev.device_name,
        )


def _join_keys(source: SourceDTO) -> List[str]:
    keys = []
    if source.serial_number:
        keys.append("serial:" + source.serial_number)
    if source.mac_address:
        keys.append("mac:" + source.mac_address)
    return keys
