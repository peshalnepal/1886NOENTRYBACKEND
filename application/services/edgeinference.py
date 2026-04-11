import os
import httpx
import asyncio
import logging

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set, Union
import uuid
from datetime import datetime, date
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_ROUTE_FALLBACK_STATUS_CODES = {404, 405}


def _health_bits(health: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(health, dict):
        return []

    bits: List[str] = []
    if "ok" in health:
        bits.append(f"ok={health.get('ok')}")
    if "pipeline_ready" in health:
        bits.append(f"pipeline_ready={health.get('pipeline_ready')}")
    startup_error = health.get("startup_error")
    if startup_error:
        bits.append(f"startup_error={startup_error}")
    return bits


class EdgeCameraInventoryError(RuntimeError):
    def __init__(
        self,
        detail: str,
        *,
        health: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.health = dict(health) if isinstance(health, dict) else None
        self.cause = cause

        msg = str(detail or "").strip() or "Edge camera inventory request failed"
        bits = _health_bits(self.health)
        if bits:
            msg = f"{msg} (health: {', '.join(bits)})"
        super().__init__(msg)


def to_jsonable(obj):
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            out[str(k)] = to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except Exception:
        value = float(default)
    return max(float(minimum), float(value))

class EdgeInferenceClient:
    """
    Talks to the Jetson TensorRT (inference) service.

    Environment (defaults are guesses; set them to match your Jetson routes):
      - EDGE_ADD_PATH (default: /cameras)
      - EDGE_PATCH_PATH (default: /cameras/{camera_uuid})
      - EDGE_DELETE_PATH (default: /cameras/{camera_uuid})
      - EDGE_API_KEY (optional header: x-api-key)

    Expected semantics on Jetson:
      - POST   add/upsert a camera by camera_uuid
      - PATCH  update config for camera_uuid
      - DELETE remove camera_uuid
    """

    def __init__(self):
        self.add_path = os.getenv("EDGE_ADD_PATH", "/cameras")
        self.patch_path = os.getenv("EDGE_PATCH_PATH", "/cameras/{camera_uuid}")
        self.delete_path = os.getenv("EDGE_DELETE_PATH", "/cameras/{camera_uuid}")
        self.api_key = os.getenv("EDGE_API_KEY")
        self.list_path = os.getenv("EDGE_LIST_PATH", self.add_path)
        self.request_timeout_s = _env_float("EDGE_TIMEOUT_S", 15.0)
        self.connect_timeout_s = _env_float(
            "EDGE_CONNECT_TIMEOUT_S",
            min(self.request_timeout_s, 10.0),
        )
        self.list_timeout_s = _env_float(
            "EDGE_LIST_TIMEOUT_S",
            min(self.request_timeout_s, 5.0),
        )
        self.list_connect_timeout_s = _env_float(
            "EDGE_LIST_CONNECT_TIMEOUT_S",
            min(self.connect_timeout_s, self.list_timeout_s, 3.0),
        )
        self.health_timeout_s = _env_float(
            "EDGE_HEALTH_TIMEOUT_S",
            min(self.list_timeout_s, 2.0),
        )
        self.health_connect_timeout_s = _env_float(
            "EDGE_HEALTH_CONNECT_TIMEOUT_S",
            min(self.connect_timeout_s, self.health_timeout_s, 1.5),
        )
        self.retry_count = max(1, int(os.getenv("EDGE_HTTP_RETRIES", "3")))
        # Edge-device calls should bypass ambient proxy settings unless explicitly
        # opted in. Proxy env vars are a common source of opaque ConnectError
        # failures for device-local or port-forwarded URLs.
        self.trust_env = _env_bool("EDGE_HTTP_TRUST_ENV", False)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout_s, connect=self.connect_timeout_s),
            trust_env=self.trust_env,
        )

    async def close(self) -> None:
        await self._client.aclose()



    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _format_request_error(self, method: str, url: str, exc: Exception) -> str:
        if isinstance(exc, RuntimeError):
            msg = str(exc).strip()
            if msg:
                return msg
        exc_name = type(exc).__name__
        msg = str(exc).strip()
        if msg:
            return f"{exc_name} during {method} {url}: {msg}"
        return f"{exc_name} during {method} {url}"

    def _candidate_paths(self, path: str) -> List[str]:
        raw = str(path or "").strip() or "/"
        if not raw.startswith("/"):
            raw = f"/{raw}"

        candidates = [raw]
        if raw.startswith("/api/"):
            candidates.append(raw[4:] or "/")
        elif raw == "/api":
            candidates.append("/")
        elif raw != "/":
            candidates.append(f"/api{raw}")
        return list(dict.fromkeys(candidates))

    def _candidate_urls(self, *, device_url: str, path: str) -> List[str]:
        base = device_url.rstrip("/")
        return [f"{base}{candidate}" for candidate in self._candidate_paths(path)]

    def _normalize_urls(self, url_or_urls: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(url_or_urls, str):
            raw_urls = [url_or_urls]
        else:
            raw_urls = list(url_or_urls)

        urls: List[str] = []
        seen: Set[str] = set()
        for item in raw_urls:
            url = str(item or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    def _timeout(
        self,
        *,
        request_timeout_s: Optional[float] = None,
        connect_timeout_s: Optional[float] = None,
    ) -> httpx.Timeout:
        read_timeout = max(
            0.1,
            float(self.request_timeout_s if request_timeout_s is None else request_timeout_s),
        )
        connect_timeout = max(
            0.1,
            float(self.connect_timeout_s if connect_timeout_s is None else connect_timeout_s),
        )
        connect_timeout = min(connect_timeout, read_timeout)
        return httpx.Timeout(read_timeout, connect=connect_timeout)

    async def _request_json(
        self,
        method: str,
        url_or_urls: Union[str, Iterable[str]],
        *,
        request_timeout_s: Optional[float] = None,
        connect_timeout_s: Optional[float] = None,
    ) -> Any:
        urls = self._normalize_urls(url_or_urls)
        last_exc: Optional[Exception] = None
        last_url = urls[-1] if urls else ""
        timeout = self._timeout(
            request_timeout_s=request_timeout_s,
            connect_timeout_s=connect_timeout_s,
        )
        for attempt in range(self.retry_count):
            for url in urls:
                try:
                    r = await self._client.request(
                        method,
                        url,
                        headers=self._headers(),
                        timeout=timeout,
                    )
                    if r.status_code in _ROUTE_FALLBACK_STATUS_CODES and len(urls) > 1:
                        last_url = url
                        last_exc = RuntimeError(
                            f"Edge service error {r.status_code} for {method} {url}: {r.text[:300]}"
                        )
                        continue
                    if r.status_code >= 400:
                        raise RuntimeError(f"Edge service error {r.status_code} for {method} {url}: {r.text[:300]}")
                    return r.json()
                except Exception as exc:
                    last_url = url
                    last_exc = exc
                    continue
            if attempt + 1 >= self.retry_count:
                break
            await asyncio.sleep(0.2 * (2 ** attempt))

        if last_exc is not None:
            raise RuntimeError(self._format_request_error(method, last_url, last_exc)) from last_exc
        raise RuntimeError(f"Edge service request failed for {method} {last_url}")

    async def get_health(
        self,
        *,
        device_url: str,
        request_timeout_s: Optional[float] = None,
        connect_timeout_s: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Best-effort health probe.
        Tries /health then /api/health and returns parsed JSON payload.
        Accepts 503 responses too if they include structured readiness details.
        """
        base = device_url.rstrip("/")
        timeout = self._timeout(
            request_timeout_s=self.health_timeout_s if request_timeout_s is None else request_timeout_s,
            connect_timeout_s=self.health_connect_timeout_s if connect_timeout_s is None else connect_timeout_s,
        )
        urls = [
            "{}/health".format(base),
            "{}/api/health".format(base),
        ]
        for url in urls:
            try:
                r = await self._client.get(url, headers=self._headers(), timeout=timeout)
                data = r.json()
                if isinstance(data, dict):
                    if any(k in data for k in ("ok", "pipeline_ready", "startup_error")):
                        return data
                    if r.status_code < 400:
                        return data
            except Exception:
                continue
        return None
    
    async def list_cameras(self, *, device_url: str) -> Set[str]:
        """
        Calls Jetson GET /cameras.

        Accepts responses like:
          - {"cameras": [{"camera_uuid": "...", ...}, ...]}  (your Jetson does this)
          - {"cameras": ["uuid1", "uuid2", ...]}
          - ["uuid1", "uuid2", ...]
        """
        urls = self._candidate_urls(device_url=device_url, path=self.list_path)
        try:
            data = await self._request_json(
                "GET",
                urls,
                request_timeout_s=self.list_timeout_s,
                connect_timeout_s=self.list_connect_timeout_s,
            )
        except Exception as exc:
            health = await self.get_health(device_url=device_url)
            raise EdgeCameraInventoryError(str(exc), health=health, cause=exc) from exc

        items = []
        if isinstance(data, dict):
            items = data.get("cameras") or []
        elif isinstance(data, list):
            items = data

        out: Set[str] = set()
        for it in items:
            if isinstance(it, dict):
                v = it.get("camera_uuid")
            else:
                v = it
            if not v:
                continue
            try:
                out.add(str(uuid.UUID(str(v))))
            except Exception:
                continue
        return out

    async def ensure_pipeline_ready(self, *, device_url: str) -> None:
        """
        Raise a clear error if edge reports startup failure / not-ready state.
        If health endpoint is unavailable, this is a no-op (backward compatible).
        """
        h = await self.get_health(device_url=device_url)
        if not isinstance(h, dict):
            return
        pipeline_ready = h.get("pipeline_ready")
        ok = h.get("ok")
        if pipeline_ready is False or ok is False:
            startup_error = h.get("startup_error")
            if startup_error:
                raise RuntimeError(
                    "Edge pipeline not ready at {} (startup_error: {})".format(device_url, startup_error)
                )
            raise RuntimeError("Edge pipeline not ready at {}".format(device_url))

    async def upsert_camera(self, *, device_url: str, payload: dict) -> None:
        urls = self._candidate_urls(device_url=device_url, path=self.add_path)
        await self._request("POST", urls, json=payload)

    async def patch_camera(self, *, device_url: str, camera_uuid: str, patch: dict) -> None:
        """
        Best-effort patch. If PATCH is not supported by the Jetson service, fallback to POST upsert.
        """
        urls = self._candidate_urls(
            device_url=device_url,
            path=self.patch_path.format(camera_uuid=camera_uuid),
        )
        try:
            await self._request("PATCH", urls, json=patch)
        except Exception:
            upsert_payload = dict(patch or {})
            upsert_payload.setdefault("camera_uuid", str(camera_uuid))
            upsert_urls = self._candidate_urls(device_url=device_url, path=self.add_path)
            await self._request("POST", upsert_urls, json=upsert_payload)

    async def delete_camera(self, *, device_url: str, camera_uuid: str) -> None:
        urls = self._candidate_urls(
            device_url=device_url,
            path=self.delete_path.format(camera_uuid=camera_uuid),
        )
        await self._request("DELETE", urls)

    async def _request(self, method: str, url_or_urls: Union[str, Iterable[str]], *, json: Optional[dict] = None) -> None:
        urls = self._normalize_urls(url_or_urls)
        last_exc: Optional[Exception] = None
        last_url = urls[-1] if urls else ""
        json_payload = to_jsonable(json) if json is not None else None
        saw_delete_not_found = False
        saw_non_404_failure = False
        for attempt in range(self.retry_count):
            for url in urls:
                try:
                    r = await self._client.request(method, url, headers=self._headers(), json=json_payload)
                    if method == "DELETE" and r.status_code == 404:
                        saw_delete_not_found = True
                        last_url = url
                        last_exc = RuntimeError(f"Edge service error 404 for {method} {url}: {r.text[:300]}")
                        continue
                    if r.status_code in _ROUTE_FALLBACK_STATUS_CODES and len(urls) > 1:
                        last_url = url
                        last_exc = RuntimeError(
                            f"Edge service error {r.status_code} for {method} {url}: {r.text[:300]}"
                        )
                        continue
                    if r.status_code >= 400:
                        raise RuntimeError(f"Edge service error {r.status_code} for {method} {url}: {r.text[:300]}")
                    return
                except Exception as e:
                    last_url = url
                    last_exc = e
                    saw_non_404_failure = True
                    continue
            if attempt + 1 >= self.retry_count:
                break
            await asyncio.sleep(0.2 * (2 ** attempt))
        if method == "DELETE" and saw_delete_not_found and not saw_non_404_failure:
            return
        if last_exc is not None:
            raise RuntimeError(self._format_request_error(method, last_url, last_exc)) from last_exc
        raise RuntimeError(f"Edge service request failed for {method} {url}")
