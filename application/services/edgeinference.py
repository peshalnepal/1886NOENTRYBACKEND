import os
import httpx
import asyncio
import logging

from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import uuid
from datetime import datetime, date
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
        self.request_timeout_s = float(os.getenv("EDGE_TIMEOUT_S", "15"))
        self.connect_timeout_s = float(os.getenv("EDGE_CONNECT_TIMEOUT_S", "10"))
        self.retry_count = max(1, int(os.getenv("EDGE_HTTP_RETRIES", "3")))
        self.trust_env = _env_bool("EDGE_HTTP_TRUST_ENV", True)

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

    async def _request_json(self, method: str, url: str) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self.retry_count):
            try:
                r = await self._client.request(method, url, headers=self._headers())
                if r.status_code >= 400:
                    raise RuntimeError(f"Edge service error {r.status_code} for {method} {url}: {r.text[:300]}")
                return r.json()
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= self.retry_count:
                    break
                await asyncio.sleep(0.2 * (2 ** attempt))

        if last_exc is not None:
            raise RuntimeError(self._format_request_error(method, url, last_exc)) from last_exc
        raise RuntimeError(f"Edge service request failed for {method} {url}")

    async def get_health(self, *, device_url: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort health probe.
        Tries /health then /api/health and returns parsed JSON payload.
        Accepts 503 responses too if they include structured readiness details.
        """
        base = device_url.rstrip("/")
        urls = [
            "{}/health".format(base),
            "{}/api/health".format(base),
        ]
        for url in urls:
            try:
                r = await self._client.get(url, headers=self._headers())
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
        url = f"{device_url.rstrip('/')}{self.list_path}"
        try:
            data = await self._request_json("GET", url)
        except Exception as exc:
            health = await self.get_health(device_url=device_url)
            if isinstance(health, dict):
                health_bits = []
                if "ok" in health:
                    health_bits.append(f"ok={health.get('ok')}")
                if "pipeline_ready" in health:
                    health_bits.append(f"pipeline_ready={health.get('pipeline_ready')}")
                startup_error = health.get("startup_error")
                if startup_error:
                    health_bits.append(f"startup_error={startup_error}")
                if health_bits:
                    raise RuntimeError(f"{exc} (health: {', '.join(health_bits)})") from exc
            raise

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
        url = f"{device_url.rstrip('/')}{self.add_path}"
        await self._request("POST", url, json=payload)

    async def patch_camera(self, *, device_url: str, camera_uuid: str, patch: dict) -> None:
        """
        Best-effort patch. If PATCH is not supported by the Jetson service, fallback to POST upsert.
        """
        url = f"{device_url.rstrip('/')}{self.patch_path.format(camera_uuid=camera_uuid)}"
        try:
            await self._request("PATCH", url, json=patch)
        except Exception:
            upsert_payload = dict(patch or {})
            upsert_payload.setdefault("camera_uuid", str(camera_uuid))
            upsert_url = f"{device_url.rstrip('/')}{self.add_path}"
            await self._request("POST", upsert_url, json=upsert_payload)

    async def delete_camera(self, *, device_url: str, camera_uuid: str) -> None:
        url = f"{device_url.rstrip('/')}{self.delete_path.format(camera_uuid=camera_uuid)}"
        await self._request("DELETE", url)

    async def _request(self, method: str, url: str, *, json: Optional[dict] = None) -> None:
        last_exc: Optional[Exception] = None
        json_payload = to_jsonable(json) if json is not None else None
        for attempt in range(self.retry_count):
            try:
                r = await self._client.request(method, url, headers=self._headers(), json=json_payload)
                if method == "DELETE" and r.status_code == 404:
                    return
                if r.status_code >= 400:
                    raise RuntimeError(f"Edge service error {r.status_code} for {method} {url}: {r.text[:300]}")
                return
            except Exception as e:
                last_exc = e
                if attempt + 1 >= self.retry_count:
                    break
                await asyncio.sleep(0.2 * (2 ** attempt))
        if last_exc is not None:
            raise RuntimeError(self._format_request_error(method, url, last_exc)) from last_exc
        raise RuntimeError(f"Edge service request failed for {method} {url}")
