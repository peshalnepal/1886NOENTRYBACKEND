import os
import httpx
import asyncio
import logging
import uuid
from datetime import datetime, date
from typing import Any, Dict, Iterable, List, Optional, Set, Union
from pydantic import BaseModel

from core.env import env_bool, env_float

logger = logging.getLogger(__name__)
_ROUTE_FALLBACK_STATUS_CODES = {404, 405}


def _health_bits(health: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(health, dict):
        return []
    bits = [f"{k}={v}" for k, v in health.items() if k in {"ok", "pipeline_ready", "startup_error"} and v]
    return bits


class EdgeCameraInventoryError(RuntimeError):
    def __init__(
        self, detail: str, *, health: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None
    ) -> None:
        self.health = dict(health) if isinstance(health, dict) else None
        self.cause = cause

        msg = str(detail or "").strip() or "Edge camera inventory request failed"
        bits = _health_bits(self.health)
        if bits:
            msg = f"{msg} (health: {', '.join(bits)})"
        super().__init__(msg)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


class EdgeInferenceClient:
    """
    Talks to the Jetson TensorRT (inference) service.
    """

    def __init__(self):
        self.add_path = os.getenv("EDGE_ADD_PATH", "/cameras")
        self.patch_path = os.getenv("EDGE_PATCH_PATH", "/cameras/{camera_uuid}")
        self.delete_path = os.getenv("EDGE_DELETE_PATH", "/cameras/{camera_uuid}")
        self.api_key = os.getenv("EDGE_API_KEY")
        self.list_path = os.getenv("EDGE_LIST_PATH", self.add_path)
        
        self.request_timeout_s = env_float("EDGE_TIMEOUT_S", 15.0, minimum=0.1)
        self.connect_timeout_s = env_float("EDGE_CONNECT_TIMEOUT_S", min(self.request_timeout_s, 10.0), minimum=0.1)
        
        self.list_timeout_s = env_float("EDGE_LIST_TIMEOUT_S", min(self.request_timeout_s, 5.0), minimum=0.1)
        self.list_connect_timeout_s = env_float("EDGE_LIST_CONNECT_TIMEOUT_S", min(self.connect_timeout_s, self.list_timeout_s, 3.0), minimum=0.1)
        
        self.health_timeout_s = env_float("EDGE_HEALTH_TIMEOUT_S", min(self.list_timeout_s, 2.0), minimum=0.1)
        self.health_connect_timeout_s = env_float("EDGE_HEALTH_CONNECT_TIMEOUT_S", min(self.connect_timeout_s, self.health_timeout_s, 1.5), minimum=0.1)
        
        self.retry_count = max(1, int(os.getenv("EDGE_HTTP_RETRIES", "3")))
        self.trust_env = env_bool("EDGE_HTTP_TRUST_ENV", False)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout_s, connect=self.connect_timeout_s),
            trust_env=self.trust_env,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def _format_request_error(self, method: str, url: str, exc: Exception) -> str:
        msg = str(exc).strip()
        if isinstance(exc, RuntimeError) and msg:
            return msg
        return f"{type(exc).__name__} during {method} {url}{': ' + msg if msg else ''}"

    def _candidate_paths(self, path: str) -> List[str]:
        raw = str(path or "").strip() or "/"
        raw = raw if raw.startswith("/") else f"/{raw}"

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
        raw_urls = [url_or_urls] if isinstance(url_or_urls, str) else list(url_or_urls)
        return list(dict.fromkeys(u.strip() for u in raw_urls if str(u or "").strip()))

    def _timeout(self, request_timeout_s: Optional[float] = None, connect_timeout_s: Optional[float] = None) -> httpx.Timeout:
        r_out = max(0.1, float(self.request_timeout_s if request_timeout_s is None else request_timeout_s))
        c_out = min(max(0.1, float(self.connect_timeout_s if connect_timeout_s is None else connect_timeout_s)), r_out)
        return httpx.Timeout(r_out, connect=c_out)

    async def _request(
        self,
        method: str,
        url_or_urls: Union[str, Iterable[str]],
        *,
        json_payload: Optional[dict] = None,
        request_timeout_s: Optional[float] = None,
        connect_timeout_s: Optional[float] = None,
        ignore_404: bool = False
    ) -> Optional[httpx.Response]:
        """Unified method to handle requests, retries, and fallback routes."""
        urls = self._normalize_urls(url_or_urls)
        timeout = self._timeout(request_timeout_s=request_timeout_s, connect_timeout_s=connect_timeout_s)
        payload = to_jsonable(json_payload) if json_payload is not None else None
        
        last_exc: Optional[Exception] = None
        last_url = urls[-1] if urls else ""
        saw_404, saw_non_404 = False, False

        for attempt in range(self.retry_count):
            for url in urls:
                try:
                    r = await self._client.request(
                        method, url, headers=self._headers(), json=payload, timeout=timeout
                    )
                    
                    if ignore_404 and r.status_code == 404:
                        saw_404, last_url = True, url
                        last_exc = RuntimeError(f"Edge service error 404 for {method} {url}: {r.text[:300]}")
                        continue
                        
                    if r.status_code in _ROUTE_FALLBACK_STATUS_CODES and len(urls) > 1:
                        last_url = url
                        last_exc = RuntimeError(f"Edge service error {r.status_code} for {method} {url}: {r.text[:300]}")
                        continue
                        
                    if r.status_code >= 400:
                        raise RuntimeError(f"Edge service error {r.status_code} for {method} {url}: {r.text[:300]}")
                        
                    return r
                    
                except Exception as exc:
                    last_url, last_exc = url, exc
                    saw_non_404 = True
                    continue
                    
            if attempt + 1 >= self.retry_count:
                break
            await asyncio.sleep(0.2 * (2 ** attempt))

        if ignore_404 and saw_404 and not saw_non_404:
            return None

        if last_exc is not None:
            raise RuntimeError(self._format_request_error(method, last_url, last_exc)) from last_exc
        raise RuntimeError(f"Edge service request failed for {method} {last_url}")

    async def get_health(
        self, *, device_url: str, request_timeout_s: Optional[float] = None, connect_timeout_s: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        base = device_url.rstrip("/")
        timeout = self._timeout(
            request_timeout_s=self.health_timeout_s if request_timeout_s is None else request_timeout_s,
            connect_timeout_s=self.health_connect_timeout_s if connect_timeout_s is None else connect_timeout_s,
        )
        for url in [f"{base}/health", f"{base}/api/health"]:
            try:
                r = await self._client.get(url, headers=self._headers(), timeout=timeout)
                data = r.json()
                if isinstance(data, dict):
                    if any(k in data for k in ("ok", "pipeline_ready", "startup_error")) or r.status_code < 400:
                        return data
            except Exception:
                continue
        return None

    async def list_cameras(self, *, device_url: str) -> Set[str]:
        urls = self._candidate_urls(device_url=device_url, path=self.list_path)
        try:
            r = await self._request(
                "GET", urls, request_timeout_s=self.list_timeout_s, connect_timeout_s=self.list_connect_timeout_s
            )
            data = r.json() if r else {}  # Convert to JSON exactly where it's needed
        except Exception as exc:
            health = await self.get_health(device_url=device_url)
            raise EdgeCameraInventoryError(str(exc), health=health, cause=exc) from exc

        items = data.get("cameras", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        
        out: Set[str] = set()
        for it in items:
            v = it.get("camera_uuid") if isinstance(it, dict) else it
            if v:
                try:
                    out.add(str(uuid.UUID(str(v))))
                except ValueError:
                    pass
        return out

    async def ensure_pipeline_ready(self, *, device_url: str) -> None:
        h = await self.get_health(device_url=device_url)
        if not isinstance(h, dict):
            return
            
        if h.get("pipeline_ready") is False or h.get("ok") is False:
            startup_err = h.get("startup_error")
            err_msg = f" (startup_error: {startup_err})" if startup_err else ""
            raise RuntimeError(f"Edge pipeline not ready at {device_url}{err_msg}")

    async def upsert_camera(self, *, device_url: str, payload: dict) -> None:
        urls = self._candidate_urls(device_url=device_url, path=self.add_path)
        await self._request("POST", urls, json_payload=payload)
    
    async def patch_camera(self, *, device_url: str, camera_uuid: str, patch: dict) -> None:
        urls = self._candidate_urls(device_url=device_url, path=self.patch_path.format(camera_uuid=camera_uuid))
        try:
            await self._request("PATCH", urls, json_payload=patch)
        except Exception:
            upsert_payload = dict(patch or {})
            upsert_payload.setdefault("camera_uuid", str(camera_uuid))
            await self.upsert_camera(device_url=device_url, payload=upsert_payload)

    async def delete_camera(self, *, device_url: str, camera_uuid: str) -> None:
        urls = self._candidate_urls(device_url=device_url, path=self.delete_path.format(camera_uuid=camera_uuid))
        await self._request("DELETE", urls, ignore_404=True)