import os
import httpx
import asyncio

from typing import Any, Callable, Dict, List, Optional, Tuple, Set
import uuid
from datetime import datetime, date
from pydantic import BaseModel, Field


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

class EdgeInferenceClient:
    """
    Talks to the Jetson TensorRT (inference) service.

    Environment (defaults are guesses; set them to match your Jetson routes):
      - EDGE_ADD_PATH (default: /api/cameras)
      - EDGE_PATCH_PATH (default: /api/cameras/{camera_uuid})
      - EDGE_DELETE_PATH (default: /api/cameras/{camera_uuid})
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

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))

    async def close(self) -> None:
        await self._client.aclose()



    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

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
        r = await self._client.get(url, headers=self._headers())
        if r.status_code >= 400:
            raise RuntimeError(f"Edge list_cameras error {r.status_code}: {r.text[:300]}")

        data = r.json()

        items = []
        if isinstance(data, dict):
            items = data.get("cameras") 
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
            upsert_url = f"{device_url.rstrip('/')}{self.add_path}"
            await self._request("POST", upsert_url, json=patch)

    async def delete_camera(self, *, device_url: str, camera_uuid: str) -> None:
        url = f"{device_url.rstrip('/')}{self.delete_path.format(camera_uuid=camera_uuid)}"
        await self._request("DELETE", url)

    async def _request(self, method: str, url: str, *, json: Optional[dict] = None) -> None:
        last_exc: Optional[Exception] = None
        json_payload = to_jsonable(json) if json is not None else None
        for attempt in range(3):
            try:
                r = await self._client.request(method, url, headers=self._headers(), json=json_payload)
                if method == "DELETE" and r.status_code == 404:
                    return
                if r.status_code >= 400:
                    raise RuntimeError(f"Edge service error {r.status_code}: {r.text[:300]}")
                return
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.2 * (2 ** attempt))
        raise last_exc or RuntimeError("Edge service request failed")

