"""Local discovery endpoints and a streaming HTTP proxy for Jetson APIs."""

import json
import re

import httpx
from fastapi import APIRouter, HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

from .jetson import JetsonError

router = APIRouter()
_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade", "host"}
# Jetson's public API surface; discovery always stays local.
_PROXY_GET = re.compile(
    r"(?:|health|cameras|cameras/detections/stream|cameras/[^/]+/(?:latest|snapshot\.jpg|detections/stream)|detections?/[^/]+)")
_CAMERA = re.compile(r"cameras/[^/]+")


def end_to_end(headers):
    """Drop hop-by-hop headers, including any named by Connection."""
    blocked = _HOP_HEADERS | {v.strip().lower() for v in headers.get("connection", "").split(",")}
    return {k: v for k, v in headers.items() if k.lower() not in blocked}


@router.get("/discovery")
async def discovery(request: Request):
    sweep = request.app.state.sweep
    roster = await sweep.roster()
    return {"discovered": roster, "present": [r for r in roster if r["is_present"]],
            "missing": [r for r in roster if not r["is_present"]], "status": sweep.status()}


@router.get("/discovery/status")
async def status(request: Request):
    return request.app.state.sweep.status()


@router.get("/discovery/report")
async def report(request: Request):
    result = request.app.state.sweep.last_report()
    if result is None:
        raise HTTPException(404, "No discovery sweep has completed yet")
    return result


@router.post("/discovery/scan")
async def scan(request: Request):
    """Force a sweep and wait for its report; concurrent callers share one."""
    return await request.app.state.sweep.run_once(force=True)


@router.delete("/discovery/{identity:path}")
async def forget(identity: str, request: Request):
    """Forget a local roster entry. Its Jetson channel and relay path are kept."""
    existed = await request.app.state.sweep.forget(identity)
    return {"forgotten": existed, "existed": existed, "identity": identity}


@router.post("/sync")
async def sync(request: Request):
    """Schedule a sweep; return Jetson's inventory and the previous report."""
    sweep = request.app.state.sweep
    pending = sweep.request_scan()
    try:
        cameras = await request.app.state.jetson.cameras()
    except JetsonError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    cameras = await request.app.state.relay.translate(cameras, outward=True)
    return {"cameras": cameras, "discovery": sweep.last_report(),
            "status": sweep.status(), "discovery_pending": pending}


@router.api_route("/{path:path}", methods=["GET", "POST", "PATCH", "DELETE"])
async def proxy(path: str, request: Request):
    """Forward allowed Jetson APIs as a byte stream (JSON, JPEG, SSE).

    Camera JSON is the exception: relay URLs are translated public -> LAN on the
    way in and LAN -> public on the way out.
    """
    allowed = (
        request.method == "GET" and _PROXY_GET.fullmatch(path)
        or request.method == "POST" and path == "cameras"
        or request.method in {"PATCH", "DELETE"} and _CAMERA.fullmatch(path)
    )
    if not allowed:
        raise HTTPException(404, "Unknown API route")
    client = request.app.state.jetson.http
    relay = request.app.state.relay
    # Retain the raw query (including repeated keys) and raw response bytes.
    url = client.base_url.copy_with(path="/" + path, query=request.scope["query_string"])
    camera_json = path == "cameras" or _CAMERA.fullmatch(path)
    content = await request.body()
    headers = end_to_end(request.headers)
    if camera_json and content:
        try:
            payload = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, "Invalid camera JSON") from None
        content = json.dumps(await relay.translate(payload, outward=False)).encode()
        headers.pop("content-length", None)
    upstream_request = client.build_request(request.method, url, headers=headers, content=content)
    if path.endswith("/detections/stream"):
        upstream_request.extensions["timeout"]["read"] = None
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException:
        raise HTTPException(504, "Jetson request timed out") from None
    except httpx.RequestError:
        raise HTTPException(502, "Jetson is unavailable") from None

    if camera_json and "application/json" in upstream.headers.get("content-type", ""):
        try:
            await upstream.aread()
            payload = await relay.translate(upstream.json(), outward=True)
            headers = end_to_end(upstream.headers)
            for key in ("content-length", "content-encoding", "etag"):
                headers.pop(key, None)
            return JSONResponse(payload, status_code=upstream.status_code, headers=headers)
        except (ValueError, httpx.RequestError):
            raise HTTPException(502, "Invalid Jetson camera response") from None
        finally:
            await upstream.aclose()

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body(), status_code=upstream.status_code,
                             headers=end_to_end(upstream.headers), background=BackgroundTask(upstream.aclose))
