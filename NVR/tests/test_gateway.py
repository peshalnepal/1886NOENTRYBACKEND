import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from NVR.application.dtos import CandidateDTO, ScanResultDTO, SourceDTO
from NVR.application.services.adoption_service import AdoptionService
from NVR.application.services.discovery_adapter import DiscoveryAdapter
from NVR.application.services.frame_verifier import FrameVerifier, VerifyResult
from NVR.application.services.sweep_service import SweepService
from NVR.core.database import Database
from NVR.jetson import JetsonClient
from NVR.relay import MediaRelay
from NVR.routes import router
from NVR.tests.fakes import MemoryMediaMtx, make_settings


def passthrough_relay():
    return SimpleNamespace(translate=AsyncMock(side_effect=lambda payload, outward: payload))


def test_discovery_adoption_retries_and_repoint(tmp_path):
    async def run():
        settings = make_settings(MINIPC_DB_PATH=tmp_path / "roster.db", DISCOVERY_ENABLED="false")
        db = Database(settings.minipc_db_path)
        await db.init()
        remote, calls = {}, []
        reject = [True]

        async def handle(request):
            calls.append((request.method, request.url.path))
            if request.method == "GET":
                return httpx.Response(200, json={"cameras": list(remote.values())})
            body = json.loads(request.content)
            if reject[0]:
                return httpx.Response(409, json={"error": "capacity"})
            if request.method == "POST":
                key = body["config"]["camera_uuid"]
                remote[key] = dict(body, camera_uuid=key)
                return httpx.Response(201, json=remote[key])
            key = request.url.path.rsplit("/", 1)[-1]
            remote[key]["source_url"] = body["source_url"]
            return httpx.Response(200, json=remote[key])

        jetson = JetsonClient(settings, transport=httpx.MockTransport(handle))
        adapter = DiscoveryAdapter(settings)
        source = SourceDTO(source_identity="serial:TEST", kind="direct", host="192.0.2.1",
                           rtsp_port=554, source_url="rtsp://192.0.2.1:554/Streaming/Channels/102",
                           serial_number="TEST")
        def result(s):
            return ScanResultDTO(candidates=[CandidateDTO(sources=[s])])
        adapter.scan = AsyncMock(return_value=result(source))
        verifier = SimpleNamespace(verify=AsyncMock(return_value=VerifyResult(True)))
        relay = MediaRelay(settings, db.session_factory, adapter, transport=MemoryMediaMtx().transport())
        adoption = AdoptionService(db.session_factory, jetson, relay)
        sweep = SweepService(settings, db.session_factory, adapter, verifier, relay, adoption)
        try:
            first = await sweep.run_once(force=True)
            key = first["roster"][0]["camera_uuid"]
            assert first["roster"][0]["jetson_state"] == "capacity_pending"
            assert not first["new_cameras"][0]["adopted"]
            reject[0] = False
            second = await sweep.run_once(force=True)
            assert second["roster"][0]["jetson_state"] == "pushed"
            assert list(remote) == [key]
            assert remote[key]["source_url"] == relay.jetson_url("serial:TEST")
            remote[key]["config"]["sample_fps"] = 2
            calls.clear()
            # A camera IP change moves only the relay source; Jetson's URL is stable.
            adapter.scan.return_value = result(source.model_copy(update={
                "host": "192.0.2.2", "source_url": source.source_url.replace(".1:", ".2:")}))
            await sweep.run_once(force=True)
            assert calls == [("GET", "/cameras")]
            # A channel still on the raw camera URL (pre-relay) is repointed in place.
            remote[key]["source_url"] = "rtsp://admin:old@192.0.2.2:554/Streaming/Channels/102"
            await sweep.run_once(force=True)
            assert calls[-1] == ("PATCH", f"/cameras/{key}")
            assert remote[key]["config"]["sample_fps"] == 2
            # Missing cameras age locally and are never deleted on Jetson.
            adapter.scan.return_value = ScanResultDTO()
            await sweep.run_once(force=True)
            missing = await sweep.run_once(force=True)
            assert len(missing["missing_cameras"]) == 1
            assert key in remote
            # A restored NVR process reuses the durable UUID.
            adapter.scan.return_value = result(source)
            replacement = SweepService(settings, db.session_factory, adapter, verifier, relay, adoption)
            recovered = await replacement.run_once(force=True)
            assert recovered["roster"][0]["camera_uuid"] == key
            assert len(recovered["recovered_cameras"]) == 1
        finally:
            await jetson.close()
            await relay.close()
            adapter.close()
            await db.dispose()
    asyncio.run(run())


def test_proxy_preserves_binary_errors_queries_and_local_routes():
    seen = []
    async def handle(request):
        seen.append(request)
        if request.url.path.endswith("snapshot.jpg"):
            return httpx.Response(200, stream=httpx.ByteStream(b"\xff\xd8\xff\xd9"), headers={"content-type": "image/jpeg"})
        if request.method == "POST":
            return httpx.Response(409, stream=httpx.ByteStream(b'{"error":"full"}'), headers={"content-type": "application/json"})
        return httpx.Response(200, stream=httpx.ByteStream(b"data: {}\n\n"), headers={"content-type": "text/event-stream"})
    jetson = JetsonClient(make_settings(), transport=httpx.MockTransport(handle))
    app = FastAPI()
    app.state.jetson = jetson
    app.state.relay = passthrough_relay()
    app.state.sweep = SimpleNamespace(status=lambda: {"owner": "nvr"})
    app.include_router(router, prefix="/api")
    app.include_router(router)
    with TestClient(app) as client:
        assert client.get("/api/discovery/status").json() == {"owner": "nvr"}
        assert not seen
        response = client.get("/api/cameras/abc/snapshot.jpg?a=1&a=2")
        assert response.content == b"\xff\xd8\xff\xd9"
        assert seen[-1].url.query == b"a=1&a=2"
        response = client.post("/cameras", json={"source_url": "rtsp://camera"})
        assert (response.status_code, response.json()) == (409, {"error": "full"})
        assert client.get("/api/cameras/detections/stream").content == b"data: {}\n\n"
        assert seen[-1].extensions["timeout"]["read"] is None
        assert client.get("/unrelated").status_code == 404
    asyncio.run(jetson.close())


def test_proxy_unreachable():
    async def handle(request):
        raise httpx.ConnectError("offline", request=request)
    jetson = JetsonClient(make_settings(), transport=httpx.MockTransport(handle))
    app = FastAPI()
    app.state.jetson = jetson
    app.state.relay = passthrough_relay()
    app.include_router(router)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 502
    asyncio.run(jetson.close())


def test_verification_requires_a_frame(monkeypatch):
    async def run():
        proc = SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b"frame=0\n", b"")))
        monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=proc))
        verifier = FrameVerifier()
        assert not (await verifier.verify("rtsp://camera/stream")).ok
        proc.communicate.return_value = (b"frame=1\n", b"")
        assert (await verifier.verify("rtsp://camera/stream")).ok
    asyncio.run(run())


def test_lost_post_response_is_reconciled_without_duplicate(tmp_path):
    async def run():
        settings = make_settings(MINIPC_DB_PATH=tmp_path / "retry.db")
        db = Database(settings.minipc_db_path)
        await db.init()
        remote, posts = [], []
        async def handle(request):
            if request.method == "GET":
                return httpx.Response(200, json={"cameras": remote})
            body = json.loads(request.content)
            posts.append(body)
            remote.append(dict(body, camera_uuid=body["config"]["camera_uuid"]))
            raise httpx.ReadTimeout("response lost", request=request)
        jetson = JetsonClient(settings, transport=httpx.MockTransport(handle))
        adapter = DiscoveryAdapter(settings)
        source = SourceDTO(source_identity="serial:RETRY", kind="direct", host="192.0.2.3",
                           rtsp_port=554, source_url="rtsp://192.0.2.3/Streaming/Channels/102")
        adapter.scan = AsyncMock(return_value=ScanResultDTO(candidates=[
            CandidateDTO(sources=[source])]))
        verifier = SimpleNamespace(verify=AsyncMock(return_value=VerifyResult(True)))
        relay = MediaRelay(settings, db.session_factory, adapter, transport=MemoryMediaMtx().transport())
        service = SweepService(settings, db.session_factory, adapter, verifier, relay,
                               AdoptionService(db.session_factory, jetson, relay))
        try:
            first = await service.run_once()
            assert first["roster"][0]["jetson_state"] == "error"
            second = await service.run_once()
            assert second["roster"][0]["jetson_state"] == "pushed"
            assert len(posts) == 1
            assert first["roster"][0]["camera_uuid"] == second["roster"][0]["camera_uuid"]
            verifier.verify.return_value = VerifyResult(False, "no frame")
            adapter.scan.return_value = ScanResultDTO(candidates=[CandidateDTO(
                sources=[source.model_copy(update={"source_identity": "serial:NEW"})])])
            failed = await service.run_once()
            assert failed["unverified_candidates"] == ["serial:NEW"]
            assert len(posts) == 1
        finally:
            await jetson.close()
            await relay.close()
            adapter.close()
            await db.dispose()
    asyncio.run(run())


def test_app_lifecycle_without_discovery(tmp_path):
    from NVR.main import create_app
    app = create_app(make_settings(MINIPC_DB_PATH=tmp_path / "app.db", DISCOVERY_ENABLED="false",
                                   RECORDING_CLIP_DIRECTORY=tmp_path / "clips"))
    with TestClient(app) as client:
        assert client.get("/api/discovery/status").json()["running"] is False
        assert client.get("/discovery").json()["discovered"] == []
        assert client.get("/discovery/report").status_code == 404
