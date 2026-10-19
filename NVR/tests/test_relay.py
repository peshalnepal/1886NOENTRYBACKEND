import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from NVR.application.dtos import CandidateDTO, ScanResultDTO, SourceDTO
from NVR.application.services.adoption_service import AdoptionService
from NVR.application.services.discovery_adapter import DiscoveryAdapter
from NVR.application.services.frame_verifier import VerifyResult
from NVR.application.services.sweep_service import SweepService
from NVR.config import Settings
from NVR.tests.fakes import make_settings
from NVR.core.database import Database
from NVR.jetson import JetsonClient
from NVR.relay import MediaRelay, RelayError
from NVR.routes import router


def settings_for(tmp_path):
    return make_settings(MINIPC_DB_PATH=tmp_path / 'relay.db', DISCOVERY_ENABLED='false',
                         HIK_USERNAME='camera-admin', HIK_PASSWORD='camera-secret',
                         NVR_PUBLIC_RTSP_BASE_URL='rtsp://reader:test@tower.example:8554',
                         MEDIAMTX_LAN_RTSP_BASE_URL='rtsp://reader:test@192.0.2.5:8554')


def test_standalone_import_without_tensort():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'tensort':
            raise AssertionError('NVR imported tensort')
sys.meta_path.insert(0, Block())
from NVR.main import create_app
from NVR.config import Settings
from NVR.tests.fakes import make_settings
create_app(Settings(_env_file=None, discovery_enabled=False, nvr_public_rtsp_base_url='rtsp://tower.example:8554'))
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_main_runs_as_plain_script_without_parent_folder(tmp_path):
    # Same as `cd NVR; python main.py`, from an unrelated cwd: only this folder
    # and site-packages may supply modules.
    nvr_dir = Path(__file__).resolve().parents[1]
    code = '''
import runpy, sys
from pathlib import Path
nvr_dir, outside = Path(sys.argv[1]), Path(sys.argv[1]).parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != outside]
app = runpy.run_path(str(nvr_dir / 'main.py'), run_name='nvr_script')
from NVR.config import Settings
app['create_app'](Settings(_env_file=None, discovery_enabled=False,
                           nvr_public_rtsp_base_url='rtsp://tower.example:8554'))
leaked = sorted(name for name, module in list(sys.modules.items())
                if (f := getattr(module, '__file__', None))
                and Path(f).resolve().is_relative_to(outside)
                and not Path(f).resolve().is_relative_to(nvr_dir))
assert not leaked, leaked
'''
    subprocess.run([sys.executable, '-c', code, str(nvr_dir)], check=True, cwd=tmp_path)


def test_relay_requires_public_origin():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, nvr_public_rtsp_base_url='')
    with pytest.raises(ValidationError):
        Settings(_env_file=None, nvr_public_rtsp_base_url='http://tower.example/stream')


def test_relay_sweep_repoint_retry_and_proxy(tmp_path):
    async def run():
        settings = settings_for(tmp_path)
        db = Database(settings.minipc_db_path)
        await db.init()
        paths, remote, mutations = {}, {}, []
        fail = [False]
        def mtx(request):
            if fail[0]:
                return httpx.Response(503, text='secret upstream details')
            path = request.url.path.rsplit('/', 1)[-1]
            if request.method == 'GET':
                return httpx.Response(200, json=paths[path]) if path in paths else httpx.Response(404)
            paths[path] = json.loads(request.content)
            mutations.append((request.method, path))
            return httpx.Response(200)
        def jetson_handler(request):
            if request.method == 'GET':
                return httpx.Response(200, json={'cameras': list(remote.values())})
            body = json.loads(request.content)
            key = body['config']['camera_uuid'] if request.method == 'POST' else request.url.path.rsplit('/', 1)[-1]
            remote.setdefault(key, {'camera_uuid': key}).update(body)
            return httpx.Response(200, json=remote[key])
        adapter = DiscoveryAdapter(settings)
        source = SourceDTO(source_identity='serial:CAM01', kind='direct', host='192.0.2.10',
                           rtsp_port=554, source_url='rtsp://192.0.2.10:554/Streaming/Channels/102', serial_number='CAM01')
        def scan_result():
            return ScanResultDTO(candidates=[CandidateDTO(sources=[source])])
        adapter.scan = AsyncMock(side_effect=scan_result)
        relay = MediaRelay(settings, db.session_factory, adapter, transport=httpx.MockTransport(mtx))
        jetson = JetsonClient(settings, transport=httpx.MockTransport(jetson_handler))
        adoption = AdoptionService(db.session_factory, jetson, relay)
        verifier = SimpleNamespace(verify=AsyncMock(return_value=VerifyResult(True)))
        sweep = SweepService(settings, db.session_factory, adapter, verifier, relay, adoption)
        try:
            fail[0] = True
            failed = await sweep.run_once(force=True)
            assert not remote
            assert failed['roster'][0]['relay_state'] == 'error'
            assert 'secret upstream' not in json.dumps(failed)
            fail[0] = False
            first = await sweep.run_once(force=True)
            row = first['roster'][0]
            public = relay.public_url(row['identity'])
            local = relay.jetson_url(row['identity'])
            assert row['source_url'] == public
            assert row['relay_state'] == 'configured'
            assert 'sources' not in row
            assert paths[relay.path(row['identity'])]['source'] == 'rtsp://camera-admin:camera-secret@192.0.2.10:554/Streaming/Channels/102'
            assert next(iter(remote.values()))['source_url'] == local
            assert 'camera-secret' not in json.dumps(first)
            await sweep.run_once(force=True)
            assert len(mutations) == 2  # live substream and recording main stream
            source = source.model_copy(update={'host': '192.0.2.11', 'source_url': 'rtsp://192.0.2.11:554/Streaming/Channels/102'})
            moved = await sweep.run_once(force=True)
            assert moved['roster'][0]['source_url'] == public
            assert moved['ip_changes'][0]['source_url'] == public
            assert mutations[-1][0] == 'PATCH'
            paths.clear()
            await sweep.run_once(force=True)
            assert mutations[-1][0] == 'POST'
            app = FastAPI()
            app.state.relay, app.state.jetson, app.state.sweep = relay, jetson, sweep
            app.include_router(router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://nvr') as client:
                response = await client.get('/cameras')
                assert response.json()['cameras'][0]['source_url'] == public
                key = row['camera_uuid']
                response = await client.patch(f'/cameras/{key}', json={'source_url': public, 'config': {'fps': 5}})
                assert response.json()['source_url'] == public
                assert remote[key]['source_url'] == local
                assert remote[key]['config']['fps'] == 5
                response = await client.get('/discovery/report')
                assert response.json()['roster'][0]['source_url'] == public
            assert paths[relay.path(row['identity'])]['source'].endswith('192.0.2.11:554/Streaming/Channels/102')
        finally:
            adapter.close()
            await relay.close()
            await jetson.close()
            await db.dispose()
    asyncio.run(run())


def test_media_api_failure_does_not_expose_credentials(tmp_path):
    async def run():
        def failure(request):
            raise httpx.ConnectError('rtsp://secret:password@camera', request=request)
        relay = MediaRelay(settings_for(tmp_path), None, None, transport=httpx.MockTransport(failure))
        try:
            with pytest.raises(RelayError, match='unavailable') as error:
                await relay.ensure('camera', 'rtsp://secret:password@camera/Streaming/Channels/102')
            assert 'password' not in str(error.value)
        finally:
            await relay.close()
    asyncio.run(run())
