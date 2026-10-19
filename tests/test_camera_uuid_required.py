"""Cloud creation requires the identity already assigned by NVR/Jetson."""
import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from core.database_orm import Camera
from core.schemas import CameraCreateSchema, SiteCameraCreate
from application.services.manager.controllers.adopt import CameraAdopter
from application.services.manager.controllers.channel import ChannelController


@pytest.mark.parametrize('schema', [CameraCreateSchema, SiteCameraCreate])
@pytest.mark.parametrize('value', [None, '', 'invalid'])
def test_create_api_requires_valid_camera_uuid(schema, value):
    payload = dict(device_uuid=uuid.uuid4(), source_url='rtsp://minipc:8554/nvr-one')
    if schema is CameraCreateSchema:
        payload['site_uuid'] = uuid.uuid4()
    if value is not None:
        payload['camera_uuid'] = value
    with pytest.raises(ValidationError):
        schema(**payload)


def test_database_has_no_camera_uuid_generator():
    assert Camera.__table__.c.camera_uuid.default is None
    assert Camera.__table__.c.camera_uuid.server_default is None
    assert not Camera.__table__.c.camera_uuid.nullable


@pytest.mark.parametrize('value', [None, '', 'invalid'])
def test_internal_channel_creation_rejects_missing_or_invalid_uuid(value):
    controller = ChannelController(None, None)
    event = SimpleNamespace(configs=dict(camera_uuid=value, site_uuid=uuid.uuid4(),
                                         source_url='rtsp://minipc:8554/nvr-one'))
    with pytest.raises(ValueError, match='camera_uuid'):
        asyncio.run(controller.add_channel(None, pid=uuid.uuid4(), ev=event,
                    user_id=1, camera_code_prefix='cam', model_pipeline=None))


def test_adopter_passes_edge_uuid_to_channel_creation():
    camera_uuid = uuid.uuid4()
    update = AsyncMock(return_value=SimpleNamespace(cameras=[SimpleNamespace(camera_uuid=camera_uuid)]))
    adopter = CameraAdopter(None, update)
    result = asyncio.run(adopter._create_camera(
        entry={'camera_uuid': str(camera_uuid), 'source_url': 'rtsp://minipc:8554/nvr-one'},
        site_uuid=uuid.uuid4(), device_uuid=uuid.uuid4(), user_id=1))
    assert result == camera_uuid
    event = update.call_args.args[1][0]
    assert event.configs['camera_uuid'] == camera_uuid
