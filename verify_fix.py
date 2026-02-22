import asyncio
import uuid
import logging
from unittest.mock import MagicMock, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession
from application.services.manager import Manager
from application.repositories.channel_repository import ChannelRepository
from core.database_orm import Device, Camera, CameraDevice
from domain.events import ChannelRemoveEvent

# Mock database session
class MockAsyncSession(AsyncMock):
    async def execute(self, statement):
        mock_result = MagicMock()
        mock_result.scalars().all.return_value = []
        mock_result.scalars().first.return_value = None
        mock_result.scalar_one_or_none.return_value = None
        return mock_result

    async def commit(self):
        pass
    
    async def delete(self, obj):
        pass
    
    async def flush(self):
        pass

async def test_remove_channel():
    print("Testing _remove_channel...")
    
    # Setup mocks
    db = MockAsyncSession()
    manager = Manager(lambda: db)
    manager._edge = AsyncMock()
    manager._webrtc = AsyncMock()
    manager.channel_repo = AsyncMock()
    
    # Test data
    cam_uuid = uuid.uuid4()
    pid = uuid.uuid4()
    device1 = Device(device_uuid=uuid.uuid4(), device_url="http://dev1:8080")
    device2 = Device(device_uuid=uuid.uuid4(), device_url="http://dev2:8080")
    
    # Mock repository returns
    # get_camera_full returns (camera, config, pipeline_id)
    mock_camera = Camera(camera_uuid=cam_uuid, camera_code="test_code")
    manager.channel_repo.get_camera_full.return_value = (mock_camera, None, pid)
    
    # get_associated_devices returns list of devices
    manager.channel_repo.get_associated_devices.return_value = [device1, device2]

    # Create event
    ev = ChannelRemoveEvent(channel_id=cam_uuid, event_type="Remove_Channel")
    
    # Run _remove_channel
    await manager._remove_channel(db, pid=pid, ev=ev)
    
    # Verify calls
    assert manager._edge.delete_camera.call_count == 2
    # Check calls with specific arguments
    manager._edge.delete_camera.assert_any_call(device_url="http://dev1:8080", camera_uuid=str(cam_uuid))
    manager._edge.delete_camera.assert_any_call(device_url="http://dev2:8080", camera_uuid=str(cam_uuid))
    
    print("✓ _remove_channel verified: delete called for all devices")

async def test_cleanup_device_resources():
    print("Testing cleanup_device_resources...")
    
    # Setup mocks
    db = MockAsyncSession()
    manager = Manager(lambda: db)
    manager._edge = AsyncMock()
    
    # Setup Manager._get_device mock (because it's not injected, it's a method)
    # Since _get_device is internal, we can mock it or mock the db execution inside it.
    # Easier to mock the method if we can, but Manager is the class under test.
    # Let's mock db execution to return what we need.
    
    device_uuid = uuid.uuid4()
    mock_device = Device(device_uuid=device_uuid, device_url="http://device_to_delete:8080")
    
    # We need to mock _get_device or db.execute to return the device
    # But _get_device does a query. Let's monkeypatch _get_device effectively.
    manager._get_device = AsyncMock(return_value=mock_device)
    
    # Mock the execute for fetching cameras
    mock_result = MagicMock()
    cam1 = Camera(camera_uuid=uuid.uuid4())
    cam2 = Camera(camera_uuid=uuid.uuid4())
    mock_result.scalars().all.return_value = [cam1, cam2]
    # db.execute is an async method.
    # We replace the method on the instance with a new AsyncMock that returns our result.
    db.execute = AsyncMock(return_value=mock_result)
    
    # Run cleanup
    await manager.cleanup_device_resources(db, device_uuid=device_uuid)
    
    # Verify
    assert manager._edge.delete_camera.call_count == 2
    manager._edge.delete_camera.assert_any_call(device_url="http://device_to_delete:8080", camera_uuid=str(cam1.camera_uuid))
    manager._edge.delete_camera.assert_any_call(device_url="http://device_to_delete:8080", camera_uuid=str(cam2.camera_uuid))
    
    print("✓ cleanup_device_resources verified: delete called for all cameras on device")

if __name__ == "__main__":
    asyncio.run(test_remove_channel())
    asyncio.run(test_cleanup_device_resources())
