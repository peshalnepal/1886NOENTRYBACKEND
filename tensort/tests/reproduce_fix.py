
import asyncio
import threading
import time
import sys
from unittest.mock import MagicMock, patch

# Mock libraries not available in this environment
sys.modules['cv2'] = MagicMock()
sys.modules['database'] = MagicMock()
sys.modules['database_orm'] = MagicMock()
sys.modules['trt_infer'] = MagicMock()

# Mock Config
class MockConfig:
    def __init__(self):
        self.camera_uuid = "cam1"
        self.channel_id = "chan1"
        self.rtsp_url = "rtsp://mock"
        self.gst_latency_ms = 200
        self.rtsp_transport = "tcp"
        self.decode_backend = "opencv"
        self.resize = None
        self.reconnect_base_ms = 100
        self.reconnect_max_ms = 1000
        self.sample_fps = 10
        self.emit_format = "raw"
        self.detection_enabled = True
        self.enabled = True

# Patch VideoChannelConfig before importing channel
with patch('channels.channel_config.VideoChannelConfig', MockConfig):
    # Now import the modules to test
    from pipeline import Broadcaster, SimpleInferencePipeline
    from channels.channel import VideoChannel

async def test_broadcaster():
    print("Testing Broadcaster...")
    b = Broadcaster()
    q1 = await b.subscribe()
    q2 = await b.subscribe()

    msg = {"test": "data"}
    b.broadcast(msg)          # synchronous — no task spawn on the result path

    m1 = await q1.get()
    m2 = await q2.get()

    assert m1 == msg
    assert m2 == msg
    print("Broadcaster passed.")

async def test_video_channel_stop():
    print("Testing VideoChannel stop...")
    cfg = MockConfig()
    ch = VideoChannel(cfg)

    # Mock cv2.VideoCapture
    mock_cap = MagicMock()
    # Simulate blocking read
    def blocking_read():
        time.sleep(2) # longer than default timeout
        return True, None
    
    mock_cap.read.side_effect = blocking_read
    mock_cap.isOpened.return_value = True
    
    with patch('cv2.VideoCapture', return_value=mock_cap):
        # Start channel in background
        asyncio.create_task(ch.stream().__anext__())
        
        # Give it time to start and block in read
        await asyncio.sleep(0.5)
        
        # Verify cap is set
        with ch._cap_lock:
            assert ch._cap is not None
        
        print("Channel running, calling stop...")
        start_time = time.time()
        await ch.stop()
        end_time = time.time()
        
        # Verify release called
        mock_cap.release.assert_called()
        
        # Check duration
        duration = end_time - start_time
        print(f"Stop took {duration:.2f}s")
        assert duration < 1.5, "Stop took too long, block was not interrupted"
        print("VideoChannel stop passed.")

async def main():
    await test_broadcaster()
    await test_video_channel_stop()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
