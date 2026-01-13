# application/routes/notifications.py

import asyncio
from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from application.services.notification import WebNotificationHub

router = APIRouter()

def _sse(data: str) -> str:
    return f"data: {data}\n\n"

@router.get("/api/notifications/stream")
async def notifications_stream(request: Request):
    hub: WebNotificationHub = request.app.state.notification_hub
    q = await hub.subscribe()

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await q.get()
                yield _sse(msg.model_dump_json())
        finally:
            await hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")
