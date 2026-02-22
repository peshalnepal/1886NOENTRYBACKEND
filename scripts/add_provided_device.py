import asyncio
import uuid
from core.database import async_engine
from core.database_orm import Device
from sqlalchemy.ext.asyncio import async_sessionmaker

async def add_device():
    SessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)
    async with SessionLocal() as db:
        device_url = "http://initialsecurity.dvrlists.com:19030"
        name = "Initial Security Device"
        
        # Check if exists
        from sqlalchemy import select
        q = select(Device).where(Device.device_url == device_url)
        existing = (await db.execute(q)).scalar_one_or_none()
        
        if existing:
            print(f"Device already exists: {existing.device_uuid}")
            return

        dev = Device(
            user_id=1,
            device_url=device_url,
            name=name,
            device_code=f"dev-{uuid.uuid4().hex[:6]}",
            is_enabled=True
        )
        db.add(dev)
        await db.commit()
        print(f"Added device: {dev.device_uuid}")

if __name__ == "__main__":
    asyncio.run(add_device())
