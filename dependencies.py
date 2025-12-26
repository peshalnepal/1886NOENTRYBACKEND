import logging
import os

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

# --- MODIFIED: Import the global factory function ---
from core.database_orm import User
from core.database import db_manager
from core.security.tokens import decode_access_token
from application.services.manager import Manager

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

# Configure logging
logger = logging.getLogger(__name__)

def get_db():
    """Provides a sync database session from the manager."""
    if not db_manager.SessionLocal:
        raise Exception("Sync database has not been initialized.")
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncSession:
    """Provides an async database session from the manager."""
    if not db_manager.AsyncSessionLocal:
        raise Exception("Async database has not been initialized.")
    async with db_manager.AsyncSessionLocal() as session:
        yield session


# # --- MODIFIED: Use the global instance factory ---
def get_manager(request: Request) -> Manager:
    return request.app.state.manager



async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        token = credentials.credentials
        payload = decode_access_token(token)

        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=401, detail="Could not validate credentials"
            )

        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")

        return user
    except ValueError as e:
        # Catch errors from decode_access_token
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

