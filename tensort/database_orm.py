# database_orm.py - Simplified ORM models for Jetson Nano
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CameraConfig(Base):
    """
    Camera configuration table for Jetson Nano.
    
    This stores all camera configurations received from the Azure backend,
    allowing the Jetson to persist camera settings across restarts.
    
    Fields:
        id: Auto-incrementing primary key
        channel_id: Unique channel identifier (UUID string)
        camera_uuid: Camera UUID from Azure backend
        user_id: User ID from Azure backend (for multi-tenant support)
        source_url: RTSP stream URL for the camera
        config_json: JSON blob containing all channel configuration parameters
        created_at: Timestamp when camera was first added
        updated_at: Timestamp when camera config was last updated
    """
    __tablename__ = "camera_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(String(64), unique=True, nullable=False, index=True)
    camera_uuid = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    source_url = Column(Text, nullable=False)
    config_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<CameraConfig(id={self.id}, camera_uuid={self.camera_uuid}, channel_id={self.channel_id})>"
    
    def to_dict(self):
        """Convert to dictionary for easy serialization."""
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "camera_uuid": self.camera_uuid,
            "user_id": self.user_id,
            "source_url": self.source_url,
            "config_json": self.config_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
