from __future__ import annotations
from typing import Any, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field

class Template(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: UUID
    configs: List[Any] = Field(default_factory=list)

    model_cfg: Optional[Any] = Field(default=None, alias="model_config")
