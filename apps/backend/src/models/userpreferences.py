from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class UserPreferences(BaseModel):
    """UserPreferences database model"""

    user_id: str
    preferences: dict
    theme: str

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
