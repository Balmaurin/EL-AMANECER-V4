from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SystemMetrics(BaseModel):
    """SystemMetrics database model"""

    metric_name: str
    value: float
    timestamp: datetime

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
