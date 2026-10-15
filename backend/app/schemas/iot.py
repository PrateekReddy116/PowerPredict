from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.enums import DataOrigin


class IotStatusOut(BaseModel):
    status: Literal["LIVE", "SIMULATED", "OFFLINE"]
    broker_connected: bool
    last_message_at: datetime | None = None
    last_data_origin: DataOrigin | None = None
    last_building_id: str | None = None
    topic: str
    mqtt_enabled: bool
    note: str = Field(
        description="SIMULATED is the honest demo label; LIVE is reserved for real_iot."
    )
