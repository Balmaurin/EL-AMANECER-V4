"""
Agent-to-Agent (A2A) Protocol
=============================
Standardized protocol for communication between autonomous agents.
"""

from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ERROR = "error"

class A2AMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")
    sender: str
    recipient: str
    msg_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    reply_to: Optional[str] = None

class A2AProtocol:
    """
    Protocol handler for Agent Communication
    """
    @staticmethod
    def create_message(sender: str, recipient: str, content: Dict[str, Any], msg_type: MessageType = MessageType.REQUEST) -> A2AMessage:
        return A2AMessage(
            sender=sender,
            recipient=recipient,
            msg_type=msg_type,
            content=content
        )

    @staticmethod
    def validate_message(message: Dict[str, Any]) -> bool:
        try:
            A2AMessage(**message)
            return True
        except:
            return False
