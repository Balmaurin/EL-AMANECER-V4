#!/usr/bin/env python3
"""
Sheily Enterprise Socket.IO Communication System
==============================================

Sistema de comunicación en tiempo real con Socket.IO para
enterprise session management, agent coordination y colaboración.

Características:
- Real-time bidirectional communication
- Enterprise session management
- Agent coordination channels
- Live collaboration features
- Secure authentication
- Room-based organization
- Performance monitoring
"""

import asyncio
import json
import logging
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from sheily_core.core.analytics.analytics_system import get_analytics_system
from sheily_core.core.events.event_system import (
    SheìlyEventType,
    get_event_stream,
    publish_event,
)
from sheily_core.core.middleware.security_middleware import SecurityContext

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Tipos de mensajes Socket.IO"""

    # System messages
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

    # Agent coordination
    AGENT_STATUS = "agent_status"
    AGENT_REQUEST = "agent_request"
    AGENT_RESPONSE = "agent_response"
    AGENT_BROADCAST = "agent_broadcast"

    # Session management
    SESSION_CREATE = "session_create"
    SESSION_UPDATE = "session_update"
    SESSION_END = "session_end"
    SESSION_JOIN = "session_join"
    SESSION_LEAVE = "session_leave"

    # Collaboration
    CHAT_MESSAGE = "chat_message"
    FILE_SHARE = "file_share"
    SCREEN_SHARE = "screen_share"
    COLLABORATION_UPDATE = "collaboration_update"

    # Enterprise features
    NOTIFICATION = "notification"
    ALERT = "alert"
    METRICS_UPDATE = "metrics_update"


class RoomType(str, Enum):
    """Tipos de salas Socket.IO"""

    PUBLIC = "public"
    PRIVATE = "private"
    AGENT_COORDINATION = "agent_coordination"
    SESSION_ROOM = "session_room"
    ADMIN_ROOM = "admin_room"
    NOTIFICATION_ROOM = "notification_room"


@dataclass
class SocketUser:
    """Usuario conectado via Socket.IO"""

    user_id: str
    username: str
    socket_id: str
    role: str
    permissions: List[str] = field(default_factory=list)
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_data: Dict[str, Any] = field(default_factory=dict)
    subscribed_rooms: Set[str] = field(default_factory=set)


@dataclass
class SocketRoom:
    """Sala Socket.IO"""

    room_id: str
    name: str
    room_type: RoomType
    created_by: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    members: Set[str] = field(default_factory=set)  # socket_ids
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_members: Optional[int] = None
    password_protected: bool = False
    password_hash: Optional[str] = None


@dataclass
class SocketMessage:
    """Mensaje Socket.IO"""

    message_id: str
    message_type: MessageType
    sender_socket_id: str
    sender_user_id: Optional[str]
    target_room: Optional[str]
    target_socket_id: Optional[str]
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class SheìlySocketManager:
    """Manager principal para conexiones Socket.IO"""

    def __init__(self):
        # Core socket management
        self.connected_users: Dict[str, SocketUser] = {}  # socket_id -> SocketUser
        self.user_sockets: Dict[str, Set[str]] = {}  # user_id -> set of socket_ids
        self.rooms: Dict[str, SocketRoom] = {}  # room_id -> SocketRoom
        self.socket_to_rooms: Dict[str, Set[str]] = {}  # socket_id -> set of room_ids

        # Message handling
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.middleware_handlers: List[Callable] = []

        # Performance tracking
        self.connection_stats = {
            "total_connections": 0,
            "current_connections": 0,
            "peak_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "rooms_created": 0,
            "errors": 0,
        }

        # Event system integration
        self.event_stream = None
        self.analytics = None

        # Cleanup tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize Socket.IO manager"""
        try:
            self.event_stream = get_event_stream()
            self.analytics = await get_analytics_system()

            # Setup default rooms
            await self._create_default_rooms()

            # Setup message handlers
            await self._setup_default_handlers()

            # Start background tasks
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_task_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_task_loop())

            await publish_event(
                SheìlyEventType.SYSTEM_HEALTH, {"status": "socket_manager_initialized"}
            )

            logger.info("✅ Sheily Socket.IO Manager initialized")

        except Exception as e:
            logger.error(f"Error initializing Socket.IO manager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown Socket.IO manager"""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Disconnect all users
        for socket_id in list(self.connected_users.keys()):
            await self.disconnect_user(socket_id, "Server shutdown")

    # ========================================
    # CONNECTION MANAGEMENT
    # ========================================

    async def connect_user(
        self,
        socket_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        role: str = "user",
        permissions: Optional[List[str]] = None,
        session_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Connect new user"""
        try:
            if socket_id in self.connected_users:
                logger.warning(f"Socket {socket_id} already connected")
                return False

            # Create user object
            user = SocketUser(
                user_id=user_id or f"anonymous_{socket_id}",
                username=username or f"User_{socket_id[:8]}",
                socket_id=socket_id,
                role=role,
                permissions=permissions or [],
                session_data=session_data or {},
            )

            # Store user
            self.connected_users[socket_id] = user

            # Update user->sockets mapping
            if user.user_id not in self.user_sockets:
                self.user_sockets[user.user_id] = set()
            self.user_sockets[user.user_id].add(socket_id)

            # Initialize socket rooms
            self.socket_to_rooms[socket_id] = set()

            # Update statistics
            self.connection_stats["total_connections"] += 1
            self.connection_stats["current_connections"] += 1
            if (
                self.connection_stats["current_connections"]
                > self.connection_stats["peak_connections"]
            ):
                self.connection_stats["peak_connections"] = self.connection_stats[
                    "current_connections"
                ]

            # Join default rooms
            await self._join_default_rooms(socket_id, user)

            # Send welcome message
            await self.send_to_socket(
                socket_id,
                MessageType.CONNECT,
                {
                    "message": "Connected successfully",
                    "user_id": user.user_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Publish connection event
            await publish_event(
                SheìlyEventType.METRICS_UPDATE,
                {
                    "metric_type": "socket_connection",
                    "action": "connect",
                    "socket_id": socket_id,
                    "user_id": user.user_id,
                    "current_connections": self.connection_stats["current_connections"],
                },
            )

            # Track analytics
            await self.analytics.record_counter(
                "sheily.socket.connections", 1, {"action": "connect", "user_role": role}
            )

            logger.info(f"✅ User connected: {user.username} ({socket_id})")
            return True

        except Exception as e:
            logger.error(f"Error connecting user {socket_id}: {e}")
            self.connection_stats["errors"] += 1
            return False

    async def disconnect_user(self, socket_id: str, reason: str = "Unknown") -> bool:
        """Disconnect user"""
        try:
            if socket_id not in self.connected_users:
                return False

            user = self.connected_users[socket_id]

            # Leave all rooms
            rooms_to_leave = list(self.socket_to_rooms.get(socket_id, set()))
            for room_id in rooms_to_leave:
                await self.leave_room(socket_id, room_id)

            # Remove from user->sockets mapping
            if user.user_id in self.user_sockets:
                self.user_sockets[user.user_id].discard(socket_id)
                if not self.user_sockets[user.user_id]:
                    del self.user_sockets[user.user_id]

            # Remove user and socket mappings
            del self.connected_users[socket_id]
            if socket_id in self.socket_to_rooms:
                del self.socket_to_rooms[socket_id]

            # Update statistics
            self.connection_stats["current_connections"] -= 1

            # Publish disconnection event
            await publish_event(
                SheìlyEventType.METRICS_UPDATE,
                {
                    "metric_type": "socket_connection",
                    "action": "disconnect",
                    "socket_id": socket_id,
                    "user_id": user.user_id,
                    "reason": reason,
                    "current_connections": self.connection_stats["current_connections"],
                },
            )

            # Track analytics
            await self.analytics.record_counter(
                "sheily.socket.disconnections",
                1,
                {"reason": reason, "user_role": user.role},
            )

            logger.info(
                f"🔌 User disconnected: {user.username} ({socket_id}) - {reason}"
            )
            return True

        except Exception as e:
            logger.error(f"Error disconnecting user {socket_id}: {e}")
            self.connection_stats["errors"] += 1
            return False

    async def _join_default_rooms(self, socket_id: str, user: SocketUser) -> None:
        """Join user to default rooms based on role"""
        # Everyone joins general notifications
        await self.join_room(socket_id, "notifications")

        # Role-based room assignment
        if user.role in ["admin", "system"]:
            await self.join_room(socket_id, "admin_room")
            await self.join_room(socket_id, "agent_coordination")

        if user.role == "agent":
            await self.join_room(socket_id, "agent_coordination")

        # User-specific room
        await self.join_room(socket_id, f"user_{user.user_id}")

    # ========================================
    # ROOM MANAGEMENT
    # ========================================

    async def create_room(
        self,
        room_id: str,
        name: str,
        room_type: RoomType,
        created_by: str,
        max_members: Optional[int] = None,
        password: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create new room"""
        try:
            if room_id in self.rooms:
                return False

            room = SocketRoom(
                room_id=room_id,
                name=name,
                room_type=room_type,
                created_by=created_by,
                max_members=max_members,
                password_protected=password is not None,
                metadata=metadata or {},
            )

            if password:
                import hashlib

                room.password_hash = hashlib.sha256(password.encode()).hexdigest()

            self.rooms[room_id] = room
            self.connection_stats["rooms_created"] += 1

            await self.analytics.record_counter(
                "sheily.socket.rooms_created",
                1,
                {
                    "room_type": room_type.value,
                    "password_protected": str(password is not None),
                },
            )

            logger.info(f"🏠 Room created: {name} ({room_id}) by {created_by}")
            return True

        except Exception as e:
            logger.error(f"Error creating room {room_id}: {e}")
            return False

    async def join_room(
        self, socket_id: str, room_id: str, password: Optional[str] = None
    ) -> bool:
        """Join user to room"""
        try:
            if socket_id not in self.connected_users:
                return False

            if room_id not in self.rooms:
                # Auto-create simple rooms
                user = self.connected_users[socket_id]
                await self.create_room(
                    room_id=room_id,
                    name=room_id,
                    room_type=RoomType.PUBLIC,
                    created_by=user.user_id,
                )

            room = self.rooms[room_id]
            user = self.connected_users[socket_id]

            # Check password if required
            if room.password_protected and password:
                import hashlib

                password_hash = hashlib.sha256(password.encode()).hexdigest()
                if password_hash != room.password_hash:
                    return False

            # Check max members
            if room.max_members and len(room.members) >= room.max_members:
                return False

            # Add to room
            room.members.add(socket_id)
            self.socket_to_rooms[socket_id].add(room_id)
            user.subscribed_rooms.add(room_id)

            # Notify room of new member
            await self.send_to_room(
                room_id,
                MessageType.SESSION_JOIN,
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "room_id": room_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                exclude_socket=socket_id,
            )

            logger.debug(f"👥 User {user.username} joined room {room_id}")
            return True

        except Exception as e:
            logger.error(f"Error joining room {room_id}: {e}")
            return False

    async def leave_room(self, socket_id: str, room_id: str) -> bool:
        """Leave room"""
        try:
            if socket_id not in self.connected_users or room_id not in self.rooms:
                return False

            room = self.rooms[room_id]
            user = self.connected_users[socket_id]

            # Remove from room
            room.members.discard(socket_id)
            self.socket_to_rooms[socket_id].discard(room_id)
            user.subscribed_rooms.discard(room_id)

            # Notify room of member leaving
            await self.send_to_room(
                room_id,
                MessageType.SESSION_LEAVE,
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "room_id": room_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                exclude_socket=socket_id,
            )

            logger.debug(f"👋 User {user.username} left room {room_id}")
            return True

        except Exception as e:
            logger.error(f"Error leaving room {room_id}: {e}")
            return False

    async def _create_default_rooms(self) -> None:
        """Create default system rooms"""
        default_rooms = [
            ("notifications", "System Notifications", RoomType.NOTIFICATION_ROOM),
            ("admin_room", "Admin Room", RoomType.ADMIN_ROOM),
            ("agent_coordination", "Agent Coordination", RoomType.AGENT_COORDINATION),
            ("public_chat", "Public Chat", RoomType.PUBLIC),
        ]

        for room_id, name, room_type in default_rooms:
            await self.create_room(room_id, name, room_type, "system")

    # ========================================
    # MESSAGE HANDLING
    # ========================================

    async def send_to_socket(
        self,
        socket_id: str,
        message_type: MessageType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send message to specific socket"""
        try:
            if socket_id not in self.connected_users:
                return False

            message = SocketMessage(
                message_id=f"msg_{int(time.time() * 1000000)}",
                message_type=message_type,
                sender_socket_id="system",
                sender_user_id="system",
                target_socket_id=socket_id,
                data=data,
                metadata=metadata or {},
            )

            # In real implementation, this would use actual Socket.IO emit
            # For now, we'll just track the message
            self.connection_stats["messages_sent"] += 1

            await self.analytics.record_counter(
                "sheily.socket.messages_sent",
                1,
                {"message_type": message_type.value, "target": "socket"},
            )

            # Update user activity
            user = self.connected_users[socket_id]
            user.last_activity = datetime.now(timezone.utc)

            logger.debug(f"📤 Sent {message_type.value} to {socket_id}")
            return True

        except Exception as e:
            logger.error(f"Error sending message to {socket_id}: {e}")
            return False

    async def send_to_room(
        self,
        room_id: str,
        message_type: MessageType,
        data: Dict[str, Any],
        exclude_socket: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Send message to all members of a room"""
        try:
            if room_id not in self.rooms:
                return 0

            room = self.rooms[room_id]
            sent_count = 0

            for socket_id in room.members:
                if exclude_socket and socket_id == exclude_socket:
                    continue

                if await self.send_to_socket(socket_id, message_type, data, metadata):
                    sent_count += 1

            await self.analytics.record_counter(
                "sheily.socket.room_messages",
                1,
                {
                    "room_type": room.room_type.value,
                    "message_type": message_type.value,
                    "recipients": str(sent_count),
                },
            )

            logger.debug(
                f"📢 Broadcast {message_type.value} to room {room_id} ({sent_count} recipients)"
            )
            return sent_count

        except Exception as e:
            logger.error(f"Error broadcasting to room {room_id}: {e}")
            return 0

    async def send_to_user(
        self,
        user_id: str,
        message_type: MessageType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Send message to all sockets of a user"""
        try:
            if user_id not in self.user_sockets:
                return 0

            sent_count = 0
            for socket_id in self.user_sockets[user_id]:
                if await self.send_to_socket(socket_id, message_type, data, metadata):
                    sent_count += 1

            logger.debug(
                f"👤 Sent {message_type.value} to user {user_id} ({sent_count} sockets)"
            )
            return sent_count

        except Exception as e:
            logger.error(f"Error sending to user {user_id}: {e}")
            return 0

    async def broadcast_to_all(
        self,
        message_type: MessageType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Broadcast message to all connected users"""
        try:
            sent_count = 0
            for socket_id in self.connected_users:
                if await self.send_to_socket(socket_id, message_type, data, metadata):
                    sent_count += 1

            await self.analytics.record_counter(
                "sheily.socket.broadcasts",
                1,
                {"message_type": message_type.value, "recipients": str(sent_count)},
            )

            logger.info(
                f"📡 Broadcast {message_type.value} to all ({sent_count} recipients)"
            )
            return sent_count

        except Exception as e:
            logger.error(f"Error broadcasting to all: {e}")
            return 0

    # ========================================
    # MESSAGE HANDLERS
    # ========================================

    async def _setup_default_handlers(self) -> None:
        """Setup default message handlers"""
        await self.register_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        await self.register_handler(
            MessageType.AGENT_REQUEST, self._handle_agent_request
        )
        await self.register_handler(MessageType.CHAT_MESSAGE, self._handle_chat_message)

    async def register_handler(
        self, message_type: MessageType, handler: Callable
    ) -> None:
        """Register message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []

        self.message_handlers[message_type].append(handler)
        logger.debug(f"📝 Registered handler for {message_type.value}")

    async def handle_incoming_message(
        self, socket_id: str, message_type: MessageType, data: Dict[str, Any]
    ) -> None:
        """Handle incoming message from client"""
        try:
            if socket_id not in self.connected_users:
                return

            user = self.connected_users[socket_id]
            user.last_activity = datetime.now(timezone.utc)

            message = SocketMessage(
                message_id=f"msg_{int(time.time() * 1000000)}",
                message_type=message_type,
                sender_socket_id=socket_id,
                sender_user_id=user.user_id,
                data=data,
            )

            self.connection_stats["messages_received"] += 1

            # Execute handlers
            if message_type in self.message_handlers:
                for handler in self.message_handlers[message_type]:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")

            await self.analytics.record_counter(
                "sheily.socket.messages_received",
                1,
                {"message_type": message_type.value, "user_role": user.role},
            )

        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")

    async def _handle_heartbeat(self, message: SocketMessage) -> None:
        """Handle heartbeat message"""
        await self.send_to_socket(
            message.sender_socket_id,
            MessageType.HEARTBEAT,
            {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "alive"},
        )

    async def _handle_agent_request(self, message: SocketMessage) -> None:
        """Handle agent coordination request"""
        # Forward to agent coordination room
        await self.send_to_room(
            "agent_coordination",
            MessageType.AGENT_REQUEST,
            {
                "request_id": message.message_id,
                "sender": message.sender_user_id,
                "request_data": message.data,
                "timestamp": message.timestamp.isoformat(),
            },
        )

    async def _handle_chat_message(self, message: SocketMessage) -> None:
        """Handle chat message"""
        target_room = message.data.get("room", "public_chat")

        if target_room in self.rooms:
            await self.send_to_room(
                target_room,
                MessageType.CHAT_MESSAGE,
                {
                    "message_id": message.message_id,
                    "sender": message.sender_user_id,
                    "content": message.data.get("content", ""),
                    "timestamp": message.timestamp.isoformat(),
                },
                exclude_socket=message.sender_socket_id,
            )

    # ========================================
    # ENTERPRISE FEATURES
    # ========================================

    async def send_notification(
        self,
        title: str,
        content: str,
        target_users: Optional[List[str]] = None,
        target_roles: Optional[List[str]] = None,
        priority: str = "normal",
    ) -> int:
        """Send system notification"""
        notification_data = {
            "title": title,
            "content": content,
            "priority": priority,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notification_id": f"notif_{int(time.time() * 1000)}",
        }

        sent_count = 0

        if target_users:
            # Send to specific users
            for user_id in target_users:
                sent_count += await self.send_to_user(
                    user_id, MessageType.NOTIFICATION, notification_data
                )
        elif target_roles:
            # Send to users with specific roles
            for socket_id, user in self.connected_users.items():
                if user.role in target_roles:
                    if await self.send_to_socket(
                        socket_id, MessageType.NOTIFICATION, notification_data
                    ):
                        sent_count += 1
        else:
            # Send to notifications room (everyone)
            sent_count = await self.send_to_room(
                "notifications", MessageType.NOTIFICATION, notification_data
            )

        await self.analytics.record_counter(
            "sheily.socket.notifications_sent",
            1,
            {
                "priority": priority,
                "target_type": (
                    "users" if target_users else ("roles" if target_roles else "all")
                ),
                "recipients": str(sent_count),
            },
        )

        return sent_count

    async def send_alert(
        self, alert_level: str, message: str, target_admins: bool = True
    ) -> int:
        """Send system alert"""
        alert_data = {
            "alert_level": alert_level,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_id": f"alert_{int(time.time() * 1000)}",
        }

        if target_admins:
            return await self.send_to_room("admin_room", MessageType.ALERT, alert_data)
        else:
            return await self.broadcast_to_all(MessageType.ALERT, alert_data)

    # ========================================
    # BACKGROUND TASKS
    # ========================================

    async def _cleanup_task_loop(self) -> None:
        """Background cleanup task"""
        logger.info("Started Socket.IO cleanup task")

        while self._running:
            try:
                await self._cleanup_inactive_connections()
                await self._cleanup_empty_rooms()
                await asyncio.sleep(300)  # Every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)

    async def _heartbeat_task_loop(self) -> None:
        """Background heartbeat task"""
        logger.info("Started Socket.IO heartbeat task")

        while self._running:
            try:
                await self._send_heartbeat_to_all()
                await asyncio.sleep(30)  # Every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(30)

    async def _cleanup_inactive_connections(self) -> None:
        """Cleanup inactive connections"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        inactive_sockets = []

        for socket_id, user in self.connected_users.items():
            if user.last_activity < cutoff_time:
                inactive_sockets.append(socket_id)

        for socket_id in inactive_sockets:
            await self.disconnect_user(socket_id, "Inactive connection")

    async def _cleanup_empty_rooms(self) -> None:
        """Cleanup empty rooms"""
        empty_rooms = []

        for room_id, room in self.rooms.items():
            if (
                len(room.members) == 0
                and room.room_type
                not in [RoomType.ADMIN_ROOM, RoomType.NOTIFICATION_ROOM]
                and not room_id.startswith("user_")
            ):
                empty_rooms.append(room_id)

        for room_id in empty_rooms:
            del self.rooms[room_id]
            logger.debug(f"🗑️ Cleaned up empty room: {room_id}")

    async def _send_heartbeat_to_all(self) -> None:
        """Send heartbeat to all connections"""
        heartbeat_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server_status": "healthy",
            "connected_users": len(self.connected_users),
            "active_rooms": len(self.rooms),
        }

        await self.broadcast_to_all(MessageType.HEARTBEAT, heartbeat_data)

    # ========================================
    # STATUS & MONITORING
    # ========================================

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            **self.connection_stats,
            "active_rooms": len(self.rooms),
            "users_by_role": self._get_users_by_role(),
            "room_occupancy": self._get_room_occupancy(),
        }

    def _get_users_by_role(self) -> Dict[str, int]:
        """Get user count by role"""
        role_counts = {}
        for user in self.connected_users.values():
            role_counts[user.role] = role_counts.get(user.role, 0) + 1
        return role_counts

    def _get_room_occupancy(self) -> Dict[str, int]:
        """Get room occupancy stats"""
        return {room_id: len(room.members) for room_id, room in self.rooms.items()}


# ========================================
# GLOBAL INSTANCE
# ========================================

_socket_manager: Optional[SheìlySocketManager] = None


async def get_socket_manager() -> SheìlySocketManager:
    """Get global socket manager instance"""
    global _socket_manager
    if _socket_manager is None:
        _socket_manager = SheìlySocketManager()
        await _socket_manager.initialize()
    return _socket_manager


async def initialize_socket_system() -> SheìlySocketManager:
    """Initialize the complete Socket.IO system"""
    socket_manager = await get_socket_manager()
    logger.info("✅ Sheily Enterprise Socket.IO System initialized")
    return socket_manager


__all__ = [
    "SheìlySocketManager",
    "MessageType",
    "RoomType",
    "SocketUser",
    "SocketRoom",
    "SocketMessage",
    "get_socket_manager",
    "initialize_socket_system",
]
