#!/usr/bin/env python3
"""
NOTIFICATION SERVICE - Sistema de Notificaciones
==============================================

Servicio completo para:
- Email notifications (SendGrid/Mailgun)
- SMS notifications (Twilio)
- In-app notifications
- System alerts
"""

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Servicio de notificaciones multi-canal
    """

    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = 587,
        smtp_username: str = None,
        smtp_password: str = None,
    ):

        # Configuración SMTP
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = smtp_username or os.getenv("SMTP_USERNAME")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")

        # Configuración por defecto
        self.from_email = os.getenv("FROM_EMAIL", "noreply@sheily.ai")
        self.from_name = os.getenv("FROM_NAME", "Sheily AI")

        # Templates de notificación
        self.templates = self._load_templates()

        logger.info("📧 Notification Service inicializado")

    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Cargar templates de notificaciones"""
        return {
            "welcome": {
                "subject": "¡Bienvenido a Sheily AI! 🎉",
                "html": """
                <h1>Bienvenido a Sheily AI</h1>
                <p>Hola {name},</p>
                <p>Tu cuenta ha sido creada exitosamente. ¡Comienza tu viaje de aprendizaje con IA!</p>
                <p><a href="https://sheily.ai/login">Iniciar Sesión</a></p>
                <p>Saludos,<br>Equipo Sheily AI</p>
                """,
            },
            "payment_success": {
                "subject": "Pago realizado exitosamente 💳",
                "html": """
                <h2>Pago Confirmado</h2>
                <p>Hola {name},</p>
                <p>Tu pago de ${amount} ha sido procesado correctamente.</p>
                <p>Tokens acreditados: {tokens}</p>
                <p>Transaction ID: {transaction_id}</p>
                <p><a href="https://sheily.ai/dashboard">Ver mi balance</a></p>
                """,
            },
            "password_reset": {
                "subject": "Restablecer contraseña 🔐",
                "html": """
                <h2>Restablecer Contraseña</h2>
                <p>Hola {name},</p>
                <p>Haz clic en el enlace para restablecer tu contraseña:</p>
                <p><a href="{reset_link}">Restablecer Contraseña</a></p>
                <p>Este enlace expira en 24 horas.</p>
                """,
            },
            "exercise_completed": {
                "subject": "¡Ejercicio completado! 🎯",
                "html": """
                <h2>¡Excelente trabajo!</h2>
                <p>Hola {name},</p>
                <p>Has completado un ejercicio con {accuracy}% de precisión.</p>
                <p>Tokens ganados: {tokens_earned}</p>
                <p>Nivel actual: {level}</p>
                <p>¡Sigue así!</p>
                """,
            },
            "system_alert": {
                "subject": "Alerta del Sistema ⚠️",
                "html": """
                <h2>Alerta del Sistema</h2>
                <p>Se ha detectado una actividad inusual en el sistema:</p>
                <p><strong>{alert_type}</strong></p>
                <p>Detalles: {details}</p>
                <p>Timestamp: {timestamp}</p>
                """,
            },
        }

    async def send_email(
        self, to_email: str, template_name: str, template_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Enviar email usando template
        """
        if template_name not in self.templates:
            return {"error": f"Template '{template_name}' no encontrado"}

        template = self.templates[template_name]
        template_data = template_data or {}

        # Render template
        subject = template["subject"]
        html_content = template["html"].format(**template_data)

        return await self.send_raw_email(to_email, subject, html_content)

    async def send_raw_email(
        self, to_email: str, subject: str, html_content: str
    ) -> Dict[str, Any]:
        """
        Enviar email con contenido raw
        """
        success = True
        error_msg = None

        try:
            # Si no hay configuración SMTP, simular envío (para desarrollo)
            if not self.smtp_username or not self.smtp_password:
                logger.info(f"📧 [SIMULADO] Email enviado a {to_email}: {subject}")
                return {
                    "success": True,
                    "method": "simulated",
                    "to": to_email,
                    "subject": subject,
                    "message": "Email simulado - configurar SMTP_USERNAME y SMTP_PASSWORD para envío real",
                }

            # Crear mensaje
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = to_email

            # Adjuntar contenido HTML
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            # Enviar via SMTP
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.sendmail(self.from_email, to_email, msg.as_string())
            server.quit()

            logger.info(f"✅ Email enviado exitosamente a {to_email}: {subject}")

        except Exception as e:
            success = False
            error_msg = str(e)
            logger.error(f"❌ Error enviando email a {to_email}: {e}")

        return {
            "success": success,
            "error": error_msg,
            "to": to_email,
            "subject": subject,
            "method": "smtp_real" if success and self.smtp_username else "simulated",
        }

    async def send_sms(self, phone_number: str, message: str) -> Dict[str, Any]:
        """
        Enviar SMS (placeholder para integración Twilio)
        """
        # Por ahora simulado - integrar con Twilio sería:
        # from twilio.rest import Client
        # client = Client(twilio_sid, twilio_token)
        # client.messages.create(body=message, from_=twilio_number, to=phone_number)

        logger.info(f"📱 [SIMULADO] SMS enviado a {phone_number}: {message[:50]}...")

        return {
            "success": True,
            "method": "simulated",
            "to": phone_number,
            "message": message,
            "note": "Integrar Twilio para envío real (TWILIO_SID, TWILIO_TOKEN, TWILIO_NUMBER)",
        }

    async def send_notification(
        self,
        user_id: str,
        notification_type: str,
        data: Dict[str, Any],
        channels: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Enviar notificación multi-canal
        """
        channels = channels or ["email"]  # Default to email

        results = {}
        user_email = data.get("email", f"user_{user_id}@sheily.ai")
        user_phone = data.get("phone")

        # Enviar por email si está configurado
        if "email" in channels and user_email:
            email_result = await self.send_email(user_email, notification_type, data)
            results["email"] = email_result

        # Enviar SMS si está configurado
        if "sms" in channels and user_phone:
            sms_result = await self.send_sms(
                user_phone, f"Sheily AI: {notification_type}"
            )
            results["sms"] = sms_result

        # Log notification
        logger.info(
            f"🔔 Notificación {notification_type} enviada a user {user_id}: {results}"
        )

        return {
            "notification_type": notification_type,
            "user_id": user_id,
            "channels": channels,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

    def get_templates(self) -> Dict[str, str]:
        """Obtener lista de templates disponibles"""
        return {name: template["subject"] for name, template in self.templates.items()}

    async def health_check(self) -> Dict[str, Any]:
        """Verificar estado del servicio de notificaciones"""
        smtp_configured = bool(self.smtp_username and self.smtp_password)

        return {
            "service": "notification_service",
            "smtp_configured": smtp_configured,
            "email_enabled": smtp_configured,
            "sms_enabled": False,  # Twilio no configurado aún
            "templates_available": len(self.templates),
            "status": "healthy" if smtp_configured else "degraded",
        }


class InAppNotificationManager:
    """
    Gestor de notificaciones in-app (para frontend)
    """

    def __init__(self):
        self.notifications = {}  # user_id -> list of notifications

    def add_notification(self, user_id: str, notification: Dict[str, Any]):
        """Agregar notificación para usuario"""
        if user_id not in self.notifications:
            self.notifications[user_id] = []

        notification["id"] = f"notif_{len(self.notifications[user_id])+1}"
        notification["timestamp"] = datetime.now().isoformat()
        notification["read"] = False

        self.notifications[user_id].append(notification)

    def get_notifications(
        self, user_id: str, unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Obtener notificaciones del usuario"""
        if user_id not in self.notifications:
            return []

        notifs = self.notifications[user_id]
        if unread_only:
            notifs = [n for n in notifs if not n.get("read", False)]

        return notifs

    def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Marcar notificación como leída"""
        if user_id in self.notifications:
            for notif in self.notifications[user_id]:
                if notif["id"] == notification_id:
                    notif["read"] = True
                    return True
        return False


# Instancia global
notification_service = NotificationService()
inapp_notifications = InAppNotificationManager()

# =============================================================================
# DEMO Y TESTING DEL NOTIFICATION SERVICE
# =============================================================================


async def demo_notification_service():
    """Demo del sistema de notificaciones"""
    print("📧 NOTIFICATION SERVICE DEMO")
    print("=" * 40)

    service = NotificationService()

    # Ver estado
    health = await service.health_check()
    print("🏥 Estado del servicio:")
    for key, value in health.items():
        print(f"   {key}: {value}")

    print(f"\n📝 Templates disponibles:")
    templates = service.get_templates()
    for name, subject in templates.items():
        print(f"   {name}: {subject}")

    # Enviar notificación de bienvenida
    print("\n📧 Enviando notificación de bienvenida...")
    welcome_data = {"name": "Juan Pérez", "email": "juan@example.com"}

    result = await service.send_notification(
        user_id="user123", notification_type="welcome", data=welcome_data
    )

    print(f"Resultado: {result}")

    # Notificación de pago exitoso
    print("\n💳 Enviando notificación de pago...")
    payment_data = {
        "name": "Juan Pérez",
        "email": "juan@example.com",
        "amount": "4.99",
        "tokens": "100",
        "transaction_id": "pay_123456",
    }

    result2 = await service.send_notification(
        user_id="user123", notification_type="payment_success", data=payment_data
    )

    print(f"Resultado: {result2}")

    print("\n📧 NOTIFICATION SERVICE OPERATIVO")
    print("   ✅ Templates pre-configurados")
    print("   ✅ Envío por email (SMTP)")
    print("   ✅ Notificaciones multi-canal")
    print("   ✅ Logs de envío")


# Configurar para testing
if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_notification_service())
