"""
Alert handling and notification system for monitoring events.
"""

import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.notification_channels = []

    def add_email_channel(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        recipients: List[str],
    ):
        """Add email notification channel"""
        self.notification_channels.append(
            EmailNotifier(smtp_host, smtp_port, username, password, recipients)
        )

    def add_slack_channel(self, webhook_url: str):
        """Add Slack notification channel"""
        self.notification_channels.append(SlackNotifier(webhook_url))

    def send_alert(
        self,
        alert_name: str,
        severity: str,
        message: str,
        details: Optional[Dict] = None,
    ):
        """Send alert through all configured channels"""
        timestamp = datetime.now().isoformat()
        alert_data = {
            "name": alert_name,
            "severity": severity,
            "message": message,
            "timestamp": timestamp,
            "details": details or {},
        }

        for channel in self.notification_channels:
            try:
                channel.send_notification(alert_data)
            except Exception as e:
                logger.error(
                    f"Failed to send alert through {channel.__class__.__name__}: {e}"
                )


class EmailNotifier:
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        recipients: List[str],
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients

    def send_notification(self, alert_data: Dict[str, Any]):
        """Send email notification"""
        subject = f"RAG Alert: {alert_data['name']} - {alert_data['severity']}"

        body = f"""
        Alert: {alert_data['name']}
        Severity: {alert_data['severity']}
        Time: {alert_data['timestamp']}

        Message: {alert_data['message']}

        Details:
        {self._format_details(alert_data['details'])}
        """

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = ", ".join(self.recipients)

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)

    def _format_details(self, details: Dict) -> str:
        """Format alert details for email body"""
        return "\n".join(f"- {k}: {v}" for k, v in details.items())


class SlackNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_notification(self, alert_data: Dict[str, Any]):
        """Send Slack notification"""
        color = self._severity_to_color(alert_data["severity"])

        attachment = {
            "color": color,
            "title": f"RAG Alert: {alert_data['name']}",
            "text": alert_data["message"],
            "fields": [
                {"title": "Severity", "value": alert_data["severity"], "short": True},
                {"title": "Time", "value": alert_data["timestamp"], "short": True},
            ]
            + [
                {"title": k, "value": str(v), "short": True}
                for k, v in alert_data["details"].items()
            ],
        }

        payload = {"attachments": [attachment]}

        response = requests.post(self.webhook_url, json=payload, timeout=30)
        response.raise_for_status()

    def _severity_to_color(self, severity: str) -> str:
        """Convert severity level to Slack color"""
        return {
            "critical": "#FF0000",
            "error": "#FF9900",
            "warning": "#FFFF00",
            "info": "#00FF00",
        }.get(severity.lower(), "#808080")
