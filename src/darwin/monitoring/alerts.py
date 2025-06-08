"""
Darwin Alert Management System

This module provides comprehensive alert management capabilities for the Darwin platform,
including alert rules, notification channels, escalation policies, alert aggregation,
and integration with monitoring metrics and health checks.

Features:
- Configurable alert rules with thresholds and conditions
- Multiple notification channels (email, Slack, webhooks)
- Alert escalation policies with timeout handling
- Alert aggregation and deduplication
- Alert history and audit logging
- Integration with metrics and health check systems
- Real-time alert processing and delivery
- Alert acknowledgment and resolution tracking
"""

import asyncio
import hashlib
import logging
import smtplib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Union

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Supported notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DISCORD = "discord"


@dataclass
class AlertRule:
    """Alert rule definition with conditions and thresholds."""

    name: str
    description: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne", "gte", "lte"
    threshold: Union[int, float]
    severity: AlertSeverity
    duration: int = 300  # seconds - how long condition must persist
    channels: List[NotificationChannel] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    # Advanced configuration
    evaluation_interval: int = 60  # seconds
    cooldown_period: int = 3600  # seconds - minimum time between alerts
    max_alerts_per_hour: int = 10
    escalation_timeout: int = 1800  # seconds - time before escalation

    # Internal state
    last_triggered: Optional[datetime] = None
    alert_count: int = 0
    condition_start_time: Optional[datetime] = None

    def __post_init__(self):
        """Initialize rule after creation."""
        self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID for the alert rule."""
        rule_string = (
            f"{self.name}_{self.metric_name}_{self.condition}_{self.threshold}"
        )
        return hashlib.md5(rule_string.encode()).hexdigest()[:8]

    def evaluate_condition(self, current_value: Union[int, float]) -> bool:
        """Evaluate if the current value triggers the alert condition."""
        if self.condition == "gt":
            return current_value > self.threshold
        elif self.condition == "lt":
            return current_value < self.threshold
        elif self.condition == "gte":
            return current_value >= self.threshold
        elif self.condition == "lte":
            return current_value <= self.threshold
        elif self.condition == "eq":
            return current_value == self.threshold
        elif self.condition == "ne":
            return current_value != self.threshold
        else:
            logger.warning(f"Unknown condition: {self.condition}")
            return False

    def should_trigger(self, current_value: Union[int, float]) -> bool:
        """Determine if alert should trigger based on duration and cooldown."""
        now = datetime.now(timezone.utc)

        # Check if rule is enabled
        if not self.enabled:
            return False

        # Check cooldown period
        if self.last_triggered:
            time_since_last = (now - self.last_triggered).total_seconds()
            if time_since_last < self.cooldown_period:
                return False

        # Check alert rate limiting
        if self.alert_count >= self.max_alerts_per_hour:
            return False

        # Evaluate condition
        condition_met = self.evaluate_condition(current_value)

        if condition_met:
            if self.condition_start_time is None:
                self.condition_start_time = now
            else:
                # Check if condition has persisted for required duration
                condition_duration = (now - self.condition_start_time).total_seconds()
                return condition_duration >= self.duration
        else:
            # Reset condition timer if condition is no longer met
            self.condition_start_time = None

        return False

    def trigger(self):
        """Mark rule as triggered."""
        self.last_triggered = datetime.now(timezone.utc)
        self.alert_count += 1
        self.condition_start_time = None  # Reset for next evaluation

    def reset_rate_limit(self):
        """Reset hourly alert count."""
        self.alert_count = 0


@dataclass
class Alert:
    """Individual alert instance."""

    id: str
    rule_name: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolution_note: Optional[str] = None

    # Notification tracking
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    escalation_level: int = 0
    next_escalation_time: Optional[datetime] = None

    def __post_init__(self):
        """Initialize alert after creation."""
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID for the alert."""
        alert_string = (
            f"{self.rule_name}_{self.metric_name}_{self.created_at.timestamp()}"
        )
        return hashlib.md5(alert_string.encode()).hexdigest()[:12]

    def acknowledge(self, user: str = None, note: str = None):
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(timezone.utc)
        self.acknowledged_by = user
        if note:
            self.resolution_note = note

    def resolve(self, user: str = None, note: str = None):
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)
        if note:
            self.resolution_note = note

    def suppress(self):
        """Suppress the alert."""
        self.status = AlertStatus.SUPPRESSED

    def add_notification(
        self,
        channel: NotificationChannel,
        success: bool,
        details: Dict[str, Any] = None,
    ):
        """Record notification attempt."""
        self.notifications_sent.append(
            {
                "channel": channel.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": success,
                "details": details or {},
            }
        )

    def should_escalate(self, escalation_timeout: int) -> bool:
        """Check if alert should be escalated."""
        if self.status != AlertStatus.ACTIVE:
            return False

        if self.next_escalation_time is None:
            self.next_escalation_time = self.created_at + timedelta(
                seconds=escalation_timeout
            )

        return datetime.now(timezone.utc) >= self.next_escalation_time

    def escalate(self, escalation_timeout: int):
        """Escalate the alert to the next level."""
        self.escalation_level += 1
        self.next_escalation_time = datetime.now(timezone.utc) + timedelta(
            seconds=escalation_timeout
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "labels": self.labels,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat()
            if self.acknowledged_at
            else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolution_note": self.resolution_note,
            "escalation_level": self.escalation_level,
            "notifications_count": len(self.notifications_sent),
        }


class NotificationService:
    """Service for sending notifications through various channels."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize notification service with configuration."""
        self.config = config or {}
        self.email_config = self.config.get("email", {})
        self.slack_config = self.config.get("slack", {})
        self.webhook_config = self.config.get("webhook", {})

        logger.info("Notification service initialized")

    async def send_notification(
        self, alert: Alert, channel: NotificationChannel
    ) -> bool:
        """Send notification for an alert through specified channel."""
        try:
            if channel == NotificationChannel.EMAIL:
                return await self._send_email(alert)
            elif channel == NotificationChannel.SLACK:
                return await self._send_slack(alert)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook(alert)
            else:
                logger.warning(f"Notification channel not implemented: {channel}")
                return False
        except Exception as e:
            logger.error(f"Failed to send notification via {channel}: {e}")
            return False

    async def _send_email(self, alert: Alert) -> bool:
        """Send email notification."""
        if not self.email_config:
            logger.warning("Email configuration not provided")
            return False

        try:
            smtp_server = self.email_config.get("smtp_server")
            smtp_port = self.email_config.get("smtp_port", 587)
            username = self.email_config.get("username")
            password = self.email_config.get("password")
            from_email = self.email_config.get("from_email")
            to_emails = self.email_config.get("to_emails", [])

            if not all([smtp_server, username, password, from_email, to_emails]):
                logger.error("Incomplete email configuration")
                return False

            # Create message
            try:
                from email.mime.multipart import MimeMultipart
                from email.mime.text import MimeText

                msg = MimeMultipart()
                msg["From"] = from_email
                msg["To"] = ", ".join(to_emails)
                msg[
                    "Subject"
                ] = f"[{alert.severity.value.upper()}] Darwin Alert: {alert.rule_name}"

                # Email body
                body = self._format_email_body(alert)
                msg.attach(MimeText(body, "html"))

                # Send email
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
                server.quit()
            except ImportError:
                logger.error("Email MIME modules not available")
                return False

            logger.info(f"Email notification sent for alert {alert.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def _format_email_body(self, alert: Alert) -> str:
        """Format email body for alert notification."""
        return f"""
        <html>
        <body>
        <h2>Darwin Platform Alert</h2>
        <table style="border-collapse: collapse; width: 100%;">
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><b>Alert ID:</b></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.id}</td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><b>Rule:</b></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.rule_name}</td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><b>Severity:</b></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.severity.value.upper()}</td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><b>Metric:</b></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.metric_name}</td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><b>Current Value:</b></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.current_value}</td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><b>Threshold:</b></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.threshold}</td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><b>Message:</b></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.message}</td></tr>
        <tr><td style="border: 1px solid #ddd; padding: 8px;"><b>Time:</b></td><td style="border: 1px solid #ddd; padding: 8px;">{alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</td></tr>
        </table>
        <p><a href="http://your-dashboard/alerts/{alert.id}">View Alert Details</a></p>
        </body>
        </html>
        """

    async def _send_slack(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not self.slack_config:
            logger.warning("Slack configuration not provided")
            return False

        if not HTTPX_AVAILABLE:
            logger.error("httpx not available for Slack notifications")
            return False

        try:
            webhook_url = self.slack_config.get("webhook_url")
            channel = self.slack_config.get("channel", "#alerts")

            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False

            # Determine color based on severity
            color_map = {
                AlertSeverity.LOW: "#36a64f",  # green
                AlertSeverity.MEDIUM: "#ff9900",  # orange
                AlertSeverity.HIGH: "#ff0000",  # red
                AlertSeverity.CRITICAL: "#800000",  # dark red
            }

            payload = {
                "channel": channel,
                "username": "Darwin Alerts",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#ff0000"),
                        "title": f"[{alert.severity.value.upper()}] {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Metric",
                                "value": alert.metric_name,
                                "short": True,
                            },
                            {
                                "title": "Current Value",
                                "value": str(alert.current_value),
                                "short": True,
                            },
                            {
                                "title": "Threshold",
                                "value": str(alert.threshold),
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": alert.created_at.strftime(
                                    "%Y-%m-%d %H:%M:%S UTC"
                                ),
                                "short": True,
                            },
                        ],
                        "footer": "Darwin Platform",
                        "ts": int(alert.created_at.timestamp()),
                    }
                ],
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload)
                response.raise_for_status()

            logger.info(f"Slack notification sent for alert {alert.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    async def _send_webhook(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not self.webhook_config:
            logger.warning("Webhook configuration not provided")
            return False

        if not HTTPX_AVAILABLE:
            logger.error("httpx not available for webhook notifications")
            return False

        try:
            url = self.webhook_config.get("url")
            headers = self.webhook_config.get("headers", {})

            if not url:
                logger.error("Webhook URL not configured")
                return False

            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "darwin-platform",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

            logger.info(f"Webhook notification sent for alert {alert.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class AlertManager:
    """Main alert management system."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.config = config or {}
        self.evaluation_interval = self.config.get("evaluation_interval", 60)
        self.alert_retention_days = self.config.get("alert_retention_days", 30)
        self.enable_notifications = self.config.get("enable_notifications", True)

        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self._lock = Lock()

        # Services
        self.notification_service = NotificationService(config.get("notifications", {}))

        # Metrics collector reference (to be set externally)
        self.metrics_collector = None
        self.health_checker = None

        # Rate limiting for alert evaluation
        self.last_rate_limit_reset = datetime.now(timezone.utc)

        logger.info("Alert manager initialized")

    def set_metrics_collector(self, metrics_collector):
        """Set metrics collector for alert evaluation."""
        self.metrics_collector = metrics_collector

    def set_health_checker(self, health_checker):
        """Set health checker for alert evaluation."""
        self.health_checker = health_checker

    def register_rule(self, rule: AlertRule):
        """Register a new alert rule."""
        with self._lock:
            self.rules[rule.id] = rule
            logger.info(f"Registered alert rule: {rule.name} ({rule.id})")

    def create_rule(
        self,
        name: str,
        description: str,
        metric_name: str,
        condition: str,
        threshold: Union[int, float],
        severity: AlertSeverity,
        channels: List[NotificationChannel] = None,
        **kwargs,
    ) -> AlertRule:
        """Create and register a new alert rule."""
        rule = AlertRule(
            name=name,
            description=description,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            channels=channels or [NotificationChannel.EMAIL],
            **kwargs,
        )

        self.register_rule(rule)
        return rule

    def remove_rule(self, rule_id: str):
        """Remove an alert rule."""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID."""
        return self.rules.get(rule_id)

    def get_all_rules(self) -> Dict[str, AlertRule]:
        """Get all alert rules."""
        return self.rules.copy()

    async def evaluate_rules(self):
        """Evaluate all alert rules against current metrics."""
        if not self.metrics_collector:
            logger.warning("No metrics collector configured for alert evaluation")
            return

        # Reset rate limits hourly
        now = datetime.now(timezone.utc)
        if (now - self.last_rate_limit_reset).total_seconds() >= 3600:
            for rule in self.rules.values():
                rule.reset_rate_limit()
            self.last_rate_limit_reset = now

        for rule in self.rules.values():
            await self._evaluate_rule(rule)

    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        try:
            # Get current metric value
            metric = self.metrics_collector.get_metric(rule.metric_name)
            if not metric:
                logger.warning(
                    f"Metric not found for rule {rule.name}: {rule.metric_name}"
                )
                return

            current_value = metric.get_current_value()

            # Check if rule should trigger
            if rule.should_trigger(current_value):
                await self._trigger_alert(rule, current_value)

        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.name}: {e}")

    async def _trigger_alert(self, rule: AlertRule, current_value: Union[int, float]):
        """Trigger a new alert."""
        try:
            # Check for existing active alert for this rule
            existing_alert = None
            for alert in self.active_alerts.values():
                if alert.rule_name == rule.name and alert.status == AlertStatus.ACTIVE:
                    existing_alert = alert
                    break

            if existing_alert:
                logger.debug(f"Alert already active for rule {rule.name}")
                return

            # Create new alert
            alert = Alert(
                id="",  # Will be generated in __post_init__
                rule_name=rule.name,
                metric_name=rule.metric_name,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                message=f"{rule.description} - Current value: {current_value}, Threshold: {rule.threshold}",
                current_value=current_value,
                threshold=rule.threshold,
                labels=rule.labels.copy(),
            )

            # Store alert
            with self._lock:
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)

            # Mark rule as triggered
            rule.trigger()

            # Send notifications
            if self.enable_notifications:
                await self._send_alert_notifications(alert, rule.channels)

            logger.warning(f"Alert triggered: {alert.id} - {rule.name}")

        except Exception as e:
            logger.error(f"Failed to trigger alert for rule {rule.name}: {e}")

    async def _send_alert_notifications(
        self, alert: Alert, channels: List[NotificationChannel]
    ):
        """Send notifications for an alert."""
        for channel in channels:
            try:
                success = await self.notification_service.send_notification(
                    alert, channel
                )
                alert.add_notification(channel, success)

                if success:
                    logger.info(
                        f"Notification sent via {channel.value} for alert {alert.id}"
                    )
                else:
                    logger.error(
                        f"Failed to send notification via {channel.value} for alert {alert.id}"
                    )

            except Exception as e:
                logger.error(f"Error sending notification via {channel.value}: {e}")
                alert.add_notification(channel, False, {"error": str(e)})

    async def acknowledge_alert(
        self, alert_id: str, user: str = None, note: str = None
    ) -> bool:
        """Acknowledge an alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False

        alert.acknowledge(user, note)
        logger.info(f"Alert acknowledged: {alert_id} by {user or 'system'}")
        return True

    async def resolve_alert(
        self, alert_id: str, user: str = None, note: str = None
    ) -> bool:
        """Resolve an alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False

        alert.resolve(user, note)

        # Remove from active alerts
        with self._lock:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]

        logger.info(f"Alert resolved: {alert_id} by {user or 'system'}")
        return True

    async def suppress_alert(self, alert_id: str) -> bool:
        """Suppress an alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False

        alert.suppress()
        logger.info(f"Alert suppressed: {alert_id}")
        return True

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        return [
            alert for alert in self.active_alerts.values() if alert.severity == severity
        ]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_alerts = list(self.active_alerts.values())

        stats = {
            "total_active": len(active_alerts),
            "total_history": len(self.alert_history),
            "by_severity": {
                severity.value: len(
                    [a for a in active_alerts if a.severity == severity]
                )
                for severity in AlertSeverity
            },
            "by_status": {
                status.value: len([a for a in active_alerts if a.status == status])
                for status in AlertStatus
            },
        }

        return stats

    async def process_escalations(self):
        """Process alert escalations."""
        for alert in self.active_alerts.values():
            rule = self.get_rule_by_name(alert.rule_name)
            if not rule:
                continue

            if alert.should_escalate(rule.escalation_timeout):
                await self._escalate_alert(alert, rule)

    def get_rule_by_name(self, rule_name: str) -> Optional[AlertRule]:
        """Get rule by name."""
        for rule in self.rules.values():
            if rule.name == rule_name:
                return rule
        return None

    async def _escalate_alert(self, alert: Alert, rule: AlertRule):
        """Escalate an alert to the next level."""
        try:
            alert.escalate(rule.escalation_timeout)

            # Send escalation notifications
            if self.enable_notifications:
                await self._send_alert_notifications(alert, rule.channels)

            logger.warning(
                f"Alert escalated: {alert.id} to level {alert.escalation_level}"
            )

        except Exception as e:
            logger.error(f"Failed to escalate alert {alert.id}: {e}")

    async def cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(
            days=self.alert_retention_days
        )

        with self._lock:
            # Remove old alerts from history
            self.alert_history = [
                alert
                for alert in self.alert_history
                if alert.created_at > cutoff_date or alert.status == AlertStatus.ACTIVE
            ]

        logger.info(
            f"Cleaned up old alerts (retention: {self.alert_retention_days} days)"
        )

    async def start_periodic_evaluation(self):
        """Start periodic alert rule evaluation."""
        logger.info(
            f"Starting periodic alert evaluation every {self.evaluation_interval} seconds"
        )

        while True:
            try:
                await self.evaluate_rules()
                await self.process_escalations()

                # Cleanup old alerts once per hour
                if int(time.time()) % 3600 == 0:
                    await self.cleanup_old_alerts()

                await asyncio.sleep(self.evaluation_interval)

            except Exception as e:
                logger.error(f"Periodic alert evaluation failed: {e}")
                await asyncio.sleep(self.evaluation_interval)
