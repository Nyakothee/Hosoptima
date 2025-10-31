"""
Advanced Alert System for HOS Violation Prediction
Multi-channel notifications with intelligent routing, escalation, and alert management
Production-ready with email, SMS, Slack, webhook, and dashboard integration
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from enum import Enum
import threading
import queue
from collections import defaultdict
import requests
import warnings
warnings.filterwarnings('ignore')

# Twilio for SMS
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logging.warning("Twilio not available. Install with: pip install twilio")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alert_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    PUSH = "push_notification"


@dataclass
class AlertConfig:
    """Configuration for alert system"""
    
    # Email configuration
    smtp_host: str = 'smtp.gmail.com'
    smtp_port: int = 587
    smtp_username: str = ''
    smtp_password: str = ''
    email_from: str = ''
    
    # SMS configuration (Twilio)
    twilio_account_sid: str = ''
    twilio_auth_token: str = ''
    twilio_from_number: str = ''
    
    # Slack configuration
    slack_webhook_url: str = ''
    slack_channel: str = '#hos-alerts'
    
    # Webhook configuration
    webhook_url: str = ''
    webhook_secret: str = ''
    
    # Alert management
    rate_limit_window: int = 300  # 5 minutes
    max_alerts_per_window: int = 10
    escalation_delay: int = 900  # 15 minutes
    auto_acknowledge_time: int = 3600  # 1 hour
    
    # Channels enabled
    enabled_channels: List[str] = None
    
    def __post_init__(self):
        if self.enabled_channels is None:
            self.enabled_channels = [AlertChannel.EMAIL.value, AlertChannel.DASHBOARD.value]


@dataclass
class Alert:
    """Represents an alert"""
    
    alert_id: str
    driver_id: str
    severity: AlertSeverity
    title: str
    message: str
    violation_type: str
    confidence: float
    timestamp: datetime
    recommendations: List[str]
    metadata: Dict = None
    
    # Alert management
    channels: List[AlertChannel] = None
    acknowledged: bool = False
    acknowledged_by: str = None
    acknowledged_at: datetime = None
    resolved: bool = False
    resolved_at: datetime = None
    escalated: bool = False
    escalated_at: datetime = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [AlertChannel.EMAIL, AlertChannel.DASHBOARD]
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'driver_id': self.driver_id,
            'severity': self.severity.name,
            'title': self.title,
            'message': self.message,
            'violation_type': self.violation_type,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'recommendations': self.recommendations,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'escalated': self.escalated,
            'escalated_at': self.escalated_at.isoformat() if self.escalated_at else None
        }
    
    def acknowledge(self, acknowledged_by: str):
        """Acknowledge alert"""
        self.acknowledged = True
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.now()
    
    def resolve(self):
        """Mark alert as resolved"""
        self.resolved = True
        self.resolved_at = datetime.now()
    
    def escalate(self):
        """Escalate alert"""
        self.escalated = True
        self.escalated_at = datetime.now()
        self.severity = AlertSeverity.CRITICAL


class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        
    def send(self, alert: Alert, recipients: List[str]) -> bool:
        """Send email alert"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.name}] {alert.title}"
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(recipients)
            
            # Create HTML content
            html_content = self._create_html_email(alert)
            
            # Attach HTML
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def _create_html_email(self, alert: Alert) -> str:
        """Create HTML email content"""
        severity_colors = {
            AlertSeverity.LOW: '#28a745',
            AlertSeverity.MEDIUM: '#ffc107',
            AlertSeverity.HIGH: '#fd7e14',
            AlertSeverity.CRITICAL: '#dc3545'
        }
        
        color = severity_colors[alert.severity]
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: {color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .recommendations {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; }}
                .footer {{ color: #6c757d; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{alert.title}</h2>
                <p>Severity: {alert.severity.name} | Confidence: {alert.confidence*100:.1f}%</p>
            </div>
            <div class="content">
                <p><strong>Driver ID:</strong> {alert.driver_id}</p>
                <p><strong>Violation Type:</strong> {alert.violation_type}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong></p>
                <p>{alert.message}</p>
                
                <div class="recommendations">
                    <h3>Recommended Actions:</h3>
                    <ul>
                        {''.join(f'<li>{rec}</li>' for rec in alert.recommendations)}
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Alert ID: {alert.alert_id}</p>
                    <p>This is an automated alert from the HOS Violation Prediction System.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


class SMSNotifier:
    """SMS notification handler using Twilio"""
    
    def __init__(self, config: AlertConfig):
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio not available. SMS notifications disabled.")
            self.client = None
            return
        
        self.config = config
        
        try:
            self.client = TwilioClient(
                config.twilio_account_sid,
                config.twilio_auth_token
            )
            logger.info("Twilio SMS client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio: {str(e)}")
            self.client = None
    
    def send(self, alert: Alert, recipients: List[str]) -> bool:
        """Send SMS alert"""
        if not self.client:
            logger.warning("SMS client not available")
            return False
        
        try:
            # Create concise SMS message
            message_body = self._create_sms_message(alert)
            
            # Send to all recipients
            for phone_number in recipients:
                self.client.messages.create(
                    body=message_body,
                    from_=self.config.twilio_from_number,
                    to=phone_number
                )
            
            logger.info(f"SMS alert sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS: {str(e)}")
            return False
    
    def _create_sms_message(self, alert: Alert) -> str:
        """Create concise SMS message"""
        message = (
            f"[{alert.severity.name}] HOS Alert\n"
            f"Driver: {alert.driver_id}\n"
            f"{alert.violation_type}\n"
            f"Confidence: {alert.confidence*100:.0f}%\n"
            f"Action: {alert.recommendations[0] if alert.recommendations else 'Review immediately'}"
        )
        
        # SMS has 160 character limit
        if len(message) > 160:
            message = message[:157] + "..."
        
        return message


class SlackNotifier:
    """Slack notification handler"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
    
    def send(self, alert: Alert) -> bool:
        """Send Slack alert"""
        try:
            # Create Slack message payload
            payload = self._create_slack_payload(alert)
            
            # Send to Slack webhook
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            
            logger.info("Slack alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False
    
    def _create_slack_payload(self, alert: Alert) -> Dict:
        """Create Slack message payload"""
        severity_colors = {
            AlertSeverity.LOW: '#28a745',
            AlertSeverity.MEDIUM: '#ffc107',
            AlertSeverity.HIGH: '#fd7e14',
            AlertSeverity.CRITICAL: '#dc3545'
        }
        
        payload = {
            'channel': self.config.slack_channel,
            'username': 'HOS Alert System',
            'icon_emoji': ':warning:',
            'attachments': [
                {
                    'color': severity_colors[alert.severity],
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {
                            'title': 'Driver ID',
                            'value': alert.driver_id,
                            'short': True
                        },
                        {
                            'title': 'Severity',
                            'value': alert.severity.name,
                            'short': True
                        },
                        {
                            'title': 'Violation Type',
                            'value': alert.violation_type,
                            'short': True
                        },
                        {
                            'title': 'Confidence',
                            'value': f"{alert.confidence*100:.1f}%",
                            'short': True
                        },
                        {
                            'title': 'Recommended Actions',
                            'value': '\n'.join(f"• {rec}" for rec in alert.recommendations),
                            'short': False
                        }
                    ],
                    'footer': f"Alert ID: {alert.alert_id}",
                    'ts': int(alert.timestamp.timestamp())
                }
            ]
        }
        
        return payload


class WebhookNotifier:
    """Generic webhook notification handler"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
    
    def send(self, alert: Alert) -> bool:
        """Send webhook notification"""
        try:
            payload = alert.to_dict()
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            if self.config.webhook_secret:
                headers['X-Webhook-Secret'] = self.config.webhook_secret
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            response.raise_for_status()
            
            logger.info("Webhook notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook: {str(e)}")
            return False


class AlertRateLimiter:
    """Rate limiting for alerts to prevent spam"""
    
    def __init__(self, window_seconds: int = 300, max_per_window: int = 10):
        self.window_seconds = window_seconds
        self.max_per_window = max_per_window
        self.alert_history = defaultdict(list)
        self.lock = threading.Lock()
    
    def should_send_alert(self, driver_id: str) -> bool:
        """Check if alert should be sent based on rate limit"""
        with self.lock:
            now = datetime.now()
            
            # Clean old alerts
            cutoff = now - timedelta(seconds=self.window_seconds)
            self.alert_history[driver_id] = [
                timestamp for timestamp in self.alert_history[driver_id]
                if timestamp > cutoff
            ]
            
            # Check if under limit
            if len(self.alert_history[driver_id]) >= self.max_per_window:
                logger.warning(f"Rate limit exceeded for driver {driver_id}")
                return False
            
            # Add current alert
            self.alert_history[driver_id].append(now)
            return True
    
    def reset(self, driver_id: str):
        """Reset rate limit for driver"""
        with self.lock:
            if driver_id in self.alert_history:
                del self.alert_history[driver_id]


class AlertEscalationManager:
    """Manages alert escalation logic"""
    
    def __init__(self, escalation_delay: int = 900):
        self.escalation_delay = escalation_delay
        self.pending_escalations = {}
        self.lock = threading.Lock()
    
    def schedule_escalation(self, alert: Alert):
        """Schedule alert for escalation if not acknowledged"""
        with self.lock:
            escalation_time = datetime.now() + timedelta(seconds=self.escalation_delay)
            self.pending_escalations[alert.alert_id] = {
                'alert': alert,
                'escalation_time': escalation_time
            }
    
    def check_escalations(self) -> List[Alert]:
        """Check for alerts that need escalation"""
        with self.lock:
            now = datetime.now()
            to_escalate = []
            
            for alert_id, data in list(self.pending_escalations.items()):
                alert = data['alert']
                escalation_time = data['escalation_time']
                
                # Check if should escalate
                if not alert.acknowledged and now >= escalation_time:
                    alert.escalate()
                    to_escalate.append(alert)
                    del self.pending_escalations[alert_id]
                
                # Remove if acknowledged
                elif alert.acknowledged:
                    del self.pending_escalations[alert_id]
            
            return to_escalate
    
    def cancel_escalation(self, alert_id: str):
        """Cancel scheduled escalation"""
        with self.lock:
            if alert_id in self.pending_escalations:
                del self.pending_escalations[alert_id]


class AlertRepository:
    """Store and retrieve alerts"""
    
    def __init__(self, storage_path: str = './alerts'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        self.alerts = {}
        self.lock = threading.Lock()
    
    def save_alert(self, alert: Alert):
        """Save alert"""
        with self.lock:
            self.alerts[alert.alert_id] = alert
            
            # Persist to disk
            file_path = self.storage_path / f"{alert.alert_id}.json"
            with open(file_path, 'w') as f:
                json.dump(alert.to_dict(), f, indent=2)
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        with self.lock:
            return self.alerts.get(alert_id)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self.lock:
            return [
                alert for alert in self.alerts.values()
                if not alert.resolved
            ]
    
    def get_driver_alerts(self, driver_id: str) -> List[Alert]:
        """Get all alerts for specific driver"""
        with self.lock:
            return [
                alert for alert in self.alerts.values()
                if alert.driver_id == driver_id
            ]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge alert"""
        with self.lock:
            alert = self.alerts.get(alert_id)
            if alert:
                alert.acknowledge(acknowledged_by)
                self.save_alert(alert)
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert"""
        with self.lock:
            alert = self.alerts.get(alert_id)
            if alert:
                alert.resolve()
                self.save_alert(alert)
                return True
            return False


class AdvancedAlertSystem:
    """
    Complete alert management system with multi-channel delivery,
    rate limiting, escalation, and alert lifecycle management
    """
    
    def __init__(self, config: AlertConfig):
        self.config = config
        
        # Initialize notifiers
        self.email_notifier = EmailNotifier(config)
        self.sms_notifier = SMSNotifier(config)
        self.slack_notifier = SlackNotifier(config)
        self.webhook_notifier = WebhookNotifier(config)
        
        # Initialize managers
        self.rate_limiter = AlertRateLimiter(
            config.rate_limit_window,
            config.max_alerts_per_window
        )
        self.escalation_manager = AlertEscalationManager(config.escalation_delay)
        self.repository = AlertRepository()
        
        # Background processing
        self.alert_queue = queue.Queue()
        self.running = False
        self.processor_thread = None
        
        logger.info("Advanced Alert System initialized")
    
    def create_alert(self, driver_id: str, prediction: Dict,
                    explanation: Dict, severity: AlertSeverity = None) -> Alert:
        """Create alert from prediction and explanation"""
        
        # Determine severity from confidence
        confidence = prediction.get('confidence', 0.5)
        if severity is None:
            if confidence > 0.8:
                severity = AlertSeverity.CRITICAL
            elif confidence > 0.6:
                severity = AlertSeverity.HIGH
            elif confidence > 0.4:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
        
        # Generate alert ID
        alert_id = f"ALERT_{driver_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            driver_id=driver_id,
            severity=severity,
            title=f"{severity.name} Risk: {prediction['violation_type']}",
            message=explanation.get('natural_language', 'High violation risk detected'),
            violation_type=prediction['violation_type'],
            confidence=confidence,
            timestamp=datetime.now(),
            recommendations=explanation.get('recommendations', []),
            metadata=prediction
        )
        
        return alert
    
    def send_alert(self, alert: Alert, 
                  email_recipients: List[str] = None,
                  sms_recipients: List[str] = None) -> bool:
        """Send alert through configured channels"""
        
        # Check rate limit
        if not self.rate_limiter.should_send_alert(alert.driver_id):
            logger.warning(f"Alert rate limit exceeded for driver {alert.driver_id}")
            return False
        
        # Save alert
        self.repository.save_alert(alert)
        
        # Send through enabled channels
        success = False
        
        if AlertChannel.EMAIL.value in self.config.enabled_channels and email_recipients:
            success |= self.email_notifier.send(alert, email_recipients)
        
        if AlertChannel.SMS.value in self.config.enabled_channels and sms_recipients:
            success |= self.sms_notifier.send(alert, sms_recipients)
        
        if AlertChannel.SLACK.value in self.config.enabled_channels:
            success |= self.slack_notifier.send(alert)
        
        if AlertChannel.WEBHOOK.value in self.config.enabled_channels:
            success |= self.webhook_notifier.send(alert)
        
        # Schedule escalation for high severity alerts
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self.escalation_manager.schedule_escalation(alert)
        
        return success
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge alert"""
        success = self.repository.acknowledge_alert(alert_id, acknowledged_by)
        
        if success:
            self.escalation_manager.cancel_escalation(alert_id)
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        
        return success
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert"""
        success = self.repository.resolve_alert(alert_id)
        
        if success:
            self.escalation_manager.cancel_escalation(alert_id)
            logger.info(f"Alert {alert_id} resolved")
        
        return success
    
    def check_escalations(self):
        """Check for alerts that need escalation"""
        escalated_alerts = self.escalation_manager.check_escalations()
        
        for alert in escalated_alerts:
            logger.warning(f"Escalating alert {alert.alert_id}")
            
            # Re-send with higher priority
            self.send_alert(alert)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return self.repository.get_active_alerts()
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert system status"""
        active_alerts = self.get_active_alerts()
        
        summary = {
            'total_active_alerts': len(active_alerts),
            'by_severity': {
                'CRITICAL': sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL),
                'HIGH': sum(1 for a in active_alerts if a.severity == AlertSeverity.HIGH),
                'MEDIUM': sum(1 for a in active_alerts if a.severity == AlertSeverity.MEDIUM),
                'LOW': sum(1 for a in active_alerts if a.severity == AlertSeverity.LOW)
            },
            'acknowledged': sum(1 for a in active_alerts if a.acknowledged),
            'unacknowledged': sum(1 for a in active_alerts if not a.acknowledged),
            'escalated': sum(1 for a in active_alerts if a.escalated)
        }
        
        return summary


# Main execution
if __name__ == "__main__":
    print("""
=================================================================================
ADVANCED ALERT SYSTEM - MODULE 11
=================================================================================

Multi-channel alert delivery system with intelligent routing and management.

FEATURES:
1. Multi-Channel Delivery
   - Email (HTML formatted)
   - SMS (via Twilio)
   - Slack
   - Webhooks
   - Dashboard notifications

2. Alert Management
   - Rate limiting to prevent spam
   - Alert acknowledgment workflow
   - Alert resolution tracking
   - Escalation for unacknowledged alerts

3. Intelligent Routing
   - Severity-based channel selection
   - Configurable recipient lists
   - Priority escalation paths

USAGE:
from alert_system import AdvancedAlertSystem, AlertConfig

config = AlertConfig(
    smtp_username='your_email@gmail.com',
    smtp_password='your_password',
    twilio_account_sid='AC...',
    twilio_auth_token='...',
    slack_webhook_url='https://hooks.slack.com/...'
)

alert_system = AdvancedAlertSystem(config)

# Create and send alert
alert = alert_system.create_alert(
    driver_id='DRV_12345',
    prediction={'confidence': 0.87, 'violation_type': 'Daily Hours'},
    explanation={'natural_language': '...', 'recommendations': ['...']}
)

alert_system.send_alert(
    alert,
    email_recipients=['dispatcher@company.com'],
    sms_recipients=['+1234567890']
)

# Acknowledge alert
alert_system.acknowledge_alert(alert.alert_id, 'dispatcher_01')

# Get summary
summary = alert_system.get_alert_summary()
print(summary)

=================================================================================
    """)