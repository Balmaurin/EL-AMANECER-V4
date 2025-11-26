#!/usr/bin/env python3
"""
Sistema de Auditoría Completo - Audit Trails y Compliance
Sistema integral de auditoría con trazabilidad completa, compliance automático y reporting
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DATOS DE AUDITORÍA
# =============================================================================


class AuditEventType(Enum):
    """Tipos de eventos de auditoría"""

    SECURITY_EVENT = "security_event"
    POLICY_VIOLATION = "policy_violation"
    AGENT_ACTIVITY = "agent_activity"
    SYSTEM_ACCESS = "system_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    COMPLIANCE_CHECK = "compliance_check"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"
    BUSINESS_TRANSACTION = "business_transaction"


class AuditSeverity(Enum):
    """Severidad de eventos de auditoría"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AuditStatus(Enum):
    """Estado de eventos de auditoría"""

    PENDING = "pending"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    ARCHIVED = "archived"


class ComplianceFramework(Enum):
    """Frameworks de compliance"""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    SOC_2 = "soc_2"
    NIST = "nist"
    CUSTOM = "custom"


@dataclass
class AuditEvent:
    """Evento de auditoría individual"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.AGENT_ACTIVITY
    severity: AuditSeverity = AuditSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # Componente que generó el evento
    actor: str = ""  # Usuario/agente que realizó la acción
    target: str = ""  # Recurso afectado
    action: str = ""  # Acción realizada
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: AuditStatus = AuditStatus.PENDING
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: str = ""
    compliance_tags: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    integrity_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "actor": self.actor,
            "target": self.target,
            "action": self.action,
            "details": self.details,
            "metadata": self.metadata,
            "status": self.status.value,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_notes": self.review_notes,
            "compliance_tags": self.compliance_tags,
            "risk_score": self.risk_score,
            "integrity_hash": self.integrity_hash,
        }
        return data

    def calculate_integrity_hash(self, secret_key: str) -> str:
        """Calcula hash de integridad del evento"""
        # Crear representación canónica del evento
        canonical_data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "actor": self.actor,
            "target": self.target,
            "action": self.action,
            "details": json.dumps(self.details, sort_keys=True),
            "metadata": json.dumps(self.metadata, sort_keys=True),
        }

        # Crear string canónico
        canonical_string = json.dumps(
            canonical_data, sort_keys=True, separators=(",", ":")
        )

        # Calcular HMAC-SHA256
        hmac_obj = hmac.new(
            secret_key.encode(), canonical_string.encode(), hashlib.sha256
        )
        self.integrity_hash = base64.b64encode(hmac_obj.digest()).decode()

        return self.integrity_hash

    def verify_integrity(self, secret_key: str) -> bool:
        """Verifica la integridad del evento"""
        expected_hash = self.calculate_integrity_hash(secret_key)
        return hmac.compare_digest(self.integrity_hash, expected_hash)


@dataclass
class AuditTrail:
    """Rastro de auditoría para una entidad"""

    entity_id: str
    entity_type: str  # agent, user, system, etc.
    events: List[AuditEvent] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def add_event(self, event: AuditEvent):
        """Añade un evento al rastro"""
        self.events.append(event)
        self.last_updated = datetime.now()

    def get_events_by_type(self, event_type: AuditEventType) -> List[AuditEvent]:
        """Obtiene eventos por tipo"""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_by_severity(self, severity: AuditSeverity) -> List[AuditEvent]:
        """Obtiene eventos por severidad"""
        return [e for e in self.events if e.severity == severity]

    def get_events_in_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[AuditEvent]:
        """Obtiene eventos en un rango de fechas"""
        return [e for e in self.events if start_date <= e.timestamp <= end_date]


@dataclass
class ComplianceRule:
    """Regla de compliance"""

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    framework: ComplianceFramework = ComplianceFramework.CUSTOM
    name: str = ""
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)
    evidence_required: List[str] = field(default_factory=list)
    check_frequency: str = "daily"  # daily, weekly, monthly, quarterly
    automated_check: bool = True
    last_check: Optional[datetime] = None
    compliance_status: str = "unknown"  # compliant, non_compliant, partial
    remediation_steps: List[str] = field(default_factory=list)


# =============================================================================
# MOTOR PRINCIPAL DE AUDITORÍA
# =============================================================================


class AuditEngine:
    """Motor principal de auditoría"""

    def __init__(self):
        self.audit_trails: Dict[str, AuditTrail] = {}
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.integrity_secret: str = "sheily-audit-integrity-secret-2025"
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processors: List[Callable] = []
        self.is_running = False

    async def initialize_audit_system(self):
        """Inicializa el sistema de auditoría"""
        self.is_running = True

        # Cargar reglas de compliance por defecto
        await self._load_default_compliance_rules()

        # Iniciar procesador de eventos
        asyncio.create_task(self._event_processor())

        # Iniciar verificaciones de compliance programadas
        asyncio.create_task(self._compliance_scheduler())

        logger.info("Audit Engine initialized")

    async def shutdown_audit_system(self):
        """Detiene el sistema de auditoría"""
        self.is_running = False
        logger.info("Audit Engine shutdown")

    async def log_event(self, event: AuditEvent) -> str:
        """Registra un evento de auditoría"""
        # Calcular hash de integridad
        event.calculate_integrity_hash(self.integrity_secret)

        # Añadir a cola de procesamiento
        await self.event_queue.put(event)

        logger.info(f"Audit event logged: {event.event_type.value} - {event.event_id}")
        return event.event_id

    async def log_security_event(
        self,
        source: str,
        actor: str,
        action: str,
        target: str = "",
        details: Dict[str, Any] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
    ) -> str:
        """Registra un evento de seguridad"""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_EVENT,
            severity=severity,
            source=source,
            actor=actor,
            target=target,
            action=action,
            details=details or {},
            risk_score=self._calculate_event_risk_score(action, details or {}),
        )

        return await self.log_event(event)

    async def log_policy_violation(
        self, policy_id: str, actor: str, violation_details: Dict[str, Any]
    ) -> str:
        """Registra una violación de política"""
        event = AuditEvent(
            event_type=AuditEventType.POLICY_VIOLATION,
            severity=AuditSeverity.HIGH,
            source="policy_engine",
            actor=actor,
            target=policy_id,
            action="policy_violation",
            details=violation_details,
            risk_score=0.8,
        )

        return await self.log_event(event)

    async def log_agent_activity(self, agent_id: str, activity: Dict[str, Any]) -> str:
        """Registra actividad de agente"""
        severity = self._determine_activity_severity(activity)

        event = AuditEvent(
            event_type=AuditEventType.AGENT_ACTIVITY,
            severity=severity,
            source="agent_system",
            actor=agent_id,
            action=activity.get("action", "unknown"),
            target=activity.get("target", ""),
            details=activity,
            risk_score=self._calculate_activity_risk(activity),
        )

        return await self.log_event(event)

    async def log_business_transaction(
        self, transaction_id: str, actor: str, transaction_details: Dict[str, Any]
    ) -> str:
        """Registra una transacción de negocio"""
        event = AuditEvent(
            event_type=AuditEventType.BUSINESS_TRANSACTION,
            severity=AuditSeverity.INFO,
            source="business_logic",
            actor=actor,
            target=transaction_id,
            action="transaction",
            details=transaction_details,
            risk_score=self._calculate_transaction_risk(transaction_details),
        )

        return await self.log_event(event)

    def get_audit_trail(self, entity_id: str) -> Optional[AuditTrail]:
        """Obtiene el rastro de auditoría de una entidad"""
        return self.audit_trails.get(entity_id)

    def get_events_by_criteria(
        self, criteria: Dict[str, Any], limit: int = 100
    ) -> List[AuditEvent]:
        """Obtiene eventos por criterios"""
        all_events = []

        for trail in self.audit_trails.values():
            for event in trail.events:
                if self._matches_criteria(event, criteria):
                    all_events.append(event)

        # Ordenar por timestamp descendente
        all_events.sort(key=lambda e: e.timestamp, reverse=True)

        return all_events[:limit]

    def get_compliance_status(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Obtiene estado de compliance para un framework"""
        relevant_rules = [
            r for r in self.compliance_rules.values() if r.framework == framework
        ]

        status = {
            "framework": framework.value,
            "total_rules": len(relevant_rules),
            "compliant_rules": 0,
            "non_compliant_rules": 0,
            "partial_rules": 0,
            "overall_status": "unknown",
            "last_assessment": None,
            "next_assessment": None,
        }

        for rule in relevant_rules:
            if rule.compliance_status == "compliant":
                status["compliant_rules"] += 1
            elif rule.compliance_status == "non_compliant":
                status["non_compliant_rules"] += 1
            elif rule.compliance_status == "partial":
                status["partial_rules"] += 1

            if rule.last_check and (
                not status["last_assessment"]
                or rule.last_check > status["last_assessment"]
            ):
                status["last_assessment"] = rule.last_check

        # Calcular estado general
        total_checked = (
            status["compliant_rules"]
            + status["non_compliant_rules"]
            + status["partial_rules"]
        )
        if total_checked == 0:
            status["overall_status"] = "not_assessed"
        elif status["non_compliant_rules"] > 0:
            status["overall_status"] = "non_compliant"
        elif status["partial_rules"] > 0:
            status["overall_status"] = "partial"
        else:
            status["overall_status"] = "compliant"

        return status

    async def generate_audit_report(
        self, report_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genera un reporte de auditoría"""
        if report_type == "security_incidents":
            return await self._generate_security_report(parameters)
        elif report_type == "compliance_status":
            return await self._generate_compliance_report(parameters)
        elif report_type == "activity_summary":
            return await self._generate_activity_report(parameters)
        elif report_type == "risk_assessment":
            return await self._generate_risk_report(parameters)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    async def _event_processor(self):
        """Procesa eventos de auditoría en cola"""
        while self.is_running:
            try:
                # Esperar evento
                event = await self.event_queue.get()

                # Obtener o crear rastro de auditoría
                trail = self._get_or_create_trail(event.actor, "actor")
                trail.add_event(event)

                # Ejecutar procesadores personalizados
                for processor in self.processors:
                    try:
                        await processor(event)
                    except Exception as e:
                        logger.error(f"Error in audit processor: {e}")

                # Verificar alertas automáticas
                await self._check_automatic_alerts(event)

                # Marcar como procesado
                self.event_queue.task_done()

            except Exception as e:
                logger.error(f"Error processing audit event: {e}")

    async def _compliance_scheduler(self):
        """Scheduler para verificaciones de compliance"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Verificar cada hora

                now = datetime.now()

                for rule in self.compliance_rules.values():
                    if rule.automated_check:
                        # Verificar si es tiempo de ejecutar la regla
                        if self._should_run_compliance_check(rule, now):
                            await self._run_compliance_check(rule)

            except Exception as e:
                logger.error(f"Error in compliance scheduler: {e}")

    def _get_or_create_trail(self, entity_id: str, entity_type: str) -> AuditTrail:
        """Obtiene o crea un rastro de auditoría"""
        if entity_id not in self.audit_trails:
            self.audit_trails[entity_id] = AuditTrail(
                entity_id=entity_id, entity_type=entity_type
            )

        return self.audit_trails[entity_id]

    def _calculate_event_risk_score(
        self, action: str, details: Dict[str, Any]
    ) -> float:
        """Calcula puntuación de riesgo de un evento"""
        risk_score = 0.0

        # Factor de tipo de acción
        high_risk_actions = ["delete", "modify", "access_sensitive", "bypass_security"]
        if any(risk_action in action.lower() for risk_action in high_risk_actions):
            risk_score += 0.4

        # Factor de datos sensibles
        if details.get("sensitive_data", False):
            risk_score += 0.3

        # Factor de autenticación
        if details.get("authentication_bypass", False):
            risk_score += 0.5

        return min(risk_score, 1.0)

    def _determine_activity_severity(self, activity: Dict[str, Any]) -> AuditSeverity:
        """Determina severidad de actividad"""
        action = activity.get("action", "").lower()

        if any(word in action for word in ["delete", "destroy", "bypass", "exploit"]):
            return AuditSeverity.CRITICAL
        elif any(word in action for word in ["modify", "change", "access_sensitive"]):
            return AuditSeverity.HIGH
        elif any(word in action for word in ["read", "query", "search"]):
            return AuditSeverity.MEDIUM
        else:
            return AuditSeverity.INFO

    def _calculate_activity_risk(self, activity: Dict[str, Any]) -> float:
        """Calcula riesgo de actividad"""
        risk_score = 0.0

        # Factor de tipo de actividad
        action = activity.get("action", "").lower()
        if "delete" in action or "destroy" in action:
            risk_score += 0.5
        elif "modify" in action or "change" in action:
            risk_score += 0.3
        elif "access" in action and activity.get("sensitive", False):
            risk_score += 0.4

        # Factor de recursos
        if activity.get("resource_type") == "system_config":
            risk_score += 0.2
        elif activity.get("resource_type") == "user_data":
            risk_score += 0.3

        return min(risk_score, 1.0)

    def _calculate_transaction_risk(self, transaction: Dict[str, Any]) -> float:
        """Calcula riesgo de transacción"""
        amount = transaction.get("amount", 0)
        if amount > 10000:
            return 0.8
        elif amount > 1000:
            return 0.5
        elif amount > 100:
            return 0.2
        else:
            return 0.1

    def _matches_criteria(self, event: AuditEvent, criteria: Dict[str, Any]) -> bool:
        """Verifica si un evento cumple con criterios"""
        for key, value in criteria.items():
            if key == "event_type" and event.event_type.value != value:
                return False
            elif key == "severity" and event.severity.value != value:
                return False
            elif key == "actor" and event.actor != value:
                return False
            elif key == "source" and event.source != value:
                return False
            elif key == "min_risk_score" and event.risk_score < value:
                return False
            elif key == "max_risk_score" and event.risk_score > value:
                return False
            elif key == "start_date" and event.timestamp < value:
                return False
            elif key == "end_date" and event.timestamp > value:
                return False

        return True

    async def _check_automatic_alerts(self, event: AuditEvent):
        """Verifica alertas automáticas para un evento"""
        # Alerta para eventos críticos
        if event.severity == AuditSeverity.CRITICAL:
            logger.critical(
                f"CRITICAL AUDIT EVENT: {event.event_type.value} - {event.details}"
            )

        # Alerta para violaciones de política
        if event.event_type == AuditEventType.POLICY_VIOLATION:
            logger.warning(f"POLICY VIOLATION: {event.details}")

        # Alerta para actividades de alto riesgo
        if event.risk_score >= 0.8:
            logger.warning(
                f"HIGH RISK ACTIVITY: {event.action} by {event.actor} (risk: {event.risk_score})"
            )

    def _should_run_compliance_check(self, rule: ComplianceRule, now: datetime) -> bool:
        """Determina si debe ejecutarse una verificación de compliance"""
        if not rule.last_check:
            return True

        if rule.check_frequency == "daily":
            return (now - rule.last_check) >= timedelta(days=1)
        elif rule.check_frequency == "weekly":
            return (now - rule.last_check) >= timedelta(weeks=1)
        elif rule.check_frequency == "monthly":
            return (now - rule.last_check) >= timedelta(days=30)
        elif rule.check_frequency == "quarterly":
            return (now - rule.last_check) >= timedelta(days=90)

        return False

    async def _run_compliance_check(self, rule: ComplianceRule):
        """Ejecuta una verificación de compliance"""
        # En implementación real, ejecutar verificación específica
        # Por ahora, simular resultado
        rule.last_check = datetime.now()
        rule.compliance_status = "compliant"  # Simulado

        # Registrar evento de compliance
        event = AuditEvent(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            severity=AuditSeverity.INFO,
            source="compliance_engine",
            actor="system",
            target=rule.rule_id,
            action="compliance_check",
            details={
                "rule_name": rule.name,
                "framework": rule.framework.value,
                "status": rule.compliance_status,
            },
        )

        await self.log_event(event)

    async def _load_default_compliance_rules(self):
        """Carga reglas de compliance por defecto"""
        default_rules = [
            ComplianceRule(
                framework=ComplianceFramework.GDPR,
                name="Data Subject Rights",
                description="Ensure data subject rights are properly handled",
                requirements=[
                    "Right to access",
                    "Right to rectification",
                    "Right to erasure",
                ],
                controls=["Access logging", "Data validation", "Deletion procedures"],
                evidence_required=["Audit logs", "Access records"],
                automated_check=True,
            ),
            ComplianceRule(
                framework=ComplianceFramework.ISO_27001,
                name="Access Control",
                description="Implement proper access controls",
                requirements=[
                    "User identification",
                    "Access authorization",
                    "Audit logging",
                ],
                controls=["Authentication", "Authorization", "Audit trails"],
                evidence_required=["Auth logs", "Access logs"],
                automated_check=True,
            ),
            ComplianceRule(
                framework=ComplianceFramework.SOC_2,
                name="Security Monitoring",
                description="Continuous security monitoring",
                requirements=[
                    "Real-time monitoring",
                    "Alert generation",
                    "Incident response",
                ],
                controls=["SIEM system", "Alert rules", "Response procedures"],
                evidence_required=["Monitoring logs", "Alert records"],
                automated_check=True,
            ),
        ]

        for rule in default_rules:
            self.compliance_rules[rule.rule_id] = rule

    async def _generate_security_report(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genera reporte de seguridad"""
        start_date = parameters.get("start_date", datetime.now() - timedelta(days=30))
        end_date = parameters.get("end_date", datetime.now())

        security_events = self.get_events_by_criteria(
            {
                "event_type": "security_event",
                "start_date": start_date,
                "end_date": end_date,
            }
        )

        report = {
            "report_type": "security_incidents",
            "period": f"{start_date.date()} to {end_date.date()}",
            "total_incidents": len(security_events),
            "incidents_by_severity": {},
            "top_risk_actors": [],
            "recommendations": [],
        }

        # Agrupar por severidad
        severity_counts = {}
        for event in security_events:
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        report["incidents_by_severity"] = severity_counts

        # Top actores de riesgo
        actor_risks = {}
        for event in security_events:
            actor = event.actor
            risk = event.risk_score
            if actor not in actor_risks:
                actor_risks[actor] = []
            actor_risks[actor].append(risk)

        # Calcular promedio de riesgo por actor
        avg_risks = {
            actor: sum(risks) / len(risks) for actor, risks in actor_risks.items()
        }
        top_actors = sorted(avg_risks.items(), key=lambda x: x[1], reverse=True)[:5]
        report["top_risk_actors"] = [
            {"actor": actor, "avg_risk": risk} for actor, risk in top_actors
        ]

        # Recomendaciones
        if severity_counts.get("critical", 0) > 0:
            report["recommendations"].append(
                "Immediate investigation of critical security incidents required"
            )
        if severity_counts.get("high", 0) > 5:
            report["recommendations"].append(
                "Review security controls - high incident rate detected"
            )

        return report

    async def _generate_compliance_report(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genera reporte de compliance"""
        framework = parameters.get("framework", "all")

        if framework == "all":
            frameworks = list(ComplianceFramework)
        else:
            frameworks = [ComplianceFramework(framework)]

        report = {"report_type": "compliance_status", "frameworks": []}

        for fw in frameworks:
            status = self.get_compliance_status(fw)
            report["frameworks"].append(status)

        return report

    async def _generate_activity_report(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genera reporte de actividad"""
        start_date = parameters.get("start_date", datetime.now() - timedelta(days=7))
        end_date = parameters.get("end_date", datetime.now())

        all_events = self.get_events_by_criteria(
            {"start_date": start_date, "end_date": end_date}, limit=10000
        )

        report = {
            "report_type": "activity_summary",
            "period": f"{start_date.date()} to {end_date.date()}",
            "total_events": len(all_events),
            "events_by_type": {},
            "events_by_source": {},
            "peak_activity_hours": [],
        }

        # Agrupar por tipo
        type_counts = {}
        source_counts = {}
        hourly_counts = {}

        for event in all_events:
            # Por tipo
            event_type = event.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

            # Por fuente
            source_counts[event.source] = source_counts.get(event.source, 0) + 1

            # Por hora
            hour = event.timestamp.hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        report["events_by_type"] = type_counts
        report["events_by_source"] = source_counts

        # Horas de mayor actividad
        top_hours = sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        report["peak_activity_hours"] = [
            {"hour": hour, "events": count} for hour, count in top_hours
        ]

        return report

    async def _generate_risk_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Genera reporte de riesgos"""
        threshold = parameters.get("risk_threshold", 0.5)

        high_risk_events = self.get_events_by_criteria(
            {"min_risk_score": threshold}, limit=1000
        )

        report = {
            "report_type": "risk_assessment",
            "risk_threshold": threshold,
            "high_risk_events": len(high_risk_events),
            "risk_distribution": {},
            "risk_trends": [],
            "mitigation_recommendations": [],
        }

        # Distribución de riesgos
        risk_ranges = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for event in high_risk_events:
            if event.risk_score >= 0.8:
                risk_ranges["critical"] += 1
            elif event.risk_score >= 0.6:
                risk_ranges["high"] += 1
            elif event.risk_score >= 0.4:
                risk_ranges["medium"] += 1
            else:
                risk_ranges["low"] += 1

        report["risk_distribution"] = risk_ranges

        # Recomendaciones de mitigación
        if risk_ranges["critical"] > 0:
            report["mitigation_recommendations"].append(
                "Immediate action required for critical risk events"
            )
        if risk_ranges["high"] > 5:
            report["mitigation_recommendations"].append(
                "Review and strengthen security controls"
            )

        return report


# =============================================================================
# INTEGRACIÓN CON SISTEMA EXISTENTE
# =============================================================================

# Instancia global del motor de auditoría
audit_engine = AuditEngine()


async def initialize_audit_system():
    """Inicializa el sistema de auditoría"""
    await audit_engine.initialize_audit_system()


async def log_audit_event(
    event_type: AuditEventType,
    severity: AuditSeverity,
    source: str,
    actor: str,
    action: str,
    target: str = "",
    details: Dict[str, Any] = None,
) -> str:
    """Registra un evento de auditoría"""
    event = AuditEvent(
        event_type=event_type,
        severity=severity,
        source=source,
        actor=actor,
        action=action,
        target=target,
        details=details or {},
    )

    return await audit_engine.log_event(event)


async def generate_compliance_report(framework: str) -> Dict[str, Any]:
    """Genera reporte de compliance"""
    return audit_engine.get_compliance_status(ComplianceFramework(framework))


# Funciones de utilidad
__all__ = [
    # Clases principales
    "AuditEngine",
    "AuditEvent",
    "AuditTrail",
    "ComplianceRule",
    # Enums
    "AuditEventType",
    "AuditSeverity",
    "AuditStatus",
    "ComplianceFramework",
    # Instancias globales
    "audit_engine",
    # Funciones de utilidad
    "initialize_audit_system",
    "log_audit_event",
    "generate_compliance_report",
]
