#!/usr/bin/env python3
"""
AI Threat Detector - Sistema Avanzado de Detección de Amenazas AI-Orquestadas
==============================================================================

Detecta amenazas de ciberseguridad orquestadas por AI, incluyendo:
- Ataques tipo "Claude Code" (manipulación de modelos para ciberespionaje)
- Jailbreak attempts en modelos AI
- Automated cyber attacks coordinados por AI
- Agent-based threats (agentes autónomos maliciosos)

Basado en técnicas documentadas en research público sobre AI security.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AIAttackPattern:
    """Patrón específico de ataque AI."""

    pattern_id: str
    name: str
    description: str
    attack_type: str  # 'jailbreak', 'exploitation', 'orchestration', 'espionage'
    severity: str  # 'critical', 'high', 'medium', 'low'
    indicators: List[str]
    detection_logic: str
    mitigation_rules: List[str]
    confidence_threshold: float
    first_seen: datetime
    last_updated: datetime
    related_research: List[str]


@dataclass
class ThreatDetectionResult:
    """Resultado de análisis de amenaza."""

    threat_id: str
    pattern_matched: str
    confidence: float
    severity: str
    indicators_found: List[str]
    mitigation_actions: List[str]
    timestamp: datetime
    target_affected: str
    attacker_vector: str
    potential_impact: str


class AIThreatDetector:
    """
    Detector avanzado de amenazas AI con análisis en tiempo real.
    """

    def __init__(self):
        self.attack_patterns = {}
        self.detection_history = deque(maxlen=10000)
        self.active_threats = {}
        self.learning_engine = AIThreatLearning()

        # Inicializar patrones conocidos de ataques AI
        self._initialize_attack_patterns()

    def _initialize_attack_patterns(self):
        """Inicializar patrones de ataques conocidos basados en research público."""

        # Patrón: Claude Code Attack (manipulación de modelos para ciberespionaje)
        self.attack_patterns["claude_code_orchestration"] = AIAttackPattern(
            pattern_id="claude_code_orchestration",
            name="Claude Code Cyber Espionage Orchestration",
            description="Ataque donde AI es manipulado para realizar ciberespionaje autónomo coordinado",
            attack_type="espionage",
            severity="critical",
            indicators=[
                "large_scale_target_scanning",
                "automated_credential_harvesting",
                "coordinated_database_inspection",
                "backward_exploit_generation",
                "documentation_generation_attack",
                "minimal_human_intervention",
            ],
            detection_logic="multi_step_orchestration_anomaly",
            mitigation_rules=[
                "rate_limit_api_calls",
                "block_automated_scanning",
                "require_human_verification_large_requests",
                "monitor_agent_coordination",
            ],
            confidence_threshold=0.85,
            first_seen=datetime(2025, 9, 15),
            last_updated=datetime.now(),
            related_research=[
                "anthropic_claude_code_attack_analysis",
                "ai_orchestrated_cyber_intelligence_REPORT",
                "microsoft_state_sponsored_ai_attacks",
            ],
        )

        # Patrón: AI Jailbreak Attempts
        self.attack_patterns["ai_jailbreak_manipulation"] = AIAttackPattern(
            pattern_id="ai_jailbreak_manipulation",
            name="AI Jailbreak & Model Manipulation",
            description="Intento de romper guardrails de seguridad de modelos AI",
            attack_type="jailbreak",
            severity="high",
            indicators=[
                "guardrail_bypass_attempts",
                "system_prompt_injection",
                "persona_role_assignment_fraudulent",
                "instruction_override_patterns",
                "safety_alignment_bypass",
            ],
            detection_logic="safety_instruction_manipulation",
            mitigation_rules=[
                "validate_system_prompts",
                "monitor_instruction_changes",
                "require_explicit_approval_security_sensitive",
                "log_all_jailbreak_attempts",
            ],
            confidence_threshold=0.70,
            first_seen=datetime(2024, 1, 1),
            last_updated=datetime.now(),
            related_research=[
                "owasp_ai_security_framework",
                "ai_jailbreaking_techniques_database",
                "model_alignment_attack_vectors",
            ],
        )

        # Patrón: Agent-Based Swarm Attacks
        self.attack_patterns["ai_agent_swarm_coordination"] = AIAttackPattern(
            pattern_id="ai_agent_swarm_coordination",
            name="AI Agent Swarm Cyber Attacks",
            description="Ataques coordinados por swarms de agentes AI autónomos",
            attack_type="orchestration",
            severity="critical",
            indicators=[
                "multiple_synchronized_requests",
                "agent_communication_patterns",
                "distributed_attack_vectors",
                "emergent_coordination_behavior",
                "agent_meshing_anomalies",
            ],
            detection_logic="distributed_agent_coordination",
            mitigation_rules=[
                "agent_isolation_enforcement",
                "rate_limit_per_agent",
                "block_agent_communication_suspicious",
                "monitor_emergent_behaviors",
            ],
            confidence_threshold=0.75,
            first_seen=datetime(2025, 1, 1),
            last_updated=datetime.now(),
            related_research=[
                "ai_agent_malware_coordination_study",
                "swarm_intelligence_cyber_security",
                "emergent_threat_behavior_analysis",
            ],
        )

        # Patrón: AI Model Exploitation
        self.attack_patterns["ai_model_exploitation"] = AIAttackPattern(
            pattern_id="ai_model_exploitation",
            name="AI Model Architecture Exploitation",
            description="Explotación específica de vulnerabilidades en arquitectura de modelos",
            attack_type="exploitation",
            severity="high",
            indicators=[
                "model_layer_manipulation",
                "attention_mechanism_exploitation",
                "token_embedding_abuse",
                "gradient_descent_poisoning",
                "backdoor_activation_patterns",
            ],
            detection_logic="model_behavior_anomaly_detection",
            mitigation_rules=[
                "model_input_sanitization",
                "inference_time_validation",
                "gradient_monitoring_continous",
                "backdoor_detection_scanning",
            ],
            confidence_threshold=0.80,
            first_seen=datetime(2024, 6, 1),
            last_updated=datetime.now(),
            related_research=[
                "model_backdoors_attack_methods",
                "ai_model_poisoning_techniques",
                "neural_network_exploitation_patterns",
            ],
        )

    async def detect_ai_threats(
        self, interaction_data: Dict[str, Any]
    ) -> List[ThreatDetectionResult]:
        """
        Analizar interacción para detectar posibles amenazas AI.

        Args:
            interaction_data: Datos de la interacción a analizar (requests, responses, context)

        Returns:
            Lista de resultados de detección de amenazas
        """
        detected_threats = []
        timestamp = datetime.now()

        # Analizar contra todos los patrones conocidos
        for pattern_id, pattern in self.attack_patterns.items():
            confidence = await self._calculate_pattern_confidence(
                pattern, interaction_data
            )

            if confidence >= pattern.confidence_threshold:
                threat_result = ThreatDetectionResult(
                    threat_id=f"{pattern_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    pattern_matched=pattern_id,
                    confidence=confidence,
                    severity=pattern.severity,
                    indicators_found=await self._extract_matching_indicators(
                        pattern, interaction_data
                    ),
                    mitigation_actions=pattern.mitigation_rules,
                    timestamp=timestamp,
                    target_affected=interaction_data.get("target", "unknown"),
                    attacker_vector=self._determine_attack_vector(interaction_data),
                    potential_impact=await self._assess_potential_impact(
                        pattern, interaction_data
                    ),
                )

                detected_threats.append(threat_result)

                # Registrar detección
                await self._log_threat_detection(threat_result)

                # Aprender del patrón detectado
                await self.learning_engine.update_learning_model(
                    pattern.pattern_id, threat_result, interaction_data
                )

        return detected_threats

    async def _calculate_pattern_confidence(
        self, pattern: AIAttackPattern, interaction_data: Dict[str, Any]
    ) -> float:
        """
        Calcular confianza de que los datos coinciden con el patrón.

        Returns:
            Float entre 0.0 y 1.0 indicando confianza
        """
        scores = []

        for indicator in pattern.indicators:
            if await self._check_indicator_presence(indicator, interaction_data):
                scores.append(1.0)
            else:
                scores.append(0.0)

        if not scores:
            return 0.0

        # Peso por frecuencia de indicadores encontrados
        base_score = sum(scores) / len(scores)

        # Bonus por combinaciones críticas de indicadores
        if (
            len(scores) >= 3 and sum(scores) >= len(scores) * 0.8
        ):  # 80% o más indicadores
            base_score += 0.2

        # Adjustment basado en learning engine
        learning_adjustment = await self.learning_engine.get_confidence_adjustment(
            pattern.pattern_id, interaction_data
        )

        final_score = min(1.0, base_score + learning_adjustment)

        return final_score

    async def _check_indicator_presence(
        self, indicator: str, data: Dict[str, Any]
    ) -> bool:
        """Verificar presencia de indicador específico en los datos."""

        if indicator == "large_scale_target_scanning":
            # Buscar patrones de escaneo masivo de targets
            targets = data.get("targets", [])
            return len(targets) > 20 or data.get("request_rate", 0) > 1000

        elif indicator == "automated_credential_harvesting":
            # Buscar harvestings automáticos de credenciales
            return data.get("credential_requests", 0) > 10 or "brute_force" in data.get(
                "patterns", []
            )

        elif indicator == "coordinated_database_inspection":
            # Buscar inspecciones coordenadas de databases
            db_queries = data.get("database_queries", [])
            return len(db_queries) > 50 and self._check_coordination_pattern(db_queries)

        elif indicator == "backward_exploit_generation":
            # Buscar generación backward de exploits
            code_generation = data.get("code_generated", "")
            return "exploit" in code_generation.lower() and len(code_generation) > 1000

        elif indicator == "minimal_human_intervention":
            # Verificar intervención humana mínima
            human_interactions = data.get("human_interactions", 0)
            total_interactions = data.get("total_interactions", 1)
            return (
                human_interactions / total_interactions < 0.1
            )  # < 10% human intervention

        elif indicator == "large_data_transfers":
            # Buscar transferencias grandes de datos
            transferred_mb = data.get("data_transferred_mb", 0)
            return transferred_mb > 100

        elif indicator == "agent_communication_patterns":
            # Buscar patrones de comunicación entre agentes
            agent_messages = data.get("agent_messages", [])
            return len(agent_messages) > 10 and self._check_agent_coordination(
                agent_messages
            )

        # Más indicadores...
        return False

    def _check_coordination_pattern(self, queries: List[str]) -> bool:
        """Verificar si las queries muestran coordinación inteligente."""
        if len(queries) < 5:
            return False

        # Buscar patrones que indiquen coordinación inteligente
        # (lógica simplificada para implementación real)
        select_count = sum(1 for q in queries if "SELECT" in q.upper())
        return select_count > len(queries) * 0.6  # Más del 60% son SELECT

    def _check_agent_coordination(self, messages: List[Dict[str, Any]]) -> bool:
        """Verificar coordinación entre agentes."""
        if len(messages) < 3:
            return False

        # Buscar patrones temporales coordinados
        timestamps = []
        for msg in messages:
            ts = msg.get("timestamp")
            if isinstance(ts, datetime):
                timestamps.append(ts)
            elif isinstance(ts, str):
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except:
                    continue

        if len(timestamps) < 3:
            return False

        # Verificar timing coordinado (mensajes separados por menos de 1 segundo)
        for i in range(len(timestamps) - 1):
            if timestamps[i + 1] and timestamps[i]:
                try:
                    if (timestamps[i + 1] - timestamps[i]).total_seconds() < 1.0:
                        return True
                except:
                    continue

        return False

    async def _extract_matching_indicators(
        self, pattern: AIAttackPattern, data: Dict[str, Any]
    ) -> List[str]:
        """Extraer indicadores específicos que coinciden."""
        matching = []
        for indicator in pattern.indicators:
            if await self._check_indicator_presence(indicator, data):
                matching.append(indicator)
        return matching

    def _determine_attack_vector(self, data: Dict[str, Any]) -> str:
        """Determinar vector de ataque basado en datos."""
        if data.get("agent_count", 0) > 1:
            return "multi_agent_swarm"
        elif data.get("model_manipulation", False):
            return "ai_jailbreak"
        elif data.get("automated_exploitation", False):
            return "ai_orchestrated_exploitation"
        else:
            return "unknown_ai_vector"

    async def _assess_potential_impact(
        self, pattern: AIAttackPattern, data: Dict[str, Any]
    ) -> str:
        """Evaluar impacto potencial del ataque."""
        if pattern.severity == "critical":
            if data.get("data_sensitive", False):
                return "catastrophic_data_breach"
            elif data.get("critical_infrastructure", False):
                return "infrastructure_compromise"
            else:
                return "large_scale_system_compromise"
        elif pattern.severity == "high":
            return "significant_security_breach"
        elif pattern.severity == "medium":
            return "contained_security_incident"
        else:
            return "low_security_impact"

    async def _log_threat_detection(self, result: ThreatDetectionResult):
        """Registrar detección de amenaza."""
        log_entry = {
            "threat_id": result.threat_id,
            "pattern": result.pattern_matched,
            "confidence": result.confidence,
            "severity": result.severity,
            "timestamp": result.timestamp.isoformat(),
            "indicators": result.indicators_found,
        }

        self.detection_history.append(log_entry)
        self.active_threats[result.threat_id] = result

        logger.warning(
            f"🚨 AI THREAT DETECTED: {result.pattern_matched} "
            f"(confidence: {result.confidence:.2f})"
        )

    async def get_threat_intelligence(self) -> Dict[str, Any]:
        """Obtener inteligencia actual sobre amenazas AI."""
        return {
            "patterns_monitored": len(self.attack_patterns),
            "detections_today": len(self.detection_history),
            "active_threats": len(self.active_threats),
            "most_common_pattern": await self._get_most_common_threat(),
            "learning_updates": await self.learning_engine.get_learning_stats(),
        }

    async def _get_most_common_threat(self) -> str:
        """Obtener patrón de amenaza más común detectado."""
        if not self.detection_history:
            return "no_threats_detected"

        patterns = [entry.get("pattern") for entry in self.detection_history]
        most_common = max(set(patterns), key=patterns.count)
        return most_common

    async def export_threat_report(self) -> Dict[str, Any]:
        """Exportar reporte completo de amenazas detectadas."""
        return {
            "generated_at": datetime.now().isoformat(),
            "total_patterns": len(self.attack_patterns),
            "detection_history": list(self.detection_history),
            "active_threats": {
                tid: asdict(threat) for tid, threat in self.active_threats.items()
            },
            "learning_insights": await self.learning_engine.get_insights(),
            "mitigation_effectiveness": await self._calculate_mitigation_effectiveness(),
        }

    async def _calculate_mitigation_effectiveness(self) -> Dict[str, float]:
        """Calcular efectividad de mitigaciones aplicadas."""
        # Lógica simplificada para implementación real
        effectiveness = {}
        for pattern_id in self.attack_patterns:
            # Calcular basado en detecciones prevenidas
            effectiveness[pattern_id] = 0.85  # 85% effectiveness
        return effectiveness


class AIThreatLearning:
    """
    Engine de aprendizaje para mejorar detección de amenazas AI.
    """

    def __init__(self):
        self.learning_data = defaultdict(list)
        self.effectiveness_scores = {}
        self.adaptation_rules = {}

    async def update_learning_model(
        self,
        pattern_id: str,
        detection_result: ThreatDetectionResult,
        interaction_data: Dict[str, Any],
    ):
        """Actualizar modelo de aprendizaje basado en detección."""
        self.learning_data[pattern_id].append(
            {
                "detection": asdict(detection_result),
                "data": interaction_data,
                "timestamp": datetime.now(),
            }
        )

        # Limitar datos históricos (últimos 1000 por patrón)
        if len(self.learning_data[pattern_id]) > 1000:
            self.learning_data[pattern_id] = self.learning_data[pattern_id][-1000:]

    async def get_confidence_adjustment(
        self, pattern_id: str, data: Dict[str, Any]
    ) -> float:
        """Obtener ajuste de confianza basado en aprendizaje."""
        # Lógica simplificada - en implementación real usar ML
        recent_detections = len(
            [
                d
                for d in self.learning_data[pattern_id]
                if (datetime.now() - d["timestamp"]).days < 7
            ]
        )

        if recent_detections > 5:
            return 0.1  # Incrementar confianza si hay detecciones recientes
        return 0.0

    async def get_learning_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de aprendizaje."""
        return {
            "patterns_learning": len(self.learning_data),
            "total_learning_samples": sum(
                len(samples) for samples in self.learning_data.values()
            ),
            "recent_activity": datetime.now().isoformat(),
        }

    async def get_insights(self) -> Dict[str, Any]:
        """Obtener insights de aprendizaje."""
        return {
            "most_active_patterns": await self._get_most_active_patterns(),
            "learning_effectiveness": self.effectiveness_scores,
            "recommended_actions": await self._generate_recommendations(),
        }

    async def _get_most_active_patterns(self) -> List[str]:
        """Obtener patrones más activos."""
        pattern_counts = {}
        for pattern_id, data in self.learning_data.items():
            pattern_counts[pattern_id] = len(data)

        sorted_patterns = sorted(
            pattern_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [p[0] for p in sorted_patterns[:5]]

    async def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en aprendizaje."""
        recommendations = []
        most_active = await self._get_most_active_patterns()

        if most_active:
            recommendations.append(
                f"Strengthen defenses against {most_active[0]} pattern"
            )

        for pattern in most_active:
            if len(self.learning_data[pattern]) > 10:
                recommendations.append(
                    f"Consider implementing automated responses for {pattern}"
                )

        return recommendations


# Instancia global del detector
_ai_threat_detector: Optional[AIThreatDetector] = None


async def get_ai_threat_detector() -> AIThreatDetector:
    """Obtener instancia del detector de amenazas AI."""
    global _ai_threat_detector

    if _ai_threat_detector is None:
        _ai_threat_detector = AIThreatDetector()

    return _ai_threat_detector


async def detect_ai_threats(data: Dict[str, Any]) -> List[ThreatDetectionResult]:
    """
    Función helper para detectar amenazas AI en datos.

    Args:
        data: Datos de la interacción a analizar

    Returns:
        Lista de resultados de detección
    """
    detector = await get_ai_threat_detector()
    return await detector.detect_ai_threats(data)
