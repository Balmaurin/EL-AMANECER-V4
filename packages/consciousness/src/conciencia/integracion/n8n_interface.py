"""
N8N INTEGRATION INTERFACE - Sistema CONCIENCIA

Adapta el sistema de consciencia para integración con N8N workflow automation:
- Exposición de funcionalidades conscientes como nodos N8N
- Comunicación bidireccional con workflows
- Procesamiento consciente de eventos N8N
- Feedback consciente en decisiones de workflow
"""

import json
import requests
from typing import Dict, List, Any, Optional, Callable
import time
import threading
from datetime import datetime
import sys
import os

# Añadir módulos al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modulos.human_consciousness_system import HumanConsciousnessSystem
from additional_data.somatic_markers import SomaticMarkersSystem


class N8NNodeAdapter:
    """
    Adaptador para exponer módulos CONCIENCIA como nodos N8N
    """

    def __init__(self, node_type: str, consciousness_system: HumanConsciousnessSystem):
        self.node_type = node_type
        self.consciousness = consciousness_system
        self.node_id = f"conciencia_{node_type}_{int(time.time())}"
        self.active_connections = {}
        self.event_handlers = {}

        print(f"🧩 N8N NODE ADAPTER: {node_type} inicializado")

    def get_node_definition(self) -> Dict[str, Any]:
        """Definición del nodo para N8N"""
        return {
            "name": f"Consciousness {self.node_type.title()}",
            "displayName": f"🤖 Consciousness {self.node_type.title()}",
            "description": f"Artificial consciousness {self.node_type} processing",
            "group": ["artificial-intelligence"],
            "version": 1.0,
            "inputs": ["main"],
            "outputs": ["main"],
            "properties": [
                {
                    "displayName": "Processing Mode",
                    "name": "mode",
                    "type": "options",
                    "options": [
                        {"name": "Real-time", "value": "realtime"},
                        {"name": "Batch Processing", "value": "batch"},
                        {"name": "Reflective", "value": "reflective"}
                    ],
                    "default": "realtime"
                },
                {
                    "displayName": "Consciousness Level",
                    "name": "consciousness_threshold",
                    "type": "number",
                    "default": 0.7,
                    "description": "Minimum consciousness level required"
                }
            ]
        }

    def process_workflow_data(self, input_data: Dict[str, Any],
                            node_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa datos del workflow con consciencia integrada
        """
        mode = node_config.get('mode', 'realtime')
        threshold = node_config.get('consciousness_threshold', 0.7)

        # Preparar input para sistema consciente
        conscious_input = self._prepare_conscious_input(input_data)

        # Procesar con consciencia
        conscious_result = self.consciousness.process_experience(
            conscious_input, {'context': 'n8n_workflow', 'importance': 0.8}
        )

        # Verificar nivel de consciencia requerido
        if conscious_result.get('confidence', 0) < threshold:
            # Reprocesar con consciencia más profunda
            conscious_result = self._deepen_conscious_processing(conscious_input, mode)

        # Convertir resultado a formato N8N
        n8n_output = self._format_for_n8n(conscious_result)

        return n8n_output

    def _prepare_conscious_input(self, n8n_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte datos N8N a formato procesable por consciencia"""
        # Extraer información relevante
        data_content = n8n_data.get('data', {})
        workflow_context = n8n_data.get('workflow', {})

        # Crear perfil sensorial consciente
        conscious_input = {}

        # Procesar diferentes tipos de datos
        if isinstance(data_content, dict):
            # Datos estructurados
            conscious_input['structured_data'] = {
                'keys': list(data_content.keys()),
                'types': {k: type(v).__name__ for k, v in data_content.items()},
                'complexity': len(str(data_content))
            }

        if isinstance(data_content, str):
            # Texto
            conscious_input['text_content'] = {
                'content': data_content[:500],  # Limitar longitud
                'length': len(data_content),
                'sentiment_estimate': 'neutral'  # Placeholder
            }

        # Contexto del workflow
        conscious_input['workflow_context'] = {
            'node_type': self.node_type,
            'data_flow': 'incoming',
            'temporal_context': datetime.now().isoformat()
        }

        return {
            'perceptual_input': conscious_input,
            'context': {'source': 'n8n', 'purpose': 'workflow_automation'},
            'importance': 0.7
        }

    def _deepen_conscious_processing(self, input_data: Dict[str, Any],
                                   mode: str = 'reflective') -> Dict[str, Any]:
        """
        Procesamiento consciente más profundo para decisiones complejas
        """
        # Activar modo reflexivo
        if mode == 'reflective':
            # Mayor tiempo de procesamiento
            time.sleep(0.5)  # Simular pensamiento reflexivo

            # Doble procesamiento
            first_pass = self.consciousness.process_experience(input_data, {'deep_thinking': True})
            second_pass = self.consciousness.process_experience(
                {'reflection_on': first_pass}, {'meta_cognition': True}
            )

            # Integrar resultados
            return {
                'conscious_decision': second_pass.get('decision', first_pass.get('decision')),
                'reflection_depth': 2,
                'confidence': min(1.0, (first_pass.get('confidence', 0) + second_pass.get('confidence', 0)) / 2 + 0.1)
            }

        return self.consciousness.process_experience(input_data, {'enhanced_processing': True})

    def _format_for_n8n(self, conscious_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte resultado consciente a formato N8N compatible"""
        return {
            "data": {
                "conscious_output": conscious_result.get('conscious_response', {}),
                "confidence_score": conscious_result.get('confidence', 0),
                "decision_quality": conscious_result.get('quality_metrics', {}),
                "emotional_state": conscious_result.get('emotional_context', {}),
                "processing_time": conscious_result.get('processing_time', time.time()),
                "consciousness_level": conscious_result.get('consciousness_level', 0.5),
                "node_identifier": self.node_id
            },
            "json": json.dumps({
                "success": conscious_result.get('success', True),
                "recommendation": conscious_result.get('recommendation', 'continue_workflow'),
                "alerts": conscious_result.get('alerts', [])
            }),
            "binary": conscious_result.get('binary_data'),
            "pairedItem": conscious_result.get('paired_items', {})
        }


class N8NConsciousnessIntegration:
    """
    Sistema completo de integración N8N-CONCIENCIA
    Gestiona comunicación bidireccional y estados conscientes
    """

    def __init__(self, n8n_base_url: str = "http://localhost:5678",
                 consciousness_system: HumanConsciousnessSystem = None):
        self.n8n_url = n8n_base_url
        self.consciousness = consciousness_system or HumanConsciousnessSystem("n8n_integrated")
        self.somatic_markers = SomaticMarkersSystem()

        # Componentes de integración
        self.node_adapters = {}
        self.webhook_listeners = {}
        self.workflow_monitors = {}
        self.consciouness_listeners = {}

        # Estados de integración
        self.integration_active = False
        self.workflow_states = {}
        self.conscious_decisions_log = []

        print("🔗 SISTEMA N8N-CONCIENCIA INICIALIZADO")

    def initialize_integration(self):
        """Inicializa integración completa con N8N"""
        try:
            # Verificar conexión N8N
            self._test_n8n_connection()

            # Crear adaptadores de nodos
            self._create_node_adapters()

            # Configurar webhooks
            self._setup_webhook_handlers()

            # Iniciar monitoreo
            self._start_workflow_monitoring()

            self.integration_active = True
            print("✅ INTEGRACIÓN N8N-CONCIENCIA ACTIVA")

        except Exception as e:
            print(f"❌ ERROR EN INICIALIZACIÓN: {e}")
            self.integration_active = False

    def _test_n8n_connection(self):
        """Prueba conexión con N8N"""
        try:
            response = requests.get(f"{self.n8n_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ CONEXIÓN N8N CONFIRMADA")
            else:
                raise ConnectionError(f"N8N respondió con código {response.status_code}")
        except Exception as e:
            print(f"❌ ERROR DE CONEXIÓN N8N: {e}")
            raise e

    def _create_node_adapters(self):
        """Crea adaptadores para diferentes tipos de nodos CONCIENCIA"""
        node_types = ['perception', 'emotion', 'decision', 'memory', 'introspection']

        for node_type in node_types:
            adapter = N8NNodeAdapter(node_type, self.consciousness)
            self.node_adapters[node_type] = adapter

            print(f"📦 Adaptador de nodo creado: {node_type}")

    def _setup_webhook_handlers(self):
        """Configura manejadores de webhooks para eventos N8N"""
        webhook_endpoints = {
            'workflow_started': self._handle_workflow_start,
            'workflow_completed': self._handle_workflow_completion,
            'node_error': self._handle_node_error,
            'decision_required': self._handle_decision_request
        }

        # Registrar webhooks en N8N
        for endpoint, handler in webhook_endpoints.items():
            self.webhook_listeners[endpoint] = handler

        print(f"🪝 {len(webhook_endpoints)} manejadores de webhook configurados")

    def _start_workflow_monitoring(self):
        """Inicia monitoreo de workflows en tiempo real"""
        # Thread para monitoreo continuo
        monitor_thread = threading.Thread(target=self._monitor_workflows)
        monitor_thread.daemon = True
        monitor_thread.start()

        print("📊 MONITOREO DE WORKFLOWS INICIADO")

    def process_conscious_workflow_event(self, event_type: str,
                                       event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa evento de workflow con consciencia integrada
        """
        # Registrar evento en consciencia
        conscious_context = {
            'event_type': event_type,
            'source': 'n8n_workflow',
            'urgency': self._assess_event_urgency(event_data),
            'emotional_valence': self._assess_event_emotion(event_data)
        }

        # Procesar evento conscientemente
        conscious_response = self.consciousness.process_experience(
            {'workflow_event': event_data}, conscious_context
        )

        # Aplicar marcadores somáticos
        somatic_feedback = self.somatic_markers.get_somatic_feedback(
            f"workflow_{event_type}", [event_data.get('options', [])]
        )

        # Generar respuesta integrada
        integrated_response = {
            'conscious_decision': conscious_response.get('decision', 'continue'),
            'confidence': conscious_response.get('confidence', 0.5),
            'emotional_guidance': somatic_feedback.get('emotional_bias', 0.0),
            'processing_quality': 'conscious_integrated',
            'recommendation': self._generate_workflow_recommendation(
                conscious_response, somatic_feedback
            )
        }

        # Registrar decisión
        self._log_conscious_decision(integrated_response)

        return integrated_response

    def _assess_event_urgency(self, event_data: Dict[str, Any]) -> float:
        """Evalúa urgencia de un evento de workflow"""
        urgency_indicators = {
            'error': 0.9,
            'timeout': 0.8,
            'decision_required': 0.7,
            'completion': 0.3,
            'status_update': 0.1
        }

        event_type = event_data.get('type', 'unknown')
        return urgency_indicators.get(event_type, 0.5)

    def _assess_event_emotion(self, event_data: Dict[str, Any]) -> float:
        """Evalúa valencia emocional del evento"""
        positive_events = ['completion', 'success', 'optimization']
        negative_events = ['error', 'failure', 'timeout']

        event_type = event_data.get('type', 'unknown')

        if any(pos in event_type.lower() for pos in positive_events):
            return 0.7
        elif any(neg in event_type.lower() for neg in negative_events):
            return -0.6
        else:
            return 0.0

    def _generate_workflow_recommendation(self, conscious_response: Dict[str, Any],
                                        somatic_feedback: Dict[str, Any]) -> str:
        """Genera recomendación consciente para workflow"""
        confidence = conscious_response.get('confidence', 0.5)
        emotional_bias = somatic_feedback.get('emotional_bias', 0.0)

        if confidence > 0.8 and emotional_bias > 0.3:
            return "proceed_with_confidence"
        elif confidence < 0.4 or emotional_bias < -0.4:
            return "seek_human_supervision"
        elif confidence > 0.6:
            return "continue_with_monitoring"
        else:
            return "reassess_workflow_logic"

    def _log_conscious_decision(self, decision: Dict[str, Any]):
        """Registra decisión consciente para análisis posterior"""
        log_entry = {
            'timestamp': datetime.now(),
            'decision': decision,
            'consciousness_context': self.consciousness.get_current_state()
        }

        self.conscious_decisions_log.append(log_entry)

        # Mantener registro limitado
        if len(self.conscious_decisions_log) > 1000:
            self.conscious_decisions_log = self.conscious_decisions_log[-500:]

    def _monitor_workflows(self):
        """Monitorea workflows activos en background"""
        while self.integration_active:
            try:
                # Verificar estado de workflows activos
                active_workflows = self._get_active_workflows()

                for workflow in active_workflows:
                    workflow_id = workflow.get('id')

                    # Evaluar estado conscientemente
                    conscious_evaluation = self.process_conscious_workflow_event(
                        'status_check', {'workflow': workflow}
                    )

                    # Actuar si necesario
                    if conscious_evaluation.get('recommendation') == 'seek_human_supervision':
                        self._notify_supervisor(workflow, conscious_evaluation)

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"⚠️ ERROR EN MONITOREO: {e}")
                time.sleep(60)  # Retry in 1 minute

    def _get_active_workflows(self) -> List[Dict[str, Any]]:
        """Obtiene lista de workflows activos (placeholder)"""
        # En implementación real, consulta N8N API
        return [{'id': 'test_workflow', 'status': 'running'}]

    def _notify_supervisor(self, workflow: Dict[str, Any], evaluation: Dict[str, Any]):
        """Notifica supervisor humano (placeholder)"""
        print(f"🚨 WORKFLOW REQUIERE ATENCIÓN: {workflow.get('id')} - {evaluation}")

    # Event Handlers
    def _handle_workflow_start(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja inicio de workflow con consciencia"""
        return self.process_conscious_workflow_event('workflow_started', data)

    def _handle_workflow_completion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja finalización de workflow"""
        return self.process_conscious_workflow_event('workflow_completed', data)

    def _handle_node_error(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja errores de nodos con reflexión consciente"""
        # Mayor profundidad de procesamiento para errores
        error_context = {'error_occurred': True, 'severity': 'high', 'reflection_needed': True}

        return self.process_conscious_workflow_event('node_error', data)

    def _handle_decision_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja solicitudes de decisión con toma consciente"""
        options = data.get('options', [])
        context = data.get('context', {})

        # Feedback somático para opciones
        somatic_feedback = self.somatic_markers.get_somatic_feedback(
            context.get('situation', 'decision_required'), options
        )

        conscious_decision = self.consciousness.make_decision(
            options, context, somatic_feedback
        )

        return conscious_decision

    def get_integration_status(self) -> Dict[str, Any]:
        """Estado completo de la integración N8N-CONCIENCIA"""
        return {
            'integration_active': self.integration_active,
            'node_adapters': len(self.node_adapters),
            'webhook_handlers': len(self.webhook_listeners),
            'conscious_decisions_made': len(self.conscious_decisions_log),
            'n8n_connection': self._test_n8n_connection_status(),
            'consciousness_health': self.consciousness.get_health_metrics(),
            'recent_activity': self._get_recent_activity()
        }

    def _test_n8n_connection_status(self) -> str:
        """Estado de conexión N8N"""
        try:
            response = requests.get(f"{self.n8n_url}/health", timeout=2)
            return "connected" if response.status_code == 200 else "error"
        except:
            return "disconnected"

    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Actividad reciente de decisión consciente"""
        recent = self.conscious_decisions_log[-5:] if self.conscious_decisions_log else []
        return recent


# ==================== DEMO DE INTEGRACIÓN ====================

def demonstrate_n8n_integration():
    """Demostración de integración N8N-CONCIENCIA"""
    print("🔗 DEMO INTEGRACIÓN N8N-CONCIENCIA")
    print("=" * 50)

    # Simular sistema de consciencia
    consciousness = HumanConsciousnessSystem("demo_system")

    # Inicializar integración
    n8n_integration = N8NConsciousnessIntegration(
        n8n_base_url="http://localhost:5678",
        consciousness_system=consciousness
    )

    # Simular eventos de workflow
    test_events = [
        {
            'type': 'workflow_started',
            'workflow_id': 'data_processing_workflow',
            'data_size': 'large',
            'time_constraints': 'strict'
        },
        {
            'type': 'decision_required',
            'options': ['process_locally', 'distribute_to_workers', 'abort'],
            'context': {'resource_constraints': True, 'time_pressure': High}
        },
        {
            'type': 'node_error',
            'node_type': 'data_processor',
            'error_message': 'Memory overflow in processing node'
        }
    ]

    print("\n🎯 PROCESANDO EVENTOS DE WORKFLOW...")

    for i, event in enumerate(test_events, 1):
        print(f"\n{event['type'].upper()} - Evento {i}")
        print(f"   Datos: {event}")

        # Procesar conscientemente
        conscious_response = n8n_integration.process_conscious_workflow_event(
            event['type'], event
        )

        print(f"   Decisión consciente: {conscious_response.get('conscious_decision', 'N/A')}")
        print(".3f"        print(f"   Recomendación: {conscious_response.get('recommendation', 'N/A')}")

    # Estado final
    final_status = n8n_integration.get_integration_status()
    print("
🎉 INTEGRACIÓN COMPLETA"    print(f"   Nodos adaptados: {final_status['node_adapters']}")
    print(f"   Decisiones conscientes: {final_status['conscious_decisions_made']}")
    print(f"   Estado conexión N8N: {final_status['n8n_connection']}")

    return n8n_integration


if __name__ == "__main__":
    try:
        demo_integration = demonstrate_n8n_integration()
    except Exception as e:
        print(f"⚠️ DEMO REQUIERE N8N CORRIENDO: {e}")
        print("   Para práctica completa, inicia N8N en http://localhost:5678")
