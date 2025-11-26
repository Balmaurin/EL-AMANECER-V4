"""
Cliente MCP para Conectar con Servicios Externos
===============================================================

Este módulo implementa las conexiones críticas faltantes identificadas:
1. MCP ↔ Training Service (PyTorch Real Training)
2. MCP ↔ RAG Service (Corpus + Embeddings)
3. Consciousness ↔ Unified Memory Service

FUNCIONALIDADES CRÍTICAS ADQUIRIDAS:
- request_fine_tune(): Conecta con training neurarial real
- rag_retrieve(): Conecta con memoria externa infinita
- save_interaction(): Conecta con memoria consciente unificada

DISIGN:
- Conexiones HTTP opcionales con fallback graceful
- Zero breaking changes al sistema existente
- Mantiene toda la funcionalidad local si servicios caen
- Timeout y error handling enterprise-grade
"""

import os
import logging
from typing import Dict, Any, Optional, List

# Setup logging
logger = logging.getLogger(__name__)

# Service URLs con support para environment variables
DEFAULT_SERVICES = {
    "training": os.getenv("TRAINING_SERVICE_URL", "http://localhost:9001"),
    "rag": os.getenv("RAG_SERVICE_URL", "http://localhost:9100"),
    "memory": os.getenv("MEMORY_SERVICE_URL", "http://localhost:9200")
}

class MCPServiceConnector:
    """
    Conectar MCP con Servicios Externos

    Esta clase maneja las 3 conexiones críticas que hacen viable el sistema:
    - Training: Auto-mejora neuronal continua
    - RAG: Memoria externa infinita
    - Memory: Aprendizaje consciente de conversaciones
    """

    def __init__(self):
        """Inicializar connector con fallback inteligente"""
        self._http_available = self._check_http_availability()
        self.services = DEFAULT_SERVICES.copy()
        self.connection_status = self._test_connections()

        # Log inicial de conexiones
        self._log_connection_status()

    def _check_http_availability(self) -> bool:
        """Check si requests está disponible"""
        try:
            import requests
            self._requests = requests
            return True
        except ImportError:
            logger.warning("⚠️  HTTP requests no instalado - servicios externos deshabilitados")
            return False

    def _test_connections(self) -> Dict[str, bool]:
        """Test conexión a cada servicio y report status"""
        status = {}

        if not self._http_available:
            return {service: False for service in self.services.keys()}

        # Test Training Service
        try:
            resp = self._requests.get(f"{self.services['training']}/health", timeout=3)
            status['training'] = resp.status_code == 200
        except Exception:
            status['training'] = False

        # Test RAG Service
        try:
            resp = self._requests.get(f"{self.services['rag']}/health", timeout=3)
            status['rag'] = resp.status_code == 200
        except Exception:
            status['rag'] = False

        # Test Memory Service
        try:
            resp = self._requests.get(f"{self.services['memory']}/health", timeout=3)
            status['memory'] = resp.status_code == 200
        except Exception:
            status['memory'] = False

        return status

    def _log_connection_status(self):
        """Log estado inicial de conexiones"""
        total_connected = sum(self.connection_status.values())
        total_services = len(self.connection_status)

        if total_connected == 0:
            logger.info("🔌 Ningún servicio externo conectado - modo local puro")
        elif total_connected == total_services:
            logger.info("✅ Todos los servicios externos conectados - modo híbrido completo")
        else:
            logger.info(f"⚠️  Servicios conectados: {total_connected}/{total_services} - modo híbrido parcial")

        for service, connected in self.connection_status.items():
            status_icon = "✅" if connected else "❌"
            logger.info(f"   {status_icon} {service}: {self.services[service]}")

    def request_fine_tune(self, model_name: str, dataset_path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        🎯 CONEXIÓN CRÍTICA 1: Conectar con Training Service

        Solicita fine-tuning neuronal real del modelo. Si falla,
        retorna None para que el sistema use implementación local.
        """
        if not self._http_available or not self.connection_status.get('training'):
            logger.debug("Training Service no disponible - using local implementation")
            return {"fallback": True, "reason": "service_unavailable"}

        try:
            url = f"{self.services['training']}/train"
            data = {
                "model": model_name,
                "dataset_path": dataset_path,
                "params": params
            }

            resp = self._requests.post(url, json=data, timeout=120)
            resp.raise_for_status()

            result = resp.json()
            logger.info(f"🧠 Training job iniciado: {result.get('job_id', 'unknown')}")
            return result

        except self._requests.exceptions.Timeout:
            logger.error("⏰ Training service timeout")
        except self._requests.exceptions.ConnectionError:
            logger.error("🔌 Training service connection failed")
        except self._requests.exceptions.HTTPError as e:
            logger.error(f"🏥 Training service HTTP error: {e}")
        except Exception as e:
            logger.error(f"⚠️  Training request failed: {e}")

        # Return fallback indicator
        return {"fallback": True, "reason": str(e) if 'e' in locals() else "unknown_error"}

    def rag_retrieve(self, query: str, top_k: int = 5) -> list:
        """
        🎯 CONEXIÓN CRÍTICA 2: Conectar con RAG Service

        Recupera documentos relevantes del corpus de 484+ archivos.
        Si falla, retorna lista vacía para que el sistema use knowledge base local.
        """
        if not self._http_available or not self.connection_status.get('rag'):
            logger.debug("RAG Service no disponible - using local retrieval")
            return []

        try:
            url = f"{self.services['rag']}/retrieve"
            data = {"query": query, "top_k": top_k}

            resp = self._requests.post(url, json=data, timeout=10)
            resp.raise_for_status()

            documents = resp.json().get("documents", [])
            logger.debug(f"📚 RAG retrieved {len(documents)} documents for query: '{query[:50]}...'")
            return documents

        except self._requests.exceptions.Timeout:
            logger.error("⏰ RAG service timeout")
        except self._requests.exceptions.ConnectionError:
            logger.error("🔌 RAG service connection failed")
        except self._requests.exceptions.HTTPError as e:
            logger.error(f"🏥 RAG service HTTP error: {e}")
        except ValueError as e:
            logger.error(f"📝 RAG response parse error: {e}")
        except Exception as e:
            logger.error(f"⚠️  RAG request failed: {e}")

        # Return empty list as fallback
        return []

    def save_interaction(self, session_id: str, user_input: str, response: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 CONEXIÓN CRÍTICA 3: Conectar con Memory Service

        Guarda interacciones para aprendizaje consciente continuo.
        Si falla, es gracioso (no afecta funcionalidad inmediata).
        """
        if not self._http_available or not self.connection_status.get('memory'):
            logger.debug("Memory Service no disponible - usando memoria local")
            return {"status": "local_fallback"}

        try:
            url = f"{self.services['memory']}/memory"
            data = {
                "session_id": session_id,
                "user_input": user_input,
                "response": response,
                "meta": meta
            }

            resp = self._requests.post(url, json=data, timeout=5)
            resp.raise_for_status()

            result = resp.json()
            logger.debug(f"🧠 Memoria guardada: session {session_id}")
            return result

        except self._requests.exceptions.Timeout:
            logger.debug("⏰ Memory service timeout (continuando con local)")
        except self._requests.exceptions.ConnectionError:
            logger.debug("🔌 Memory service connection failed (continuando con local)")
        except self._requests.exceptions.HTTPError as e:
            logger.debug(f"🏥 Memory service HTTP error: {e} (continuando con local)")
        except Exception as e:
            logger.debug(f"⚠️  Memory save failed: {e} (continuando con local)")

        # Silent fallback - no error para no romper flow
        return {"status": "fallback_ok", "reason": str(e) if 'e' in locals() else "unknown"}

    def get_system_status(self) -> Dict[str, Any]:
        """Get complete status of MCP connectors for monitoring"""
        status = self._test_connections()  # Refresh status

        return {
            "http_available": self._http_available,
            "services": self.services,
            "connection_status": status,
            "total_connected": sum(status.values()),
            "total_services": len(status),
            "hybrid_mode": sum(status.values()) > 0,
            "fallback_mode": sum(status.values()) == 0
        }

# ================================
# FUNCIONES GLOBALES DE ACCESO
# ================================

# Instancia global del MCP connector
_mcp_connector = MCPServiceConnector()

def request_fine_tune(model_name: str, dataset_path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """🎯 Función global: Solicitar fine-tuning neuronal"""
    return _mcp_connector.request_fine_tune(model_name, dataset_path, params)

def rag_retrieve(query: str, top_k: int = 5) -> List[dict]:
    """🎯 Función global: Recuperar documentos RAG"""
    return _mcp_connector.rag_retrieve(query, top_k)

def save_interaction(session_id: str, user_input: str, response: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """🎯 Función global: Guardar interacción en memoria consciente"""
    return _mcp_connector.save_interaction(session_id, user_input, response, meta)

def get_mcp_connector_status() -> Dict[str, Any]:
    """Estado completo del sistema MCP"""
    return _mcp_connector.get_system_status()

# ===========================
# BACKWARD COMPATIBILITY
# ===========================

# Alias para mantener compatibility
mcp_connector = _mcp_connector
