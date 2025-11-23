#!/usr/bin/env python3
"""
RATE LIMITING SERVICE - Protección DDoS y Control de Uso
========================================================

Servicio completo para:
- Rate limiting por IP/usuario
- Control de uso de APIs
- Protección contra abuso
- Métricas de requests
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter basado en sliding window con memoria
    """

    def __init__(self, window_size: int = 60, max_requests: int = 100):
        # Ventana deslizante de 60 segundos, máximo 100 requests
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = defaultdict(lambda: deque())  # IP -> deque de timestamps
        self._lock = threading.Lock()

    def is_allowed(self, identifier: str) -> bool:
        """
        Verificar si el identificador puede hacer un request
        """
        current_time = time.time()

        with self._lock:
            # Limpiar timestamps antiguos
            request_queue = self.requests[identifier]
            while request_queue and current_time - request_queue[0] > self.window_size:
                request_queue.popleft()

            # Verificar límite
            if len(request_queue) >= self.max_requests:
                return False

            # Registrar nuevo request
            request_queue.append(current_time)
            return True

    def get_remaining_requests(self, identifier: str) -> int:
        """
        Obtener requests restantes en la ventana actual
        """
        with self._lock:
            request_queue = self.requests[identifier]
            current_time = time.time()

            # Limpiar timestamps antiguos
            while request_queue and current_time - request_queue[0] > self.window_size:
                request_queue.popleft()

            return max(0, self.max_requests - len(request_queue))

    def get_reset_time(self, identifier: str) -> float:
        """
        Tiempo hasta reset de ventana (segundos)
        """
        with self._lock:
            request_queue = self.requests[identifier]
            if not request_queue:
                return 0

            oldest_request = request_queue[0]
            return max(0, self.window_size - (time.time() - oldest_request))


class RateLimitingService:
    """
    Servicio completo de rate limiting para toda la aplicación
    """

    def __init__(self):
        # Configuración por ruta/endpoint
        self.limiters = {
            "global": RateLimiter(60, 1000),  # 1000 req/min global
            "api": RateLimiter(60, 500),  # 500 req/min para APIs
            "rag_search": RateLimiter(60, 50),  # 50 búsquedas/min
            "payments": RateLimiter(60, 20),  # 20 pagos/min por IP
            "exercises": RateLimiter(60, 30),  # 30 ejercicios/min
        }

        # Estadísticas
        self.stats = {
            "blocked_requests": 0,
            "total_requests": 0,
            "requests_by_ip": defaultdict(int),
        }

        logger.info("🛡️ Rate Limiting Service inicializado")

    async def check_rate_limit(
        self, endpoint: str, client_ip: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verificar rate limit para un request específico
        """
        identifier = user_id or client_ip  # Usar user_id si disponible, sino IP

        # Verificar límite global
        if not self.limiters["global"].is_allowed(client_ip):
            self.stats["blocked_requests"] += 1
            return {
                "allowed": False,
                "reason": "global_rate_limit_exceeded",
                "retry_after": int(self.limiters["global"].get_reset_time(client_ip)),
            }

        # Verificar límite específico del endpoint
        endpoint_limiter = self.limiters.get(endpoint, self.limiters["api"])
        if not endpoint_limiter.is_allowed(identifier):
            self.stats["blocked_requests"] += 1
            return {
                "allowed": False,
                "reason": f"{endpoint}_rate_limit_exceeded",
                "retry_after": int(endpoint_limiter.get_reset_time(identifier)),
            }

        # Request permitido
        self.stats["total_requests"] += 1
        self.stats["requests_by_ip"][client_ip] += 1

        remaining = endpoint_limiter.get_remaining_requests(identifier)

        return {
            "allowed": True,
            "remaining_requests": remaining,
            "reset_in_seconds": int(endpoint_limiter.get_reset_time(identifier)),
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de rate limiting
        """
        total_blocked = self.stats["blocked_requests"]
        total_allowed = self.stats["total_requests"]
        total_requests = total_blocked + total_allowed

        blockage_rate = (
            (total_blocked / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "total_requests": total_requests,
            "allowed_requests": total_allowed,
            "blocked_requests": total_blocked,
            "blockage_rate_percent": round(blockage_rate, 2),
            "active_ips": len(self.stats["requests_by_ip"]),
            "top_ips": sorted(
                self.stats["requests_by_ip"].items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def reset_stats(self):
        """Reset estadísticas"""
        self.stats = {
            "blocked_requests": 0,
            "total_requests": 0,
            "requests_by_ip": defaultdict(int),
        }


# =============================================================================
# DEMO Y TESTING DEL RATE LIMITING SERVICE
# =============================================================================


async def demo_rate_limiting():
    """Demo del sistema de rate limiting"""
    print("🛡️ RATE LIMITING SERVICE DEMO")
    print("=" * 40)

    service = RateLimitingService()

    # Simular requests normales
    print("\n📊 Simulando requests normales...")
    client_ip = "192.168.1.100"

    for i in range(5):
        result = await service.check_rate_limit("api", client_ip)
        status = "✅ ALLOWED" if result["allowed"] else "❌ BLOCKED"
        print(
            f"Request {i+1}: {status} | Remaining: {result.get('remaining_requests', 0)}"
        )

    # Simular ataque DDoS
    print("\n🚨 Simulando ataque DDoS...")
    for i in range(10):
        result = await service.check_rate_limit("api", f"attacker_{i}")
        status = "✅ ALLOWED" if result["allowed"] else "❌ BLOCKED"
        print(f"DDoS {i+1}: {status}")

    # Ver estadísticas
    print("\n📈 Estadísticas finales:")
    stats = service.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n🛡️ RATE LIMITING OPERATIVO")
    print("   ✅ Protección DDoS activa")
    print("   ✅ Control por endpoint")
    print("   ✅ Métricas en tiempo real")


# Configurar para testing
if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_rate_limiting())
