#!/usr/bin/env python3
"""
Sistema Real de Recompensas Sheilys - Sheily AI
===============================================

Sistema completamente funcional de recompensas para aprendizaje incremental.
Ejecuta interacciones reales y calcula recompensas Sheilys automáticamente.

USO:
    python sheily_rewards.py

FUNCIONALIDADES:
- Procesamiento de interacciones en múltiples dominios
- Cálculo automático de puntuaciones Sheilys
- Almacenamiento persistente en vault
- Optimización adaptativa en tiempo real
- Estadísticas y reportes detallados
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from sheily_core.rewards.adaptive_rewards import AdaptiveRewardsOptimizer
from sheily_core.rewards.contextual_accuracy import evaluate_contextual_accuracy
from sheily_core.rewards.integration_example import SheilyRewardsIntegration
from sheily_core.rewards.reward_system import SheilyRewardSystem
from sheily_core.rewards.tracker import SessionTracker


class RealSheilyRewardsSystem:
    """Sistema real y completamente funcional de recompensas Sheilys"""

    def __init__(self):
        # Inicializar componentes del sistema
        self.reward_system = SheilyRewardSystem()
        self.session_tracker = SessionTracker()
        self.adaptive_optimizer = AdaptiveRewardsOptimizer()
        self.integration_system = SheilyRewardsIntegration()

        # Estadísticas de funcionamiento
        self.stats = {
            "interactions_processed": 0,
            "total_sheilys_generated": 0.0,
            "domains_covered": set(),
            "start_time": datetime.now(),
        }

    async def process_real_interaction(
        self, domain: str, query: str, response: str, quality_score: float = None
    ) -> dict:
        """
        Procesar una interacción real y calcular recompensas

        Args:
            domain (str): Dominio de la interacción
            query (str): Consulta del usuario
            response (str): Respuesta generada
            quality_score (float, optional): Puntuación de calidad (calculada automáticamente si None)

        Returns:
            dict: Resultados del procesamiento
        """
        print(f"\n🔄 Procesando interacción en dominio: {domain}")

        # Calcular calidad automáticamente si no se proporciona
        if quality_score is None:
            try:
                contextual_score = evaluate_contextual_accuracy(query, response)
                quality_score = contextual_score * 0.7 + 0.3  # Combinar con baseline
                print(f"   📊 Score contextual: {contextual_score:.3f}")
            except Exception as e:
                quality_score = 0.7  # Valor por defecto
                print(
                    f"⚠️ Error calculando calidad contextual: {e}, usando valor por defecto"
                )

        # Preparar datos de sesión
        session_data = {
            "domain": domain,
            "query": query,
            "response": response,
            "quality_score": quality_score,
            "tokens_used": len(query.split()) + len(response.split()),
        }

        # 1. Registrar sesión
        tracked_session = self.session_tracker.track_session(**session_data)
        session_id = tracked_session["session_id"]
        print(f"📝 Sesión registrada: {session_id[:12]}...")

        # 2. Calcular y registrar recompensa
        reward = self.reward_system.record_reward(tracked_session)
        sheilys = reward["sheilys"]

        # 3. Actualizar optimizador adaptativo
        interaction_data = {
            **session_data,
            "sheilys_earned": sheilys,
            "contextual_score": evaluate_contextual_accuracy(query, response),
        }
        self.adaptive_optimizer.update_performance(domain, interaction_data)

        # 4. Procesar con sistema integrado
        integrated_result = self.integration_system.process_interaction(
            domain, query, response
        )

        # 5. Actualizar estadísticas
        self.stats["interactions_processed"] += 1
        self.stats["total_sheilys_generated"] += sheilys
        self.stats["domains_covered"].add(domain)

        result = {
            "session_id": session_id,
            "domain": domain,
            "quality_score": round(quality_score, 3),
            "sheilys_earned": sheilys,
            "reward_id": reward["reward_id"],
            "processing_timestamp": datetime.now().isoformat(),
            "integrated_score": integrated_result["sheilys_earned"],
        }

        print(f"💎 Sheilys generados: {sheilys}")
        print(f"✅ Interacción procesada exitosamente")

        return result

    async def run_interactive_demo(self):
        """Ejecutar demo interactiva del sistema de recompensas"""

        print("🎯 SISTEMA REAL DE RECOMPENSAS SHEILYS - DEMO INTERACTIVA")
        print("=" * 70)

        # Interacciones de ejemplo realistas
        sample_interactions = [
            {
                "domain": "medicina",
                "query": "¿Cuáles son los síntomas principales de la hipertensión arterial?",
                "response": "Los síntomas principales incluyen dolor de cabeza, mareos, visión borrosa, fatiga y, en casos graves, dificultad para respirar. Sin embargo, muchos pacientes no presentan síntomas evidentes, por lo que es fundamental la medición regular de la presión arterial para su diagnóstico temprano.",
            },
            {
                "domain": "programación",
                "query": "Explícame el patrón de diseño Singleton en Python",
                "response": "El patrón Singleton asegura que una clase tenga solo una instancia y proporciona un punto de acceso global a ella. En Python, se puede implementar usando una variable de clase o decoradores. Es útil para recursos compartidos como conexiones a base de datos, pools de hilos, o configuraciones globales.",
            },
            {
                "domain": "ciberseguridad",
                "query": "¿Qué medidas de seguridad debo implementar en mi red doméstica?",
                "response": "Implementa WPA3 en tu router, cambia la contraseña por defecto, activa el firewall, usa VPN para conexiones públicas, mantiene el firmware actualizado, configura redes guest separadas, y considera usar un sistema de detección de intrusiones. Además, educa a todos los miembros de la familia sobre phishing y navegación segura.",
            },
            {
                "domain": "matemáticas",
                "query": "Demuestra el teorema de Pitágoras",
                "response": "Para un triángulo rectángulo con catetos a y b, e hipotenusa c, se cumple que a² + b² = c². La demostración geométrica clásica divide el cuadrado de la hipotenusa en figuras congruentes que se reacomodan para formar los cuadrados de los catetos, demostrando visualmente la equivalencia.",
            },
            {
                "domain": "vida_diaria",
                "query": "¿Cómo puedo ahorrar energía en mi hogar?",
                "response": "Utiliza electrodomésticos eficientes (A++), apaga dispositivos cuando no los uses, aprovecha la luz natural, instala aislamiento térmico, usa termostatos programables, elige electrodomésticos LED, y considera paneles solares para generación propia. Pequeños cambios pueden reducir tu consumo hasta un 30%.",
            },
        ]

        results = []

        print(f"\n🚀 Procesando {len(sample_interactions)} interacciones reales...\n")

        for i, interaction in enumerate(sample_interactions, 1):
            print(f"[{i}/{len(sample_interactions)}] Procesando...")

            result = await self.process_real_interaction(
                interaction["domain"], interaction["query"], interaction["response"]
            )
            results.append(result)

            # Pequeña pausa para mejor visualización
            await asyncio.sleep(0.5)

        # Optimización final
        print("\n🎯 Ejecutando optimización adaptativa final...")
        optimized_config = self.adaptive_optimizer.optimize_reward_factors()

        # Mostrar resultados finales
        await self.show_final_results(results, optimized_config)

    async def show_final_results(self, results: list, optimized_config: dict):
        """Mostrar resultados finales del procesamiento"""

        print("\n" + "=" * 70)
        print("📊 RESULTADOS FINALES - SISTEMA DE RECOMPENSAS SHEILYS")
        print("=" * 70)

        # Estadísticas generales
        total_sheilys = sum(r["sheilys_earned"] for r in results)
        avg_quality = sum(r["quality_score"] for r in results) / len(results)
        domains = set(r["domain"] for r in results)

        print("\n🏆 MÉTRICAS GENERALES:")
        print(f"   • Interacciones procesadas: {len(results)}")
        print(f"   • Total Sheilys generados: {total_sheilys:.2f}")
        print(f"   • Calidad promedio: {avg_quality:.3f}")
        print(f"   • Dominios cubiertos: {len(domains)}")
        print(f"   • Lista de dominios: {', '.join(domains)}")

        # Detalles por dominio
        print("\n🏅 RESULTADOS POR DOMINIO:")
        domain_stats = {}
        for result in results:
            domain = result["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = []
            domain_stats[domain].append(result["sheilys_earned"])

        for domain, sheilys_list in domain_stats.items():
            avg_domain = sum(sheilys_list) / len(sheilys_list)
            max_domain = max(sheilys_list)
            print(f"   • {domain}:")
            print(f"     └─ Promedio: {avg_domain:.2f} Sheilys")
            print(f"     └─ Máximo: {max_domain:.2f} Sheilys")
            print(f"     └─ Interacciones: {len(sheilys_list)}")

        # Optimización adaptativa
        print("\n🎯 OPTIMIZACIÓN ADAPTATIVA:")
        print("   • Factores de recompensa optimizados:")
        for factor, weight in optimized_config["factors"].items():
            print(f"     └─ {factor}: {weight:.3f}")
        # Estadísticas del sistema
        print("\n📈 ESTADÍSTICAS DEL SISTEMA:")
        system_health = self.reward_system.get_system_health()
        print(f"   • Recompensas totales almacenadas: {system_health['total_rewards']}")
        print(f"   • Tamaño del vault: {system_health['vault_size_mb']:.2f} MB")
        print(f"   • Dominios procesados: {system_health['domains_processed']}")
        print(f"   • Retención configurada: {system_health['retention_days']} días")

        # Mejores interacciones
        print("\n🏆 TOP INTERACCIONES:")
        sorted_results = sorted(
            results, key=lambda x: x["sheilys_earned"], reverse=True
        )
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"   {i}. {result['domain']}: {result['sheilys_earned']:.2f} Sheilys")
            print(f"      └─ ID: {result['session_id'][:12]}...")

        print("\n✅ SISTEMA DE RECOMPENSAS SHEILYS OPERATIVO")
        print("💎 Todas las recompensas han sido almacenadas en el vault")
        print("🔄 Optimización adaptativa completada")
        print("📊 Estadísticas y reportes disponibles")

    async def show_system_status(self):
        """Mostrar estado actual del sistema"""

        print("\n🔍 ESTADO DEL SISTEMA DE RECOMPENSAS SHEILYS")
        print("=" * 50)

        # Estado del vault
        health = self.reward_system.get_system_health()
        print("\n💾 VAULT:")
        print(f"   Tamaño: {health['vault_size_mb']:.2f} MB")
        print(f"   Recompensas: {health['total_rewards']}")
        print(f"   Sheilys totales: {health['total_sheilys']:.2f}")
        print(f"   Dominios: {health['domains_processed']}")

        # Estadísticas de sesión
        total_sheilys = self.reward_system.get_total_sheilys()
        domain_stats = self.reward_system.get_domain_stats()

        print("\n📊 ESTADÍSTICAS:")
        print(f"   Sheilys acumulados: {total_sheilys:.2f}")
        print(f"   Dominios con datos: {len(domain_stats)}")

        if domain_stats:
            print("\n🏅 TOP DOMINIOS:")
            sorted_domains = sorted(
                domain_stats.items(), key=lambda x: x[1]["total_sheilys"], reverse=True
            )
            for domain, stats in sorted_domains[:5]:
                print(
                    f"   • {domain}: {stats['total_sheilys']:.2f} Sheilys "
                    f"({stats['total_rewards']} interacciones)"
                )

        # Estado de optimización
        print("\n🎯 OPTIMIZACIÓN:")
        optimized = self.adaptive_optimizer.optimize_reward_factors()
        print("   • Factores optimizados: ✅")
        print("   • Dominios analizados: ✅")

    async def cleanup_system(self):
        """Limpiar y optimizar el sistema"""

        print("\n🧹 REALIZANDO LIMPIEZA DEL SISTEMA...")

        # Limpiar recompensas antiguas
        self.reward_system.cleanup_old_rewards()

        # Limpiar sesiones antiguas
        self.session_tracker.cleanup_old_sessions()

        print("✅ Limpieza completada")
        print("🔄 Sistema optimizado y listo para nuevas interacciones")


async def main():
    """Función principal del sistema real de recompensas"""

    # Verificar argumentos
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "status":
            # Mostrar estado del sistema
            system = RealSheilyRewardsSystem()
            await system.show_system_status()
            return

        elif command == "cleanup":
            # Limpiar sistema
            system = RealSheilyRewardsSystem()
            await system.cleanup_system()
            return

        elif command == "help":
            print("Sistema Real de Recompensas Sheilys")
            print("===================================")
            print("Comandos disponibles:")
            print("  python sheily_rewards.py         - Ejecutar demo interactiva")
            print("  python sheily_rewards.py status  - Mostrar estado del sistema")
            print("  python sheily_rewards.py cleanup - Limpiar sistema")
            print("  python sheily_rewards.py help    - Mostrar esta ayuda")
            return

    # Ejecutar demo interactiva por defecto
    system = RealSheilyRewardsSystem()
    await system.run_interactive_demo()


if __name__ == "__main__":
    asyncio.run(main())
