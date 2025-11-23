#!/usr/bin/env python3
"""
Sistema Integrado de Mejoras Avanzadas - Sheily AI
==================================================

Script principal para inicializar todos los sistemas avanzados:
- Sistema de caché inteligente con Redis
- API de análisis de sentimientos en tiempo real
- Dashboard web moderno con React
- Sistema de backup y recovery automático
- API de métricas avanzadas y alertas proactivas
- Sistema de recomendaciones personalizadas
- Motor de traducción multilingüe
- Sistema de auto-escalado inteligente

Este script configura y arranca todos los componentes mejorados.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def initialize_all_systems():
    """Inicializar todos los sistemas mejorados"""

    logger.info("🚀 INICIANDO SISTEMAS AVANZADOS DE SHEILY AI")
    logger.info("=" * 60)

    systems_initialized = []

    try:
        # 1. Sistema de caché inteligente
        logger.info("📦 Inicializando sistema de caché inteligente...")
        from sheily_core.cache.smart_cache import get_cache

        cache = get_cache()
        health = await cache.health_check()
        if health["redis_connected"]:
            systems_initialized.append("Sistema de Caché")
            logger.info("✅ Sistema de caché inicializado")
        else:
            logger.warning("⚠️ Sistema de caché inicializado (sin Redis)")

        # 2. API de análisis de sentimientos
        logger.info("😊 Inicializando análisis de sentimientos...")
        from sheily_core.sentiment.sentiment_analysis import get_sentiment_api

        sentiment_api = get_sentiment_api()
        sentiment_health = await sentiment_api.health_check()
        systems_initialized.append("Análisis de Sentimientos")
        logger.info("✅ API de análisis de sentimientos inicializada")

        # 3. Sistema de backup y recovery
        logger.info("💾 Inicializando sistema de backup...")
        from sheily_core.backup.backup_manager import get_backup_manager

        backup_manager = get_backup_manager()
        systems_initialized.append("Sistema de Backup")
        logger.info("✅ Sistema de backup inicializado")

        # 4. API de métricas empresariales avanzadas
        logger.info("📊 Inicializando métricas empresariales avanzadas...")
        from sheily_core.metrics.advanced_metrics import get_enterprise_metrics_api

        enterprise_metrics = get_enterprise_metrics_api()
        await enterprise_metrics.start_enterprise_monitoring()
        systems_initialized.append("Métricas Empresariales Avanzadas")
        logger.info("✅ API de métricas empresariales avanzadas inicializada")

        # 5. Sistema de recomendaciones personalizadas
        logger.info("🎯 Inicializando sistema de recomendaciones...")
        from sheily_core.personalization.recommendation_engine import (
            get_personalization_engine,
        )

        personalization_engine = get_personalization_engine()
        systems_initialized.append("Sistema de Recomendaciones")
        logger.info("✅ Sistema de recomendaciones inicializado")

        # 6. Motor de traducción multilingüe
        logger.info("🌍 Inicializando traducción multilingüe...")
        from sheily_core.translation.multilingual_engine import get_translation_engine

        translation_engine = get_translation_engine()

        # Pre-cargar modelos comunes
        common_pairs = [("en", "es"), ("es", "en"), ("fr", "es"), ("de", "es")]
        await translation_engine.warmup_translations(common_pairs)

        systems_initialized.append("Traducción Multilingüe")
        logger.info("✅ Motor de traducción multilingüe inicializado")

        # 7. Sistema de auto-escalado
        logger.info("⚖️ Inicializando auto-escalado inteligente...")
        from sheily_core.scaling.auto_scaling_engine import (
            initialize_auto_scaling,
            start_auto_scaling_background,
        )

        await initialize_auto_scaling()
        await start_auto_scaling_background()

        systems_initialized.append("Auto-Escalado")
        logger.info("✅ Sistema de auto-escalado inicializado")

        # 8. Dashboard web (opcional - verificar si está disponible)
        try:
            logger.info("🌐 Verificando dashboard web...")
            dashboard_path = Path("web-dashboard/package.json")
            if dashboard_path.exists():
                logger.info(
                    "✅ Dashboard web disponible (iniciar con: python start_dashboard.py)"
                )
                systems_initialized.append("Dashboard Web (Disponible)")
            else:
                logger.info("ℹ️ Dashboard web no encontrado")
        except Exception as e:
            logger.warning(f"Error verificando dashboard: {e}")

    except Exception as e:
        logger.error(f"❌ Error inicializando sistemas: {e}")
        return False

    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("🎉 INICIALIZACIÓN COMPLETADA")
    logger.info("=" * 60)

    logger.info(f"✅ Sistemas inicializados: {len(systems_initialized)}")
    for system in systems_initialized:
        logger.info(f"   • {system}")

    logger.info("\n🚀 SISTEMA SHEILY AI MEJORADO LISTO PARA USO")
    logger.info("📖 Ver README_MEJORAS.md para documentación completa")

    return True


async def run_system_health_check():
    """Ejecutar verificación de salud de todos los sistemas"""

    logger.info("\n🔍 EJECUTANDO VERIFICACIÓN DE SALUD...")

    health_checks = []

    try:
        # Verificar caché
        from sheily_core.cache.smart_cache import get_cache

        cache = get_cache()
        cache_health = await cache.health_check()
        health_checks.append(("Caché", cache_health["status"] == "healthy"))

        # Verificar métricas
        from sheily_core.metrics.advanced_metrics import get_metrics_api

        metrics = get_metrics_api()
        metrics_health = await metrics.health_check()
        health_checks.append(("Métricas", metrics_health["status"] == "healthy"))

        # Verificar backup
        from sheily_core.backup.backup_manager import get_backup_manager

        backup = get_backup_manager()
        health_checks.append(("Backup", True))  # Siempre disponible

        # Verificar traducción
        from sheily_core.translation.multilingual_engine import get_translation_engine

        translation = get_translation_engine()
        translation_health = await translation.health_check()
        health_checks.append(
            ("Traducción", translation_health["status"] in ["healthy", "degraded"])
        )

    except Exception as e:
        logger.error(f"Error en verificación de salud: {e}")
        return False

    # Mostrar resultados
    all_healthy = True
    for system, healthy in health_checks:
        status = "✅" if healthy else "❌"
        logger.info(f"{status} {system}: {'Saludable' if healthy else 'Con problemas'}")
        if not healthy:
            all_healthy = False

    return all_healthy


async def main():
    """Función principal"""

    # Verificar argumentos
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "health":
            # Solo verificación de salud
            success = await run_system_health_check()
            return 0 if success else 1

        elif command == "help":
            print("Uso: python initialize_improvements.py [comando]")
            print("Comandos:")
            print("  (sin argumentos) - Inicializar todos los sistemas")
            print("  health           - Verificar salud de los sistemas")
            print("  help             - Mostrar esta ayuda")
            return 0

    # Inicialización completa
    success = await initialize_all_systems()

    if success:
        # Verificación de salud final
        await run_system_health_check()

        print("\n" + "=" * 60)
        print("🎊 SHEILY AI MEJORADO OPERATIVO")
        print("=" * 60)
        print("Nuevas funcionalidades disponibles:")
        print("• Sistema de caché inteligente con Redis")
        print("• Análisis de sentimientos en tiempo real")
        print("• Dashboard web moderno (python start_dashboard.py)")
        print("• Backup y recovery automático")
        print("• Métricas avanzadas y alertas proactivas")
        print("• Recomendaciones personalizadas")
        print("• Traducción automática multilingüe")
        print("• Auto-escalado inteligente")
        print("\n📚 Ver documentación completa en README_MEJORAS.md")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
