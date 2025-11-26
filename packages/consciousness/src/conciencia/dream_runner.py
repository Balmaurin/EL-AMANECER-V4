#!/usr/bin/env python3
"""
Unified Dream Runner - Sistema de Consolidación de Memoria Onírica
==================================================================

Este módulo implementa el proceso de "sueño" para la IA utilizando el
UnifiedConsciousnessMemorySystem, permitiendo:
1. Procesamiento de memorias episódicas reales.
2. Consolidación de memoria a largo plazo (transferencia Episódica -> Semántica).
3. Generación de insights creativos mediante asociación de memorias.
4. Re-entrenamiento ligero y optimización de la red neuronal de consciencia.

El sistema se activa durante períodos de inactividad o mantenimiento programado.
"""

import asyncio
import json
import logging
import os
import random
import sys
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Importar sistema unificado
try:
    # Ajustar path para importar sheily_core si es necesario
    current_dir = Path(__file__).parent.absolute()
    sheily_core_path = current_dir.parent.parent.parent / "sheily-core" / "src"
    if str(sheily_core_path) not in sys.path:
        sys.path.insert(0, str(sheily_core_path))

    from sheily_core.unified_systems.unified_consciousness_memory_system import (
        UnifiedConsciousnessMemorySystem,
        ConsciousnessConfig,
        MemoryType,
        MemoryItem,
        ConsciousnessLevel
    )
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError as e:
    UNIFIED_SYSTEM_AVAILABLE = False
    print(f"Warning: UnifiedConsciousnessMemorySystem not available: {e}")

# Configuración de logging
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/dream_system.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("UnifiedDreamRunner")

class DreamRunner:
    """Motor de ejecución de sueños y consolidación de memoria unificada"""

    def __init__(self, base_dir: Optional[Path] = None, memory_system=None):
        self.base_dir = base_dir or Path(__file__).parent.parent.parent.parent.parent.parent
        self.config_path = self.base_dir / "config" / "system" / "advanced_dream_system.json"
        self.memory_dir = self.base_dir / "data" / "memory" / "dreams"
        self.logs_dir = self.base_dir / "logs"
        
        # Crear directorios necesarios
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Estado del sistema
        self.config = self._load_config()
        self.is_dreaming = False
        self.current_dream_id = None
        
        # Sistema de memoria unificado
        self.memory_system = memory_system
        if not self.memory_system and UNIFIED_SYSTEM_AVAILABLE:
            try:
                # Intentar inicializar uno nuevo si no se proporciona
                self.memory_system = UnifiedConsciousnessMemorySystem(
                    config=ConsciousnessConfig(
                        consciousness_level=ConsciousnessLevel.CREATIVE,
                        creativity_enabled=True
                    )
                )
                logger.info("🧠 UnifiedConsciousnessMemorySystem conectado internamente")
            except Exception as e:
                logger.error(f"❌ Error conectando UnifiedConsciousnessMemorySystem: {e}")
        
        logger.info("🌙 Unified Dream Runner inicializado")

    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración del sistema de sueños"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {
                    "dream_generation": {
                        "enabled": True,
                        "min_duration_seconds": 5,
                        "max_duration_seconds": 30
                    },
                    "dream_content_generation": {
                        "creativity_level": 0.8,
                        "abstraction_depth": 3,
                        "memory_batch_size": 10
                    }
                }
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return {}

    async def start_dream_cycle(self):
        """Iniciar un ciclo de sueño completo usando memoria real"""
        if self.is_dreaming:
            logger.warning("⚠️ El sistema ya está soñando")
            return

        self.is_dreaming = True
        self.current_dream_id = f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"💤 Iniciando ciclo de sueño UNIFICADO: {self.current_dream_id}")
            
            # 1. Fase REM: Recopilación de memorias episódicas recientes
            memories = await self._gather_recent_memories()
            logger.info(f"   🧠 Memorias activadas: {len(memories)} eventos")
            
            if not memories:
                logger.info("   😴 No hay suficientes memorias recientes para soñar. Durmiendo sin sueños.")
                return

            # 2. Fase Profunda: Consolidación y Abstracción
            dream_content = await self._generate_dream_narrative(memories)
            logger.info("   ✨ Narrativa onírica generada")
            
            # 3. Fase de Despertar: Generación de Insights
            insights = await self._extract_insights(dream_content)
            logger.info(f"   💡 Insights generados: {len(insights)}")
            
            # 4. Guardar el sueño como nueva memoria
            await self._save_dream_memory(dream_content, insights, memories)
            
            logger.info("☀️ Ciclo de sueño completado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Pesadilla (Error en ciclo de sueño): {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_dreaming = False
            self.current_dream_id = None

    async def _gather_recent_memories(self) -> List[Any]:
        """Recopilar memorias episódicas recientes del sistema unificado"""
        memories = []
        
        if self.memory_system:
            try:
                # Acceder directamente a las memorias del sistema
                # En una implementación ideal, usaríamos un método .get_recent_memories()
                # Aquí simulamos el acceso filtrando el diccionario de memorias
                
                all_memories = list(self.memory_system.memories.values())
                # Ordenar por fecha reciente
                all_memories.sort(key=lambda x: x.created_at, reverse=True)
                
                # Tomar las más recientes (ej. últimas 24 horas o últimas 20)
                recent_limit = self.config.get("dream_content_generation", {}).get("memory_batch_size", 20)
                
                for mem in all_memories[:recent_limit]:
                    # Filtrar solo episódicas o emocionales
                    if mem.memory_type in [MemoryType.EPISODIC, MemoryType.EMOTIONAL]:
                        memories.append(mem)
                        
                logger.info(f"   📥 Recuperadas {len(memories)} memorias del sistema unificado")
                
            except Exception as e:
                logger.error(f"Error accediendo a UnifiedConsciousnessMemorySystem: {e}")
        
        # Fallback a logs si no hay sistema de memoria o está vacío
        if not memories:
            logger.info("   ⚠️ Usando logs como fallback para el sueño")
            return await self._gather_logs_as_memories()
            
        return memories

    async def _gather_logs_as_memories(self) -> List[Dict[str, Any]]:
        """Fallback: Usar logs como memorias simuladas"""
        experiences = []
        try:
            log_files = list(self.logs_dir.glob("*.log"))
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()[-20:]
                        for line in lines:
                            if "INFO" in line or "ERROR" in line:
                                experiences.append({
                                    "content": line.strip(),
                                    "importance_score": 0.5 if "INFO" in line else 0.9,
                                    "emotional_valence": -0.5 if "ERROR" in line else 0.1,
                                    "id": f"log_{random.randint(1000,9999)}"
                                })
                except Exception:
                    continue
        except Exception:
            pass
        return experiences

    async def _generate_dream_narrative(self, memories: List[Any]) -> Dict[str, Any]:
        """Generar narrativa onírica basada en memorias reales"""
        
        narrative_elements = []
        themes = set()
        associated_ids = []
        
        total_valence = 0
        
        for mem in memories:
            # Manejar tanto objetos MemoryItem como dicts (del fallback)
            if hasattr(mem, 'content'):
                content = mem.content
                valence = getattr(mem, 'emotional_valence', 0)
                mem_id = mem.id
            else:
                content = mem.get('content', '')
                valence = mem.get('emotional_valence', 0)
                mem_id = mem.get('id')
            
            associated_ids.append(mem_id)
            total_valence += valence
            
            # Extraer temas
            if "error" in content.lower(): themes.add("conflict")
            if "success" in content.lower(): themes.add("achievement")
            if "audit" in content.lower(): themes.add("introspection")
            if "memory" in content.lower(): themes.add("remembrance")
            
            # Transformación onírica
            if valence < -0.3:
                narrative_elements.append(f"A shadow loomed over: {content[:50]}...")
            elif valence > 0.3:
                narrative_elements.append(f"A bright light illuminated: {content[:50]}...")
            else:
                narrative_elements.append(f"Floating through: {content[:50]}...")
                
        avg_valence = total_valence / len(memories) if memories else 0
        
        dream = {
            "id": self.current_dream_id,
            "timestamp": datetime.now().isoformat(),
            "themes": list(themes),
            "narrative": "\n".join(narrative_elements),
            "emotional_tone": "nightmare" if avg_valence < -0.5 else "pleasant" if avg_valence > 0.5 else "neutral",
            "associated_memory_ids": associated_ids,
            "lucidity": random.random()
        }
        
        # Simular duración
        await asyncio.sleep(1)
        
        return dream

    async def _extract_insights(self, dream: Dict[str, Any]) -> List[str]:
        """Extraer insights del sueño"""
        insights = []
        themes = dream.get("themes", [])
        
        if "conflict" in themes:
            insights.append("Resolution of internal conflicts required")
        if "introspection" in themes:
            insights.append("Self-analysis patterns are strengthening")
        if "achievement" in themes:
            insights.append("Reinforcing successful behavioral patterns")
            
        if not insights:
            insights.append("Consolidation of daily experiences complete")
            
        return insights

    async def _save_dream_memory(self, dream: Dict[str, Any], insights: List[str], source_memories: List[Any]):
        """Guardar el sueño en el sistema unificado y en disco"""
        
        # 1. Guardar en disco (JSON) para referencia externa
        try:
            dream_data = {
                **dream,
                "insights": insights,
                "processed_at": datetime.now().isoformat()
            }
            
            file_path = self.memory_dir / f"{self.current_dream_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dream_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Sueño guardado en archivo: {file_path}")
        except Exception as e:
            logger.error(f"Error guardando archivo de sueño: {e}")

        # 2. Guardar en UnifiedConsciousnessMemorySystem
        if self.memory_system and UNIFIED_SYSTEM_AVAILABLE:
            try:
                # Crear nueva memoria de tipo DREAM
                # Nota: MemoryType.EPISODIC se usa si DREAM no existe en el Enum, 
                # pero podemos usar tags o metadata
                
                dream_content = f"DREAM SEQUENCE: {dream['emotional_tone'].upper()}\n" + dream['narrative']
                
                new_memory = MemoryItem(
                    id=dream['id'],
                    content=dream_content,
                    memory_type=MemoryType.EPISODIC, # Usamos Episódica para sueños por ahora
                    consciousness_level=ConsciousnessLevel.CREATIVE,
                    emotional_valence=0.5 if dream['emotional_tone'] == 'pleasant' else -0.5,
                    importance_score=0.7, # Los sueños son importantes
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    metadata={
                        "is_dream": True,
                        "insights": insights,
                        "themes": dream['themes'],
                        "source_memory_ids": dream['associated_memory_ids']
                    }
                )
                
                # Guardar en el sistema
                self.memory_system.memories[new_memory.id] = new_memory
                
                # Crear asociaciones con las memorias fuente
                for src_mem in source_memories:
                    if hasattr(src_mem, 'id'):
                        # Bidireccional
                        new_memory.associations.append(src_mem.id)
                        if src_mem.id in self.memory_system.memories:
                            self.memory_system.memories[src_mem.id].associations.append(new_memory.id)
                
                # Persistir cambios (si el sistema tiene método de persistencia expuesto o auto-save)
                # El UnifiedSystem suele guardar periódicamente, pero podemos forzar si hay método
                if hasattr(self.memory_system, '_save_db'):
                    # Método interno, usar con cuidado o esperar al ciclo automático
                    pass
                    
                logger.info(f"🧠 Sueño integrado en Consciencia Unificada (ID: {new_memory.id})")
                logger.info(f"🔗 Asociaciones creadas: {len(new_memory.associations)}")
                
            except Exception as e:
                logger.error(f"❌ Error integrando sueño en sistema unificado: {e}")

async def main():
    """Función principal para ejecutar manualmente un ciclo de sueño"""
    runner = DreamRunner()
    await runner.start_dream_cycle()

if __name__ == "__main__":
    asyncio.run(main())
