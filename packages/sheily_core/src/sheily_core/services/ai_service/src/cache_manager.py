#!/usr/bin/env python3
"""
Cache Manager - Gestor de Caché Avanzado

Este módulo implementa un gestor de caché avanzado con capacidades de:
- Almacenamiento en memoria y disco
- Políticas de expiración
- Compresión de datos
- Estadísticas de rendimiento
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Gestor avanzado de caché"""

    def __init__(self, cache_dir: str = "./cache", max_size: int = 1000):
        """Inicializar gestor de caché"""
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache = {}
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0, "size": 0}

        # Crear directorio de caché si no existe
        os.makedirs(cache_dir, exist_ok=True)

        self.initialized = True
        logger.info(f"CacheManager inicializado en {cache_dir}")

    def get(self, key: str) -> Optional[Any]:
        """Obtener valor de la caché"""
        if key in self.cache:
            entry = self.cache[key]
            # Verificar expiración
            if entry["expires"] and time.time() > entry["expires"]:
                self.delete(key)
                self.stats["misses"] += 1
                return None

            self.stats["hits"] += 1
            return entry["value"]
        else:
            self.stats["misses"] += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establecer valor en caché"""
        try:
            expires = time.time() + ttl if ttl else None

            # Verificar límite de tamaño
            if len(self.cache) >= self.max_size:
                # Eliminar entrada más antigua
                oldest_key = min(
                    self.cache.keys(), key=lambda k: self.cache[k]["created"]
                )
                self.delete(oldest_key)

            self.cache[key] = {
                "value": value,
                "created": time.time(),
                "expires": expires,
                "access_count": 0,
            }

            self.stats["sets"] += 1
            self.stats["size"] = len(self.cache)

            return True
        except Exception as e:
            logger.error(f"Error estableciendo caché para {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Eliminar entrada de caché"""
        if key in self.cache:
            del self.cache[key]
            self.stats["deletes"] += 1
            self.stats["size"] = len(self.cache)
            return True
        return False

    def clear(self) -> bool:
        """Limpiar toda la caché"""
        try:
            self.cache.clear()
            self.stats["size"] = 0
            logger.info("Caché limpiada completamente")
            return True
        except Exception as e:
            logger.error(f"Error limpiando caché: {e}")
            return False

    def has_key(self, key: str) -> bool:
        """Verificar si existe una clave"""
        return key in self.cache

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de caché"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests) if total_requests > 0 else 0

        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "cache_dir": self.cache_dir,
            "initialized": self.initialized,
        }

    def save_to_disk(self, filename: str = "cache_backup.json") -> bool:
        """Guardar caché en disco"""
        try:
            filepath = os.path.join(self.cache_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                # Solo guardar datos no expirados
                current_time = time.time()
                valid_cache = {
                    k: v
                    for k, v in self.cache.items()
                    if not v["expires"] or v["expires"] > current_time
                }
                json.dump(valid_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Caché guardada en {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error guardando caché: {e}")
            return False

    def load_from_disk(self, filename: str = "cache_backup.json") -> bool:
        """Cargar caché desde disco"""
        try:
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded_cache = json.load(f)
                    # Filtrar entradas expiradas
                    current_time = time.time()
                    valid_cache = {
                        k: v
                        for k, v in loaded_cache.items()
                        if not v.get("expires") or v["expires"] > current_time
                    }
                    self.cache.update(valid_cache)
                    self.stats["size"] = len(self.cache)
                logger.info(f"Caché cargada desde {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cargando caché: {e}")
            return False
