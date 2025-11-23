#!/usr/bin/env python3
"""
VECTOR MEMORY SYSTEM - MEMORIA A LARGO PLAZO REAL
=================================================
Implementación de memoria vectorial persistente usando ChromaDB.
Permite almacenamiento, recuperación semántica, refuerzo y consolidación de experiencias.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class VectorMemorySystem:
    """
    Sistema de Memoria Vectorial Persistente.
    Almacena experiencias como vectores para recuperación semántica.
    """

    def __init__(self, persistence_path: str = "./data/chroma_memory"):
        self.logger = logging.getLogger("Sheily.VectorMemory")
        self.persistence_path = persistence_path
        
        # Asegurar directorio de datos
        os.makedirs(persistence_path, exist_ok=True)

        # Inicializar cliente ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=persistence_path)
            self.logger.info(f"ChromaDB initialized at {persistence_path}")
        except Exception as e:
            self.logger.error(f"Failed to init ChromaDB: {e}")
            raise

        # Configurar función de embedding
        # Usamos el default de Chroma (all-MiniLM-L6-v2) que es ligero y efectivo
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Inicializar colecciones
        self.episodic = self._get_or_create_collection("episodic_memory")
        self.semantic = self._get_or_create_collection("semantic_memory")
        self.procedural = self._get_or_create_collection("procedural_memory")

    def _get_or_create_collection(self, name: str):
        """Obtener o crear una colección"""
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Usar similitud coseno
        )

    def add_memory(self, 
                   content: str, 
                   memory_type: str = "episodic", 
                   metadata: Dict[str, Any] = None) -> str:
        """
        Añadir una nueva memoria al sistema.
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Metadatos base
        full_metadata = {
            "timestamp": timestamp,
            "type": memory_type,
            "version": "1.0",
            "utility_score": 1.0, # Score inicial
            "usage_count": 0
        }
        if metadata:
            # Filtrar metadatos para asegurar tipos compatibles con Chroma
            clean_metadata = {k: v for k, v in metadata.items() 
                            if isinstance(v, (str, int, float, bool))}
            full_metadata.update(clean_metadata)

        collection = self._get_collection_by_type(memory_type)
        
        try:
            collection.add(
                documents=[content],
                metadatas=[full_metadata],
                ids=[memory_id]
            )
            self.logger.info(f"Memory stored [{memory_type}]: {memory_id}")
            return memory_id
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return None

    def query_memory(self, 
                     query_text: str, 
                     memory_type: str = "episodic", 
                     n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Buscar memorias similares semánticamente.
        """
        collection = self._get_collection_by_type(memory_type)
        
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Formatear resultados
            formatted_results = []
            if results["ids"]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results["distances"] else None
                    })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error querying memory: {e}")
            return []

    def reinforce_memory(self, memory_id: str, outcome: str, memory_type: str = "episodic"):
        """
        Reforzar o debilitar una memoria basada en el resultado de su uso.
        """
        collection = self._get_collection_by_type(memory_type)
        
        try:
            # Obtener memoria actual
            result = collection.get(ids=[memory_id], include=["metadatas"])
            if not result["ids"]:
                return
                
            current_metadata = result["metadatas"][0]
            
            # Calcular nuevo score de utilidad
            utility_score = current_metadata.get("utility_score", 1.0)
            usage_count = current_metadata.get("usage_count", 0) + 1
            
            if outcome == "success":
                utility_score *= 1.1  # Aumentar 10%
            else:
                utility_score *= 0.8  # Disminuir 20%
                
            # Actualizar metadatos
            current_metadata["utility_score"] = utility_score
            current_metadata["usage_count"] = usage_count
            current_metadata["last_accessed"] = datetime.now().isoformat()
            
            collection.update(
                ids=[memory_id],
                metadatas=[current_metadata]
            )
            self.logger.info(f"Memory {memory_id} reinforced: score={utility_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error reinforcing memory: {e}")

    def consolidate_memories(self):
        """
        'Sueño Digital': Analiza memorias episódicas para generar conocimiento semántico.
        Busca patrones repetidos y crea reglas generales.
        """
        try:
            # Obtener últimas memorias episódicas
            results = self.episodic.get(limit=100, include=["documents", "metadatas"])
            if not results["ids"]:
                return

            documents = results["documents"]
            
            # Análisis simple de patrones (frecuencia de términos clave)
            keywords = ["error", "cpu", "disk", "network", "success", "failed", "clean", "backup"]
            patterns = {}
            
            for doc in documents:
                doc_lower = doc.lower()
                for key in keywords:
                    if key in doc_lower:
                        patterns[key] = patterns.get(key, 0) + 1
            
            # Generar memoria semántica si hay patrones fuertes
            for key, count in patterns.items():
                if count > 2: # Umbral bajo para demo
                    insight = f"Pattern detected: Frequent occurrences of '{key}' related events."
                    
                    # Verificar si ya sabemos esto (búsqueda semántica)
                    existing = self.query_memory(insight, "semantic", n_results=1)
                    if not existing or (existing[0]['distance'] > 0.2):
                        self.add_memory(
                            content=insight, 
                            memory_type="semantic", 
                            metadata={"source": "consolidation", "confidence": 0.8}
                        )
                        self.logger.info(f"🧠 Knowledge Consolidated: {insight}")

        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")

    def prune_memories(self, threshold_score: float = 0.5):
        """
        Olvido Activo: Elimina memorias con bajo score de utilidad.
        """
        for collection in [self.episodic, self.semantic, self.procedural]:
            try:
                # Implementación simplificada: obtener todo y filtrar
                all_mems = collection.get(include=["metadatas"])
                ids_to_delete = []
                
                if not all_mems["ids"]:
                    continue

                for i, meta in enumerate(all_mems["metadatas"]):
                    score = meta.get("utility_score", 1.0)
                    timestamp_str = meta.get("timestamp", datetime.now().isoformat())
                    
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        age_days = (datetime.now() - timestamp).days
                        
                        # Eliminar si tiene score bajo Y es vieja (> 7 días)
                        if score < threshold_score and age_days > 7:
                            ids_to_delete.append(all_mems["ids"][i])
                    except:
                        pass 
                
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
                    self.logger.info(f"Pruned {len(ids_to_delete)} weak memories")
                    
            except Exception as e:
                self.logger.error(f"Error pruning memories: {e}")

    def _get_collection_by_type(self, memory_type: str):
        """Seleccionar colección según tipo"""
        if memory_type == "semantic":
            return self.semantic
        elif memory_type == "procedural":
            return self.procedural
        else:
            return self.episodic

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de memoria"""
        return {
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
            "procedural_count": self.procedural.count(),
            "storage_path": self.persistence_path
        }

# Instancia global
vector_memory = None

def get_vector_memory():
    global vector_memory
    if vector_memory is None:
        vector_memory = VectorMemorySystem()
    return vector_memory
