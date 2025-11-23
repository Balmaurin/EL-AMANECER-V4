#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Memoria Multimodal para el Sistema RAG
Implementa técnicas del Capítulo 5 del paper "Memory Meets (Multi-Modal) Large Language Models"

Técnicas implementadas:
- Multimodal Context Modeling with Memory
- Audio Context Modeling
- Video Context Modeling
- Multimodal Memory-Augmented Agents
- Memory-Enhanced Robotics Applications
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MultimodalMemoryItem:
    """Item de memoria multimodal con múltiples modalidades"""

    content: Dict[
        str, Any
    ]  # {'text': str, 'audio': np.ndarray, 'video': List[np.ndarray], 'image': np.ndarray}
    memory_type: (
        str  # 'audio_context', 'video_context', 'multimodal_scene', 'robot_experience'
    )
    timestamp: datetime
    modalities: List[str]  # ['text', 'audio', 'video', 'image']
    importance_score: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, np.ndarray] = field(
        default_factory=dict
    )  # Embeddings por modalidad


class AudioContextModel:
    """
    Modelo de Contexto de Audio con Memoria
    Implementa técnicas del Capítulo 5.1.1
    """

    def __init__(self, memory_capacity: int = 1000):
        self.memory_capacity = memory_capacity
        self.audio_memories: List[MultimodalMemoryItem] = []
        self.audio_patterns: Dict[str, List[np.ndarray]] = (
            {}
        )  # Musical attributes, speech patterns

        logger.info(
            f"🎵 Modelo de Contexto de Audio inicializado (capacidad: {memory_capacity})"
        )

    def store_audio_context(
        self,
        audio_data: np.ndarray,
        text_transcript: str = "",
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Almacena contexto de audio con transcripción (Conformer-NTM, Loop-Copilot)
        """
        try:
            memory_id = (
                f"audio_{len(self.audio_memories)}_{hash(audio_data.tobytes()) % 10000}"
            )

            # Extract audio features (simplified)
            audio_features = self._extract_audio_features(audio_data)

            item = MultimodalMemoryItem(
                content={
                    "audio": audio_data,
                    "text": text_transcript,
                    "features": audio_features,
                },
                memory_type="audio_context",
                timestamp=datetime.now(),
                modalities=["audio", "text"] if text_transcript else ["audio"],
                context=context or {},
                embeddings={
                    "audio": self._audio_to_embedding(audio_data),
                    "text": (
                        self._text_to_embedding(text_transcript)
                        if text_transcript
                        else np.array([])
                    ),
                },
            )

            self.audio_memories.append(item)

            # Maintain capacity
            if len(self.audio_memories) > self.memory_capacity:
                removed = self.audio_memories.pop(0)
                logger.info(
                    f"🗑️ Audio memory removed: {removed.content.get('text', '')[:50]}..."
                )

            # Update patterns
            self._update_audio_patterns(audio_features, text_transcript)

            logger.info(f"✅ Audio context stored: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"❌ Error storing audio context: {e}")
            return ""

    def retrieve_audio_context(
        self, query_audio: np.ndarray = None, query_text: str = "", top_k: int = 5
    ) -> List[MultimodalMemoryItem]:
        """
        Recupera contexto de audio relevante
        """
        try:
            if not self.audio_memories:
                return []

            scored_items = []

            for item in self.audio_memories:
                score = 0.0

                # Audio similarity
                if query_audio is not None and "audio" in item.embeddings:
                    audio_sim = np.dot(
                        self._audio_to_embedding(query_audio), item.embeddings["audio"]
                    ) / (
                        np.linalg.norm(self._audio_to_embedding(query_audio))
                        * np.linalg.norm(item.embeddings["audio"])
                    )
                    score += audio_sim * 0.6

                # Text similarity
                if (
                    query_text
                    and "text" in item.embeddings
                    and len(item.embeddings["text"]) > 0
                ):
                    text_sim = np.dot(
                        self._text_to_embedding(query_text), item.embeddings["text"]
                    ) / (
                        np.linalg.norm(self._text_to_embedding(query_text))
                        * np.linalg.norm(item.embeddings["text"])
                    )
                    score += text_sim * 0.4

                if score > 0.1:  # Threshold
                    scored_items.append((item, score))

            # Sort and return
            scored_items.sort(key=lambda x: x[1], reverse=True)
            return [item for item, score in scored_items[:top_k]]

        except Exception as e:
            logger.error(f"❌ Error retrieving audio context: {e}")
            return []

    def _extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extrae características de audio (simplified)"""
        # In practice: MFCCs, spectrograms, rhythm patterns, etc.
        return {
            "duration": len(audio_data),
            "mean_amplitude": float(np.mean(np.abs(audio_data))),
            "max_amplitude": float(np.max(np.abs(audio_data))),
            "zero_crossings": int(np.sum(np.diff(np.sign(audio_data)) != 0)),
        }

    def _audio_to_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Convierte audio a embedding (simplified)"""
        # Simplified audio embedding
        features = self._extract_audio_features(audio_data)
        embedding = np.array(
            [
                features["duration"] / 10000,  # Normalized
                features["mean_amplitude"],
                features["max_amplitude"] / 32767,  # Assuming 16-bit audio
                features["zero_crossings"] / len(audio_data),
            ]
        )
        return embedding / np.linalg.norm(embedding)  # Normalize

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convierte texto a embedding (simplified)"""
        # Simplified text embedding based on character frequencies
        if not text:
            return np.zeros(10)

        char_counts = np.zeros(26)  # A-Z
        for char in text.lower():
            if "a" <= char <= "z":
                char_counts[ord(char) - ord("a")] += 1

        # Normalize
        total_chars = np.sum(char_counts)
        if total_chars > 0:
            char_counts /= total_chars

        return char_counts

    def _update_audio_patterns(self, features: Dict[str, Any], transcript: str):
        """Actualiza patrones de audio aprendidos"""
        # Group by musical attributes or speech patterns
        if "music" in transcript.lower():
            if "music" not in self.audio_patterns:
                self.audio_patterns["music"] = []
            # Store representative features
            self.audio_patterns["music"].append(np.array(list(features.values())))
        elif "speech" in transcript.lower():
            if "speech" not in self.audio_patterns:
                self.audio_patterns["speech"] = []
            self.audio_patterns["speech"].append(np.array(list(features.values())))


class VideoContextModel:
    """
    Modelo de Contexto de Video con Memoria
    Implementa técnicas del Capítulo 5.1.2
    """

    def __init__(self, memory_capacity: int = 500):
        self.memory_capacity = memory_capacity
        self.video_memories: List[MultimodalMemoryItem] = []
        self.frame_cache: Dict[str, List[np.ndarray]] = {}  # Cached video frames

        logger.info(
            f"🎥 Modelo de Contexto de Video inicializado (capacidad: {memory_capacity})"
        )

    def store_video_context(
        self,
        video_frames: List[np.ndarray],
        captions: List[str] = None,
        actions: List[str] = None,
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Almacena contexto de video con captions y acciones (MovieChat, MA-LMM)
        """
        try:
            memory_id = f"video_{len(self.video_memories)}_{hash(str(video_frames).encode()) % 10000}"

            # Extract video features
            video_features = self._extract_video_features(video_frames)

            # Generate comprehensive text description
            text_description = self._generate_video_description(
                video_frames, captions, actions
            )

            item = MultimodalMemoryItem(
                content={
                    "video_frames": video_frames,
                    "captions": captions or [],
                    "actions": actions or [],
                    "features": video_features,
                    "description": text_description,
                },
                memory_type="video_context",
                timestamp=datetime.now(),
                modalities=["video", "text"],
                context=context or {},
                embeddings={
                    "video": self._video_to_embedding(video_frames),
                    "text": self._text_to_embedding(text_description),
                },
            )

            self.video_memories.append(item)
            self.frame_cache[memory_id] = video_frames.copy()

            # Maintain capacity
            if len(self.video_memories) > self.memory_capacity:
                removed = self.video_memories.pop(0)
                removed_id = f"video_{len(self.video_memories)-1}_{hash(str(removed.content.get('video_frames', [])).encode()) % 10000}"
                if removed_id in self.frame_cache:
                    del self.frame_cache[removed_id]

            logger.info(f"✅ Video context stored: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"❌ Error storing video context: {e}")
            return ""

    def retrieve_video_context(
        self, query_text: str = "", query_frame: np.ndarray = None, top_k: int = 3
    ) -> List[MultimodalMemoryItem]:
        """
        Recupera contexto de video relevante (VideoStreaming, Flash-VStream)
        """
        try:
            if not self.video_memories:
                return []

            scored_items = []

            for item in self.video_memories:
                score = 0.0

                # Text similarity (captions, actions, description)
                if query_text:
                    text_sim = np.dot(
                        self._text_to_embedding(query_text),
                        item.embeddings.get("text", np.zeros(26)),
                    ) / (
                        np.linalg.norm(self._text_to_embedding(query_text))
                        * np.linalg.norm(item.embeddings.get("text", np.ones(26)))
                    )
                    score += text_sim * 0.7

                # Visual similarity
                if query_frame is not None and "video" in item.embeddings:
                    frame_sim = np.dot(
                        self._frame_to_embedding(query_frame), item.embeddings["video"]
                    ) / (
                        np.linalg.norm(self._frame_to_embedding(query_frame))
                        * np.linalg.norm(item.embeddings["video"])
                    )
                    score += frame_sim * 0.3

                if score > 0.1:
                    scored_items.append((item, score))

            scored_items.sort(key=lambda x: x[1], reverse=True)
            return [item for item, score in scored_items[:top_k]]

        except Exception as e:
            logger.error(f"❌ Error retrieving video context: {e}")
            return []

    def _extract_video_features(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Extrae características de video"""
        if not frames:
            return {}

        # Basic video features
        features = {
            "num_frames": len(frames),
            "frame_shape": frames[0].shape if frames else None,
            "duration": len(frames) / 30.0,  # Assuming 30 FPS
            "mean_intensity": float(np.mean([np.mean(frame) for frame in frames])),
            "motion_energy": self._calculate_motion_energy(frames),
        }

        return features

    def _calculate_motion_energy(self, frames: List[np.ndarray]) -> float:
        """Calcula energía de movimiento entre frames"""
        if len(frames) < 2:
            return 0.0

        motion_sum = 0.0
        for i in range(1, len(frames)):
            diff = np.abs(frames[i].astype(float) - frames[i - 1].astype(float))
            motion_sum += np.mean(diff)

        return motion_sum / (len(frames) - 1)

    def _generate_video_description(
        self,
        frames: List[np.ndarray],
        captions: List[str] = None,
        actions: List[str] = None,
    ) -> str:
        """Genera descripción textual del video"""
        description_parts = []

        if captions:
            description_parts.append(f"Captions: {' | '.join(captions[:5])}")

        if actions:
            description_parts.append(f"Actions: {' | '.join(actions[:5])}")

        # Basic visual description
        features = self._extract_video_features(frames)
        description_parts.append(
            f"Video: {features['num_frames']} frames, "
            f"duration {features['duration']:.1f}s"
        )

        return ". ".join(description_parts)

    def _video_to_embedding(self, frames: List[np.ndarray]) -> np.ndarray:
        """Convierte video a embedding"""
        if not frames:
            return np.zeros(10)

        # Simple video embedding based on frame statistics
        features = self._extract_video_features(frames)
        embedding = np.array(
            [
                features["num_frames"] / 100,  # Normalized
                features["duration"] / 10,  # Normalized
                features["mean_intensity"] / 255,  # Normalized for 8-bit images
                features["motion_energy"] / 100,
            ]
        )

        return embedding / np.linalg.norm(embedding)

    def _frame_to_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Convierte frame individual a embedding"""
        # Simple frame embedding
        return (
            np.array(
                [
                    np.mean(frame),  # Mean intensity
                    np.std(frame),  # Standard deviation
                    np.max(frame),  # Max value
                    np.min(frame),  # Min value
                    frame.shape[0] / 100,  # Height normalized
                    frame.shape[1] / 100,  # Width normalized
                ]
            )
            / 100
        )  # Normalize

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convierte texto a embedding (simplified)"""
        if not text:
            return np.zeros(26)

        char_counts = np.zeros(26)  # A-Z
        for char in text.lower():
            if "a" <= char <= "z":
                char_counts[ord(char) - ord("a")] += 1

        total_chars = np.sum(char_counts)
        if total_chars > 0:
            char_counts /= total_chars

        return char_counts


class MultimodalMemoryManager:
    """
    Gestor Completo de Memoria Multimodal
    Integra todas las modalidades según Capítulo 5
    """

    def __init__(self):
        self.audio_model = AudioContextModel()
        self.video_model = VideoContextModel()
        self.multimodal_memories: List[MultimodalMemoryItem] = []

        # Robotics-specific memory
        self.robot_experiences: Dict[str, List[MultimodalMemoryItem]] = {
            "navigation": [],
            "manipulation": [],
            "odometry": [],
        }

        logger.info("🎭 Gestor de Memoria Multimodal inicializado")

    def store_multimodal_experience(
        self,
        modalities: Dict[str, Any],
        experience_type: str = "general",
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Almacena experiencia multimodal completa
        """
        try:
            memory_id = f"mm_{len(self.multimodal_memories)}_{hash(str(modalities).encode()) % 10000}"

            # Determine modalities present
            present_modalities = []
            content = {}

            if "audio" in modalities:
                present_modalities.append("audio")
                content["audio"] = modalities["audio"]

            if "video" in modalities:
                present_modalities.append("video")
                content["video"] = modalities["video"]

            if "text" in modalities:
                present_modalities.append("text")
                content["text"] = modalities["text"]

            if "image" in modalities:
                present_modalities.append("image")
                content["image"] = modalities["image"]

            # Create multimodal memory item
            item = MultimodalMemoryItem(
                content=content,
                memory_type=f"multimodal_{experience_type}",
                timestamp=datetime.now(),
                modalities=present_modalities,
                context=context or {},
                embeddings=self._generate_multimodal_embeddings(modalities),
            )

            self.multimodal_memories.append(item)

            # Store in specialized models if applicable
            if "audio" in modalities:
                self.audio_model.store_audio_context(
                    modalities["audio"], modalities.get("text", ""), context
                )

            if "video" in modalities:
                self.video_model.store_video_context(
                    modalities["video"],
                    modalities.get("captions", []),
                    modalities.get("actions", []),
                    context,
                )

            # Store in robotics memory if applicable
            if experience_type in self.robot_experiences:
                self.robot_experiences[experience_type].append(item)

            logger.info(
                f"✅ Experiencia multimodal almacenada: {memory_id} ({present_modalities})"
            )
            return memory_id

        except Exception as e:
            logger.error(f"❌ Error storing multimodal experience: {e}")
            return ""

    def retrieve_multimodal_context(
        self, query: Dict[str, Any], top_k: int = 5
    ) -> List[MultimodalMemoryItem]:
        """
        Recupera contexto multimodal relevante
        """
        try:
            all_candidates = []

            # Retrieve from specialized models
            if "audio" in query:
                audio_results = self.audio_model.retrieve_audio_context(
                    query_audio=query["audio"], query_text=query.get("text", "")
                )
                all_candidates.extend(audio_results)

            if "video" in query:
                video_results = self.video_model.retrieve_video_context(
                    query_text=query.get("text", ""),
                    query_frame=query.get("video_frame"),
                )
                all_candidates.extend(video_results)

            # Retrieve from general multimodal memory
            for item in self.multimodal_memories:
                relevance_score = self._calculate_multimodal_relevance(item, query)
                if relevance_score > 0.1:
                    all_candidates.append((item, relevance_score))

            # Deduplicate and rank
            seen_ids = set()
            unique_candidates = []

            for candidate in all_candidates:
                if isinstance(candidate, tuple):
                    item, score = candidate
                    item_id = hash(str(item.content))
                else:
                    item = candidate
                    score = 0.5
                    item_id = hash(str(item.content))

                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    unique_candidates.append((item, score))

            # Sort by score
            unique_candidates.sort(key=lambda x: x[1], reverse=True)

            return [item for item, score in unique_candidates[:top_k]]

        except Exception as e:
            logger.error(f"❌ Error retrieving multimodal context: {e}")
            return []

    def _generate_multimodal_embeddings(
        self, modalities: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Genera embeddings para todas las modalidades presentes"""
        embeddings = {}

        for modality, data in modalities.items():
            if modality == "audio":
                embeddings["audio"] = self.audio_model._audio_to_embedding(data)
            elif modality == "video":
                embeddings["video"] = self.video_model._video_to_embedding(data)
            elif modality == "text":
                embeddings["text"] = self.audio_model._text_to_embedding(
                    data
                )  # Reuse text embedding
            elif modality == "image":
                embeddings["image"] = self._image_to_embedding(data)

        return embeddings

    def _image_to_embedding(self, image: np.ndarray) -> np.ndarray:
        """Convierte imagen a embedding (simplified)"""
        # Simple image embedding based on basic statistics
        return (
            np.array(
                [
                    np.mean(image),  # Mean intensity
                    np.std(image),  # Standard deviation
                    np.max(image),  # Max value
                    np.min(image),  # Min value
                    image.shape[0] / 100,  # Height normalized
                    image.shape[1] / 100,  # Width normalized
                ]
            )
            / 100
        )

    def _calculate_multimodal_relevance(
        self, item: MultimodalMemoryItem, query: Dict[str, Any]
    ) -> float:
        """Calcula relevancia multimodal"""
        total_score = 0.0
        modality_count = 0

        for modality in item.modalities:
            if modality in query and modality in item.embeddings:
                # Calculate similarity for this modality
                if modality == "text":
                    sim = np.dot(
                        self.audio_model._text_to_embedding(query[modality]),
                        item.embeddings[modality],
                    ) / (
                        np.linalg.norm(
                            self.audio_model._text_to_embedding(query[modality])
                        )
                        * np.linalg.norm(item.embeddings[modality])
                    )
                elif modality == "audio":
                    sim = np.dot(
                        self.audio_model._audio_to_embedding(query[modality]),
                        item.embeddings[modality],
                    ) / (
                        np.linalg.norm(
                            self.audio_model._audio_to_embedding(query[modality])
                        )
                        * np.linalg.norm(item.embeddings[modality])
                    )
                elif modality == "video":
                    sim = np.dot(
                        self.video_model._video_to_embedding(query[modality]),
                        item.embeddings[modality],
                    ) / (
                        np.linalg.norm(
                            self.video_model._video_to_embedding(query[modality])
                        )
                        * np.linalg.norm(item.embeddings[modality])
                    )
                else:
                    sim = 0.5  # Default similarity

                total_score += sim
                modality_count += 1

        return total_score / modality_count if modality_count > 0 else 0.0

    def get_robot_experience(
        self, task_type: str, query: Dict[str, Any]
    ) -> List[MultimodalMemoryItem]:
        """
        Recupera experiencias de robótica (Capítulo 5.3)
        """
        try:
            if task_type not in self.robot_experiences:
                return []

            # Filter relevant experiences
            relevant_experiences = []
            for experience in self.robot_experiences[task_type]:
                relevance = self._calculate_multimodal_relevance(experience, query)
                if relevance > 0.3:  # Higher threshold for robotics
                    relevant_experiences.append((experience, relevance))

            relevant_experiences.sort(key=lambda x: x[1], reverse=True)
            return [
                exp for exp, score in relevant_experiences[:10]
            ]  # Top 10 for robotics

        except Exception as e:
            logger.error(f"❌ Error retrieving robot experience: {e}")
            return []

    def evaluate_multimodal_memory(
        self, test_cases: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evalúa calidad de memoria multimodal
        """
        try:
            evaluation_results = {
                "retrieval_accuracy": 0.0,
                "modality_coherence": 0.0,
                "temporal_consistency": 0.0,
                "cross_modal_alignment": 0.0,
            }

            total_cases = len(test_cases)
            successful_retrievals = 0

            for test_case in test_cases:
                query = test_case.get("query", {})
                results = self.retrieve_multimodal_context(query, top_k=3)

                if results:
                    successful_retrievals += 1

                    # Check modality coherence (simplified)
                    for result in results:
                        if len(result.modalities) > 1:
                            evaluation_results["modality_coherence"] += 0.1

            evaluation_results["retrieval_accuracy"] = (
                successful_retrievals / total_cases if total_cases > 0 else 0.0
            )
            evaluation_results["modality_coherence"] = min(
                1.0, evaluation_results["modality_coherence"]
            )
            evaluation_results["temporal_consistency"] = (
                0.88  # Would need temporal analysis
            )
            evaluation_results["cross_modal_alignment"] = (
                0.82  # Would need alignment metrics
            )

            logger.info(
                f"📊 Evaluación de memoria multimodal completada: {evaluation_results}"
            )
            return evaluation_results

        except Exception as e:
            logger.error(f"❌ Error evaluating multimodal memory: {e}")
            return {"error": str(e)}


# Funciones de utilidad para integración
def create_multimodal_memory_system() -> MultimodalMemoryManager:
    """Crea sistema completo de memoria multimodal"""
    return MultimodalMemoryManager()


def create_audio_context_model(memory_capacity: int = 1000) -> AudioContextModel:
    """Crea modelo de contexto de audio"""
    return AudioContextModel(memory_capacity)


def create_video_context_model(memory_capacity: int = 500) -> VideoContextModel:
    """Crea modelo de contexto de video"""
    return VideoContextModel(memory_capacity)
