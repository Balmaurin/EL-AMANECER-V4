#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USER PROFILE MANAGER - Memoria Autobiográfica Persistente
=========================================================

Sistema de perfiles de usuario con memoria autobiográfica que aprende
continuamente de cada conversación para construir relaciones personales.

Características:
- Perfiles persistentes con aprendizaje automático
- Detección y procesamiento de introducciones personales
- Memoria autobiográfica que construye relaciones
- Aprendizaje de patrones conversacionales
- Contexto personal para respuestas
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import uuid

from .linguistic_metacognition_system import LinguisticIntent


class ConversationMemory:
    """
    Memoria de una conversación específica con análisis profundo
    """

    def __init__(self,
                 user_id: str,
                 conversation_id: str,
                 timestamp: float,
                 user_message: str,
                 assistant_response: str,
                 intentional_analysis: Dict[str, Any] = None,
                 conscious_state: Dict[str, Any] = None,
                 style_selection: Dict[str, Any] = None,
                 success_score: float = None):

        self.user_id = user_id
        self.conversation_id = conversation_id
        self.timestamp = timestamp
        self.datetime = datetime.fromtimestamp(timestamp)
        self.user_message = user_message
        self.assistant_response = assistant_response
        self.intentional_analysis = intentional_analysis or {}
        self.conscious_state = conscious_state or {}
        self.style_selection = style_selection or {}
        self.success_score = success_score

        # Metadata aprendida
        self.learnings = {
            'intent_preference': intentional_analysis.get('intent', 'general') if isinstance(intentional_analysis, dict) else 'general',
            'emotional_tone': intentional_analysis.get('emotional_charge', 0.0) if isinstance(intentional_analysis, dict) else 0.0,
            'linguistic_complexity': intentional_analysis.get('linguistic_complexity', 0.0) if isinstance(intentional_analysis, dict) else 0.0,
            'conscious_state_phi': conscious_state.get('phi', 0.0) if isinstance(conscious_state, dict) else 0.0,
            'response_style_used': getattr(style_selection, 'primary_style', {}).get('name', 'casual') if hasattr(style_selection, 'primary_style') and isinstance(getattr(style_selection, 'primary_style', {}), dict) else 'casual'
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'timestamp': self.timestamp,
            'user_message': self.user_message,
            'assistant_response': self.assistant_response,
            'intentional_analysis': self.intentional_analysis,
            'conscious_state': self.conscious_state,
            'style_selection': self.style_selection,
            'success_score': self.success_score,
            'learnings': self.learnings
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """Crear instancia desde diccionario"""
        return cls(
            user_id=data['user_id'],
            conversation_id=data['conversation_id'],
            timestamp=data['timestamp'],
            user_message=data['user_message'],
            assistant_response=data['assistant_response'],
            intentional_analysis=data.get('intentional_analysis', {}),
            conscious_state=data.get('conscious_state', {}),
            style_selection=data.get('style_selection', {}),
            success_score=data.get('success_score')
        )


class UserProfileStore:
    """
    Almacén persistente de perfiles de usuario con aprendizaje automático
    """

    def __init__(self, data_dir: str = "data/user_profiles"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_cache: Dict[str, Dict[str, Any]] = {}
        self.conversation_memories: Dict[str, List[ConversationMemory]] = {}

        # Cargar perfiles existentes
        self._load_all_profiles()

    def _load_all_profiles(self):
        """Cargar todos los perfiles desde archivos"""
        if not self.data_dir.exists():
            return

        for profile_file in self.data_dir.glob("profile_*.json"):
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                    user_id = profile_data['user_id']
                    self.profiles_cache[user_id] = profile_data
                    self.conversation_memories[user_id] = []

                    # Cargar memorias de conversación si existen
                    memories_file = self.data_dir / f"memories_{user_id}.json"
                    if memories_file.exists():
                        with open(memories_file, 'r', encoding='utf-8') as mf:
                            memories_data = json.load(mf)
                            for mem_data in memories_data:
                                memory = ConversationMemory.from_dict(mem_data)
                                self.conversation_memories[user_id].append(memory)

            except Exception as e:
                print(f"Error loading profile {profile_file}: {e}")

    def get_or_create_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener perfil existente o crear uno nuevo
        """

        if user_id in self.profiles_cache:
            return self.profiles_cache[user_id]

        # Crear nuevo perfil
        new_profile = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_interaction': time.time(),
            'total_messages': 0,

            # Información personal aprendida
            'identified_name': None,
            'preferred_name': user_id,

            # Estadísticas conversacionales
            'preferred_intentions': {},
            'dominant_emotions': [],
            'conversation_depth': 'surface',
            'relationship_status': 'initial',

            # Preferencias aprendidas
            'style_preferences': {
                'technical': 0.0,
                'poetic': 0.0,
                'casual': 0.0,
                'analytical': 0.0
            },

            # Learning stats
            'learning_stage': 'exploration',
            'adaptability_score': 0.5,

            # Historical data
            'conversation_history': [],
            'emotional_progression': [],
            'learning_milestones': []
        }

        self.profiles_cache[user_id] = new_profile
        self.conversation_memories[user_id] = []

        # Guardar inmediatamente
        self._save_profile(user_id)

        return new_profile

    def update_profile_with_conversation(self, user_id: str, conversation_memory: ConversationMemory):
        """
        Actualizar perfil basado en nueva conversación
        """

        profile = self.get_or_create_profile(user_id)

        # Actualizar estadísticas básicas
        profile['last_interaction'] = time.time()
        profile['total_messages'] += 1

        # Agregar a historial de conversaciones
        profile['conversation_history'].append({
            'timestamp': conversation_memory.timestamp,
            'intent': conversation_memory.learnings.get('intent_preference', 'general'),
            'emotional_tone': conversation_memory.learnings.get('emotional_tone', 0.0),
            'success_score': conversation_memory.success_score
        })

        # Limitar historial reciente
        if len(profile['conversation_history']) > 50:
            profile['conversation_history'] = profile['conversation_history'][-50:]

        # Aprender de la conversación
        self._learn_from_conversation(profile, conversation_memory)

        # Actualizar métricas de relación
        self._update_relationship_metrics(profile)

        # Guardar conversaciones en memoria
        if user_id not in self.conversation_memories:
            self.conversation_memories[user_id] = []
        self.conversation_memories[user_id].append(conversation_memory)

        # Limitar memorias
        if len(self.conversation_memories[user_id]) > 100:
            self.conversation_memories[user_id] = self.conversation_memories[user_id][-100:]

        # Persistir cambios
        self._save_profile(user_id)

        # Actualizar cache
        self.profiles_cache[user_id] = profile

    def _learn_from_conversation(self, profile: Dict[str, Any], conversation: ConversationMemory):
        """
        Aprender patrones de la conversación
        """

        # Aprender intenciones preferidas
        intent = conversation.learnings.get('intent_preference', 'general')
        if intent not in profile['preferred_intentions']:
            profile['preferred_intentions'][intent] = 0

        profile['preferred_intentions'][intent] += 1

        # Aprender emociones dominantes
        emotion_valence = conversation.learnings.get('emotional_tone', 0.0)

        # Categorizar emoción
        if emotion_valence > 0.3:
            emotion_category = 'positive'
        elif emotion_valence < -0.3:
            emotion_category = 'negative'
        else:
            emotion_category = 'neutral'

        # Actualizar contador
        if emotion_category not in profile['dominant_emotions']:
            profile['dominant_emotions'] = []

        # Mantener top 3 emociones
        emotion_counts = {}
        for mem in self.conversation_memories.get(profile['user_id'], []):
            mem_emotion = mem.learnings.get('emotional_tone', 0.0)
            if mem_emotion > 0.3:
                cat = 'positive'
            elif mem_emotion < -0.3:
                cat = 'negative'
            else:
                cat = 'neutral'

            emotion_counts[cat] = emotion_counts.get(cat, 0) + 1

        # Obtener top emociones
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        profile['dominant_emotions'] = [emotion[0] for emotion in sorted_emotions[:3]]

        # Aprender complejidad conversacional
        complexity = conversation.learnings.get('linguistic_complexity', 0.0)
        if complexity > 0.6:
            profile['conversation_depth'] = 'deep'
        elif complexity > 0.3:
            profile['conversation_depth'] = 'moderate'
        else:
            profile['conversation_depth'] = 'surface'

        # Aprender preferencias de estilo
        style_used = conversation.learnings.get('response_style_used', 'casual')
        if style_used in profile['style_preferences']:
            profile['style_preferences'][style_used] += 0.1
            # Normalizar
            total = sum(profile['style_preferences'].values())
            if total > 0:
                for style in profile['style_preferences']:
                    profile['style_preferences'][style] /= total

    def _update_relationship_metrics(self, profile: Dict[str, Any]):
        """
        Actualizar métricas de profundidad de relación
        """

        messages_count = profile['total_messages']

        # Establecer nivel de relación basado en interacciones
        if messages_count < 5:
            profile['relationship_status'] = 'initial'
        elif messages_count < 20:
            profile['relationship_status'] = 'developing'
        elif messages_count < 50:
            profile['relationship_status'] = 'established'
        else:
            profile['relationship_status'] = 'deep'

        # Nivel de aprendizaje
        if messages_count < 10:
            profile['learning_stage'] = 'exploration'
        elif messages_count < 30:
            profile['learning_stage'] = 'adaptation'
        elif messages_count < 100:
            profile['learning_stage'] = 'mastery'
        else:
            profile['learning_stage'] = 'optimization'

        # Calcular score de adaptabilidad
        conversation_count = len(profile['conversation_history'])
        if conversation_count > 10:
            recent_scores = [conv.get('success_score', 0.5) for conv in profile['conversation_history'][-10:]]
            profile['adaptability_score'] = sum(recent_scores) / len(recent_scores)
        else:
            profile['adaptability_score'] = 0.5

    def get_profile_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener estadísticas completas del perfil
        """
        profile = self.get_or_create_profile(user_id)
        return dict(profile)  # Return copy

    def _save_profile(self, user_id: str):
        """Guardar perfil en archivo"""

        if user_id not in self.profiles_cache:
            return

        profile = self.profiles_cache[user_id]
        profile_file = self.data_dir / f"profile_{user_id}.json"

        try:
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving profile {user_id}: {e}")

        # Guardar memorias de conversación
        if user_id in self.conversation_memories and self.conversation_memories[user_id]:
            memories_file = self.data_dir / f"memories_{user_id}.json"
            try:
                memories_data = [mem.to_dict() for mem in self.conversation_memories[user_id][-50:]]  # Últimas 50
                with open(memories_file, 'w', encoding='utf-8') as f:
                    json.dump(memories_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error saving memories {user_id}: {e}")


class ChatDataHarvester:
    """
    Recolector de datos de conversaciones con análisis y aprendizaje
    """

    def __init__(self, profile_store: UserProfileStore):
        self.profile_store = profile_store
        self.collect_session_start = time.time()

    def harvest_interaction(self,
                          user_id: str,
                          user_message: str,
                          assistant_response: str,
                          intentional_analysis: Dict[str, Any] = None,
                          conscious_state: Dict[str, Any] = None,
                          style_selection: Dict[str, Any] = None,
                          success_score: float = None):
        """
        Cosechar datos de una interacción completa para aprendizaje
        """

        # Crear memoria de conversación
        conversation_memory = ConversationMemory(
            user_id=user_id,
            conversation_id=str(uuid.uuid4()),
            timestamp=time.time(),
            user_message=user_message,
            assistant_response=assistant_response,
            intentional_analysis=intentional_analysis,
            conscious_state=conscious_state,
            style_selection=style_selection,
            success_score=success_score
        )

        # Actualizar perfil del usuario con aprendizaje
        self.profile_store.update_profile_with_conversation(user_id, conversation_memory)

    def get_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener insights de aprendizaje para un usuario
        """

        profile = self.profile_store.get_profile_stats(user_id)
        memories = self.profile_store.conversation_memories.get(user_id, [])

        if not memories:
            return {'insights_available': False}

        # Analizar tendencias
        recent_memories = memories[-20:] if len(memories) > 20 else memories

        insights = {
            'insights_available': True,
            'conversation_count': len(memories),
            'avg_success_score': sum(m.success_score for m in recent_memories if m.success_score) / len([m for m in recent_memories if m.success_score]),
            'preferred_topics': profile.get('preferred_intentions', {}),
            'relationship_development': profile.get('relationship_status', 'initial'),
            'learning_adaptability': profile.get('adaptability_score', 0.5)
        }

        return insights


# ============================================================================
# FUNCIONES DE INTRODUCCIONES PERSONALES
# ============================================================================

def process_user_introduction(message: str, profile_store: UserProfileStore) -> Dict[str, Any]:
    """
    Procesar introducciones personales y actualizar perfiles
    """

    message_lower = message.lower()

    # Patrones de introducción
    intro_patterns = [
        r"(?:me\s+llamo|soy|hola\s+soy|mi\s+nombre\s+es)\s+([A-Za-zÁÉÍÓÚáéíóúÑñ]+)",
        r"(?:puedes\s+llamarme|llámame)\s+([A-Za-zÁÉÍÓÚáéíóúÑñ]+)",
        r"(?:encantado\s+de\s+conocerte|es\s+un\s+placer).*([A-Za-zÁÉÍÓÚáéíóúÑñ]+)"
    ]

    import re

    for pattern in intro_patterns:
        match = re.search(pattern, message_lower)
        if match:
            name = match.group(1).strip().title()

            # Actualizar perfil con nombre identificado
            profile_updated = False
            for user_id in profile_store.profiles_cache.keys():
                profile = profile_store.profiles_cache[user_id]
                if name.lower() in message_lower:
                    profile['identified_name'] = name
                    profile['preferred_name'] = name
                    profile_store._save_profile(user_id)
                    profile_updated = True

            return {
                'introduction_detected': True,
                'identified_name': name,
                'updated_user_id': name,
                'profile_updated': profile_updated
            }

    return {
        'introduction_detected': False,
        'identified_name': None,
        'updated_user_id': None,
        'profile_updated': False
    }


def enhance_prompt_with_personal_context(user_id: str, profile_store: UserProfileStore, base_prompt: str) -> str:
    """
    Mejorar prompt con contexto personal del usuario
    """

    profile = profile_store.get_profile_stats(user_id)
    personal_context = []

    # Información básica
    if profile.get('identified_name'):
        personal_context.append(f"El usuario se llama {profile['identified_name']}")

    # Estado de relación
    relationship_status = profile.get('relationship_status', 'initial')
    if relationship_status == 'established':
        personal_context.append("Han tenido varias conversaciones profundas")
    elif relationship_status == 'deep':
        personal_context.append("Existe una relación profunda y de confianza")

    # Preferencias aprendidas
    preferred_intentions = profile.get('preferred_intentions', {})
    if preferred_intentions:
        top_intent = max(preferred_intentions.keys(), key=lambda x: preferred_intentions[x])
        personal_context.append(f"Generalmente prefiere conversaciones {top_intent}")

    # Emociones dominantes
    dominant_emotions = profile.get('dominant_emotions', [])
    if dominant_emotions:
        personal_context.append(f"Emociones típicas: {', '.join(dominant_emotions[:2])}")

    # Profundidad conversacional
    depth = profile.get('conversation_depth', 'surface')
    personal_context.append(f"Nivel de profundidad conversacional: {depth}")

    # Construir prompt mejorado
    if personal_context:
        context_prefix = "\n\nCONTEXTO PERSONAL DEL USUARIO:\n" + "\n".join(f"- {ctx}" for ctx in personal_context)
        return base_prompt + context_prefix + "\n\nINSTRUCCIONES: Adapta tu respuesta considerando este contexto personal."

    return base_prompt


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

_profile_store_instance: Optional[UserProfileStore] = None
_data_harvester_instance: Optional[ChatDataHarvester] = None


def get_user_profile_store() -> UserProfileStore:
    """Obtener instancia singleton del almacén de perfiles"""
    global _profile_store_instance
    if _profile_store_instance is None:
        _profile_store_instance = UserProfileStore()
    return _profile_store_instance


def get_chat_data_harvester() -> ChatDataHarvester:
    """Obtener instancia singleton del cosechador de datos"""
    global _data_harvester_instance
    if _data_harvester_instance is None:
        _data_harvester_instance = ChatDataHarvester(get_user_profile_store())
    return _data_harvester_instance


# ============================================================================
# DEMO Y TESTING
# ============================================================================

def demo_user_profile_system():
    """Demostración del sistema de perfiles de usuario"""

    print("="*70)
    print("🧠 DEMO - SISTEMA DE PERFILES DE USUARIO CON APRENDIZAJE")
    print("="*70)

    # Inicializar sistema
    profile_store = get_user_profile_store()
    data_harvester = get_chat_data_harvester()

    # Simular conversaciones de un usuario
    test_user = "Sergio"
    conversations = [
        {
            'message': 'Me llamo Sergio',
            'response': '¡Hola Sergio! Mucho gusto en conocerte.',
            'intent': 'personal_introduction',
            'success_score': 0.9
        },
        {
            'message': '¿Qué es la inteligencia artificial?',
            'response': 'La inteligencia artificial es...',
            'intent': 'factual_objective',
            'success_score': 0.8
        },
        {
            'message': '¿Qué significa para ti la inteligencia artificial?',
            'response': 'Para mí, la IA representa...',
            'intent': 'emotional_personal',
            'success_score': 0.95
        }
    ]

    for i, conv in enumerate(conversations, 1):
        print(f"💬 Conversación {i}:")

        # Crear ConversationMemory
        memory = ConversationMemory(
            user_id=test_user,
            conversation_id=f"test_conversation_{i}",
            timestamp=time.time(),
            user_message=conv['message'],
            assistant_response=conv['response'],
            intentional_analysis={'intent': conv['intent'], 'linguistic_complexity': 0.5, 'emotional_charge': 0.1},
            conscious_state={'phi': 0.6, 'emotion': 'engaged'},
            style_selection={'primary_style': {'name': 'casual'}},
            success_score=conv['success_score']
        )

        # Harvest data
        data_harvester.harvest_interaction(
            user_id=memory.user_id,
            user_message=memory.user_message,
            assistant_response=memory.assistant_response,
            intentional_analysis=memory.intentional_analysis,
            conscious_state=memory.conscious_state,
            style_selection=memory.style_selection,
            success_score=memory.success_score
        )

    # Mostrar perfil aprendido
    profile = profile_store.get_profile_stats(test_user)
    print("\n📊 PERFIL APRENDIDO:")
    print(f"   Nombre identificado: {profile.get('identified_name')}")
    print(f"   Total mensajes: {profile['total_messages']}")
    print(f"   Preferencias: {profile.get('preferred_intentions', {})}")
    print(f"   Emociones dominantes: {profile.get('dominant_emotions', [])}")
    print(f"   Profundidad: {profile.get('conversation_depth')}")
    print(f"   Relación: {profile.get('relationship_status')}")
    print(f"   Score adaptabilidad: {profile.get('adaptability_score', 0.5):.2f}")

    # Probar contexto personal
    test_prompt = "Responde de manera natural"
    enhanced_prompt = enhance_prompt_with_personal_context(test_user, profile_store, test_prompt)
    print("\n🎭 CONTEXTO PERSONAL GENERADO:")
    print(f"   {enhanced_prompt.replace(test_prompt, '...').strip()}")

    print("\n✅ DEMO COMPLETADA - Sistema de perfiles operativo")
    print("="*70)

    return profile


if __name__ == "__main__":
    demo_user_profile_system()
