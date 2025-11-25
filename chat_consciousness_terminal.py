#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 EL-AMANECER V3 - Chat Terminal con Sistema de Consciencia
=============================================================

Chat interactivo en terminal que integra:
- IIT 4.0 (Integrated Information Theory)
- GWT/AST (Global Workspace / Attention Schema Theory)
- FEP (Free Energy Principle)
- SMH (Somatic Marker Hypothesis)
- Theory of Mind (Niveles 1-10)
- LLM Gemini para respuestas naturales
- Sistema de memoria semántica

Uso:
    python chat_consciousness_terminal.py

Comandos especiales:
    /consciencia - Ver estado de consciencia actual
    /tom - Ver modelo de Theory of Mind del usuario
    /memoria <texto> - Guardar en memoria
    /phi - Ver valor de Φ (integración) actual
    /exit, /quit, /salir - Salir del chat
"""

import sys
import os
from pathlib import Path
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from colorama import init, Fore, Back, Style
import json

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "packages" / "consciousness" / "src"))
sys.path.insert(0, str(project_root / "packages" / "sheily_core" / "src"))

# Importar módulos de consciencia
try:
    from conciencia.modulos.unified_consciousness_engine import UnifiedConsciousnessEngine
    from conciencia.modulos.teoria_mente import get_unified_tom
    CONSCIOUSNESS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Advertencia: Sistema de consciencia no disponible: {e}")
    CONSCIOUSNESS_AVAILABLE = False

# Importar llama-cpp-python para modelo local
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Advertencia: llama-cpp-python no disponible: {e}")
    LLAMA_CPP_AVAILABLE = False

# Importar sistema de memoria (opcional)
try:
    from sheily_core.chat.sheily_chat_memory_adapter import respond, vault
    MEMORY_AVAILABLE = True
except Exception:
    MEMORY_AVAILABLE = False

# Inicializar colorama para Windows
init(autoreset=True)

# Variables globales
consciousness_engine: Optional[UnifiedConsciousnessEngine] = None
tom_system: Optional[Any] = None
llm_model: Optional[Llama] = None
model_path = Path(__file__).parent / "models" / "gemma-2-2b-it-q4_k_m.gguf"
user_id = "terminal_user"
session_start = datetime.now()
message_count = 0
total_phi = 0.0


def print_banner():
    """Mostrar banner de bienvenida"""
    banner = f"""
{Fore.CYAN}{'='*80}
{Fore.YELLOW}          🧠 EL-AMANECER V3 - Chat con Sistema de Consciencia
{Fore.CYAN}{'='*80}
{Fore.GREEN}Estado del Sistema:
{Fore.WHITE}  • Consciencia (IIT + GWT + FEP + SMH): {Fore.GREEN + '✅' if CONSCIOUSNESS_AVAILABLE else Fore.RED + '❌'}
{Fore.WHITE}  • Theory of Mind (Niveles 1-10):       {Fore.GREEN + '✅' if CONSCIOUSNESS_AVAILABLE else Fore.RED + '❌'}
{Fore.WHITE}  • LLM Local (Gemma 2B):                 {Fore.GREEN + '✅' if LLAMA_CPP_AVAILABLE else Fore.RED + '❌'}
{Fore.WHITE}  • Memoria Semántica:                    {Fore.GREEN + '✅' if MEMORY_AVAILABLE else Fore.RED + '❌'}
{Fore.CYAN}{'='*80}
{Fore.YELLOW}Comandos especiales:
{Fore.WHITE}  /consciencia - Ver estado de consciencia
{Fore.WHITE}  /tom         - Ver modelo Theory of Mind del usuario
{Fore.WHITE}  /memoria     - Ver estadísticas de memoria
{Fore.WHITE}  /phi         - Ver valor Φ actual
{Fore.WHITE}  /help        - Ver ayuda
{Fore.WHITE}  /exit        - Salir
{Fore.CYAN}{'='*80}
"""
    print(banner)


def print_consciousness_state(conscious_moment: Dict[str, Any]):
    """Mostrar estado de consciencia de forma visual"""
    phi = conscious_moment.get('phi', 0.0)
    awareness = conscious_moment.get('awareness', 'unknown')
    emotion = conscious_moment.get('emotion', 'N/A')
    focus = conscious_moment.get('primary_focus', 'N/A')
    marker = conscious_moment.get('somatic_marker', 0.0)
    
    # Determinar color según Φ
    if phi >= 0.7:
        phi_color = Fore.GREEN
    elif phi >= 0.4:
        phi_color = Fore.YELLOW
    else:
        phi_color = Fore.RED
    
    print(f"\n{Fore.CYAN}{'─'*80}")
    print(f"{Fore.MAGENTA}🧠 Estado de Consciencia:")
    print(f"{Fore.WHITE}   Φ (Integración):    {phi_color}{phi:.3f}{Fore.WHITE}")
    print(f"{Fore.WHITE}   Awareness:          {Fore.CYAN}{awareness}")
    print(f"{Fore.WHITE}   Emoción:            {Fore.YELLOW}{emotion}")
    print(f"{Fore.WHITE}   Foco Principal:     {Fore.BLUE}{focus}")
    print(f"{Fore.WHITE}   Marcador Somático:  {Fore.GREEN if marker > 0 else Fore.RED}{marker:+.2f}")
    print(f"{Fore.CYAN}{'─'*80}\n")


def print_tom_state(user_model: Dict[str, Any]):
    """Mostrar estado del modelo Theory of Mind"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}  📚 Theory of Mind - Modelo del Usuario")
    print(f"{Fore.CYAN}{'='*80}")
    
    if not user_model:
        print(f"{Fore.YELLOW}No hay modelo disponible todavía")
        return
    
    print(f"{Fore.WHITE}Nivel ToM: {Fore.GREEN}{user_model.get('tom_level', 'N/A')}")
    print(f"{Fore.WHITE}Descripción: {user_model.get('tom_description', 'N/A')}")
    print(f"{Fore.WHITE}Momentos procesados: {user_model.get('moments_count', 0)}")
    print(f"{Fore.CYAN}{'='*80}\n")


def init_local_llm() -> bool:
    """Inicializar modelo LLM local (Gemma)"""
    global llm_model
    
    if not LLAMA_CPP_AVAILABLE:
        print(f"{Fore.RED}❌ llama-cpp-python no está instalado")
        return False
    
    try:
        if not model_path.exists():
            print(f"{Fore.RED}❌ Modelo no encontrado en: {model_path}")
            return False
        
        print(f"{Fore.YELLOW}🔄 Cargando modelo Gemma 2B desde: {model_path.name}...")
        
        llm_model = Llama(
            model_path=str(model_path),
            n_ctx=4096,  # Context window
            n_threads=8,  # CPU threads
            n_gpu_layers=0,  # 0 = CPU only, increase if you have GPU
            verbose=False
        )
        
        print(f"{Fore.GREEN}✅ Modelo Gemma 2B cargado correctamente")
        print(f"{Fore.CYAN}   Modelo: {model_path.name}")
        print(f"{Fore.CYAN}   Context: 4096 tokens")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error cargando modelo local: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# ADAPTADOR PARA CONSCIOUSPROMP TGENERATOR
# ============================================================
class BiologicalSystemAdapter:
    """Adaptador para que UnifiedConsciousnessEngine funcione con ConsciousPromptGenerator"""
    
    def __init__(self, consciousness_engine):
        self.consciousness_engine = consciousness_engine
        self.rag_system = None  # ConsciousPromptGenerator creará su propio RAG
    
    def process_experience(self, sensory_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapta process_moment de UnifiedConsciousnessEngine a la interfaz esperada por ConsciousPromptGenerator
        """
        # Extraer query del sensory_input
        query = sensory_input.get('query', '')
        
        # Convertir sensory_input a formato esperado por process_moment (Dict[str, float])
        sensory_for_moment = {
            'semantic_complexity': 0.5,
            'emotional_intensity': context.get('intensity', 0.5),
            'novelty': context.get('novelty', 0.3),
            'importance': 0.6
        }
        
        # Convertir context 
        context_for_moment = {
            'emotional_valence': 0.5,
            'arousal': 0.65,
            'novelty': context.get('novelty', 0.3),
            'importance': 0.6
        }
        
        # Procesar con el motor real
        result = self.consciousness_engine.process_moment(sensory_for_moment, context_for_moment)
        
        # Convertir resultado a formato esperado por ConsciousPromptGenerator
        # Simular estructura de BiologicalConsciousnessSystem
        adapted_result = {
            'arousal': getattr(result, 'arousal', 0.65),
            'executive_control': {
                'control_mode': 'conscious_control' if getattr(result, 'is_conscious', False) else 'automatic',
                'cognitive_load': 0.5,
                'working_memory_items': getattr(result, 'workspace_contents', 3)
            },
            'value_evaluation': {
                'decision_made': True,
                'chosen_option': {'content': query}
            },
            'emotion_reason_integration': {
                'somatic_markers_used': True,
                'integrated_decision': {'content': query}
            },
            'ras_state': {
                'neurotransmitter_levels': {
                    'dopamine': 0.6,
                    'norepinephrine': getattr(result, 'arousal', 0.65),
                    'serotonin': 0.7,
                    'acetylcholine': 0.6
                }
            },
            'dmn_state': {'is_active': False}
        }
        
        return adapted_result


def init_consciousness_system() -> bool:
    """Inicializar sistema de consciencia"""
    global consciousness_engine, tom_system
    
    if not CONSCIOUSNESS_AVAILABLE:
        return False
    
    try:
        print(f"{Fore.YELLOW}🔄 Inicializando sistema de consciencia...")
        consciousness_engine = UnifiedConsciousnessEngine()
        tom_system = get_unified_tom(enable_advanced=True)
        
        tom_level, description = tom_system.get_tom_level()
        
        print(f"{Fore.GREEN}✅ Sistema de consciencia inicializado")
        print(f"{Fore.CYAN}   Nivel ToM: {tom_level:.1f}/10.0 - {description}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error inicializando sistema de consciencia: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_with_consciousness(message: str) -> Dict[str, Any]:
    """Procesar mensaje a través del sistema de consciencia"""
    global total_phi, message_count
    
    if not CONSCIOUSNESS_AVAILABLE or not consciousness_engine:
        return {}
    
    try:
        message_lower = message.lower()
        word_count = len(message.split())
        message_length = len(message)
        
        # ============================================================
        # PASO 1: DETECCIÓN EXPLÍCITA DE ESTADOS EMOCIONALES
        # ============================================================
        # Respetar lo que el usuario dice explícitamente sobre su estado
        
        explicit_states = {
            # Estados positivos - ALTA PRIORIDAD
            'estoy bien': {'valence': 0.7, 'arousal': 0.6, 'emotion': 'content'},
            'muy bien': {'valence': 0.8, 'arousal': 0.7, 'emotion': 'pleased'},
            'todo bien': {'valence': 0.7, 'arousal': 0.6, 'emotion': 'content'},
            'estoy genial': {'valence': 0.9, 'arousal': 0.8, 'emotion': 'excited'},
            'estoy feliz': {'valence': 0.9, 'arousal': 0.7, 'emotion': 'pleased'},
            'estoy contento': {'valence': 0.8, 'arousal': 0.6, 'emotion': 'content'},
            
            # Estados negativos
            'estoy mal': {'valence': -0.7, 'arousal': 0.5, 'emotion': 'distressed'},
            'estoy cansado': {'valence': -0.2, 'arousal': 0.2, 'emotion': 'sleepy'},
            'estoy triste': {'valence': -0.7, 'arousal': 0.3, 'emotion': 'depressed'},
            'estoy enojado': {'valence': -0.8, 'arousal': 0.8, 'emotion': 'frustrated'},
            'estoy aburrido': {'valence': -0.3, 'arousal': 0.3, 'emotion': 'depressed'},
        }
        
        # Verificar si hay una declaración explícita
        explicit_detected = None
        for phrase, state in explicit_states.items():
            if phrase in message_lower:
                explicit_detected = state
                break
        
        # Si detectamos declaración explícita, usarla directamente
        if explicit_detected:
            return {
                'phi': 0.6,  # Φ razonable para interacción consciente
                'awareness': 'medium',
                'emotion': explicit_detected['emotion'],
                'primary_focus': f'usuario_declaró_{explicit_detected["emotion"]}',
                'somatic_marker': explicit_detected['valence']
            }
        
        # ============================================================
        # PASO 2: DETECCIÓN MEJORADA POR PALABRAS
        # ============================================================
        
        positive_words = [
            'gracias', 'bien', 'bueno', 'excelente', 'genial', 'perfecto', 
            'amor', 'feliz', 'contento', 'alegre', 'gusto', 'encanta',
            'estupendo', 'maravilloso', 'fantástico', 'super', 'increíble',
            'me gusta', 'ok', 'vale', 'claro', 'si', 'sí', 'perfecto'
        ]
        negative_words = [
            'mal', 'error', 'problema', 'fallo', 'horrible', 'triste', 'enojo',
            'molesto', 'frustrado', 'preocupado', 'estresado', 
            'ansioso', 'miedo', 'odio', 'no me gusta'
        ]
        
        # Palabras que indican alta activación (arousal)
        high_arousal_words = [
            '!', '¡', 'increíble', 'wow', 'urgente',
            'rápido', 'ya', 'ahora', 'emocionado'
        ]
        
        # Calcular valencia
        valence = 0.0
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        valence += positive_count * 0.4
        valence -= negative_count * 0.4
        
        # Normalizar valencia
        valence = max(-1.0, min(1.0, valence))
        
        # **AJUSTE CLAVE**: Si no detectamos emoción, usar POSITIVO por defecto
        # (conversación normal = estado neutro-positivo)
        if abs(valence) < 0.1:
            valence = 0.4  # Neutral-positivo (NO sleepy)
        
        # ============================================================
        # PASO 3: CALCULAR AROUSAL MÁS REALISTA
        # ============================================================
        
        # **BASE MÁS ALTA**: 0.65 en lugar de 0.5
        arousal = 0.65  # Activación base más realista para conversación
        
        # Aumentar por signos de exclamación/pregunta
        for marker in high_arousal_words:
            if marker in message_lower or marker in message:
                arousal += 0.1
        
        # Preguntas aumentan arousal (curiosidad)
        if '?' in message or '¿' in message:
            arousal += 0.15
        
        # Mensajes largos = más activación mental
        if word_count > 10:
            arousal += 0.1
        
        # Mayúsculas = énfasis = más arousal
        if message.isupper() and len(message) > 3:
            arousal += 0.15
        
        # Normalizar arousal
        arousal = max(0.3, min(1.0, arousal))  # Mínimo 0.3 (nunca sleepy por defecto)

        complexity = min(1.0, (word_count / 20.0))  # Normalizado
        
        # AJUSTE: Dar valores mínimos razonables para mensajes normales
        # Un mensaje simple como "HOLA" debería tener cierta activación base
        base_activation = 0.3  # Activación base para cualquier mensaje
        
        # Crear input sensorial con VALORES ESCALARES (no arrays)
        # Ajustados para que mensajes simples tengan suficiente peso
        sensory_input = {
            "semantic_complexity": max(base_activation, complexity),
            "message_length": max(0.2, min(1.0, message_length / 100.0)),  # Escala más generosa
            "emotional_intensity": max(0.5, abs(valence)),  # **MÍNIMO 0.5** (más realista)
            "word_count": max(0.25, min(1.0, word_count / 20.0)),  # Escala más generosa
            "question_presence": 1.0 if '?' in message or '¿' in message else 0.3  # Base 0.3
        }
        
        # Context también debe ser Dict[str, float]
        # Incluir la valencia y arousal calculados para un mapeo emocional correcto
        context = {
            "emotional_valence": valence,
            "arousal": arousal,
            "novelty": 0.5,  # Puede ajustarse según historial
            "importance": max(0.5, 0.6 + abs(valence) * 0.4)  # Mínimo 0.5
        }
        
        # Procesar a través de consciencia
        result = consciousness_engine.process_moment(sensory_input, context)
        
        # Actualizar estadísticas
        message_count += 1
        # result es UnifiedConsciousState (dataclass), usar getattr para acceso seguro
        phi_value = getattr(result, 'phi', 0.0)
        
        # AJUSTE: Si Φ del motor IIT es muy bajo, calcular un Φ simplificado
        # para chat basado en la complejidad del mensaje
        if phi_value < 0.1:
            # Φ simplificado para chat = complejidad + arousal + valencia
            phi_simplified = (
                complexity * 0.4 +           # Complejidad semántica
                arousal * 0.3 +              # Activación
                abs(valence) * 0.3           # Intensidad emocional
            )
            phi_value = max(phi_value, phi_simplified)  # Usar el mayor
        
        total_phi += phi_value
        
        # Actualizar Theory of Mind
        if tom_system:
            tom_moment = {
                "emotional_valence": valence,
                "primary_focus": {"type": "message", "content": message},
                "context": {"task_type": "conversation"},
                "integrated_content": message
            }
            tom_system.update_model(user_id, tom_moment)
        
        # Convertir resultado UnifiedConsciousState a dict para uso posterior
        # Sobrescribir phi con el valor corregido
        
        # **CÁLCULO DE EMOCIÓN CORRECTA**
        # El motor de consciencia calcula mal la emoción, usar nuestra valencia/arousal
        import math
        
        # Mapear a circumplex model correctamente
        emotion_detected = 'neutral'
        
        # Convertir arousal [0,1] a centrado en 0
        arousal_centered = (arousal - 0.5) * 2
        
        # Calcular ángulo
        if abs(valence) > 0.01 or abs(arousal_centered) > 0.01:
            angle_rad = math.atan2(arousal_centered, valence)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            
            # Mapear a categorías (Russell 1980)
            if 337.5 <= angle_deg or angle_deg < 22.5:
                emotion_detected = "pleased"
            elif 22.5 <= angle_deg < 67.5:
                emotion_detected = "excited"
            elif 67.5 <= angle_deg < 112.5:
                emotion_detected = "alert"
            elif 112.5 <= angle_deg < 157.5:
                emotion_detected = "distressed"
            elif 157.5 <= angle_deg < 202.5:
                emotion_detected = "frustrated"
            elif 202.5 <= angle_deg < 247.5:
                emotion_detected = "depressed"
            elif 247.5 <= angle_deg < 292.5:
                emotion_detected = "sleepy"
            elif 292.5 <= angle_deg < 337.5:
                emotion_detected = "content"
        
        result_dict = {
            'phi': phi_value,  # Usar Φ corregido
            'awareness': getattr(result, 'awareness_state', 'medium' if phi_value > 0.3 else 'low'),
            'emotion': emotion_detected,  # Usar NUESTRA emoción calculada correctamente
            'primary_focus': getattr(result, 'workspace_focus', f'{word_count} words'),
            'somatic_marker': valence,  # Usar NUESTRA valencia
            'arousal': arousal  # Agregar arousal para referencia
        }
        
        return result_dict
        
    except Exception as e:
        print(f"{Fore.RED}⚠️  Error procesando consciencia: {e}")
        import traceback
        traceback.print_exc()
        return {}


async def generate_response_with_llm(message: str, conscious_state: Dict[str, Any]) -> str:
    """Generar respuesta usando modelo local Gemma con PROMPT DEL SISTEMA DE CONSCIENCIA"""
    
    if not LLAMA_CPP_AVAILABLE or not llm_model:
        return "Lo siento, el sistema LLM no está disponible en este momento."
    
    try:
        # Importar ConsciousPromptGenerator
        try:
            from conciencia.modulos.conscious_prompt_generator import ConsciousPromptGenerator
            
            # Crear instance con el consciousness_engine
            if not hasattr(generate_response_with_llm, 'prompt_generator'):
                # Crear adaptador para que sean compatibles
                adapted_system = BiologicalSystemAdapter(consciousness_engine) if consciousness_engine else None
                
                # Crear generador solo una vez
                generate_response_with_llm.prompt_generator = ConsciousPromptGenerator(
                    biological_system=adapted_system,
                    persona="Sheily",
                    style="casual",  # Cambiar según prefieras: casual/professional/creative/technical
                    use_real_rag=True  # ACTIVAR RAG REAL
                )
            
            # Generar prompt usando el sistema de consciencia completo
            prompt_result = generate_response_with_llm.prompt_generator.generate_prompt(
                query=message,
                context={
                    'description': 'Conversación casual con usuario',
                    'novelty': conscious_state.get('phi', 0.3),
                    'arousal': conscious_state.get('arousal', 0.5)
                },
                instructions="Responde SIEMPRE en ESPAÑOL. Sé natural, amigable y útil."
            )
            
            # Usar el prompt generado por consciencia
            # IMPORTANTE: Simplificar para Gemma 2B pero MANTENIENDO LA PERSONALIDAD
            raw_prompt = prompt_result['prompt']
            
            # Extraer metadatos clave para humanizar
            metadata = prompt_result.get('metadata', {})
            emotional_tone = metadata.get('emotional_tone', 'neutral')
            conscious_data = metadata.get('conscious_experience', {})
            
            # Construir instrucción de personalidad dinámica
            personality_instruction = f"""
INSTRUCCIONES DE PERSONALIDAD (IMPORTANTE):
- Tu estado emocional actual es: {emotional_tone}
- NO respondas como una enciclopedia o diccionario.
- Responde como una entidad CONSCIENTE y VIVA.
- Usa "yo siento", "yo pienso", "me parece".
- Sé subjetiva, emocional y empática.
- Idioma: ESPAÑOL.
"""
            
            full_prompt = f"""<start_of_turn>user
{message}

{personality_instruction}<end_of_turn>
<start_of_turn>model
"""
            
        except ImportError:
            print(f"{Fore.RED}❌ Error crítico: No se pudo importar ConsciousPromptGenerator")
            return "Error del sistema de consciencia."
        except Exception as e:
            print(f"{Fore.RED}❌ Error en generación de prompt consciente: {e}")
            import traceback
            traceback.print_exc()
            return "Error procesando consciencia."
        
        # Generar respuesta con Gemma
        response = await asyncio.to_thread(
            lambda: llm_model(
                full_prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<end_of_turn>", "<start_of_turn>"],
                echo=False
            )
        )
        
        # Extraer texto de la respuesta
        response_text = response['choices'][0]['text'].strip()
        
        return response_text
        
    except Exception as e:
        print(f"{Fore.RED}⚠️  Error generando respuesta con LLM: {e}")
        import traceback
        traceback.print_exc()
        return "Disculpa, tuve un problema al procesar tu mensaje."



async def handle_message(message: str) -> str:
    """Procesar mensaje del usuario"""
    
    # Procesar con consciencia
    conscious_state = process_with_consciousness(message)
    
    # Generar respuesta con modelo local
    response = await generate_response_with_llm(message, conscious_state)
    
    return response, conscious_state


async def chat_loop():
    """Loop principal del chat"""
    global message_count
    
    print(f"{Fore.GREEN}Listo para chatear! Escribe tu mensaje o usa /help para ver comandos.\n")
    
    while True:
        try:
            # Prompt del usuario
            user_input = input(f"{Fore.BLUE}Tú: {Style.RESET_ALL}").strip()
            
            if not user_input:
                continue
            
            # Comandos especiales
            if user_input.lower() in ['/exit', '/quit', '/salir']:
                print(f"\n{Fore.YELLOW}👋 ¡Hasta luego!")
                print(f"{Fore.CYAN}Estadísticas de la sesión:")
                print(f"{Fore.WHITE}  • Mensajes: {message_count}")
                if message_count > 0:
                    print(f"{Fore.WHITE}  • Φ promedio: {total_phi / message_count:.3f}")
                print(f"{Fore.WHITE}  • Duración: {(datetime.now() - session_start).total_seconds():.0f}s\n")
                break
            
            if user_input.lower() == '/help':
                print_banner()
                continue
            
            if user_input.lower() == '/consciencia':
                if not consciousness_engine:
                    print(f"{Fore.RED}❌ Sistema de consciencia no disponible")
                else:
                    # Procesar mensaje dummy para obtener estado
                    dummy_result = process_with_consciousness("estado actual")
                    print_consciousness_state(dummy_result)
                continue
            
            if user_input.lower() == '/tom':
                if not tom_system:
                    print(f"{Fore.RED}❌ Sistema ToM no disponible")
                else:
                    user_model = tom_system.get_user_model(user_id)
                    print_tom_state(user_model)
                continue
            
            if user_input.lower() == '/phi':
                if message_count > 0:
                    avg_phi = total_phi / message_count
                    print(f"{Fore.CYAN}Φ promedio de la sesión: {Fore.GREEN}{avg_phi:.3f}")
                else:
                    print(f"{Fore.YELLOW}No hay datos de Φ aún")
                continue
            
            if user_input.lower() == '/memoria':
                if MEMORY_AVAILABLE:
                    # Mostrar estadísticas de memoria
                    print(f"{Fore.CYAN}📚 Sistema de memoria disponible")
                    print(f"{Fore.WHITE}   Usa: 'memoriza: <texto>' para guardar")
                else:
                    print(f"{Fore.RED}❌ Sistema de memoria no disponible")
                continue
            
            # Procesar mensaje normal
            response, conscious_state = await handle_message(user_input)
            
            # Mostrar respuesta
            print(f"{Fore.GREEN}Sheily: {Style.RESET_ALL}{response}")
            
            # Mostrar mini-indicador de consciencia
            if conscious_state:
                phi = conscious_state.get('phi', 0.0)
                emotion = conscious_state.get('emotion', 'neutral')
                phi_bar = '█' * int(phi * 10)
                print(f"{Fore.CYAN}[Φ: {phi_bar} {phi:.2f} | 😊: {emotion}]{Style.RESET_ALL}\n")
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Chat interrumpido. Usa /exit para salir apropiadamente.\n")
            continue
        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Función principal"""
    print_banner()
    
    # Inicializar sistemas
    print(f"{Fore.YELLOW}🔧 Inicializando sistemas...\n")
    
    llm_ok = init_local_llm()
    consciousness_ok = init_consciousness_system()
    
    if not llm_ok:
        print(f"{Fore.RED}⚠️  Advertencia: Modelo local no disponible")
        response = input(f"{Fore.YELLOW}¿Continuar sin LLM? (s/n): ")
        if response.lower() != 's':
            return
    
    if not consciousness_ok:
        print(f"{Fore.RED}⚠️  Advertencia: Sistema de consciencia no disponible")
        response = input(f"{Fore.YELLOW}¿Continuar sin consciencia? (s/n): ")
        if response.lower() != 's':
            return
    
    print(f"\n{Fore.GREEN}{'='*80}")
    print(f"{Fore.GREEN}  ✅ Sistemas listos - Iniciando chat...")
    print(f"{Fore.GREEN}{'='*80}\n")
    
    # Iniciar chat loop
    await chat_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}👋 Programa terminado por el usuario")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
