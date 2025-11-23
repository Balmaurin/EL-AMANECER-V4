#!/usr/bin/env python3
"""
MCP Chat Orchestrator - Sistema de Chat Orquestado por MCP Enterprise Master
=============================================================================

Este módulo conecta el sistema de chat con todos los componentes enterprise:
- Finance Agent para consultas financieras
- Quantitative Agent para análisis cuantitativo
- Unified Memory System para contexto histórico
- Llama 3.2 (Local LLM) para respuestas generales e inteligentes

El MCP Master orquesta inteligentemente qué componente debe responder.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sheily_core.logger import get_logger
from sheily_core.chat.chat_engine import create_chat_engine

logger = get_logger("mcp_chat_orchestrator")


@dataclass
class MCPChatResponse:
    """Respuesta completa del sistema MCP"""
    query: str
    response: str
    agent_used: str
    confidence: float
    processing_time: float
    memories_accessed: int
    insights_used: List[str]
    metadata: Dict[str, Any]


class MCPChatOrchestrator:
    """
    Orquestador de Chat basado en MCP Enterprise Master
    
    Coordina todos los sistemas del proyecto para dar respuestas inteligentes.
    """
    
    def __init__(self, mcp_master=None):
        """
        Inicializar orquestador
        
        Args:
            mcp_master: Instancia de MCPEnterpriseMaster (opcional, se crea si no existe)
        """
        self.mcp_master = mcp_master
        self.response_cache = {}
        
        # Inicializar motor de chat LLM
        try:
            self.chat_engine = create_chat_engine()
            logger.info("✅ Motor de Chat Llama 3.2 inicializado")
        except Exception as e:
            logger.error(f"❌ Error inicializando Llama 3.2: {e}")
            self.chat_engine = None
            
        logger.info("🎯 MCP Chat Orchestrator inicializado")
    
    async def _ensure_mcp_master(self):
        """Asegurar que MCPEnterpriseMaster esté inicializado"""
        if not self.mcp_master:
            from sheily_core.core.mcp.mcp_enterprise_master import MCPEnterpriseMaster
            self.mcp_master = MCPEnterpriseMaster()
            await self.mcp_master.initialize_enterprise_system()
            logger.info("✅ MCPEnterpriseMaster inicializado automáticamente")
    
    async def _detect_intent(self, query: str) -> Dict[str, Any]:
        """
        Detectar la intención del usuario y el agente apropiado
        """
        query_lower = query.lower()
        
        # Palabras clave financieras
        finance_keywords = [
            'riesgo', 'portfolio', 'inversión', 'acción', 'mercado',
            'finanz', 'bolsa', 'dividendo', 'rendimiento', 'var'
        ]
        
        # Palabras clave cuantitativas
        quant_keywords = [
            'predicción', 'modelo', 'estrategia', 'algoritmo', 'trading',
            'machine learning', 'ml', 'análisis', 'señal', 'arbitraje'
        ]
        
        # Palabras clave de memoria/historia
        memory_keywords = [
            'recuerdo', 'anterior', 'historia', 'pasado', 'dijiste',
            'antes', 'memoriza', 'aprendiste'
        ]
        
        # Calcular puntajes
        finance_score = sum(1 for kw in finance_keywords if kw in query_lower)
        quant_score = sum(1 for kw in quant_keywords if kw in query_lower)
        memory_score = sum(1 for kw in memory_keywords if kw in query_lower)
        
        # Determinar agente
        if finance_score > quant_score and finance_score > memory_score:
            return {
                'agent': 'finance',
                'confidence': min(finance_score / 3.0, 1.0),
                'keywords': [kw for kw in finance_keywords if kw in query_lower]
            }
        elif quant_score > memory_score:
            return {
                'agent': 'quant',
                'confidence': min(quant_score / 3.0, 1.0),
                'keywords': [kw for kw in quant_keywords if kw in query_lower]
            }
        elif memory_score > 0:
            return {
                'agent': 'memory',
                'confidence': min(memory_score / 2.0, 1.0),
                'keywords': [kw for kw in memory_keywords if kw in query_lower]
            }
        else:
            return {
                'agent': 'general',
                'confidence': 0.3,
                'keywords': []
            }
    
    async def _query_finance_agent(self, query: str) -> Dict[str, Any]:
        """Consultar al agente financiero"""
        if not self.mcp_master or not self.mcp_master.finance_agent:
            return {'error': 'Finance agent no disponible'}
        
        try:
            # Crear tarea para el agente
            task = {
                'task_type': 'risk_assessment',
                'parameters': {
                    'query': query,
                    'portfolio': 'GENERAL_PORTFOLIO'
                }
            }
            
            result = await self.mcp_master.finance_agent.execute_task(task)
            return result
        except Exception as e:
            logger.error(f"Error consultando Finance Agent: {e}")
            return {'error': str(e)}
    
    async def _query_quant_agent(self, query: str) -> Dict[str, Any]:
        """Consultar al agente cuantitativo"""
        if not self.mcp_master or not self.mcp_master.quant_agent:
            return {'error': 'Quantitative agent no disponible'}
        
        try:
            task = {
                'task_type': 'trading_strategy',
                'strategy': 'machine_learning',
                'assets': ['SPY', 'QQQ']
            }
            
            result = await self.mcp_master.quant_agent.process_task(task)
            return result
        except Exception as e:
            logger.error(f"Error consultando Quant Agent: {e}")
            return {'error': str(e)}
    
    async def _query_memory_system(self, query: str) -> Dict[str, Any]:
        """Consultar el sistema de memoria unificado"""
        if not self.mcp_master or not hasattr(self.mcp_master, 'memory_core'):
            return {'memories': [], 'count': 0}
        
        try:
            memory_system = self.mcp_master.memory_core.unified_memory
            
            # Buscar memorias relevantes (simplificado)
            relevant_memories = []
            for mem_id, memory in list(memory_system.memories.items())[:10]:
                if any(keyword in memory.content.lower() for keyword in query.lower().split()):
                    relevant_memories.append({
                        'id': mem_id,
                        'content': memory.content[:200],
                        'type': memory.memory_type.value,
                        'importance': memory.importance_score
                    })
            
            return {
                'memories': relevant_memories,
                'count': len(relevant_memories),
                'total_memories': len(memory_system.memories)
            }
        except Exception as e:
            logger.error(f"Error consultando Memory System: {e}")
            return {'memories': [], 'count': 0, 'error': str(e)}
    
    async def _generate_general_response(self, query: str, context: Dict) -> str:
        """Generar respuesta general usando LLM Llama 3.2"""
        
        # Si tenemos el motor de chat LLM, usarlo
        if self.chat_engine:
            try:
                logger.info(f"🧠 Consultando Llama 3.2 para: '{query}'")
                # Ejecutar en thread pool para no bloquear asyncio
                response_obj = await asyncio.to_thread(self.chat_engine, query)
                
                if response_obj and response_obj.response:
                    return response_obj.response
                else:
                    return "Lo siento, no pude generar una respuesta coherente en este momento."
            except Exception as e:
                logger.error(f"Error generando respuesta LLM: {e}")
                return f"Tuve un problema pensando la respuesta: {e}"
        
        # Fallback si no hay LLM
        return (
            f"👋 **Hola, soy Sheily AI Enterprise**\n\n"
            f"He recibido tu consulta: *'{query}'*\n\n"
            f"⚠️ **Nota**: El modelo de lenguaje Llama 3.2 no está disponible en este momento.\n"
            f"Por favor verifica la configuración de rutas en `config.py`.\n\n"
            f"Puedo ayudarte con:\n"
            f"• **Finanzas**: Análisis de riesgo, portfolio.\n"
            f"• **Trading**: Estrategias ML.\n"
        )
    
    def _format_agent_response(self, agent_name: str, data: Any) -> str:
        """Formatear respuesta de agente a texto legible"""
        if isinstance(data, dict):
            # Intentar extraer resumen o mensaje principal
            content = data.get('summary') or data.get('message') or data.get('result')
            
            if isinstance(content, dict):
                # Si el contenido sigue siendo dict, formatearlo bonito
                formatted = "\n".join(f"• **{k.title()}**: {v}" for k, v in content.items() if isinstance(v, (str, int, float)))
                return f"**Reporte de {agent_name}**\n\n{formatted}"
            elif content:
                return f"**{agent_name}**\n\n{content}"
            else:
                # Fallback a mostrar claves
                keys = ", ".join(data.keys())
                return f"**{agent_name}** procesó la solicitud.\nDatos generados: {keys}"
        return str(data)

    async def process_query(self, query: str, user_id: str = "default") -> MCPChatResponse:
        """
        Procesar consulta usando todo el poder del MCP Enterprise Master
        """
        start_time = time.time()
        
        logger.info(f"📨 Procesando consulta de {user_id}: '{query[:50]}...'")
        
        # Asegurar que MCP Master esté listo
        await self._ensure_mcp_master()
        
        # Detectar intención
        intent = await self._detect_intent(query)
        logger.info(f"🎯 Intención detectada: {intent['agent']} (confianza: {intent['confidence']:.2f})")
        
        # Consultar memoria para contexto
        memory_context = await self._query_memory_system(query)
        
        # Enrutar a agente apropiado
        agent_response = None
        agent_used = intent['agent']
        response_text = ""
        
        if intent['agent'] == 'finance':
            agent_response = await self._query_finance_agent(query)
        elif intent['agent'] == 'quant':
            agent_response = await self._query_quant_agent(query)
        elif intent['agent'] == 'memory':
            # Respuesta basada en memorias
            if memory_context['count'] > 0:
                response_text = f"**🧠 Memoria del Sistema**\n\nEncontré {memory_context['count']} recuerdos relevantes:\n\n"
                for mem in memory_context['memories'][:3]:
                    response_text += f"• *[{mem['type']}]* {mem['content']}\n"
            else:
                response_text = "🧠 **Memoria**: No encontré recuerdos específicos sobre eso, pero estoy aprendiendo de esta conversación."
        else:
            # Respuesta general -> LLM
            response_text = await self._generate_general_response(query, {
                'intent': intent,
                'memories': memory_context
            })
            agent_used = "llama-3.2"
        
        # Formatear respuesta del agente si hubo uno (Finance/Quant)
        if agent_response:
            if 'error' not in agent_response:
                if intent['agent'] == 'finance':
                    result_data = agent_response.get('result', agent_response)
                    response_text = self._format_agent_response("💰 Análisis Financiero", result_data)
                elif intent['agent'] == 'quant':
                    result_data = agent_response.get('result', agent_response)
                    response_text = self._format_agent_response("📊 Estrategia Cuantitativa", result_data)
            else:
                # Fallback si el agente falló
                error_msg = agent_response['error']
                response_text = f"⚠️ **Error en Agente {intent['agent'].title()}**\n\nNo pude completar el análisis: {error_msg}\n\nPero aquí estoy para ayudarte con otras consultas."
                agent_used = 'fallback'
        
        # Si no hay respuesta aún, usar general
        if not response_text:
            response_text = await self._generate_general_response(query, {})
        
        processing_time = time.time() - start_time
        
        chat_response = MCPChatResponse(
            query=query,
            response=response_text,
            agent_used=agent_used,
            confidence=intent['confidence'],
            processing_time=processing_time,
            memories_accessed=memory_context['count'],
            insights_used=intent['keywords'],
            metadata={
                'intent': intent,
                'memory_context': memory_context,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"✅ Consulta procesada en {processing_time:.2f}s por agente '{agent_used}'")
        
        return chat_response


# Singleton global
_orchestrator_instance = None

def get_mcp_chat_orchestrator(mcp_master=None) -> MCPChatOrchestrator:
    """Obtener instancia singleton del orquestador"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MCPChatOrchestrator(mcp_master)
    return _orchestrator_instance
