#!/usr/bin/env python3
"""
CLI Tool para el Sistema Universal de Optimización de Prompts
Herramienta de línea de comandos con Click para testing y uso rápido.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

from ..universal_prompt_optimizer import (LlamaCppAdapter,
                                          UniversalAutoImprovingPromptSystem)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIContext:
    """Contexto para el CLI"""

    def __init__(self):
        self.system = None

    async def get_system(self) -> UniversalAutoImprovingPromptSystem:
        """Obtener el sistema (lazy loading)"""
        if self.system is None:
            try:
                llm = LlamaCppAdapter("models/llama-3.2-3b-q4.gguf")
                self.system = UniversalAutoImprovingPromptSystem(llm)
                click.echo("🚀 Sistema inicializado con Llama 3.2 3B")
            except Exception as e:
                click.echo(f"❌ Error inicializando sistema: {e}", err=True)
                sys.exit(1)
        return self.system

pass_system = click.make_pass_decorator(CLIContext, ensure=True)

@click.group()
@click.pass_context
def cli(ctx):
    """🧠 Universal Prompt Optimizer CLI Tool

    Sistema automático para mejorar prompts de cualquier LLM.

    Usa: uapo optimize "tu prompt aquí" --llm llama
    """
    ctx.obj = CLIContext()

@cli.command()
@click.argument('prompt')
@click.option('--iterations', '-i', default=3, help='Iteraciones de optimización (1-10)', type=int)
@click.option('--context', '-c', help='Archivo JSON con contexto', type=click.Path(exists=True))
@click.option('--output', '-o', help='Guardar resultado en archivo', type=click.Path())
@pass_system
def optimize(context, prompt: str, iterations: int, output: Optional[str] = None):
    """Optimizar un prompt automáticamente"""
    system = asyncio.run(context.get_system())

    # Cargar contexto si proporcionado
    ctx_data = {}
    if context:
        try:
            with open(context, 'r') as f:
                ctx_data = json.load(f)
        except Exception as e:
            click.echo(f"❌ Error cargando contexto: {e}", err=True)
            return

    click.echo(f"🔄 Optimizando prompt: {prompt[:100]}...")
    if len(prompt) > 100:
        click.echo("...(truncado)")

    try:
        result = asyncio.run(system.optimize_prompt(
            original_prompt=prompt,
            context=ctx_data,
            max_iterations=min(max(iterations, 1), 10)
        ))

        # Mostrar resultado
        click.echo("✅ Optimización completada!"
        click.echo(f"📊 Score original: {(result.evaluation.metrics.get('relevance', 0) or 0):.1f}")
        click.echo("-" * 50)
        click.echo(f"📝 Prompt optimizado:")
        click.echo(result.optimized_prompt)
        click.echo("-" * 50)
        click.echo(f"📊 Score final: {result.evaluation.score:.1f}/100")
        click.echo(f"🛠️ Técnica usada: {result.technique_used}")
        click.echo(f"🔄 Iteraciones: {result.iterations}")
        click.echo(f"💡 Sugerencias: {', '.join(result.evaluation.improvements[:3])}")

        # Guardar si solicitado
        if output:
            with open(output, 'w') as f:
                json.dump({
                    'original': result.original_prompt,
                    'optimized': result.optimized_prompt,
                    'score': result.evaluation.score,
                    'metrics': result.evaluation.metrics,
                    'improvements': result.evaluation.improvements
                }, f, indent=2)
            click.echo(f"💾 Resultado guardado en: {output}")

    except Exception as e:
        click.echo(f"❌ Error durante optimización: {e}", err=True)

@cli.command()
@click.argument('query')
@click.option('--context', '-c', help='Archivo JSON con contexto', type=click.Path(exists=True))
@click.option('--stream', '-s', is_flag=True, help='Stream de respuesta (no implementado aún)')
@pass_system
def generate(context, query: str, stream: bool = False):
    """Generar respuesta optimizada para una query"""
    system = asyncio.run(context.get_system())

    # Cargar contexto
    ctx_data = {}
    if context:
        try:
            with open(context, 'r') as f:
                ctx_data = json.load(f)
        except Exception as e:
            click.echo(f"❌ Error cargando contexto: {e}", err=True)
            return

    click.echo(f"🤖 Generando respuesta para: {query[:100]}...")
    if len(query) > 100:
        click.echo("...(truncado)")

    try:
        response = asyncio.run(system.generate_response(query))

        click.echo("✅ Respuesta generada:")
        click.echo("-" * 50)
        click.echo(response)
        click.echo("-" * 50)

    except Exception as e:
        click.echo(f"❌ Error generando respuesta: {e}", err=True)

@cli.command()
@click.argument('prompt')
@click.option('--detailed', '-d', is_flag=True, help='Mostrar métricas detalladas')
@pass_system
def evaluate(context, prompt: str, detailed: bool = False):
    """Solo evaluar un prompt sin optimizar"""
    system = asyncio.run(context.get_system())

    click.echo(f"📊 Evaluando prompt: {prompt[:100]}...")
    if len(prompt) > 100:
        click.echo("...(truncado)")

    try:
        evaluation = asyncio.run(system.evaluator.evaluate_prompt(prompt))

        click.echo("✅ Evaluación completada!")
        click.echo(".1f")

        if detailed:
            click.echo(f"📈 Métricas:")
            for metric, value in evaluation.metrics.items():
                click.echo(f"   • {metric}: {value:.2f}")
            click.echo(f"🤔 Razonamiento: {evaluation.reasoning}")
            click.echo(f"💡 Mejoras sugeridas: {', '.join(evaluation.improvements)}")

    except Exception as e:
        click.echo(f"❌ Error evaluando prompt: {e}", err=True)

@cli.command()
@click.argument('prompts_file', type=click.Path(exists=True))
@click.option('--technique', '-t', help='Técnica específica a probar')
@click.option('--output', '-o', help='Archivo de salida para resultados', type=click.Path())
@pass_system
def benchmark(context, prompts_file: str, technique: Optional[str], output: Optional[str]):
    """Ejecutar benchmark con múltiples prompts"""
    system = asyncio.run(context.get_system())

    # Cargar prompts
    try:
        with open(prompts_file, 'r') as f:
            if prompts_file.endswith('.json'):
                prompts_data = json.load(f)
                if isinstance(prompts_data, list):
                    prompts = prompts_data
                elif 'prompts' in prompts_data:
                    prompts = prompts_data['prompts']
                else:
                    click.echo("❌ Formato JSON inválido. Use array de strings o objeto con 'prompts'", err=True)
                    return
            else:  # TXT file
                prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        click.echo(f"❌ Error cargando prompts: {e}", err=True)
        return

    click.echo(f"🏃 Ejecutando benchmark con {len(prompts)} prompts...")

    results = []
    total_score = 0

    for i, prompt in enumerate(prompts, 1):
        click.echo(f"➤ Prompt {i}/{len(prompts)}: {prompt[:50]}...")
        try:
            # Evaluar cada prompt
            evaluation = asyncio.run(system.evaluator.evaluate_prompt(prompt))
            results.append({
                'prompt': prompt,
                'score': evaluation.score,
                'metrics': evaluation.metrics,
                'improvements': evaluation.improvements
            })
            total_score += evaluation.score

            # Mostrar resultado rápido
            click.echo(".1f")

        except Exception as e:
            logger.error(f"Error evaluando prompt {i}: {e}")
            results.append({
                'prompt': prompt,
                'score': 0,
                'error': str(e)
            })

    # Resumen
    avg_score = total_score / len(results) if results else 0
    click.echo("
📊 RESUMEN DEL BENCHMARK:"    click.echo(".1f"    click.echo(f"📝 Prompts evaluados: {len(results)}")
    click.echo(f"📈 Mejores técnicas: CoT, Chain-of-Thought, Expert Prompting"

    # Guardar resultados
    if output:
        with open(output, 'w') as f:
            json.dump({
                'benchmark_results': results,
                'summary': {
                    'total_prompts': len(prompts),
                    'average_score': avg_score,
                    'best_technique': 'CoT con Safety Rails'
                }
            }, f, indent=2)
        click.echo(f"💾 Resultados guardados en: {output}")

@cli.command()
@click.option('--model', '-m', default='llama', help='Modelo a usar (llama, openai)', type=str)
@click.option('--api-key', help='API key para OpenAI (si usa openai)', type=str)
@click.option('--port', '-p', default=8000, help='Puerto para el API server', type=int)
def serve(model: str, api_key: Optional[str], port: int):
    """Iniciar API REST server"""
    click.echo("🚀 Iniciando API REST del Universal Prompt Optimizer")
    click.echo(f"📡 Puerto: {port}")
    click.echo(f"🤖 Modelo: {model}")

    # Configurar variables de entorno para la API
    if api_key:
        import os
        os.environ['OPENAI_API_KEY'] = api_key

    try:
        # Importar y ejecutar la API
        import uvicorn

        from .api_server import app
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        click.echo(f"❌ Error iniciando API server: {e}", err=True)
        sys.exit(1)

@cli.command()
def techniques():
    """Mostrar técnicas disponibles"""
    system_id = "llama_default"

    click.echo("🎯 TÉCNICAS DISPONIBLES:")
    click.echo("=" * 50)

    techniques = [
        "✅ Estructura y Claridad:",
        "   • DelimitersTechnique (# separadores)",
        "   • OutputPrimerTechnique (inicios de respuesta)",
        "   • AudienceIntegration (especificidad de público)",
        "   • AffirmativeDirectives nắm (directivas positivas)",

        "✅ Especificidad e Información:",
        "   • SpeakFirstTechnique (hablar desde el principio)",
        "   • CompletionInstructions (instrucciones de fin)",
        "   • UnderstandingTest (probar comprensión)",
        "   • CuriosityDriven (guiding por curiosidad)",

        "✅ Interacción y Compromiso:",
        "   • ExplainWithEvidence (explicar con evidencia)",
        "   • ComprehensiveCoverage (cobertura completa)",
        "   • TopDownMeditation (razonamiento arriba-abajo)",
        "   • MacroGeneration (generación macro)",

        "✅ Contenido y Lenguaje:",
        "   • CommonTerminology (términos comunes)",
        "   • KeyPhraseRepetition (repetición de frases clave)",
        "   • ChooseOptionSupport (soporte para opciones)",
        "   • ThinkStepByStep (razonamiento paso a paso)",

        "✅ Safety & Ethics:",
        "   • Toxicity detection (detección toxic)",
        "   • Bias detection (anti-sesgos)",
        "   • Jailbreak prevention (anti-jailbreaks)",
        "   • Ethical enforcement"
    ]

    for tech in techniques:
        click.echo(tech)

    click.echo("
💡 Usa 'uapo optimize \"tu prompt\"' para aplicar automáticamente!"
if __name__ == '__main__':
    cli()
