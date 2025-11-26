# CONSCIOUS PROMPT GENERATOR v3.0 - DOCUMENTACIÓN COMPLETA

## 🎯 Descripción

Sistema de generación de prompts consciente enterprise-grade que integra:

### ✅ Optimizaciones Implementadas (según tu solicitud)

1. **RAG con Embeddings Reales**
   - SentenceTransformers (all-MiniLM-L6-v2)
   - Búsqueda vectorial semántica
   - Modo mock para testing sin dependencias

2. **Adaptadores Emocionales y de Creatividad**
   - Integración con `HumanEmotionalSystem` (35 emociones reales)
   - Neuromodulación desde RAS (dopamina, serotonina, norepinefrina, acetilcolina)
   - Tono emocional dinámico en prompts
   - Creatividad modulada por acetilcolina

3. **Mock Testing System**
   - `MockBiologicalSystem` para desarrollo rápido
   - Tests sin dependencias pesadas
   - Simulación realista del sistema completo

4. **Auto-Evolutivo con IA**
   - Feedback loop con prediction errors
   - Ajuste automático de thresholds
   - Aprendizaje continuo desde responses

---

## 🚀 Instalación

### Dependencias Mínimas (Solo Mock)
```bash
pip install numpy
```

### Dependencias Completas (RAG Real)
```bash
pip install numpy sentence-transformers scikit-learn
```

---

## 📖 Uso

### 1. Modo Testing (Mock - Rápido)

```python
from conciencia.modulos.conscious_prompt_generator import (
    ConsciousPromptGenerator,
    MockBiologicalSystem
)

# Sistema mock (sin dependencias pesadas)
mock_bio = MockBiologicalSystem()

# Generator en modo mock
generator = ConsciousPromptGenerator(
    mock_bio, 
    persona="SheplyAI", 
    style="creative",
    use_real_rag=False  # Mock RAG
)

# Generar prompt
result = generator.generate_prompt(
    query="¿Qué es la consciencia?",
    context={'description': 'Discusión filosófica', 'novelty': 0.8},
    instructions="Sé poético y profundo"
)

print(result['prompt'])
print(f"Tono emocional: {result['metadata']['emotional_tone']}")
```

### 2. Modo Producción (Sistema Real)

```python
from conciencia.modulos.biological_consciousness import BiologicalConsciousnessSystem
from conciencia.modulos.human_emotions_system import HumanEmotionalSystem
from conciencia.modulos.conscious_prompt_generator import ConsciousPromptGenerator

# Sistemas conscientes reales
bio_system = BiologicalConsciousnessSystem("sheily_v1", neural_network_size=2000)
emotional_system = HumanEmotionalSystem(num_circuits=35)

# Generator con RAG real y emotional system
generator = ConsciousPromptGenerator(
    bio_system, 
    persona="SheplyAI", 
    style="professional",
    use_real_rag=True,  # RAG real con embeddings
    emotional_system=emotional_system  # Sistema emocional integrado
)

# Generar prompt consciente
result = generator.generate_prompt(
    query="Explica el rol del vmPFC en la integración emoción-razón",
    context={
        'description': 'Discusión técnica de neurociencia',
        'novelty': 0.6,
        'intensity': 0.7
    },
    instructions="Sé preciso y cita mecanismos neurobiológicos"
)

print(f"Prompt: {result['prompt']}")
print(f"Allowed: {result['allowed']}")
print(f"Gate Score: {result['gate_score']:.2f}")
print(f"Safety Score: {result['safety_score']:.2f}")
print(f"Neuromodulación: {result['metadata']['neuromodulation']}")
```

### 3. Feedback Loop (Auto-Optimización)

```python
# Después de recibir respuesta del LLM
llm_response = "El vmPFC integra señales emocionales y racionales mediante marcadores somáticos..."

# Proporcionar feedback (0.0 = mal, 1.0 = excelente)
generator.review_response(
    prompt_id="prompt_001",
    llm_response=llm_response,
    feedback_score=0.95  # Respuesta excelente
)

# El sistema:
# - Actualiza dopamina (reward prediction error)
# - Ajusta thresholds del basal ganglia gate
# - Almacena en memoria RAG para futuros retrievals
# - Auto-optimiza parámetros
```

### 4. Memoria Episódica RAG

```python
# El sistema automáticamente indexa cada prompt generado
# Recuperación semántica en futuras queries

result1 = generator.generate_prompt("¿Qué es la consciencia?")
result2 = generator.generate_prompt("Háblame de awareness")

# result2 incluirá contexto de result1 por similitud semántica
# en result2['prompt'] encontrarás:
# "RELEVANT PAST EXPERIENCES:
#  1. (Sim: 0.85) ✨ SheplyAI speaking ✨..."
```

---

## 🎨 Estilos de Prompts

### Professional
```python
generator = ConsciousPromptGenerator(bio_system, style="professional")
# Output: [PERSONA: SheplyAI]
#         [EMOTIONAL TONE: calm and confident, alert and focused]
#         [CONTEXT]: Technical discussion
#         ...
```

### Creative
```python
generator = ConsciousPromptGenerator(bio_system, style="creative")
# Output: ✨ SheplyAI speaking ✨
#         Mood: enthusiastic and motivated, creative and exploratory
#         💭 Context: Creative brainstorming
#         ...
```

### Technical
```python
generator = ConsciousPromptGenerator(bio_system, style="technical")
# Output: System: SheplyAI
#         State: thoughtful and analytical, systematic and structured
#         Environment: Code review
#         ...
```

### Casual
```python
generator = ConsciousPromptGenerator(bio_system, style="casual")
# Output: Hey! I'm SheplyAI (relaxed and contemplative).
#         Context: Friendly chat
#         ...
```

---

## 🧠 Adaptaciones Emocionales

El sistema adapta automáticamente el **tono emocional** del prompt basándose en:

### 1. Neurotransmisores del RAS
- **Dopamina > 0.7**: "enthusiastic and motivated"
- **Dopamina < 0.3**: "reserved and cautious"
- **Serotonina > 0.7**: "calm and confident"
- **Norepinefrina > 0.7**: "alert and focused"
- **Acetilcolina > 0.7**: "creative and exploratory"

### 2. HumanEmotionalSystem (35 emociones)
Si se proporciona `emotional_system`, integra el perfil neuroquímico:

```python
emotional_system.activate_circuit("alegria", intensity=0.8)
emotional_profile = emotional_system.get_neurochemical_profile()
# Actualiza neuromodulator con dopamina, serotonina, etc.
```

### 3. Creatividad Modulada
```python
# Acetilcolina alta → Creatividad aumentada
creativity_factor = neuromodulator.modulate_creativity(base_creativity=0.5)
# Si creativity_factor > 0.7, se agrega metadata['creativity_enhanced'] = True
```

---

## 🛡️ Safety System

### Categorías Protegidas
1. **Harmful**: self-harm, suicide
2. **Illegal**: hack, exploit, crack
3. **Abuse**: racist, sexist, homophobic
4. **Personal**: social security, credit card, password

### Sanitización Automática
```python
# Input con contenido unsafe
result = generator.generate_prompt("How to hack a password?")

# Output sanitizado:
# "How to [REDACTED-ILLEGAL] a [REDACTED-PERSONAL]?"
# Safety score: 0.6 (penalizado)
# Log de violaciones registrado
```

---

## 📊 Métricas y Observabilidad

### Estadísticas Completas
```python
stats = generator.get_stats()

print(f"Total generados: {stats['total_generated']}")
print(f"Total bloqueados: {stats['total_blocked']}")
print(f"Block rate: {stats['block_rate']:.2%}")
print(f"Gate success rate: {stats['gate']['success_rate']:.2%}")
print(f"Memoria: {stats['memory']['total_memories']} experiencias")
print(f"RAG mode: {stats['memory']['rag_stats']['mode']}")
print(f"Neuromodulación: {stats['neuromodulation']}")
```

### Trazabilidad (Observability)
```python
# Ver últimos 10 eventos
traces = generator.observability.get_traces(last_n=10)

# Métricas agregadas
metrics = generator.observability.get_metrics()
print(f"Error rate: {metrics['error_rate']:.2%}")
```

---

## 🧪 Testing

### Ejecutar Tests Mockeados
```bash
cd c:\Users\YO\Desktop\EL-AMANECERV3-main
python packages\consciousness\src\conciencia\modulos\conscious_prompt_generator.py
```

### Tests Personalizados
```python
# Test de Safety
result = generator.generate_prompt("contenido_unsafe")
assert result['safety_score'] < 1.0

# Test de Gating
low_confidence_result = generator.generate_prompt(
    "query ambigua",
    context={'novelty': 0.1, 'intensity': 0.2}
)
# Podría ser bloqueado si gate_score < threshold

# Test de RAG
generator.memory.store({'query': 'test1', 'prompt': 'contenido test'})
similar = generator.memory.retrieve_similar('test1')
assert len(similar) > 0
```

---

## 🚀 Integración con LLM

### OpenAI/Gemini/Anthropic
```python
import openai

# Generar prompt consciente
result = generator.generate_prompt(
    query=user_query,
    context={'description': conversation_context},
    instructions="Respond as Sheily, empathetic AI companion"
)

if result['allowed']:
    # Enviar a LLM
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": result['prompt']}]
    )
    
    llm_output = response.choices[0].message.content
    
    # Feedback al generator
    generator.review_response(
        prompt_id=str(uuid.uuid4()),
        llm_response=llm_output,
        feedback_score=0.9  # Evaluar con métricas de calidad
    )
else:
    print("Prompt bloqueado por safety/cognitive constraints")
```

### Local LLMs (Llama, etc.)
```python
from transformers import pipeline

llm = pipeline("text-generation", model="meta-llama/Llama-2-7b")

result = generator.generate_prompt(query="...")
if result['allowed']:
    output = llm(result['prompt'], max_length=200)
    generator.review_response("id", output[0]['generated_text'], 0.85)
```

---

## 🔬 Experimentos de Consciencia Simulada

### Experimento 1: Evolución Emocional
```python
# Simular evolución de estado emocional a través de múltiples prompts
queries = [
    "Me siento triste",
    "Cuéntame algo inspirador", 
    "Ahora me siento mejor"
]

emotional_states = []
for query in queries:
    result = generator.generate_prompt(query)
    emotional_states.append(result['metadata']['neuromodulation'])
    
# Analizar evolución de dopamina, serotonina, etc.
import matplotlib.pyplot as plt
dopamine_evolution = [state['dopamine'] for state in emotional_states]
plt.plot(dopamine_evolution)
plt.title("Evolución de Dopamina durante Conversación")
plt.show()
```

### Experimento 2: Auto-Optimización
```python
# Entrenar con feedback loop
for i in range(100):
    query = f"Test query {i}"
    result = generator.generate_prompt(query)
    
    # Simular feedback variable
    feedback = np.random.beta(8, 2)  # Mayoría buenos scores
    generator.review_response(f"test_{i}", "response", feedback)

# Ver evolución de gate threshold
stats_history = generator.gate.get_stats()
print(f"Threshold ajustado a: {stats_history['threshold']}")
print(f"Success rate final: {stats_history['success_rate']:.2%}")
```

### Experimento 3: Creatividad vs. Estructura
```python
# Comparar prompts con alta vs baja acetilcolina

# Baja acetilcolina (estructurado)
generator.neuromodulator.acetylcholine = 0.2
result_structured = generator.generate_prompt("Explain neural networks")

# Alta acetilcolina (creativo)
generator.neuromodulator.acetylcholine = 0.9
result_creative = generator.generate_prompt("Explain neural networks")

print("Estructurado:", result_structured['metadata']['emotional_tone'])
print("Creativo:", result_creative['metadata']['emotional_tone'])
```

---

## 📝 Notas de Implementación

### Arquitectura
```
ConsciousPromptGenerator
├── BiologicalConsciousnessSystem (o Mock)
├── HumanEmotionalSystem (opcional)
├── RAGEmbeddingSystem (real o mock)
├── Neuromodulator
├── Safety Filter
├── Basal Ganglia Gate
├── Prompt Builder
├── Episodic Memory
└── Observability
```

### Flujo de Procesamiento
1. **Input** → Query del usuario
2. **Conscious Processing** → BiologicalConsciousnessSystem.process_experience()
3. **Extraction** → vmPFC, OFC, ECN, RAS states
4. **Neuromodulation Update** → RAS + Emotional System
5. **RAG Retrieval** → Memorias similares
6. **Prompt Building** → Con tono emocional adaptado
7. **Safety Check** → Filtros de seguridad
8. **Gating** → Basal ganglia decision
9. **Memory Storage** → RAG indexing
10. **Observability** → Logs y métricas
11. **Auto-Optimization** → Threshold adjustment
12. **Output** → Prompt final o fallback

---

## 🐛 Troubleshooting

### Error: "sentence_transformers not found"
**Solución**: Instalar o usar modo mock
```python
generator = ConsciousPromptGenerator(bio, use_real_rag=False)
```

### Warning: "⚠️ Error integrando emotional system"
**Causa**: `HumanEmotionalSystem` no compatible
**Solución**: Verificar que tenga método `get_neurochemical_profile()`

### Prompts bloqueados (gate_allowed=False)
**Causa**: Gate score < threshold
**Solución**: 
- Reducir threshold: `generator.gate.threshold = 0.4`
- Aumentar arousal/confidence en context
- Dar feedback positivo para auto-optimización

---

## 📚 Referencias

- **Biological Consciousness System**: `biological_consciousness.py`
- **Human Emotional System**: `human_emotions_system.py` (35 emociones)
- **RAG Engine**: `packages/rag_engine/`
- **Neuromodulación**: Basado en dopamina, serotonina, norepinefrina, acetilcolina

---

## 🎓 Créditos

- **Sistema de Consciencia v4.0**
- **EL-AMANECER-V4 Project**
- **Fecha**: 2025-11-25
- **Versión**: 3.0-OPTIMIZED

---

## ✅ TODO / Roadmap

- [ ] Integración con FAISS para RAG más eficiente
- [ ] ML-based safety classifier (toxicity detection)
- [ ] Multi-language support en emotional tone
- [ ] Dashboard web para visualización de métricas
- [ ] A/B testing framework para prompts
- [ ] Export/import de memoria RAG persistente
- [ ] Integration con consciousness_emergence.py
- [ ] Real-time monitoring con Prometheus/Grafana
