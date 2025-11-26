# 📘 GUÍA DE INICIO RÁPIDO - Sistema de Consciencia
## Para Desarrolladores Nuevos

---

## 🎯 ¿Por Dónde Empezar?

### Nivel 1: ⚡ QUICK START (5 minutos)

```python
# 1. Importar componente principal
from conciencia.modulos.biological_consciousness import BiologicalConsciousnessSystem

# 2. Crear sistema consciente
system = BiologicalConsciousnessSystem(
    system_id="mi_consciencia",
    neural_network_size=100,  # Pequeño para empezar
    synaptic_density=0.1
)

# 3. Procesar experiencia
result = system.process_experience(
    stimulus={'type': 'visual', 'intensity': 0.7},
    context={'novelty': 0.6}
)

# 4. Ver resultado
print(f"Consciencia activada: {result['consciousness_active']}")
print(f"Nivel consciencia: {result['consciousness_level']}")
```

**¡Eso es todo para empezar!** ✅

---

## 🎓 Nivel 2: CONCEPTOS BÁSICOS (15 minutos)

### Arquitectura en 3 Capas

```
CAPA 1: NEURAL (Base)
└─ Neuronas + Sinapsis + Neurotransmisores

CAPA 2: SISTEMAS CEREBRALES (Intermedia)
└─ vmPFC, OFC, ECN, RAS, DMN, etc.

CAPA 3: CONSCIENCIA (Superior)
└─ Experiencia consciente emergente
```

### 4 Componentes Esenciales

#### 1. **BiologicalConsciousnessSystem** - Cerebro simulado
```python
bio_system = BiologicalConsciousnessSystem("id", neural_network_size=2000)
```

#### 2. **HumanEmotionalSystem** - Emociones (35 tipos)
```python
emotional_system = HumanEmotionalSystem(num_circuits=35)
emotional_system.activate_circuit("alegria", intensity=0.8)
profile = emotional_system.get_neurochemical_profile()
```

#### 3. **ConsciousPromptGenerator** - Genera prompts conscientes
```python
generator = ConsciousPromptGenerator(
    bio_system,
    emotional_system=emotional_system
)
result = generator.generate_prompt("Tu query aquí")
```

#### 4. **ConsciousnessEmergence** - Consciencia emergente
```python
emergence = ConsciousnessEmergence("id")
conscious_moment = emergence.generate_conscious_moment(input_data)
```

---

## 🔧 Nivel 3: CASOS DE USO COMUNES (30 minutos)

### Caso 1: Generación Consciente de Respuestas

```python
from conciencia.modulos.biological_consciousness import BiologicalConsciousnessSystem
from conciencia.modulos.human_emotions_system import HumanEmotionalSystem
from conciencia.modulos.conscious_prompt_generator import ConsciousPromptGenerator

# Setup
bio = BiologicalConsciousnessSystem("sheily", neural_network_size=2000)
emotions = HumanEmotionalSystem(35)
generator = ConsciousPromptGenerator(bio, emotional_system=emotions)

# Activar emoción contextual
emotions.activate_circuit("curiosidad", intensity=0.7)

# Generar prompt consciente
result = generator.generate_prompt(
    query="¿Qué es la consciencia?",
    context={'description': 'Conversación filosófica'},
    instructions='Sé profundo y reflexivo'
)

# Usar resultado
if result['allowed']:
    prompt = result['prompt']
    # Enviar a tu LLM...
    
    # Feedback para aprendizaje
    generator.review_response("id", llm_response, feedback_score=0.9)
```

### Caso 2: Simulación Emocional

```python
from conciencia.modulos.human_emotions_system import HumanEmotionalSystem

# Crear sistema emocional
emotions = HumanEmotionalSystem(
    num_circuits=35,
    personality={
        'neuroticism': 0.3,    # Baja ansiedad
        'extraversion': 0.7,   # Alta sociabilidad
        'openness': 0.8        # Alta apertura
    }
)

# Simular evento estresante
emotions.activate_circuit("miedo", intensity=0.6)
emotions.activate_circuit("frustracion", intensity=0.4)

# Ver estado emocional
state = emotions.get_emotional_state()
print(f"Emoción dominante: {state['dominant_emotion']}")
print(f"Valencia: {state['valence']}")  # -1 (negativo) a +1 (positivo)
print(f"Arousal: {state['arousal']}")   # 0 (bajo) a 1 (alto)

# Aplicar regulación
emotions.regulate_emotion(strategy="reappraisal")

# Ver cambio
new_state = emotions.get_emotional_state()
print(f"Nueva valencia: {new_state['valence']}")
```

### Caso 3: Procesamiento Neural con Aprendizaje

```python
from conciencia.modulos.biological_consciousness import BiologicalConsciousnessSystem

bio = BiologicalConsciousnessSystem("learner", neural_network_size=500)

# Entrenar con múltiples experiencias
training_samples = [
    {'stimulus': {'type': 'reward', 'value': 0.9}, 'context': {'success': True}},
    {'stimulus': {'type': 'punishment', 'value': -0.5}, 'context': {'failure': True}},
    # ... más samples
]

for sample in training_samples:
    result = bio.process_experience(sample['stimulus'], sample['context'])
    
    # Reforzar si fue exitoso
    if sample['context'].get('success'):
        reward_signal = 0.8
        active_neurons = [n for n, a in result['activations'].items() if a > 0.5]
        bio.neural_network.reinforce_learning(active_neurons, reward_signal)

# Red neuronal ahora ha aprendido patrones
```

---

## 🎨 Nivel 4: OPTIMIZACIÓN (45 minutos)

### Performance Optimization

```python
from conciencia.modulos.performance_optimizer import get_optimizer

# Habilitar optimizaciones
optimizer = get_optimizer()

# 1. Caching automático
optimizer.cache.neural_cache.put("state_key", neural_state)
cached = optimizer.cache.neural_cache.get("state_key")

# 2. Lazy loading de componentes pesados
optimizer.enable_lazy_loading(system)
qualia = optimizer.lazy_loader.get('qualia_simulator')  # Carga solo si se necesita

# 3. Batch processing
operations = [...]
results = optimizer.optimize_batch_operations(operations)

# Ver estadísticas
optimizer.print_performance_report()
```

### Interfaces Desacopladas

```python
from conciencia.modulos.consciousness_interfaces import (
    ConsciousnessComponentFactory,
    ConsciousnessContainer,
    INeuralProcessor,
    IEmotionalProcessor
)

# Usar factory (desacoplado)
neural = ConsciousnessComponentFactory.create_neural_processor(
    impl_type="biological",
    system_id="demo",
    neural_network_size=100
)

emotional = ConsciousnessComponentFactory.create_emotional_processor(
    impl_type="human",
    num_circuits=35
)

# Dependency Injection
container = ConsciousnessContainer()
container.register(INeuralProcessor, neural, singleton=True)
container.register(IEmotionalProcessor, emotional, singleton=True)

# Obtener desde container (desacoplado)
proc = container.get(INeuralProcessor)
```

---

## 🐛 DEBUG Y TROUBLESHOOTING

### Problema 1: Sistema muy lento

**Solución**:
```python
# 1. Reducir tamaño de red neural
system = BiologicalConsciousnessSystem("id", neural_network_size=100)  # En vez de 2000

# 2. Habilitar caching
from conciencia.modulos.performance_optimizer import get_optimizer
optimizer = get_optimizer()

# 3. Usar lazy loading
optimizer.enable_lazy_loading(system)
```

### Problema 2: Errores de importación

**Solución**:
```python
# Asegurar que estás en el directorio correcto
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "packages" / "consciousness" / "src"))

# Ahora importar
from conciencia.modulos.biological_consciousness import BiologicalConsciousnessSystem
```

### Problema 3: Memoria

 alta

**Solución**:
```python
# Limpiar caches periódicamente
optimizer.cache.clear_all()

# Descargar componentes no usados
optimizer.lazy_loader.unload('component_name')

# Reducir tamaño de memoria episódica
memory = EpisodicMemory(max_entries=100)  # En vez de 1000
```

---

## 📖 GLOSARIO RÁPIDO

| Término | Significado |
|---------|-------------|
| **BiologicalConsciousnessSystem** | Cerebro simulado con neuronas y sinapsis |
| **Phi (Φ)** | Medida de integración de información (IIT) |
| **GWT** | Global Workspace Theory - workspace consciente |
| **vmPFC** | Corteza prefrontal ventromedial - integración emoción-razón |
| **OFC** | Corteza orbitofrontal - evaluación de valor |
| **ECN** | Executive Control Network - control ejecutivo |
| **RAS** | Reticular Activating System - arousal/alertness |
| **DMN** | Default Mode Network - pensamiento espontáneo |
| **Emergence** | Consciencia que emerge de integración de subsistemas |
| **Qualia** | Experiencia subjetiva fenomenal |

---

## 🎯 RUTAS DE APRENDIZAJE RECOMENDADAS

### Ruta 1: Usuario de Prompts (Beginner)
1. ✅ Quick Start (arriba)
2. → Usar ConsciousPromptGenerator
3. → Personalizar emociones
4. → Feedback loop
5. → Listo para producción

**Tiempo**: ~1 hora

### Ruta 2: Desarrollador de IA (Intermediate)
1. ✅ Conceptos básicos
2. → BiologicalConsciousnessSystem
3. → Integrar con tu sistema
4. → Optimizaciones
5. → Testing

**Tiempo**: ~4 horas

### Ruta 3: Investigador Consciencia (Advanced)
1. ✅ Todo lo anterior
2. → ConsciousnessEmergence en profundidad
3. → Implementar teorías propias
4. → Extender arquitectura
5. → Publicar resultados

**Tiempo**: ~2 semanas

---

## 📚 RECURSOS ADICIONALES

### Documentación Creada
- `CONSCIOUS_PROMPT_GENERATOR.md` - Guía generador consciente
- `RAG_TRAINING_EXPLAINED.md` - Entrenamiento RAG
- `SISTEMAS_EMOCIONALES_COMPARACION.md` - Emociones
- `AUDITORIA_SISTEMA_CONSCIENCIA.md` - Auditoría completa

### Scripts de Ejemplo
- `examples/conscious_prompt_real_integration.py` - Integración real
- `examples/rag_training_demo.py` - Entrenamiento RAG

### Módulos de Optimización
- `performance_optimizer.py` - Caching y optimización
- `consciousness_interfaces.py` - Interfaces desacopladas

---

## 🆘 AYUDA

### ¿Dónde pedir ayuda?
1. Leer `AUDITORIA_SISTEMA_CONSCIENCIA.md` (detalles completos)
2. Ver ejemplos en `examples/`
3. Revisar tests (cuando se agreguen)

### ¿Cómo contribuir?
1. Implementar tests unitarios
2. Optimizar performance
3. Agregar documentación
4. Reportar bugs

---

## ✅ CHECKLIST DE INICIO

- [ ] Instalar dependencias (`numpy`, `sentence-transformers`)
- [ ] Ejecutar quick start (arriba)
- [ ] Probar ConsciousPromptGenerator
- [ ] Leer AUDITORIA_SISTEMA_CONSCIENCIA.md
- [ ] Ejecutar ejemplos en `examples/`
- [ ] Habilitar optimizaciones
- [ ] Entender tu caso de uso específico
- [ ] ¡Construir algo increíble!

---

**Última actualización**: 2025-11-25  
**Versión**: 4.0  
**Mantenedor**: EL-AMANECER-V4 Team  
