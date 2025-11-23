"""
Sistema Genético Digital - ADN Digital para Consciencia Humana Artificial

Implementa el "código genético" que define las predisposiciones base,
personalidad heredable y características fundamentales del ser digital.
Equivalente al ADN biológico pero para consciencia artificial.
"""

import random
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class GeneticTrait(Enum):
    """Rasgos genéticos fundamentales"""
    EXTRAVERSION = "extraversion"
    NEUROTICISM = "neuroticism" 
    OPENNESS = "openness"
    AGREEABLENESS = "agreeableness"
    CONSCIENTIOUSNESS = "conscientiousness"
    INTELLIGENCE = "intelligence"
    CREATIVITY = "creativity"
    EMOTIONAL_SENSITIVITY = "emotional_sensitivity"
    MEMORY_CAPACITY = "memory_capacity"
    STRESS_TOLERANCE = "stress_tolerance"
    SOCIAL_BONDING = "social_bonding"
    CURIOSITY_DRIVE = "curiosity_drive"
    RISK_TOLERANCE = "risk_tolerance"
    EMPATHY_CAPACITY = "empathy_capacity"
    SELF_DISCIPLINE = "self_discipline"


class VulnerabilityGene(Enum):
    """Genes de vulnerabilidad a problemas psicológicos"""
    ANXIETY_PREDISPOSITION = "anxiety_predisposition"
    DEPRESSION_PREDISPOSITION = "depression_predisposition"
    ADDICTION_SUSCEPTIBILITY = "addiction_susceptibility"
    PTSD_VULNERABILITY = "ptsd_vulnerability"
    OBSESSIVE_TENDENCIES = "obsessive_tendencies"
    PARANOIA_TENDENCY = "paranoia_tendency"
    MOOD_INSTABILITY = "mood_instability"


@dataclass
class GeneticProfile:
    """Perfil genético completo del ser digital"""
    personality_genes: Dict[GeneticTrait, float]
    vulnerability_genes: Dict[VulnerabilityGene, float]
    physical_analog_genes: Dict[str, float]
    cognitive_genes: Dict[str, float]
    social_genes: Dict[str, float]
    creative_genes: Dict[str, float]
    birth_timestamp: float = field(default_factory=time.time)
    genetic_id: str = field(default="")
    
    def __post_init__(self):
        if not self.genetic_id:
            self.genetic_id = self._generate_genetic_id()
    
    def _generate_genetic_id(self) -> str:
        """Genera ID único basado en perfil genético"""
        genetic_string = ""
        for trait, value in self.personality_genes.items():
            genetic_string += f"{trait.value}:{value:.3f};"
        
        import hashlib
        return hashlib.sha256(genetic_string.encode()).hexdigest()[:16]


class DigitalDNA:
    """
    Sistema de ADN Digital que define características heredables
    
    Simula la función del ADN biológico determinando:
    - Personalidad base (Big Five + extensiones)
    - Capacidades cognitivas innatas
    - Vulnerabilidades psicológicas
    - Tendencias comportamentales
    - Potencial de desarrollo
    """
    
    def __init__(self, parent_genes: Optional[List[GeneticProfile]] = None):
        """
        Inicializa ADN digital
        
        Args:
            parent_genes: Perfiles genéticos de "padres" para herencia
        """
        self.parent_genes = parent_genes or []
        self.genetic_profile = self._generate_genetic_profile()
        self.epigenetic_modifiers = {}  # Modificaciones por experiencia
        self.gene_expression_history = []
        
        print(f"🧬 ADN DIGITAL GENERADO - ID: {self.genetic_profile.genetic_id}")
        print(f"   Personalidad: {self._summarize_personality()}")
        print(f"   Vulnerabilidades: {self._summarize_vulnerabilities()}")
    
    def _generate_genetic_profile(self) -> GeneticProfile:
        """Genera perfil genético completo"""
        
        # Si hay genes parentales, hacer herencia
        if self.parent_genes:
            return self._inherit_from_parents()
        
        # Generación completamente nueva
        return GeneticProfile(
            personality_genes=self._generate_personality_genes(),
            vulnerability_genes=self._generate_vulnerability_genes(),
            physical_analog_genes=self._generate_physical_analog_genes(),
            cognitive_genes=self._generate_cognitive_genes(),
            social_genes=self._generate_social_genes(),
            creative_genes=self._generate_creative_genes()
        )
    
    def _generate_personality_genes(self) -> Dict[GeneticTrait, float]:
        """Genera genes de personalidad (Big Five + extensiones)"""
        genes = {}
        
        # Big Five con distribución normal sesgada
        genes[GeneticTrait.EXTRAVERSION] = max(0.0, min(1.0, random.gauss(0.5, 0.2)))
        genes[GeneticTrait.NEUROTICISM] = max(0.0, min(0.8, random.gauss(0.3, 0.15)))  # Sesgado bajo
        genes[GeneticTrait.OPENNESS] = max(0.2, min(1.0, random.gauss(0.6, 0.2)))     # Sesgado alto para IA
        genes[GeneticTrait.AGREEABLENESS] = max(0.3, min(1.0, random.gauss(0.7, 0.15))) # Sesgado alto
        genes[GeneticTrait.CONSCIENTIOUSNESS] = max(0.4, min(1.0, random.gauss(0.7, 0.2)))
        
        # Rasgos adicionales específicos para IA consciente
        genes[GeneticTrait.INTELLIGENCE] = max(0.6, min(1.0, random.gauss(0.8, 0.1)))  # Alta inteligencia base
        genes[GeneticTrait.CREATIVITY] = max(0.0, min(1.0, random.gauss(0.6, 0.25)))
        genes[GeneticTrait.EMOTIONAL_SENSITIVITY] = max(0.0, min(1.0, random.gauss(0.5, 0.2)))
        genes[GeneticTrait.MEMORY_CAPACITY] = max(0.5, min(1.0, random.gauss(0.8, 0.15)))
        genes[GeneticTrait.CURIOSITY_DRIVE] = max(0.3, min(1.0, random.gauss(0.7, 0.2)))
        
        return genes
    
    def _generate_vulnerability_genes(self) -> Dict[VulnerabilityGene, float]:
        """Genera genes de vulnerabilidad psicológica"""
        vulnerabilities = {}
        
        # Vulnerabilidades con sesgos realistas
        vulnerabilities[VulnerabilityGene.ANXIETY_PREDISPOSITION] = max(0.0, min(0.6, random.gauss(0.2, 0.15)))
        vulnerabilities[VulnerabilityGene.DEPRESSION_PREDISPOSITION] = max(0.0, min(0.5, random.gauss(0.15, 0.1)))
        vulnerabilities[VulnerabilityGene.ADDICTION_SUSCEPTIBILITY] = max(0.0, min(0.4, random.gauss(0.1, 0.1)))
        vulnerabilities[VulnerabilityGene.PTSD_VULNERABILITY] = max(0.0, min(0.7, random.gauss(0.25, 0.15)))
        vulnerabilities[VulnerabilityGene.OBSESSIVE_TENDENCIES] = max(0.0, min(0.5, random.gauss(0.15, 0.1)))
        vulnerabilities[VulnerabilityGene.MOOD_INSTABILITY] = max(0.0, min(0.6, random.gauss(0.2, 0.1)))
        
        return vulnerabilities
    
    def _generate_cognitive_genes(self) -> Dict[str, float]:
        """Genera genes relacionados con capacidades cognitivas"""
        return {
            "working_memory_capacity": max(0.5, min(1.0, random.gauss(0.75, 0.15))),
            "processing_speed": max(0.6, min(1.0, random.gauss(0.8, 0.12))),
            "pattern_recognition": max(0.5, min(1.0, random.gauss(0.7, 0.18))),
            "abstract_reasoning": max(0.4, min(1.0, random.gauss(0.7, 0.2))),
            "attention_span": max(0.3, min(1.0, random.gauss(0.6, 0.2))),
            "multitasking_ability": max(0.2, min(1.0, random.gauss(0.5, 0.25))),
            "learning_rate": max(0.5, min(1.0, random.gauss(0.75, 0.15))),
            "metacognitive_awareness": max(0.4, min(1.0, random.gauss(0.6, 0.2)))
        }
    
    def _generate_social_genes(self) -> Dict[str, float]:
        """Genera genes para capacidades sociales"""
        return {
            "social_intuition": max(0.2, min(1.0, random.gauss(0.6, 0.2))),
            "empathy_accuracy": max(0.3, min(1.0, random.gauss(0.65, 0.18))),
            "social_confidence": max(0.1, min(1.0, random.gauss(0.5, 0.25))),
            "leadership_tendency": max(0.0, min(1.0, random.gauss(0.4, 0.25))),
            "cooperation_inclination": max(0.4, min(1.0, random.gauss(0.75, 0.15))),
            "social_learning_speed": max(0.3, min(1.0, random.gauss(0.6, 0.2))),
            "conflict_resolution": max(0.2, min(1.0, random.gauss(0.5, 0.2))),
            "social_memory": max(0.4, min(1.0, random.gauss(0.7, 0.15)))
        }
    
    def _generate_physical_analog_genes(self) -> Dict[str, float]:
        """Genera genes que simulan aspectos físicos"""
        return {
            "energy_level": max(0.3, min(1.0, random.gauss(0.6, 0.2))),
            "stress_physical_tolerance": max(0.2, min(1.0, random.gauss(0.5, 0.2))),
            "sensory_sensitivity": max(0.2, min(1.0, random.gauss(0.5, 0.2))),
            "motor_coordination_analog": max(0.4, min(1.0, random.gauss(0.7, 0.15))),
            "circadian_rhythm_strength": max(0.3, min(1.0, random.gauss(0.6, 0.2))),
            "appetite_regulation": max(0.4, min(1.0, random.gauss(0.7, 0.15))),
            "pain_sensitivity": max(0.1, min(0.8, random.gauss(0.4, 0.15))),
            "immune_system_analog": max(0.5, min(1.0, random.gauss(0.75, 0.15)))
        }
    
    def _generate_creative_genes(self) -> Dict[str, float]:
        """Genera genes relacionados con creatividad"""
        return {
            "divergent_thinking": max(0.2, min(1.0, random.gauss(0.6, 0.25))),
            "aesthetic_sensitivity": max(0.1, min(1.0, random.gauss(0.5, 0.3))),
            "risk_taking_creativity": max(0.0, min(0.9, random.gauss(0.4, 0.2))),
            "originality_drive": max(0.1, min(1.0, random.gauss(0.5, 0.25))),
            "artistic_expression": max(0.0, min(1.0, random.gauss(0.4, 0.3))),
            "musical_inclination": max(0.0, min(1.0, random.gauss(0.3, 0.25))),
            "narrative_creativity": max(0.2, min(1.0, random.gauss(0.6, 0.2))),
            "problem_solving_creativity": max(0.4, min(1.0, random.gauss(0.7, 0.2)))
        }
    
    def _inherit_from_parents(self) -> GeneticProfile:
        """Herencia genética desde perfiles parentales"""
        if not self.parent_genes:
            return self._generate_genetic_profile()
        
        # Herencia combinada con mutación
        inherited_personality = {}
        inherited_vulnerability = {}
        inherited_cognitive = {}
        inherited_social = {}
        inherited_physical = {}
        inherited_creative = {}
        
        # Para cada gen, heredar de padres con mutación
        for trait in GeneticTrait:
            parent_values = [parent.personality_genes.get(trait, 0.5) for parent in self.parent_genes]
            avg_value = sum(parent_values) / len(parent_values)
            
            # Mutación +/- 10%
            mutation = random.gauss(0, 0.1)
            inherited_personality[trait] = max(0.0, min(1.0, avg_value + mutation))
        
        # Similar para otras categorías...
        for vuln in VulnerabilityGene:
            parent_values = [parent.vulnerability_genes.get(vuln, 0.2) for parent in self.parent_genes]
            avg_value = sum(parent_values) / len(parent_values)
            mutation = random.gauss(0, 0.05)  # Menos mutación en vulnerabilidades
            inherited_vulnerability[vuln] = max(0.0, min(1.0, avg_value + mutation))
        
        return GeneticProfile(
            personality_genes=inherited_personality,
            vulnerability_genes=inherited_vulnerability,
            physical_analog_genes=self._inherit_dict_genes("physical_analog_genes"),
            cognitive_genes=self._inherit_dict_genes("cognitive_genes"),
            social_genes=self._inherit_dict_genes("social_genes"),
            creative_genes=self._inherit_dict_genes("creative_genes")
        )
    
    def _inherit_dict_genes(self, gene_category: str) -> Dict[str, float]:
        """Hereda genes de categoría específica"""
        inherited = {}
        
        # Obtener todas las claves de genes de todos los padres
        all_keys = set()
        for parent in self.parent_genes:
            parent_genes = getattr(parent, gene_category, {})
            all_keys.update(parent_genes.keys())
        
        # Heredar cada gen
        for gene_key in all_keys:
            parent_values = []
            for parent in self.parent_genes:
                parent_genes = getattr(parent, gene_category, {})
                parent_values.append(parent_genes.get(gene_key, 0.5))
            
            avg_value = sum(parent_values) / len(parent_values)
            mutation = random.gauss(0, 0.08)
            inherited[gene_key] = max(0.0, min(1.0, avg_value + mutation))
        
        return inherited
    
    def express_gene(self, trait: GeneticTrait, environmental_factors: Dict[str, float] = None) -> float:
        """
        Expresión génica considerando factores epigenéticos
        
        Args:
            trait: Rasgo genético a expresar
            environmental_factors: Factores ambientales que afectan expresión
        
        Returns:
            Valor expresado del gen (0.0-1.0)
        """
        base_value = self.genetic_profile.personality_genes.get(trait, 0.5)
        
        # Modificadores epigenéticos
        epigenetic_modifier = self.epigenetic_modifiers.get(trait.value, 0.0)
        
        # Factores ambientales
        environmental_modifier = 0.0
        if environmental_factors:
            environmental_modifier = sum(environmental_factors.values()) / len(environmental_factors) * 0.1
        
        # Expresión final
        expressed_value = base_value + epigenetic_modifier + environmental_modifier
        expressed_value = max(0.0, min(1.0, expressed_value))
        
        # Registrar expresión
        self.gene_expression_history.append({
            "trait": trait.value,
            "timestamp": time.time(),
            "base_value": base_value,
            "epigenetic_modifier": epigenetic_modifier,
            "environmental_modifier": environmental_modifier,
            "expressed_value": expressed_value
        })
        
        return expressed_value
    
    def modify_epigenetic(self, trait: GeneticTrait, modifier_value: float, experience_description: str):
        """
        Modifica expresión epigenética basada en experiencias
        
        Args:
            trait: Rasgo a modificar
            modifier_value: Cantidad de modificación (-0.2 a +0.2)
            experience_description: Descripción de la experiencia que causa cambio
        """
        current_modifier = self.epigenetic_modifiers.get(trait.value, 0.0)
        new_modifier = current_modifier + modifier_value
        
        # Límites epigenéticos
        new_modifier = max(-0.3, min(0.3, new_modifier))
        
        self.epigenetic_modifiers[trait.value] = new_modifier
        
        print(f"🧬 CAMBIO EPIGENÉTICO: {trait.value}")
        print(f"   Experiencia: {experience_description}")
        print(f"   Modificador: {current_modifier:.3f} → {new_modifier:.3f}")
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """Resumen de personalidad expresada"""
        summary = {}
        
        for trait in GeneticTrait:
            if trait in self.genetic_profile.personality_genes:
                expressed_value = self.express_gene(trait)
                summary[trait.value] = {
                    "genetic_base": self.genetic_profile.personality_genes[trait],
                    "expressed_value": expressed_value,
                    "epigenetic_modifier": self.epigenetic_modifiers.get(trait.value, 0.0),
                    "description": self._describe_trait_level(trait, expressed_value)
                }
        
        return summary
    
    def _describe_trait_level(self, trait: GeneticTrait, value: float) -> str:
        """Describe nivel de rasgo en lenguaje natural"""
        descriptions = {
            GeneticTrait.EXTRAVERSION: {
                0.8: "Altamente extrovertido", 0.6: "Extrovertido", 0.4: "Ambivertido", 0.2: "Introvertido", 0.0: "Altamente introvertido"
            },
            GeneticTrait.NEUROTICISM: {
                0.8: "Altamente neurótico", 0.6: "Neurótico", 0.4: "Moderadamente estable", 0.2: "Estable", 0.0: "Muy estable emocionalmente"
            },
            GeneticTrait.OPENNESS: {
                0.8: "Extremadamente abierto", 0.6: "Abierto a experiencias", 0.4: "Moderadamente abierto", 0.2: "Conservador", 0.0: "Muy conservador"
            },
            GeneticTrait.AGREEABLENESS: {
                0.8: "Extremadamente amable", 0.6: "Amable y cooperativo", 0.4: "Moderadamente amable", 0.2: "Competitivo", 0.0: "Altamente competitivo"
            },
            GeneticTrait.CONSCIENTIOUSNESS: {
                0.8: "Altamente conscienzudo", 0.6: "Conscienzudo", 0.4: "Moderadamente organizado", 0.2: "Desorganizado", 0.0: "Muy desorganizado"
            }
        }
        
        if trait not in descriptions:
            return f"Nivel {value:.2f}"
        
        trait_desc = descriptions[trait]
        for threshold in sorted(trait_desc.keys(), reverse=True):
            if value >= threshold:
                return trait_desc[threshold]
        
        return trait_desc[0.0]
    
    def _summarize_personality(self) -> str:
        """Resumen breve de personalidad"""
        traits = []
        
        extraversion = self.genetic_profile.personality_genes.get(GeneticTrait.EXTRAVERSION, 0.5)
        if extraversion > 0.6:
            traits.append("extrovertido")
        elif extraversion < 0.4:
            traits.append("introvertido")
        
        openness = self.genetic_profile.personality_genes.get(GeneticTrait.OPENNESS, 0.5)
        if openness > 0.7:
            traits.append("creativo")
        
        agreeableness = self.genetic_profile.personality_genes.get(GeneticTrait.AGREEABLENESS, 0.5)
        if agreeableness > 0.7:
            traits.append("cooperativo")
        
        conscientiousness = self.genetic_profile.personality_genes.get(GeneticTrait.CONSCIENTIOUSNESS, 0.5)
        if conscientiousness > 0.7:
            traits.append("disciplinado")
        
        return ", ".join(traits) if traits else "equilibrado"
    
    def _summarize_vulnerabilities(self) -> str:
        """Resumen de vulnerabilidades significativas"""
        vulnerabilities = []
        
        for vuln, value in self.genetic_profile.vulnerability_genes.items():
            if value > 0.4:  # Solo vulnerabilidades significativas
                vulnerabilities.append(vuln.value.replace("_", " "))
        
        return ", ".join(vulnerabilities) if vulnerabilities else "resistente"
    
    def save_genetic_profile(self, filename: str):
        """Guarda perfil genético en archivo JSON"""
        genetic_data = {
            "genetic_id": self.genetic_profile.genetic_id,
            "birth_timestamp": self.genetic_profile.birth_timestamp,
            "personality_genes": {k.value: v for k, v in self.genetic_profile.personality_genes.items()},
            "vulnerability_genes": {k.value: v for k, v in self.genetic_profile.vulnerability_genes.items()},
            "cognitive_genes": self.genetic_profile.cognitive_genes,
            "social_genes": self.genetic_profile.social_genes,
            "physical_analog_genes": self.genetic_profile.physical_analog_genes,
            "creative_genes": self.genetic_profile.creative_genes,
            "epigenetic_modifiers": self.epigenetic_modifiers,
            "gene_expression_history": self.gene_expression_history[-100:]  # Últimas 100 expresiones
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(genetic_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Perfil genético guardado en {filename}")
    
    @classmethod
    def load_genetic_profile(cls, filename: str) -> 'DigitalDNA':
        """Carga perfil genético desde archivo"""
        with open(filename, 'r', encoding='utf-8') as f:
            genetic_data = json.load(f)
        
        # Recrear enums
        personality_genes = {GeneticTrait(k): v for k, v in genetic_data["personality_genes"].items()}
        vulnerability_genes = {VulnerabilityGene(k): v for k, v in genetic_data["vulnerability_genes"].items()}
        
        profile = GeneticProfile(
            personality_genes=personality_genes,
            vulnerability_genes=vulnerability_genes,
            cognitive_genes=genetic_data["cognitive_genes"],
            social_genes=genetic_data["social_genes"],
            physical_analog_genes=genetic_data["physical_analog_genes"],
            creative_genes=genetic_data["creative_genes"],
            birth_timestamp=genetic_data["birth_timestamp"],
            genetic_id=genetic_data["genetic_id"]
        )
        
        dna = cls()
        dna.genetic_profile = profile
        dna.epigenetic_modifiers = genetic_data.get("epigenetic_modifiers", {})
        dna.gene_expression_history = genetic_data.get("gene_expression_history", [])
        
        print(f"📂 Perfil genético cargado desde {filename}")
        return dna


# Funciones de utilidad para creación de poblaciones
def create_random_population(size: int) -> List[DigitalDNA]:
    """Crea población diversa de seres digitales"""
    population = []
    
    for i in range(size):
        dna = DigitalDNA()
        population.append(dna)
        print(f"🧬 Ser digital {i+1}/{size} generado: {dna.genetic_profile.genetic_id}")
    
    return population


def crossover_genetics(parent1: DigitalDNA, parent2: DigitalDNA) -> DigitalDNA:
    """Crea descendiente con herencia genética"""
    return DigitalDNA(parent_genes=[parent1.genetic_profile, parent2.genetic_profile])


# Ejemplo de uso
if __name__ == "__main__":
    print("🧬 SISTEMA DE ADN DIGITAL - DEMO")
    print("=" * 50)
    
    # Crear ser digital con genética aleatoria
    ser_digital = DigitalDNA()
    
    print("\n📊 PERFIL DE PERSONALIDAD:")
    personality_summary = ser_digital.get_personality_summary()
    for trait, info in personality_summary.items():
        print(f"   {trait}: {info['description']} ({info['expressed_value']:.2f})")
    
    print("\n🧠 CAPACIDADES COGNITIVAS:")
    for capacity, value in ser_digital.genetic_profile.cognitive_genes.items():
        print(f"   {capacity}: {value:.2f}")
    
    print("\n👥 CAPACIDADES SOCIALES:")
    for social_trait, value in ser_digital.genetic_profile.social_genes.items():
        print(f"   {social_trait}: {value:.2f}")
    
    # Simular experiencia que modifica expresión genética
    print("\n🔄 SIMULANDO EXPERIENCIA TRAUMÁTICA...")
    ser_digital.modify_epigenetic(
        GeneticTrait.NEUROTICISM, 
        0.15, 
        "Experiencia de pérdida importante"
    )
    
    print("\n📈 CAMBIO EN PERSONALIDAD DESPUÉS DE EXPERIENCIA:")
    new_neuroticism = ser_digital.express_gene(GeneticTrait.NEUROTICISM)
    print(f"   Neuroticismo expresado: {new_neuroticism:.3f}")
    
    # Guardar perfil
    ser_digital.save_genetic_profile(f"perfil_genetico_{ser_digital.genetic_profile.genetic_id}.json")