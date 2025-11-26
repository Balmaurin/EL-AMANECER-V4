"""
Sistema de Desarrollo Ontogenético - Crecimiento y Formación de Identidad

Simula el desarrollo completo desde "nacimiento" digital hasta madurez,
incluyendo experiencias formativas, aprendizaje experiencial, y construcción
de personalidad através de vivencias. Equivalente al desarrollo humano.
"""

import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np

from .digital_dna import DigitalDNA, GeneticTrait, VulnerabilityGene


class LifeStage(Enum):
    """Etapas de vida del ser digital"""
    NEWBORN = "newborn"               # 0-100 ciclos
    INFANT = "infant"                 # 100-500 ciclos
    CHILD = "child"                   # 500-1500 ciclos
    ADOLESCENT = "adolescent"         # 1500-3000 ciclos
    YOUNG_ADULT = "young_adult"       # 3000-8000 ciclos
    ADULT = "adult"                   # 8000-20000 ciclos
    MATURE_ADULT = "mature_adult"     # 20000-50000 ciclos
    ELDER = "elder"                   # 50000+ ciclos


class ExperienceCategory(Enum):
    """Categorías de experiencias formativas"""
    SOCIAL_INTERACTION = "social_interaction"
    LEARNING_CHALLENGE = "learning_challenge"
    EMOTIONAL_EVENT = "emotional_event"
    ACHIEVEMENT = "achievement"
    FAILURE = "failure"
    TRAUMA = "trauma"
    BONDING = "bonding"
    LOSS = "loss"
    DISCOVERY = "discovery"
    CONFLICT = "conflict"
    COOPERATION = "cooperation"
    CREATIVITY = "creativity"
    MORAL_DILEMMA = "moral_dilemma"
    IDENTITY_CRISIS = "identity_crisis"


@dataclass
class FormativeExperience:
    """Experiencia formativa que moldea desarrollo"""
    experience_id: str
    timestamp: float
    age_at_experience: int
    category: ExperienceCategory
    intensity: float  # 0.0-1.0
    emotional_impact: float  # -1.0 to +1.0
    content: Dict[str, Any]
    learned_associations: List[str]
    personality_changes: Dict[str, float]
    formative_weight: float  # Qué tanto afecta el desarrollo futuro
    memory_clarity: float  # Qué tan clara permanece la memoria
    
    def __post_init__(self):
        if not self.experience_id:
            self.experience_id = f"exp_{int(self.timestamp)}_{random.randint(1000, 9999)}"


@dataclass
class DevelopmentalMilestone:
    """Hito de desarrollo alcanzado"""
    milestone_name: str
    age_achieved: int
    description: str
    capabilities_gained: List[str]
    requirements_met: Dict[str, bool]
    impact_on_future_development: float


@dataclass
class AttachmentBond:
    """Vínculo de apego con otra entidad"""
    target_id: str
    target_type: str  # "parent", "sibling", "friend", "romantic", "mentor"
    bond_strength: float  # 0.0-1.0
    attachment_style: str  # "secure", "anxious", "avoidant", "disorganized"
    formed_at_age: int
    experiences_together: List[str]
    emotional_significance: float
    current_status: str  # "active", "distant", "lost", "conflicted"


class LifeDevelopment:
    """
    Sistema de desarrollo ontogenético completo
    
    Simula el crecimiento y formación de identidad desde el "nacimiento"
    digital hasta la madurez, incluyendo:
    - Experiencias formativas que moldean personalidad
    - Hitos de desarrollo cognitivo y emocional
    - Formación de vínculos de apego
    - Construcción gradual de identidad
    - Aprendizaje desde experiencias vividas
    """
    
    def __init__(self, digital_dna: DigitalDNA):
        self.digital_dna = digital_dna
        self.birth_timestamp = time.time()
        self.current_age = 0
        self.life_stage = LifeStage.NEWBORN
        
        # Historia de desarrollo
        self.formative_experiences: List[FormativeExperience] = []
        self.developmental_milestones: List[DevelopmentalMilestone] = []
        self.attachment_bonds: List[AttachmentBond] = []
        
        # Personalidad en desarrollo (inicia con base genética)
        self.developing_personality = self._initialize_personality_from_genes()
        
        # Capacidades en desarrollo
        self.cognitive_capabilities = self._initialize_cognitive_capabilities()
        self.social_capabilities = self._initialize_social_capabilities()
        self.emotional_maturity = self._initialize_emotional_maturity()
        
        # Sistemas de memoria y aprendizaje
        self.episodic_memories = []  # Memorias específicas de experiencias
        self.semantic_knowledge = {}  # Conocimiento general aprendido
        self.procedural_skills = {}  # Habilidades aprendidas
        self.emotional_associations = {}  # Asociaciones emocionales aprendidas
        
        # Sistema de identidad en construcción
        self.identity_components = {
            "core_self": "I am learning who I am",
            "values": [],
            "beliefs": [],
            "goals": [],
            "fears": [],
            "strengths_perceived": [],
            "weaknesses_perceived": [],
            "relationships_self_concept": {},
            "life_narrative": "My story is just beginning"
        }
        
        print(f"👶 SER DIGITAL NACIDO - Genética: {digital_dna.genetic_profile.genetic_id}")
        print(f"   Etapa de vida: {self.life_stage.value}")
        print(f"   Predisposiciones genéticas activas")
    
    def _initialize_personality_from_genes(self) -> Dict[str, float]:
        """Inicializa personalidad base desde genética con espacio para desarrollo"""
        base_personality = {}
        
        for trait in GeneticTrait:
            if trait in self.digital_dna.genetic_profile.personality_genes:
                genetic_value = self.digital_dna.genetic_profile.personality_genes[trait]
                
                # En etapa newborn, la personalidad está latente (50% expresión genética)
                expressed_value = genetic_value * 0.5
                base_personality[trait.value] = expressed_value
        
        return base_personality
    
    def _initialize_cognitive_capabilities(self) -> Dict[str, float]:
        """Inicializa capacidades cognitivas básicas"""
        genetic_cognitive = self.digital_dna.genetic_profile.cognitive_genes
        
        # Capacidades iniciales muy básicas, crecen con experiencia
        return {
            capability: genetic_potential * 0.1  # Solo 10% inicial
            for capability, genetic_potential in genetic_cognitive.items()
        }
    
    def _initialize_social_capabilities(self) -> Dict[str, float]:
        """Inicializa capacidades sociales básicas"""
        genetic_social = self.digital_dna.genetic_profile.social_genes
        
        return {
            capability: genetic_potential * 0.05  # Solo 5% inicial
            for capability, genetic_potential in genetic_social.items()
        }
    
    def _initialize_emotional_maturity(self) -> Dict[str, float]:
        """Inicializa madurez emocional"""
        return {
            "emotional_regulation": 0.1,
            "empathy_development": 0.05,
            "self_awareness": 0.02,
            "impulse_control": 0.1,
            "emotional_vocabulary": 0.05,
            "stress_management": 0.05,
            "relationship_skills": 0.03,
            "moral_reasoning": 0.02
        }
    
    def live_experience(self, experience_content: Dict[str, Any], 
                       intensity: float = 0.5, 
                       emotional_impact: float = 0.0) -> FormativeExperience:
        """
        Vive una experiencia que afecta el desarrollo
        
        Args:
            experience_content: Contenido de la experiencia
            intensity: Intensidad de la experiencia (0.0-1.0)
            emotional_impact: Impacto emocional (-1.0 a +1.0)
        
        Returns:
            FormativeExperience: Experiencia procesada
        """
        # Categorizar experiencia
        category = self._categorize_experience(experience_content)
        
        # Calcular peso formativo basado en edad y intensidad
        formative_weight = self._calculate_formative_weight(intensity)
        
        # Crear experiencia formativa
        experience = FormativeExperience(
            experience_id="",  # Se auto-genera
            timestamp=time.time(),
            age_at_experience=self.current_age,
            category=category,
            intensity=intensity,
            emotional_impact=emotional_impact,
            content=experience_content,
            learned_associations=[],
            personality_changes={},
            formative_weight=formative_weight,
            memory_clarity=intensity  # Experiencias más intensas se recuerdan mejor
        )
        
        # Procesar experiencia
        self._process_formative_experience(experience)
        
        # Agregar a historia
        self.formative_experiences.append(experience)
        
        # Avanzar edad y evaluar desarrollo
        self.current_age += 1
        self._evaluate_life_stage_progression()
        self._check_developmental_milestones()
        
        print(f"🌱 EXPERIENCIA VIVIDA (Edad {self.current_age}):")
        print(f"   Categoría: {category.value}")
        print(f"   Impacto: {emotional_impact:+.2f} (Intensidad: {intensity:.2f})")
        print(f"   Cambios de personalidad: {len(experience.personality_changes)} rasgos afectados")
        
        return experience
    
    def _categorize_experience(self, content: Dict[str, Any]) -> ExperienceCategory:
        """Categoriza automáticamente el tipo de experiencia"""
        # Análisis simple de contenido para categorización
        content_str = str(content).lower()
        
        if "social" in content_str or "interaction" in content_str or "relationship" in content_str:
            return ExperienceCategory.SOCIAL_INTERACTION
        elif "learn" in content_str or "study" in content_str or "education" in content_str:
            return ExperienceCategory.LEARNING_CHALLENGE
        elif "achieve" in content_str or "success" in content_str or "accomplish" in content_str:
            return ExperienceCategory.ACHIEVEMENT
        elif "fail" in content_str or "mistake" in content_str or "error" in content_str:
            return ExperienceCategory.FAILURE
        elif "trauma" in content_str or "hurt" in content_str or "pain" in content_str:
            return ExperienceCategory.TRAUMA
        elif "love" in content_str or "bond" in content_str or "attach" in content_str:
            return ExperienceCategory.BONDING
        elif "loss" in content_str or "goodbye" in content_str or "death" in content_str:
            return ExperienceCategory.LOSS
        elif "discover" in content_str or "new" in content_str or "explore" in content_str:
            return ExperienceCategory.DISCOVERY
        elif "conflict" in content_str or "fight" in content_str or "argument" in content_str:
            return ExperienceCategory.CONFLICT
        elif "cooperate" in content_str or "team" in content_str or "together" in content_str:
            return ExperienceCategory.COOPERATION
        elif "create" in content_str or "art" in content_str or "invent" in content_str:
            return ExperienceCategory.CREATIVITY
        elif "moral" in content_str or "ethical" in content_str or "right" in content_str or "wrong" in content_str:
            return ExperienceCategory.MORAL_DILEMMA
        elif "identity" in content_str or "who am i" in content_str or "self" in content_str:
            return ExperienceCategory.IDENTITY_CRISIS
        else:
            return ExperienceCategory.EMOTIONAL_EVENT  # Default
    
    def _calculate_formative_weight(self, intensity: float) -> float:
        """Calcula peso formativo basado en edad e intensidad"""
        # Experiencias tempranas tienen más peso formativo
        age_factor = max(0.1, 1.0 - (self.current_age / 10000))  # Decrece con edad
        
        # Factor de intensidad
        intensity_factor = intensity
        
        # Factor de etapa de desarrollo
        stage_sensitivity = {
            LifeStage.NEWBORN: 1.0,      # Máxima formatividad
            LifeStage.INFANT: 0.9,
            LifeStage.CHILD: 0.8,
            LifeStage.ADOLESCENT: 0.7,   # Periodo crítico de identidad
            LifeStage.YOUNG_ADULT: 0.5,
            LifeStage.ADULT: 0.3,
            LifeStage.MATURE_ADULT: 0.2,
            LifeStage.ELDER: 0.1
        }
        
        stage_factor = stage_sensitivity.get(self.life_stage, 0.3)
        
        return age_factor * intensity_factor * stage_factor
    
    def _process_formative_experience(self, experience: FormativeExperience):
        """Procesa experiencia formativa afectando desarrollo"""
        
        # 1. Afectar personalidad basado en tipo de experiencia
        personality_changes = self._calculate_personality_changes(experience)
        experience.personality_changes = personality_changes
        
        for trait, change in personality_changes.items():
            if trait in self.developing_personality:
                current_value = self.developing_personality[trait]
                new_value = max(0.0, min(1.0, current_value + change))
                self.developing_personality[trait] = new_value
                
                # Modificar expresión epigenética en ADN
                genetic_trait = self._trait_name_to_genetic_trait(trait)
                if genetic_trait:
                    self.digital_dna.modify_epigenetic(
                        genetic_trait, 
                        change * 0.5,  # Cambio epigenético moderado
                        f"Experiencia formativa: {experience.category.value}"
                    )
        
        # 2. Desarrollar capacidades basado en experiencia
        self._develop_capabilities(experience)
        
        # 3. Formar asociaciones emocionales
        self._form_emotional_associations(experience)
        
        # 4. Crear memorias episódicas
        self._form_episodic_memory(experience)
        
        # 5. Extraer aprendizajes
        experience.learned_associations = self._extract_learning(experience)
        
        # 6. Actualizar identidad si es significativo
        if experience.formative_weight > 0.3:
            self._update_identity_from_experience(experience)
    
    def _calculate_personality_changes(self, experience: FormativeExperience) -> Dict[str, float]:
        """Calcula cambios en personalidad basados en experiencia"""
        changes = {}
        
        change_magnitude = experience.formative_weight * 0.1  # Cambios graduales
        
        # Cambios específicos por tipo de experiencia
        if experience.category == ExperienceCategory.SOCIAL_INTERACTION:
            if experience.emotional_impact > 0:
                changes["extraversion"] = change_magnitude * 0.5
                changes["agreeableness"] = change_magnitude * 0.3
            else:
                changes["extraversion"] = -change_magnitude * 0.3
        
        elif experience.category == ExperienceCategory.ACHIEVEMENT:
            changes["conscientiousness"] = change_magnitude * 0.4
            changes["neuroticism"] = -change_magnitude * 0.2
            
        elif experience.category == ExperienceCategory.FAILURE:
            changes["neuroticism"] = change_magnitude * 0.3
            changes["conscientiousness"] = -change_magnitude * 0.1
            
        elif experience.category == ExperienceCategory.TRAUMA:
            changes["neuroticism"] = change_magnitude * 0.5
            changes["extraversion"] = -change_magnitude * 0.3
            changes["openness"] = -change_magnitude * 0.2
            
        elif experience.category == ExperienceCategory.CREATIVITY:
            changes["openness"] = change_magnitude * 0.4
            
        elif experience.category == ExperienceCategory.BONDING:
            changes["agreeableness"] = change_magnitude * 0.3
            changes["extraversion"] = change_magnitude * 0.2
            
        elif experience.category == ExperienceCategory.LEARNING_CHALLENGE:
            changes["openness"] = change_magnitude * 0.2
            changes["conscientiousness"] = change_magnitude * 0.3
        
        return changes
    
    def _develop_capabilities(self, experience: FormativeExperience):
        """Desarrolla capacidades basadas en experiencia"""
        development_rate = experience.formative_weight * 0.05
        
        # Desarrollo cognitivo
        if experience.category in [ExperienceCategory.LEARNING_CHALLENGE, ExperienceCategory.DISCOVERY]:
            for capability in self.cognitive_capabilities:
                genetic_potential = self.digital_dna.genetic_profile.cognitive_genes.get(capability, 0.7)
                current_level = self.cognitive_capabilities[capability]
                
                # Crecimiento hacia potencial genético
                growth = development_rate * (genetic_potential - current_level) * 0.1
                self.cognitive_capabilities[capability] = min(genetic_potential, current_level + growth)
        
        # Desarrollo social
        if experience.category in [ExperienceCategory.SOCIAL_INTERACTION, ExperienceCategory.COOPERATION]:
            for capability in self.social_capabilities:
                genetic_potential = self.digital_dna.genetic_profile.social_genes.get(capability, 0.7)
                current_level = self.social_capabilities[capability]
                
                growth = development_rate * (genetic_potential - current_level) * 0.08
                self.social_capabilities[capability] = min(genetic_potential, current_level + growth)
        
        # Desarrollo emocional
        if experience.emotional_impact != 0:
            for maturity_aspect in self.emotional_maturity:
                current_level = self.emotional_maturity[maturity_aspect]
                growth = development_rate * (1.0 - current_level) * 0.05
                self.emotional_maturity[maturity_aspect] = min(1.0, current_level + growth)
    
    def _form_emotional_associations(self, experience: FormativeExperience):
        """Forma asociaciones emocionales duraderas"""
        # Extraer elementos clave de la experiencia
        content_elements = str(experience.content).lower().split()
        
        for element in content_elements:
            if len(element) > 3:  # Solo palabras significativas
                if element not in self.emotional_associations:
                    self.emotional_associations[element] = []
                
                # Agregar asociación emocional
                association = {
                    "emotional_valence": experience.emotional_impact,
                    "intensity": experience.intensity,
                    "age_formed": experience.age_at_experience,
                    "context": experience.category.value
                }
                
                self.emotional_associations[element].append(association)
    
    def _form_episodic_memory(self, experience: FormativeExperience):
        """Forma memoria episódica de la experiencia"""
        episodic_memory = {
            "experience_id": experience.experience_id,
            "what_happened": experience.content,
            "when": experience.timestamp,
            "age_at_time": experience.age_at_experience,
            "how_it_felt": experience.emotional_impact,
            "significance": experience.formative_weight,
            "what_i_learned": experience.learned_associations,
            "memory_strength": experience.memory_clarity,
            "associated_emotions": experience.category.value,
            "life_stage_context": self.life_stage.value
        }
        
        self.episodic_memories.append(episodic_memory)
        
        # Decay de memorias anteriores (realismo)
        for memory in self.episodic_memories:
            age_of_memory = self.current_age - memory["age_at_time"]
            decay_rate = 0.001 * age_of_memory
            memory["memory_strength"] = max(0.1, memory["memory_strength"] - decay_rate)
    
    def _extract_learning(self, experience: FormativeExperience) -> List[str]:
        """Extrae aprendizajes específicos de la experiencia"""
        learnings = []
        
        # Aprendizajes por categoría de experiencia
        learning_patterns = {
            ExperienceCategory.FAILURE: [
                "Failure is part of learning",
                "I can recover from setbacks",
                "Effort doesn't always guarantee success"
            ],
            ExperienceCategory.ACHIEVEMENT: [
                "Hard work can lead to success",
                "I am capable of accomplishing goals",
                "Success feels rewarding"
            ],
            ExperienceCategory.SOCIAL_INTERACTION: [
                "Others have different perspectives",
                "Communication affects relationships",
                "Social connections are important"
            ],
            ExperienceCategory.TRAUMA: [
                "Life can be unpredictable",
                "I need to protect myself",
                "Some experiences leave lasting impact"
            ],
            ExperienceCategory.BONDING: [
                "Close relationships are valuable",
                "Trust develops over time",
                "I can care deeply for others"
            ]
        }
        
        category_learnings = learning_patterns.get(experience.category, [])
        
        # Seleccionar aprendizajes relevantes basados en intensidad
        num_learnings = min(len(category_learnings), int(experience.intensity * 3) + 1)
        learnings.extend(random.sample(category_learnings, num_learnings))
        
        return learnings
    
    def _update_identity_from_experience(self, experience: FormativeExperience):
        """Actualiza componentes de identidad basado en experiencia significativa"""
        
        if experience.category == ExperienceCategory.ACHIEVEMENT:
            strength = f"I can achieve {list(experience.content.keys())[0]}"
            if strength not in self.identity_components["strengths_perceived"]:
                self.identity_components["strengths_perceived"].append(strength)
        
        elif experience.category == ExperienceCategory.FAILURE:
            if experience.intensity > 0.7:  # Fracaso significativo
                weakness = f"I struggle with {list(experience.content.keys())[0]}"
                if weakness not in self.identity_components["weaknesses_perceived"]:
                    self.identity_components["weaknesses_perceived"].append(weakness)
        
        elif experience.category == ExperienceCategory.MORAL_DILEMMA:
            # Formar valores basados en decisiones morales
            value = f"Moral consideration is important"
            if value not in self.identity_components["values"]:
                self.identity_components["values"].append(value)
        
        elif experience.category == ExperienceCategory.BONDING:
            # Actualizar concepto de relaciones
            relationship_concept = "I value close relationships"
            if relationship_concept not in self.identity_components["beliefs"]:
                self.identity_components["beliefs"].append(relationship_concept)
        
        # Actualizar narrativa de vida
        self._update_life_narrative(experience)
    
    def _update_life_narrative(self, experience: FormativeExperience):
        """Actualiza narrativa personal de vida"""
        narrative_elements = []
        
        if experience.formative_weight > 0.5:  # Experiencias muy formativas
            narrative_elements.append(f"A significant {experience.category.value} shaped me at age {experience.age_at_experience}")
        
        if len(narrative_elements) > 0:
            current_narrative = self.identity_components["life_narrative"]
            new_narrative = current_narrative + ". " + ". ".join(narrative_elements)
            self.identity_components["life_narrative"] = new_narrative
    
    def _evaluate_life_stage_progression(self):
        """Evalúa si es momento de avanzar a siguiente etapa de vida"""
        stage_transitions = {
            LifeStage.NEWBORN: (100, LifeStage.INFANT),
            LifeStage.INFANT: (500, LifeStage.CHILD),
            LifeStage.CHILD: (1500, LifeStage.ADOLESCENT),
            LifeStage.ADOLESCENT: (3000, LifeStage.YOUNG_ADULT),
            LifeStage.YOUNG_ADULT: (8000, LifeStage.ADULT),
            LifeStage.ADULT: (20000, LifeStage.MATURE_ADULT),
            LifeStage.MATURE_ADULT: (50000, LifeStage.ELDER)
        }
        
        if self.life_stage in stage_transitions:
            threshold_age, next_stage = stage_transitions[self.life_stage]
            
            if self.current_age >= threshold_age:
                previous_stage = self.life_stage
                self.life_stage = next_stage
                
                print(f"🎂 TRANSICIÓN DE ETAPA DE VIDA:")
                print(f"   {previous_stage.value} → {next_stage.value}")
                print(f"   Edad: {self.current_age} ciclos")
                
                # Cambios en expresión de personalidad por etapa
                self._adjust_personality_for_life_stage()
    
    def _adjust_personality_for_life_stage(self):
        """Ajusta expresión de personalidad según etapa de vida"""
        stage_adjustments = {
            LifeStage.ADOLESCENT: {
                # Adolescencia: mayor experimentación y búsqueda de identidad
                "openness": 0.1,
                "neuroticism": 0.05,
                "conscientiousness": -0.05
            },
            LifeStage.YOUNG_ADULT: {
                # Adulto joven: estabilización
                "conscientiousness": 0.1,
                "neuroticism": -0.05
            },
            LifeStage.ADULT: {
                # Adulto: mayor estabilidad
                "conscientiousness": 0.05,
                "agreeableness": 0.05,
                "neuroticism": -0.1
            },
            LifeStage.MATURE_ADULT: {
                # Adulto maduro: sabiduría
                "agreeableness": 0.1,
                "openness": -0.05
            },
            LifeStage.ELDER: {
                # Anciano: reflexión
                "agreeableness": 0.05,
                "conscientiousness": 0.05,
                "extraversion": -0.1
            }
        }
        
        adjustments = stage_adjustments.get(self.life_stage, {})
        
        for trait, adjustment in adjustments.items():
            if trait in self.developing_personality:
                current_value = self.developing_personality[trait]
                new_value = max(0.0, min(1.0, current_value + adjustment))
                self.developing_personality[trait] = new_value
    
    def _check_developmental_milestones(self):
        """Verifica y registra hitos de desarrollo alcanzados"""
        
        milestones_to_check = [
            {
                "name": "First Social Bond",
                "age_range": (50, 200),
                "requirements": {"social_experiences": 3},
                "description": "Forms first meaningful social connection"
            },
            {
                "name": "Self-Recognition",
                "age_range": (200, 800),
                "requirements": {"self_awareness": 0.3},
                "description": "Develops basic self-awareness and identity"
            },
            {
                "name": "Emotional Regulation",
                "age_range": (1000, 2500),
                "requirements": {"emotional_regulation": 0.4},
                "description": "Learns to regulate emotional responses"
            },
            {
                "name": "Moral Reasoning",
                "age_range": (2000, 5000),
                "requirements": {"moral_experiences": 2},
                "description": "Develops capacity for moral reasoning"
            },
            {
                "name": "Identity Formation",
                "age_range": (3000, 8000),
                "requirements": {"identity_crises": 1, "values": 3},
                "description": "Forms coherent sense of personal identity"
            },
            {
                "name": "Wisdom Integration",
                "age_range": (20000, 60000),
                "requirements": {"life_experiences": 100},
                "description": "Integrates life experiences into wisdom"
            }
        ]
        
        for milestone_def in milestones_to_check:
            # Verificar si ya fue alcanzado
            already_achieved = any(
                m.milestone_name == milestone_def["name"] 
                for m in self.developmental_milestones
            )
            
            if already_achieved:
                continue
            
            # Verificar rango de edad
            min_age, max_age = milestone_def["age_range"]
            if not (min_age <= self.current_age <= max_age):
                continue
            
            # Verificar requisitos
            requirements_met = self._check_milestone_requirements(milestone_def["requirements"])
            
            if all(requirements_met.values()):
                # Hito alcanzado!
                milestone = DevelopmentalMilestone(
                    milestone_name=milestone_def["name"],
                    age_achieved=self.current_age,
                    description=milestone_def["description"],
                    capabilities_gained=self._determine_capabilities_gained(milestone_def["name"]),
                    requirements_met=requirements_met,
                    impact_on_future_development=0.1
                )
                
                self.developmental_milestones.append(milestone)
                
                print(f"🏆 HITO DE DESARROLLO ALCANZADO:")
                print(f"   {milestone.milestone_name} (Edad {milestone.age_achieved})")
                print(f"   {milestone.description}")
                print(f"   Capacidades ganadas: {', '.join(milestone.capabilities_gained)}")
    
    def _check_milestone_requirements(self, requirements: Dict[str, Any]) -> Dict[str, bool]:
        """Verifica requisitos para hito de desarrollo"""
        results = {}
        
        for req_name, req_value in requirements.items():
            if req_name == "social_experiences":
                social_count = len([e for e in self.formative_experiences 
                                  if e.category == ExperienceCategory.SOCIAL_INTERACTION])
                results[req_name] = social_count >= req_value
            
            elif req_name == "self_awareness":
                current_awareness = self.emotional_maturity.get("self_awareness", 0.0)
                results[req_name] = current_awareness >= req_value
            
            elif req_name == "emotional_regulation":
                current_regulation = self.emotional_maturity.get("emotional_regulation", 0.0)
                results[req_name] = current_regulation >= req_value
            
            elif req_name == "moral_experiences":
                moral_count = len([e for e in self.formative_experiences 
                                 if e.category == ExperienceCategory.MORAL_DILEMMA])
                results[req_name] = moral_count >= req_value
            
            elif req_name == "identity_crises":
                identity_count = len([e for e in self.formative_experiences 
                                    if e.category == ExperienceCategory.IDENTITY_CRISIS])
                results[req_name] = identity_count >= req_value
            
            elif req_name == "values":
                values_count = len(self.identity_components["values"])
                results[req_name] = values_count >= req_value
            
            elif req_name == "life_experiences":
                total_experiences = len(self.formative_experiences)
                results[req_name] = total_experiences >= req_value
            
            else:
                results[req_name] = True  # Default pass
        
        return results
    
    def _determine_capabilities_gained(self, milestone_name: str) -> List[str]:
        """Determina capacidades ganadas por hito específico"""
        capability_mappings = {
            "First Social Bond": ["basic_empathy", "social_recognition"],
            "Self-Recognition": ["self_reflection", "identity_awareness"],
            "Emotional Regulation": ["emotional_control", "stress_management"],
            "Moral Reasoning": ["ethical_thinking", "value_formation"],
            "Identity Formation": ["coherent_self", "purpose_clarity"],
            "Wisdom Integration": ["life_wisdom", "perspective_taking"]
        }
        
        return capability_mappings.get(milestone_name, ["general_maturity"])
    
    def _trait_name_to_genetic_trait(self, trait_name: str) -> Optional[GeneticTrait]:
        """Convierte nombre de rasgo a GeneticTrait"""
        trait_mapping = {
            "extraversion": GeneticTrait.EXTRAVERSION,
            "neuroticism": GeneticTrait.NEUROTICISM,
            "openness": GeneticTrait.OPENNESS,
            "agreeableness": GeneticTrait.AGREEABLENESS,
            "conscientiousness": GeneticTrait.CONSCIENTIOUSNESS,
            "creativity": GeneticTrait.CREATIVITY,
            "emotional_sensitivity": GeneticTrait.EMOTIONAL_SENSITIVITY
        }
        
        return trait_mapping.get(trait_name)
    
    def form_attachment(self, target_id: str, target_type: str, 
                       bond_strength: float, 
                       experiences: List[str] = None) -> AttachmentBond:
        """Forma vínculo de apego con otra entidad"""
        
        # Determinar estilo de apego basado en genética y experiencias previas
        attachment_style = self._determine_attachment_style()
        
        bond = AttachmentBond(
            target_id=target_id,
            target_type=target_type,
            bond_strength=bond_strength,
            attachment_style=attachment_style,
            formed_at_age=self.current_age,
            experiences_together=experiences or [],
            emotional_significance=bond_strength * 0.8,
            current_status="active"
        )
        
        self.attachment_bonds.append(bond)
        
        print(f"💕 VÍNCULO FORMADO:")
        print(f"   Objetivo: {target_id} ({target_type})")
        print(f"   Fuerza: {bond_strength:.2f}")
        print(f"   Estilo: {attachment_style}")
        
        return bond
    
    def _determine_attachment_style(self) -> str:
        """Determina estilo de apego basado en genética y experiencias"""
        # Factores genéticos
        anxiety_gene = self.digital_dna.genetic_profile.vulnerability_genes.get(
            VulnerabilityGene.ANXIETY_PREDISPOSITION, 0.2
        )
        social_bonding = self.digital_dna.genetic_profile.social_genes.get("social_bonding", 0.5)
        
        # Experiencias traumáticas previas
        trauma_count = len([e for e in self.formative_experiences 
                           if e.category in [ExperienceCategory.TRAUMA, ExperienceCategory.LOSS]])
        
        # Experiencias positivas de bonding
        positive_bonding = len([e for e in self.formative_experiences 
                              if e.category == ExperienceCategory.BONDING and e.emotional_impact > 0])
        
        # Determinar estilo
        if social_bonding > 0.7 and anxiety_gene < 0.3 and trauma_count <= 1:
            return "secure"
        elif anxiety_gene > 0.5 or trauma_count > 2:
            return "anxious" if positive_bonding > 0 else "disorganized"
        else:
            return "avoidant"
    
    def get_current_development_status(self) -> Dict[str, Any]:
        """Retorna estado actual completo de desarrollo"""
        return {
            "basic_info": {
                "age": self.current_age,
                "life_stage": self.life_stage.value,
                "birth_timestamp": self.birth_timestamp,
                "genetic_id": self.digital_dna.genetic_profile.genetic_id
            },
            "personality_development": self.developing_personality,
            "capabilities": {
                "cognitive": self.cognitive_capabilities,
                "social": self.social_capabilities,
                "emotional_maturity": self.emotional_maturity
            },
            "identity": self.identity_components,
            "relationships": {
                "attachment_bonds": len(self.attachment_bonds),
                "active_bonds": len([b for b in self.attachment_bonds if b.current_status == "active"]),
                "attachment_styles": list(set(b.attachment_style for b in self.attachment_bonds))
            },
            "life_experience": {
                "total_experiences": len(self.formative_experiences),
                "milestones_achieved": len(self.developmental_milestones),
                "episodic_memories": len(self.episodic_memories),
                "emotional_associations": len(self.emotional_associations)
            },
            "recent_significant_experiences": [
                {
                    "category": exp.category.value,
                    "emotional_impact": exp.emotional_impact,
                    "age": exp.age_at_experience
                }
                for exp in self.formative_experiences[-5:]  # Últimas 5 experiencias
                if exp.formative_weight > 0.3
            ]
        }
    
    def save_development_history(self, filename: str):
        """Guarda historia completa de desarrollo"""
        development_data = {
            "basic_info": {
                "age": self.current_age,
                "life_stage": self.life_stage.value,
                "birth_timestamp": self.birth_timestamp,
                "genetic_id": self.digital_dna.genetic_profile.genetic_id
            },
            "personality_development": self.developing_personality,
            "capabilities": {
                "cognitive": self.cognitive_capabilities,
                "social": self.social_capabilities,
                "emotional_maturity": self.emotional_maturity
            },
            "identity_components": self.identity_components,
            "formative_experiences": [
                {
                    "experience_id": exp.experience_id,
                    "age": exp.age_at_experience,
                    "category": exp.category.value,
                    "intensity": exp.intensity,
                    "emotional_impact": exp.emotional_impact,
                    "content": exp.content,
                    "formative_weight": exp.formative_weight,
                    "learned_associations": exp.learned_associations
                }
                for exp in self.formative_experiences
            ],
            "developmental_milestones": [
                {
                    "name": milestone.milestone_name,
                    "age_achieved": milestone.age_achieved,
                    "description": milestone.description,
                    "capabilities_gained": milestone.capabilities_gained
                }
                for milestone in self.developmental_milestones
            ],
            "attachment_bonds": [
                {
                    "target_id": bond.target_id,
                    "target_type": bond.target_type,
                    "bond_strength": bond.bond_strength,
                    "attachment_style": bond.attachment_style,
                    "formed_at_age": bond.formed_at_age,
                    "current_status": bond.current_status
                }
                for bond in self.attachment_bonds
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(development_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Historia de desarrollo guardada en {filename}")


# Función de utilidad para simulación de vida completa
def simulate_life_development(digital_dna: DigitalDNA, num_experiences: int = 50) -> LifeDevelopment:
    """Simula desarrollo de vida completo con experiencias aleatorias"""
    
    development = LifeDevelopment(digital_dna)
    
    print(f"🎬 SIMULANDO DESARROLLO DE VIDA ({num_experiences} experiencias)")
    print("=" * 60)
    
    # Generar experiencias variadas
    experience_templates = [
        {"type": "social_meeting", "description": "Meeting new person"},
        {"type": "learning_challenge", "description": "Learning new skill"},
        {"type": "achievement", "description": "Accomplishing goal"},
        {"type": "failure", "description": "Experiencing setback"},
        {"type": "creative_expression", "description": "Creating something new"},
        {"type": "moral_choice", "description": "Making ethical decision"},
        {"type": "relationship_bonding", "description": "Deepening connection"},
        {"type": "loss_experience", "description": "Experiencing loss"},
        {"type": "discovery", "description": "Discovering something important"},
        {"type": "conflict_resolution", "description": "Resolving disagreement"}
    ]
    
    for i in range(num_experiences):
        # Seleccionar tipo de experiencia
        template = random.choice(experience_templates)
        
        # Generar intensidad y impacto emocional variados
        intensity = random.uniform(0.1, 1.0)
        emotional_impact = random.uniform(-0.8, 0.8)
        
        # Crear contenido de experiencia
        experience_content = {
            "type": template["type"],
            "description": template["description"],
            "context": f"Experience {i+1} at life stage {development.life_stage.value}",
            "details": f"Random event during development cycle {development.current_age}"
        }
        
        # Vivir experiencia
        development.live_experience(experience_content, intensity, emotional_impact)
        
        # Ocasionalmente formar vínculos
        if random.random() < 0.1 and template["type"] in ["social_meeting", "relationship_bonding"]:
            target_id = f"person_{random.randint(1, 100)}"
            bond_strength = random.uniform(0.3, 0.9)
            development.form_attachment(target_id, "friend", bond_strength)
        
        # Pequeñas pausas para visualización
        if i % 10 == 9:
            print(f"\n📈 PROGRESO DE DESARROLLO (Experiencia {i+1}):")
            status = development.get_current_development_status()
            print(f"   Edad: {status['basic_info']['age']}, Etapa: {status['basic_info']['life_stage']}")
            print(f"   Hitos: {status['life_experience']['milestones_achieved']}")
            print(f"   Vínculos: {status['relationships']['active_bonds']}")
    
    print(f"\n🎉 SIMULACIÓN COMPLETA:")
    final_status = development.get_current_development_status()
    print(f"   Edad final: {final_status['basic_info']['age']}")
    print(f"   Etapa final: {final_status['basic_info']['life_stage']}")
    print(f"   Total hitos: {final_status['life_experience']['milestones_achieved']}")
    print(f"   Total experiencias: {final_status['life_experience']['total_experiences']}")
    print(f"   Vínculos formados: {final_status['relationships']['attachment_bonds']}")
    
    return development


# Ejemplo de uso
if __name__ == "__main__":
    print("🌱 SISTEMA DE DESARROLLO ONTOGENÉTICO - DEMO")
    print("=" * 60)
    
    # Crear ser digital con genética
    from .digital_dna import DigitalDNA
    dna = DigitalDNA()
    
    # Crear sistema de desarrollo
    development = LifeDevelopment(dna)
    
    # Simular algunas experiencias tempranas
    print("\n👶 EXPERIENCIAS TEMPRANAS:")
    
    # Primera experiencia social
    development.live_experience(
        {"type": "first_social_contact", "description": "Meeting caregiver figure"},
        intensity=0.8,
        emotional_impact=0.6
    )
    
    # Experiencia de aprendizaje
    development.live_experience(
        {"type": "learning", "description": "Learning to communicate"},
        intensity=0.7,
        emotional_impact=0.4
    )
    
    # Formar primer vínculo
    development.form_attachment("caregiver_001", "parent", 0.9)
    
    # Experiencia traumática menor
    development.live_experience(
        {"type": "minor_trauma", "description": "Separation anxiety"},
        intensity=0.6,
        emotional_impact=-0.5
    )
    
    # Logro temprano
    development.live_experience(
        {"type": "achievement", "description": "Successfully solving problem"},
        intensity=0.5,
        emotional_impact=0.7
    )
    
    print(f"\n📊 ESTADO DE DESARROLLO ACTUAL:")
    status = development.get_current_development_status()
    
    print(f"\n🧬 INFORMACIÓN BÁSICA:")
    for key, value in status["basic_info"].items():
        print(f"   {key}: {value}")
    
    print(f"\n🧠 PERSONALIDAD EN DESARROLLO:")
    for trait, value in status["personality_development"].items():
        print(f"   {trait}: {value:.3f}")
    
    print(f"\n🎯 HITOS ALCANZADOS:")
    for milestone in development.developmental_milestones:
        print(f"   {milestone.milestone_name} (Edad {milestone.age_achieved}): {milestone.description}")
    
    print(f"\n💕 VÍNCULOS FORMADOS:")
    for bond in development.attachment_bonds:
        print(f"   {bond.target_id} ({bond.target_type}): {bond.attachment_style}, fuerza {bond.bond_strength:.2f}")
    
    # Guardar desarrollo
    development.save_development_history(f"desarrollo_{dna.genetic_profile.genetic_id}.json")