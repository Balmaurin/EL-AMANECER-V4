"""
Simple Emotion Detector for Chat Integration
Wraps the advanced human_emotions_system for easy use in chat endpoint
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add consciousness package to path
consciousness_path = Path(__file__).parent.parent.parent.parent.parent.parent / "packages" / "consciousness" / "src"
if str(consciousness_path) not in sys.path:
    sys.path.append(str(consciousness_path))

try:
    from conciencia.modulos.human_emotions_system import (
        HumanEmotionalSystem,
        BasicEmotions,
        SocialEmotions,
        ComplexEmotions
    )
    EMOTIONS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Could not import emotion system: {e}")
    EMOTIONS_AVAILABLE = False


class SimpleEmotionDetector:
    """Simple wrapper for emotion detection in chat"""
    
    def __init__(self):
        if EMOTIONS_AVAILABLE:
            self.emotion_system = HumanEmotionalSystem(num_circuits=35)
        else:
            self.emotion_system = None
    
    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """
        Detect emotion from user text
        
        Returns dict with:
        - emotion: detected emotion name
        - intensity: 0.0-1.0
        - valence: -1.0 to 1.0 (negative to positive)
        - category: basic/social/complex
        """
        if not self.emotion_system:
            return {
                "emotion": "neutral",
                "intensity": 0.5,
                "valence": 0.0,
                "category": "basic",
                "confidence": 0.0
            }
        
        # Simple keyword-based emotion detection
        text_lower = text.lower()
        
        # Positive emotions
        if any(word in text_lower for word in ["feliz", "alegre", "contento", "genial", "excelente", "bien"]):
            return {
                "emotion": BasicEmotions.ALEGRIA,
                "intensity": 0.7,
                "valence": 0.8,
                "category": "basic",
                "confidence": 0.75
            }
        
        # Sad emotions
        elif any(word in text_lower for word in ["triste", "mal", "terrible", "horrible", "deprimido"]):
            return {
                "emotion": BasicEmotions.TRISTEZA,
                "intensity": 0.7,
                "valence": -0.8,
                "category": "basic",
                "confidence": 0.75
            }
        
        # Fear/anxiety
        elif any(word in text_lower for word in ["miedo", "asustado", "nervioso", "preocupado", "ansiedad"]):
            return {
                "emotion": BasicEmotions.MIEDO,
                "intensity": 0.6,
                "valence": -0.6,
                "category": "basic",
                "confidence": 0.7
            }
        
        # Anger
        elif any(word in text_lower for word in ["enojado", "furioso", "molesto", "irritado", "enojo"]):
            return {
                "emotion": BasicEmotions.ENOJO,
                "intensity": 0.7,
                "valence": -0.7,
                "category": "basic",
                "confidence": 0.75
            }
        
        # Love/affection
        elif any(word in text_lower for word in ["amor", "quiero", "amo", "cariño"]):
            return {
                "emotion": SocialEmotions.AMOR,
                "intensity": 0.8,
                "valence": 0.9,
                "category": "social",
                "confidence": 0.8
            }
        
        # Frustration
        elif any(word in text_lower for word in ["frustrado", "frustración", "no funciona", "no puedo"]):
            return {
                "emotion": ComplexEmotions.FRUSTRACION,
                "intensity": 0.6,
                "valence": -0.5,
                "category": "complex",
                "confidence": 0.7
            }
        
        # Gratitude
        elif any(word in text_lower for word in ["gracias", "agradecido", "agradezco"]):
            return {
                "emotion": SocialEmotions.GRATITUD,
                "intensity": 0.7,
                "valence": 0.7,
                "category": "social",
                "confidence": 0.8
            }
        
        # Questions/curiosity
        elif "?" in text or any(word in text_lower for word in ["como", "qué", "por qué", "cuál"]):
            return {
                "emotion": ComplexEmotions.CURIOSIDAD,
                "intensity": 0.5,
                "valence": 0.3,
                "category": "complex",
                "confidence": 0.6
            }
        
        # Default neutral
        else:
            return {
                "emotion": "neutral",
                "intensity": 0.3,
                "valence": 0.0,
                "category": "neutral",
                "confidence": 0.5
            }


# Singleton instance
_emotion_detector = None

def get_emotion_detector() -> SimpleEmotionDetector:
    """Get or create emotion detector singleton"""
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = SimpleEmotionDetector()
    return _emotion_detector
