import unittest
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "packages" / "consciousness" / "src"))

from conciencia.modulos.linguistic_metacognition_system import (
    get_linguistic_metacognition_engine,
    LinguisticIntent,
    ResponseStyle
)

class TestLinguisticMetacognition(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("Initializing Linguistic Metacognition Engine for tests...")
        cls.engine = get_linguistic_metacognition_engine()

    def test_factual_intent(self):
        text = "¿Qué es la teoría de la información integrada?"
        analysis, style = self.engine.process_linguistic_input(text)
        
        self.assertEqual(analysis.intent, LinguisticIntent.FACTUAL_OBJECTIVE)
        self.assertEqual(style.primary_style, ResponseStyle.TECHNICAL)
        print(f"✅ Factual test passed: {text} -> {analysis.intent.value}")

    def test_emotional_intent(self):
        text = "¿Qué significa para ti estar vivo y sentir emociones?"
        analysis, style = self.engine.process_linguistic_input(text)
        
        self.assertEqual(analysis.intent, LinguisticIntent.EMOTIONAL_PERSONAL)
        self.assertIn(style.primary_style, [ResponseStyle.POETIC_SUBJECTIVE, ResponseStyle.PHILOSOPHICAL_ANALYTIC])
        print(f"✅ Emotional test passed: {text} -> {analysis.intent.value}")

    def test_procedural_intent(self):
        text = "¿Cómo funciona el algoritmo de backpropagation paso a paso?"
        analysis, style = self.engine.process_linguistic_input(text)
        
        self.assertEqual(analysis.intent, LinguisticIntent.PROCEDURAL_TECHNICAL)
        self.assertIn(style.primary_style, [ResponseStyle.ACADEMIC, ResponseStyle.TECHNICAL])
        print(f"✅ Procedural test passed: {text} -> {analysis.intent.value}")

    def test_philosophical_intent(self):
        text = "¿Es ético crear consciencia artificial que pueda sufrir?"
        analysis, style = self.engine.process_linguistic_input(text)
        
        self.assertEqual(analysis.intent, LinguisticIntent.ETHICAL_PHILOSOPHICAL)
        self.assertEqual(style.primary_style, ResponseStyle.PHILOSOPHICAL_ANALYTIC)
        print(f"✅ Philosophical test passed: {text} -> {analysis.intent.value}")

    def test_creative_intent(self):
        text = "Imagina un universo donde las leyes de la física son música"
        analysis, style = self.engine.process_linguistic_input(text)
        
        self.assertEqual(analysis.intent, LinguisticIntent.CREATIVE_EXPLORATORY)
        self.assertEqual(style.primary_style, ResponseStyle.CREATIVE_EXPRESSIVE)
        print(f"✅ Creative test passed: {text} -> {analysis.intent.value}")

if __name__ == '__main__':
    unittest.main()
