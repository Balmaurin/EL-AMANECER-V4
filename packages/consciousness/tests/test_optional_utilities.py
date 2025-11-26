import unittest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "packages" / "consciousness" / "src"))

from conciencia.modulos.linguistic_metacognition_system import (
    get_linguistic_metacognition_engine,
    LinguisticIntent,
    ResponseStyle
)

class TestOptionalUtilities(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("Initializing Linguistic Metacognition Engine for optional tests...")
        cls.engine = get_linguistic_metacognition_engine()

    def test_multi_intent_division(self):
        # Test complex query splitting
        text = "Explícame qué es la IA y luego dime cómo te sientes al respecto"
        parts = self.engine.divide_multi_intent_query(text)
        
        self.assertGreater(len(parts), 1)
        print(f"✅ Multi-intent split passed: '{text}' -> {parts}")
        
        # Verify analysis handles it (should pick the most complex intent)
        # "cómo te sientes" (emotional) > "qué es IA" (factual)
        analysis, _ = self.engine.process_linguistic_input(text)
        self.assertEqual(analysis.intent, LinguisticIntent.EMOTIONAL_PERSONAL)
        print(f"✅ Multi-intent analysis passed: Dominant intent is {analysis.intent.value}")

    @patch('conciencia.modulos.linguistic_metacognition_system.get_user_profile_store', create=True)
    @patch('conciencia.modulos.linguistic_metacognition_system.user_profiles_available', True)
    def test_user_profile_preference(self, mock_get_store):
        # Mock user profile with preference
        mock_store = MagicMock()
        mock_store.get_profile.return_value = {'preferences': {'preferred_style': 'technical'}}
        mock_get_store.return_value = mock_store
        
        # Test query that would normally be casual
        text = "Hola, ¿cómo estás?"
        
        # Force context with user_id
        context = {'user_id': 'test_user'}
        
        # The engine should respect the 'technical' preference override
        # Note: "Hola" is usually SOCIAL_RELATIONAL -> CASUAL
        # But preference 'technical' should shift it towards TECHNICAL or at least affect it
        
        # Let's check the logic in select_style directly or via process
        # In our implementation:
        # if pref == 'technical' and base_style == CASUAL -> base_style = TECHNICAL
        
        analysis, style = self.engine.process_linguistic_input(text, context)
        
        self.assertEqual(style.primary_style, ResponseStyle.TECHNICAL)
        print(f"✅ User profile preference passed: '{text}' -> {style.primary_style.value} (User preferred technical)")

if __name__ == '__main__':
    unittest.main()
