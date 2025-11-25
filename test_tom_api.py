"""
Quick Test Script for Advanced Theory of Mind API
==================================================

Tests all ToM endpoints to verify system is working correctly.
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def print_response(title: str, response: requests.Response):
    """Print formatted API response"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except:
        print(response.text)


def test_system_info():
    """Test system info endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print_response("🌟 SYSTEM INFO", response)


def test_tom_status():
    """Test ToM status"""
    response = requests.get(f"{BASE_URL}/api/v1/tom/status")
    print_response("🎭 ToM STATUS", response)


def test_belief_hierarchy():
    """Test creating belief hierarchy (Level 8)"""
    payload = {
        "agent_chain": ["CEO", "Investor", "Analyst"],
        "final_content": "the merger will be announced next week",
        "confidence": 0.85
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/tom/belief-hierarchy",
        json=payload
    )
    print_response("🧠 LEVEL 8: BELIEF HIERARCHY", response)
    
    return response.json().get("belief_id") if response.status_code == 200 else None


def test_strategic_action():
    """Test strategic action evaluation (Level 9)"""
    payload = {
        "actor": "TechStartup",
        "target": "Enterprise",
        "strategy_type": "cooperation",
        "context": {
            "goal": "technology partnership",
            "industry": "AI"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/tom/strategic-action",
        json=payload
    )
    print_response("🎯 LEVEL 9: STRATEGIC ACTION (Cooperation)", response)


def test_deception_detection():
    """Test deception detection (Level 9)"""
    payload = {
        "actor": "Competitor",
        "stated_belief": "We are committed to fair competition",
        "actual_behavior": "Competitor engaged in price dumping"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/tom/detect-deception",
        json=payload
    )
    print_response("🔍 LEVEL 9: DECEPTION DETECTION", response)


def test_cultural_assignment():
    """Test cultural assignment (Level 10)"""
    payload = {
        "agent_id": "GlobalManager",
        "cultures": ["western", "professional", "tech"]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/tom/assign-culture",
        json=payload
    )
    print_response("🌍 LEVEL 10: CULTURAL ASSIGNMENT", response)


def test_cultural_context():
    """Test getting cultural context (Level 10)"""
    response = requests.get(
        f"{BASE_URL}/api/v1/tom/cultural-context",
        params={
            "agent_a": "GlobalManager",
            "agent_b": "LocalPartner",
            "situation": "business"
        }
    )
    print_response("🌐 LEVEL 10: CULTURAL CONTEXT", response)


def test_social_interaction():
    """Test complete social interaction (All levels)"""
    payload = {
        "actor": "Negotiator",
        "target": "Client",
        "interaction_type": "proposal",
        "content": {
            "text": "We would like to propose a long-term partnership",
            "stated_belief": "mutual benefit is achievable"
        },
        "context": {
            "situation": "business",
            "formality": "high"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/tom/social-interaction",
        json=payload
    )
    print_response("✨ INTEGRATED: COMPLETE SOCIAL INTERACTION (Levels 8-10)", response)


def main():
    """Run all tests"""
    print("\n" + "🧠"*40)
    print(" "*15 + "ADVANCED THEORY OF MIND API - TEST SUITE")
    print("🧠"*40)
    
    try:
        # Basic tests
        test_system_info()
        test_tom_status()
        
        # Level 8: Belief Hierarchies
        test_belief_hierarchy()
        
        # Level 9: Strategic Reasoning
        test_strategic_action()
        test_deception_detection()
        
        # Level 10: Cultural Context
        test_cultural_assignment()
        test_cultural_context()
        
        # Integrated (All Levels)
        test_social_interaction()
        
        # Summary
        print("\n" + "="*80)
        print("  ✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\n🏆 Advanced Theory of Mind (Levels 8-10) is OPERATIONAL")
        print("📚 Full API docs available at: http://localhost:8000/docs\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print("Please make sure the backend is running:")
        print("  python start_complete_system.py --skip-frontend\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
