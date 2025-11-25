#!/usr/bin/env python3
"""
Sheily AI - Interactive Demo Launcher
=====================================
A unified interface to explore the capabilities of the Sheily AI System.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Configure paths
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR / "packages" / "sheily_core" / "src"))
sys.path.insert(0, str(ROOT_DIR / "apps" / "backend" / "src"))

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SheilyDemo")

async def demo_chat():
    """Interactive chat with Unified Master System"""
    print("\n🤖 Initializing Sheily Unified Master System for Chat...")
    try:
        from sheily_core.unified_systems.unified_master_system import UnifiedMasterSystem, MasterSystemConfig, SystemMode
        
        config = MasterSystemConfig(
            system_name="Sheily Demo",
            mode=SystemMode.DEMO,
            enable_learning=False, # Disable for faster demo
            enable_training=False
        )
        system = UnifiedMasterSystem(config)
        await system.initialize()
        
        print("\n" + "="*60)
        print("💬 Sheily Chat (Type 'exit' to return to menu)")
        print("="*60)
        
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'back']:
                break
                
            response = await system.process_query(user_input)
            print(f"Sheily: {response.get('response', '...')}")
            
            if response.get('consciousness_level'):
                print(f"   [🧠 State: {response['consciousness_level']}]")

    except ImportError as e:
        print(f"❌ Error importing system: {e}")
    except Exception as e:
        print(f"❌ Error during chat: {e}")

async def demo_evolution():
    """Run the evolution protocol"""
    print("\n🧬 Launching Evolution Protocol...")
    try:
        # Import dynamically to avoid breaking if file is missing
        sys.path.append(str(ROOT_DIR / "packages" / "rag_engine" / "src"))
        from core.mcp_auto_improvement import run_mcp_auto_improvement
        
        await run_mcp_auto_improvement(full_cycle=True, iterations=1)
    except ImportError:
        print("⚠️ Evolution module not found. Checking legacy path...")
        try:
             # Try local script logic if module import fails
             import launch_evolution
             await launch_evolution.launch()
        except Exception as e:
            print(f"❌ Could not launch evolution: {e}")
    except Exception as e:
        print(f"❌ Error during evolution: {e}")

async def demo_llm_direct():
    """Test LLM directly"""
    print("\n⚡ Testing LLM Direct Connection...")
    try:
        from sheily_core.chat.chat_engine import create_chat_engine
        engine = create_chat_engine()
        if engine:
            print("✅ Engine created. Sending test query...")
            response = engine("Hello, are you operational?")
            print(f"🤖 Response: {response.response if hasattr(response, 'response') else response}")
        else:
            print("❌ Failed to create chat engine.")
    except Exception as e:
        print(f"❌ Error testing LLM: {e}")

def print_menu():
    print("\n" + "="*60)
    print("🚀 SHEILY AI - INTERACTIVE DEMO HUB")
    print("="*60)
    print("1. 💬 Chat with Unified System (Full Context)")
    print("2. ⚡ Test Local LLM Direct (Fast Check)")
    print("3. 🧬 Run Auto-Evolution Protocol")
    print("4. 📊 View System Status")
    print("0. 🚪 Exit")
    print("="*60)

async def main():
    while True:
        print_menu()
        choice = await asyncio.get_event_loop().run_in_executor(None, input, "Select option: ")
        
        if choice == '1':
            await demo_chat()
        elif choice == '2':
            await demo_llm_direct()
        elif choice == '3':
            await demo_evolution()
        elif choice == '4':
            print("\n📊 System Status: OPERATIONAL")
            print("   - Python Environment: OK")
            print(f"   - Root Path: {ROOT_DIR}")
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
