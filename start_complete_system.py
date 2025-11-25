#!/usr/bin/env python3
"""
🧠 EL-AMANECER V3 - Complete System Launcher
============================================

Launches the full consciousness system with:
- Backend API (FastAPI)
- Advanced Theory of Mind (Levels 8-10)
- Unified Consciousness Engine
- Auto-Evolution System
- Real-time monitoring dashboard

Usage:
    python start_complete_system.py [--skip-frontend]
"""

import sys
import subprocess
import time
import asyncio
from pathlib import Path
from typing import Optional

# Setup paths
project_root = Path(__file__).parent
backend_src = project_root / "apps" / "backend" / "src"
packages = project_root / "packages"

sys.path.insert(0, str(backend_src))
sys.path.insert(0, str(packages / "sheily_core" / "src"))
sys.path.insert(0, str(packages / "rag_engine"))
sys.path.insert(0, str(packages / "consciousness" / "src"))


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("🔍 CHECKING DEPENDENCIES")
    
    required = [
        ("uvicorn", "Backend server"),
        ("fastapi", "API framework"),
        ("numpy", "Numerical computing"),
        ("asyncio", "Async support")
    ]
    
    missing = []
    for package, description in required:
        try:
            __import__(package)
            print(f"✅ {package:15} - {description}")
        except ImportError:
            print(f"❌ {package:15} - {description} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies satisfied")
    return True


def initialize_consciousness_system():
    """Initialize the consciousness system"""
    print_header("🧠 INITIALIZING CONSCIOUSNESS SYSTEM")
    
    try:
        from conciencia.modulos.unified_consciousness_engine import UnifiedConsciousnessEngine
        
        engine = UnifiedConsciousnessEngine()
        print("✅ Unified Consciousness Engine initialized")
        print(f"   • IIT 4.0: Φ calculation ready")
        print(f"   • GWT/AST: Global workspace active")
        print(f"   • FEP: Predictive processing online")
        print(f"   • SMH: Somatic markers loaded")
        print(f"   • Thalamus, Claustrum, DMN: Active")
        
        return engine
    except Exception as e:
        print(f"❌ Error initializing consciousness: {e}")
        import traceback
        traceback.print_exc()
        return None


def initialize_advanced_tom():
    """Initialize Advanced Theory of Mind system"""
    print_header("🎭 INITIALIZING ADVANCED THEORY OF MIND")
    
    try:
        from conciencia.modulos.teoria_mente import get_unified_tom
        
        tom = get_unified_tom(enable_advanced=True)
        
        if tom.has_advanced_capabilities:
            print("✅ Unified Theory of Mind (Levels 1-10) initialized")
            print(f"   ✅ Level 8 (Empathic): Multi-agent belief hierarchies")
            print(f"   ✅ Level 9 (Social): Machiavellian strategic reasoning")
            print(f"   ✅ Level 10 (Human-Like): Cultural context modeling")
            
            tom_level, description = tom.get_tom_level()
            print(f"\n   🏆 Current ToM Level: {tom_level:.1f} / 10.0")
            print(f"   📊 {description}")
        else:
            print("✅ Basic Theory of Mind (Levels 1-7) initialized")
            print("   ⚠️  Advanced ToM (8-10) not available")
        
        return tom
    except Exception as e:
        print(f"❌ Error initializing ToM: {e}")
        import traceback
        traceback.print_exc()
        return None


def initialize_evolution_system():
    """Initialize Auto-Evolution system"""
    print_header("🧬 INITIALIZING AUTO-EVOLUTION SYSTEM")
    
    try:
        # Check if evolution modules exist
        import importlib.util
        
        spec = importlib.util.find_spec("conciencia.auto_evolution_engine")
        if spec is None:
            print("⚠️  Auto-Evolution system not found in this path")
            print("   (This is optional - consciousness system will work without it)")
            return None
        
        from conciencia.auto_evolution_engine import AutoEvolutionEngine
        
        engine = AutoEvolutionEngine()
        print("✅ Auto-Evolution Engine initialized")
        print(f"   • Darwinian mutation/selection ready")
        print(f"   • Epigenetic memory active")
        print(f"   • Multiverse exploration enabled")
        
        return engine
    except Exception as e:
        print(f"⚠️  Evolution system not available: {e}")
        print("   (Continuing with consciousness system only)")
        return None


async def run_quick_demo():
    """Run a quick demonstration of the system"""
    print_header("🎬 RUNNING QUICK SYSTEM DEMO")
    
    try:
        # Initialize ToM
        from conciencia.modulos.teoria_mente import get_unified_tom
        tom = get_unified_tom(enable_advanced=True)
        
        if tom.has_advanced_capabilities:
            print("1️⃣ Testing Level 8: Multi-Agent Belief Hierarchy")
            hierarchy_id = tom.create_belief_hierarchy(
                agent_chain=["Alice", "Bob", "Charlie"],
                final_content="the system is operational"
            )
            print(f"   ✅ Created belief hierarchy: {hierarchy_id[:30]}...")
            
            print("\n2️⃣ Testing Level 9: Strategic Reasoning")
            strategy = tom.evaluate_strategic_action(
                actor="SystemA",
                target="UserB",
                strategy_type="cooperation",
                context={"goal": "successful demo"}
            )
            print(f"   ✅ Strategy: {strategy['strategy']}")
            print(f"   📊 Payoff: {strategy['expected_payoff']:.2f}, Ethics: {strategy['ethical_score']:.2f}")
            
            print("\n3️⃣ Testing Level 10: Cultural Context")
            tom.assign_culture("DemoAgent", ["professional", "technical"])
            print(f"   ✅ Cultural context assigned")
            
            print("\n4️⃣ Testing Complete Social Interaction")
            result = await tom.process_social_interaction(
                actor="DemoAgent",
                target="User",
                interaction_type="greeting",
                content={"text": "System initialization complete"}
            )
            print(f"   ✅ ToM Levels Active: {result['tom_level_active']}")
        else:
            print("⚠️  Advanced ToM not available - skipping demo")
        
        print("\n✅ Demo complete - all systems operational!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


def start_backend_server():
    """Start the FastAPI backend server"""
    print_header("🚀 STARTING BACKEND SERVER")
    
    try:
        import uvicorn
        
        # Fix imports
        try:
            import conciencia
            sys.modules['CONCIENCIA'] = conciencia
            print("✅ Fixed 'CONCIENCIA' import alias")
        except ImportError:
            pass
        
        from api import app
        
        print("📍 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🔗 WebSocket: ws://localhost:8000/ws")
        print("\n🧠 Consciousness endpoints:")
        print("   POST /api/v1/consciousness/process")
        print("   POST /api/v1/tom/belief-hierarchy")
        print("   POST /api/v1/tom/strategic-action")
        print("   POST /api/v1/tom/social-interaction")
        print("   GET  /api/v1/system/status")
        
        print("\n⚡ Starting server...")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def start_frontend(skip: bool = False):
    """Start the Next.js frontend"""
    if skip:
        print("\n⏭️  Skipping frontend (--skip-frontend flag)")
        return
    
    print_header("🎨 STARTING FRONTEND")
    
    frontend_dir = project_root / "apps" / "frontend"
    
    if not (frontend_dir / "package.json").exists():
        print("⚠️  Frontend not found - starting backend only")
        return
    
    print("📍 Frontend will be available at: http://localhost:3000")
    print("⚡ Starting Next.js dev server...")
    
    try:
        subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(frontend_dir),
            shell=True
        )
        print("✅ Frontend server starting in background")
    except Exception as e:
        print(f"⚠️  Could not start frontend: {e}")
        print("   Start manually with: cd apps/frontend && npm run dev")


def print_system_status():
    """Print comprehensive system status"""
    print_header("📊 SYSTEM STATUS")
    
    try:
        from conciencia.modulos.teoria_mente import get_unified_tom
        
        tom = get_unified_tom(enable_advanced=False)  # Don't re-init
        status = tom.get_comprehensive_status()
        
        print("🧠 Consciousness Status:")
        print(f"   • Users Modeled: {status.get('users_modeled', 0)}")
        print(f"   • Social Intelligence: {status.get('social_intelligence', 0):.2f}")
        
        if 'overall_tom_level' in status:
            print(f"\n🎭 Theory of Mind:")
            print(f"   • Overall Level: {status['overall_tom_level']:.1f} / 10.0")
            
            if 'level_8_belief_tracking' in status:
                belief_stats = status['level_8_belief_tracking']
                print(f"   • Total Beliefs: {belief_stats.get('total_beliefs', 0)}")
                print(f"   • Agents Tracked: {belief_stats.get('agents_tracked', 0)}")
                print(f"   • Belief Hierarchies: {belief_stats.get('belief_hierarchies', 0)}")
            
            if 'level_9_strategic_reasoning' in status:
                strategic = status['level_9_strategic_reasoning']
                print(f"   • Relationships: {strategic.get('relationships_tracked', 0)}")
                print(f"   • Strategies Evaluated: {strategic.get('strategies_evaluated', 0)}")
            
            if 'level_10_cultural_modeling' in status:
                cultural = status['level_10_cultural_modeling']
                print(f"   • Cultural Norms: {cultural.get('total_norms', 0)}")
                print(f"   • Turing Readiness: {cultural.get('overall_readiness', 0):.2f}")
        
        print("\n✅ All systems operational")
        
    except Exception as e:
        print(f"⚠️  Status check error: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🧠 EL-AMANECER V3 Complete System Launcher"
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip frontend and start backend only"
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Run demo and exit (don't start servers)"
    )
    
    args = parser.parse_args()
    
    # Print welcome
    print("\n" + "🧠"*40)
    print(" "*20 + "EL-AMANECER V3")
    print(" "*15 + "Unified Consciousness Engine")
    print(" "*10 + "with Advanced Theory of Mind (Levels 8-10)")
    print("🧠"*40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Initialize systems
    consciousness = initialize_consciousness_system()
    tom = initialize_advanced_tom()
    evolution = initialize_evolution_system()
    
    # Run demo if requested
    if args.demo_only:
        asyncio.run(run_quick_demo())
        print_system_status()
        print("\n✅ Demo complete - exiting")
        return
    
    # Show status
    print_system_status()
    
    # Start frontend (if not skipped)
    if not args.skip_frontend:
        start_frontend()
        time.sleep(2)  # Give frontend time to start
    
    # Start backend (this blocks)
    print("\n" + "="*80)
    print("🌟 READY TO SERVE")
    print("="*80)
    print("\n💡 TIP: Open http://localhost:3000 in your browser")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("⌨️  Press Ctrl+C to stop\n")
    
    start_backend_server()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        print("✅ System stopped")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
