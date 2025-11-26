"""
MCP Consciousness Layer - Integration Module for MCP Enterprise Master

This module provides integration between the MCP Enterprise Master
and the Consciousness system (conciencia package).
"""

import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConsciousnessLayer:
    """
    Wrapper class for the consciousness system integration
    """

    def __init__(self):
        self.is_initialized = False
        self.global_workspace = None
        self.meta_cognition = None
        self.consciousness_modules = []

    async def initialize_consciousness_layer(self) -> bool:
        """
        Initialize the consciousness layer by importing and setting up
        the consciousness system modules.
        """
        try:
            logger.info("🧠 Initializing MCP Consciousness Layer...")

            # Import consciousness system
            try:
                # Try to import from the conciencia package
                import sys
                from pathlib import Path

                # Add the consciousness package to path
                consciousness_path = Path(__file__).parents[5] / "packages" / "consciousness" / "src"
                if str(consciousness_path) not in sys.path:
                    sys.path.insert(0, str(consciousness_path))

                # Import Global Workspace
                from conciencia.modulos.global_workspace import GlobalWorkspace
                self.global_workspace = GlobalWorkspace()

                # Try to import meta cognition system
                try:
                    from conciencia.meta_cognition_system import MetaCognitionSystem
                    self.meta_cognition = MetaCognitionSystem()
                except ImportError:
                    try:
                        from conciencia.meta_cognition_system_simple import MetaCognitionSystem
                        self.meta_cognition = MetaCognitionSystem()
                    except ImportError:
                        logger.warning("Meta cognition system not available")
                        self.meta_cognition = None

                # Load available modules from conciencia.modulos
                try:
                    from conciencia import modulos
                    available_modules = []
                    for attr_name in dir(modulos):
                        if not attr_name.startswith('_'):
                            attr = getattr(modulos, attr_name)
                            if hasattr(attr, '__module__') and 'conciencia.modulos' in str(attr.__module__):
                                available_modules.append(attr_name)

                    logger.info(f"Found {len(available_modules)} consciousness modules")
                    self.consciousness_modules = available_modules

                except ImportError:
                    logger.warning("Could not load consciousness modules")

                self.is_initialized = True
                logger.info("✅ MCP Consciousness Layer initialized successfully")
                return True

            except ImportError as e:
                logger.error(f"Failed to import consciousness system: {e}")
                return False

        except Exception as e:
            logger.error(f"Error initializing consciousness layer: {e}")
            return False

    async def get_consciousness_status(self) -> Dict[str, Any]:
        """Get the status of the consciousness system"""
        try:
            status = {
                "is_initialized": self.is_initialized,
                "global_workspace_active": self.global_workspace is not None,
                "meta_cognition_active": self.meta_cognition is not None,
                "modules_count": len(self.consciousness_modules),
            }

            if self.global_workspace:
                workspace_status = self.global_workspace.get_workspace_status()
                status.update({
                    "workspace_entries": workspace_status.get("workspace_entries", 0),
                    "registered_processors": workspace_status.get("registered_processors", 0),
                    "total_competitions": workspace_status.get("total_competitions", 0),
                })

            return status

        except Exception as e:
            return {
                "error": str(e),
                "is_initialized": self.is_initialized
            }

    async def process_conscious_input(self, inputs: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process inputs through the consciousness layer

        Args:
            inputs: Pre-conscious inputs to process
            context: Context information for processing

        Returns:
            Conscious processing results
        """
        try:
            if not self.is_initialized or not self.global_workspace:
                return {
                    "success": False,
                    "error": "Consciousness layer not initialized"
                }

            if context is None:
                context = {}

            # Process through Global Workspace
            result = self.global_workspace.integrate(inputs, context)

            return {
                "success": True,
                "result": result,
                "is_conscious": result.get("status") == "integrated",
                "activation_level": result.get("max_activation", 0) if result.get("status") == "sub-threshold" else result.get("confidence", 0)
            }

        except Exception as e:
            logger.error(f"Error processing conscious input: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def shutdown_consciousness_layer(self) -> bool:
        """Shutdown the consciousness layer"""
        try:
            logger.info("Shutting down MCP Consciousness Layer...")

            # Clear workspace if available
            if self.global_workspace:
                self.global_workspace.clear_workspace()

            # Clean up references
            self.global_workspace = None
            self.meta_cognition = None
            self.consciousness_modules = []
            self.is_initialized = False

            logger.info("✅ MCP Consciousness Layer shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Error shutting down consciousness layer: {e}")
            return False


# Global instance
_consciousness_layer_instance: Optional[ConsciousnessLayer] = None

async def get_consciousness_layer() -> ConsciousnessLayer:
    """
    Get or create a consciousness layer instance for the MCP Enterprise Master

    This function is called by the MCP Enterprise Master to initialize
    the consciousness system.
    """
    global _consciousness_layer_instance

    if _consciousness_layer_instance is None:
        _consciousness_layer_instance = ConsciousnessLayer()
        success = await _consciousness_layer_instance.initialize_consciousness_layer()
        if not success:
            logger.warning("Failed to initialize consciousness layer")
            # Return the instance anyway, it will handle not being initialized

    return _consciousness_layer_instance
