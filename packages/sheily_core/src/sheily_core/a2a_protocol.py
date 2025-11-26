# -*- coding: utf-8 -*-
"""
Sheily Core A2A Protocol Export
"""

from sheily_core.core.protocols import a2a_protocol

# Export singleton instance
a2a_system = None

try:
    a2a_system = a2a_protocol.a2a_system
except AttributeError:
    # Create if doesn't exist
    a2a_system = a2a_protocol.A2ASystem()
