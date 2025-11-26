"""
Agent Tracing Module
Simple tracing utilities for agent execution
"""

import contextlib
from typing import Any, Dict


@contextlib.contextmanager
def trace_agent_execution(agent_name: str, operation: str):
    """
    Simple tracing context manager for agent execution
    
    Args:
        agent_name: Name of the agent being traced
        operation: Name of the operation being performed
        
    Yields:
        Trace object with add_event method
    """
    # Create a simple trace object
    trace_obj = type('Trace', (), {
        'add_event': lambda self, name, data: None
    })()
    
    yield trace_obj


__all__ = ['trace_agent_execution']
