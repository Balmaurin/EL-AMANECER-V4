"""
Integration Module for Advanced RAG
Combines all RAG techniques with MCP agents and Federated Learning

Based on EMNLP 2024 Paper + Sheily AI Architecture
"""

from .rag_integrator import IntegratedRAGResult, RAGIntegrator

__all__ = ["RAGIntegrator", "IntegratedRAGResult"]
