"""
RAPTOR (Recursive Abstractive Processing and Topical Organization for Retrieval)

This package implements the RAPTOR algorithm for hierarchical document organization
and efficient retrieval. RAPTOR builds a tree structure of document clusters,
enabling fast and semantically meaningful document retrieval.
"""

from .build_tree import build_raptor

__all__ = ["build_raptor"]
