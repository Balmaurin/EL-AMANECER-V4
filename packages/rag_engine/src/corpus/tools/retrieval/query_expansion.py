"""
Query expansion using local LLM or fallback heuristics.
Expands queries with synonyms, variations, and reformulations.
No external API required - all processing local.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

# Synonym dictionary (could be extended with external sources)
SYNONYMS = {
    # Spanish tech terms
    "aprendizaje automático": ["machine learning", "ML", "aprendizaje de máquina"],
    "red neuronal": ["neural network", "deep learning"],
    "datos": ["data", "información", "dataset", "corpus"],
    "búsqueda": ["search", "retrieval", "query", "consulta"],
    "inteligencia artificial": ["IA", "AI", "artificial intelligence"],
    "ciberseguridad": ["cybersecurity", "seguridad informática", "security"],
    "aprendizaje federado": ["federated learning", "FL"],
    "Internet de las Cosas": ["IoT", "Internet of Things"],
    "GPU": ["graphics processing unit", "procesador gráfico"],
    "procesamiento": ["processing", "processing", "computación"],
    "modelo": ["model", "architecture", "red"],
    "entrenamiento": ["training", "entrenamieto", "aprendizaje"],
    "evaluación": ["evaluation", "assessment", "métricas"],
    "precisión": ["accuracy", "precision", "recall"],
    "velocidad": ["speed", "performance", "latency", "throughput"],
    "costo": ["cost", "price", "expense", "presupuesto"],
    "calidad": ["quality", "excellence", "validez"],
    "escalabilidad": ["scalability", "scaling", "performance"],
    "distribuido": ["distributed", "distributed computing", "paralelo"],
}


class QueryExpander:
    """Expand queries with synonyms and variations."""

    def __init__(self, use_local_llm: bool = False):
        """
        Args:
            use_local_llm: Whether to use local LLM for expansion (optional)
        """
        self.use_local_llm = use_local_llm
        self.local_llm = None

        if use_local_llm:
            try:
                from transformers import pipeline

                logger.info("🤖 Loading local LLM for query expansion")
                self.local_llm = pipeline("text2text-generation", model="gpt2")
                logger.info("[+] Local LLM loaded successfully")
            except ImportError:
                logger.warning(
                    "transformers not installed, using synonym expansion only"
                )
                self.use_local_llm = False
            except Exception as e:
                logger.warning(
                    f"Failed to load local LLM: {e}, falling back to synonyms"
                )
                self.use_local_llm = False

    def _get_synonyms(self, query: str) -> Set[str]:
        """Get synonyms for query terms."""
        query_lower = query.lower()
        synonyms = set()

        # Check exact phrase matches first
        for key, syns in SYNONYMS.items():
            if key in query_lower:
                synonyms.update(syns)

        # Check individual terms
        terms = re.split(r"\s+", query_lower)
        for term in terms:
            for key, syns in SYNONYMS.items():
                if term in key or key in term:
                    synonyms.update(syns)

        return synonyms

    def _llm_expand(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate query variations using local LLM."""
        if not self.local_llm:
            return []

        try:
            prompt = f"Generate {num_variations} alternative phrasings of this query, separated by commas: {query}"
            result = self.local_llm(prompt, max_length=200)
            variations = [v.strip() for v in result[0]["generated_text"].split(",")]
            return variations[:num_variations]
        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
            return []

    def expand(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        Expand query with synonyms and variations.

        Args:
            query: Original query
            max_expansions: Maximum number of expansions to return

        Returns:
            List of expanded query variations (including original)
        """
        variations = {query}  # Always include original

        # Add synonym-based expansions
        synonyms = self._get_synonyms(query)
        if synonyms:
            # Replace each synonym in original query
            for synonym in list(synonyms)[: max_expansions - 1]:
                variation = query
                for key, syns in SYNONYMS.items():
                    if key in query.lower() and synonym in syns:
                        # Replace key with synonym while preserving case
                        variation = re.sub(
                            re.escape(key),
                            synonym,
                            variation,
                            flags=re.IGNORECASE,
                        )
                variations.add(variation)

        # Add LLM-based expansions if enabled
        if self.use_local_llm:
            llm_variations = self._llm_expand(
                query, num_variations=max_expansions - len(variations)
            )
            variations.update(llm_variations)

        # Return sorted, deduplicated list (original first)
        result = [query]
        result.extend(sorted(set(variations) - {query}))

        return result[:max_expansions]

    def expand_and_search(
        self, search_func, query: str, top_k: int = 10, max_expansions: int = 3
    ) -> List[Dict]:
        """
        Expand query and combine results from all variations.

        Args:
            search_func: Search function to call with (query, top_k)
            query: Original query
            top_k: Results per expansion
            max_expansions: Maximum query variations

        Returns:
            Combined results, deduplicated by doc_id, sorted by score
        """
        variations = self.expand(query, max_expansions=max_expansions)
        logger.debug(f"📚 Query expansions: {variations}")

        # Search with all variations
        all_results = []
        seen_chunks = set()

        for variation in variations:
            try:
                results = search_func(variation, top_k=top_k)
                for result in results:
                    chunk_id = result.get("chunk_id")
                    if chunk_id not in seen_chunks:
                        all_results.append(result)
                        seen_chunks.add(chunk_id)
                        if len(all_results) >= top_k:
                            break
            except Exception as e:
                logger.warning(f"Search for variation '{variation}' failed: {e}")

            if len(all_results) >= top_k:
                break

        # Sort by score descending
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return all_results[:top_k]


# Singleton instance
_expander_instance = None


def get_query_expander(use_local_llm: bool = False) -> QueryExpander:
    """Get or create singleton expander instance."""
    global _expander_instance
    if _expander_instance is None:
        _expander_instance = QueryExpander(use_local_llm=use_local_llm)
    return _expander_instance

    def _compute_diversity_score(self, query: str, original: str) -> float:
        """Compute diversity score based on word overlap"""
        query_words = set(query.lower().split())
        original_words = set(original.lower().split())

        overlap = len(query_words & original_words)
        total = len(query_words | original_words)

        return 1.0 - (overlap / total if total > 0 else 0)
