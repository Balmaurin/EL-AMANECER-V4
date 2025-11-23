#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Query Processing Techniques
Based on COLING 2025 and other advanced RAG papers

Implements:
- Query Decomposition for complex queries
- Multi-turn conversation handling
- Query intent classification
- Dynamic query expansion
- Context-aware query rewriting
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class QueryAnalysisResult:
    """Result of advanced query analysis"""

    original_query: str
    query_type: str
    complexity_score: float
    sub_queries: List[str]
    intent: str
    entities: List[str]
    rewritten_queries: List[str]
    metadata: Dict[str, Any]


@dataclass
class QueryDecompositionResult:
    """Result of query decomposition"""

    original_query: str
    sub_queries: List[str]
    dependencies: Dict[str, List[str]]
    execution_order: List[str]
    confidence_scores: Dict[str, float]


class AdvancedQueryProcessor:
    """
    Advanced Query Processing for Complex RAG Queries

    Implements techniques from COLING 2025 and other papers:
    - Query decomposition for multi-hop reasoning
    - Intent classification and entity extraction
    - Dynamic query expansion
    - Context-aware rewriting
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize advanced query processor

        Args:
            model_name: Model for query processing tasks
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.nlp_pipeline = None

        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.nlp_pipeline = pipeline(
                    "text-classification", model="microsoft/DialoGPT-small"
                )
            except:
                print("Warning: Could not load query processing models")

        # Query type patterns
        self.query_patterns = {
            "factual": r"(what|who|when|where|how|why)\s",
            "comparative": r"(compare|versus|vs|difference|better|worse)",
            "causal": r"(cause|reason|why|because|result|effect)",
            "temporal": r"(before|after|during|when|timeline|history)",
            "quantitative": r"(how many|how much|count|number|percentage)",
            "definitional": r"(what is|define|meaning|explain)",
            "procedural": r"(how to|steps|process|guide|tutorial)",
            "opinion": r"(best|worst|recommend|opinion|suggest)",
            "multi_hop": r"(and then|followed by|after that|next)",
            "hypothetical": r"(if|suppose|imagine|what if|assume)",
        }

    def analyze_query(
        self, query: str, context: Optional[List[str]] = None
    ) -> QueryAnalysisResult:
        """
        Perform comprehensive query analysis

        Args:
            query: User query
            context: Previous conversation context

        Returns:
            Detailed query analysis
        """
        # Classify query type
        query_type = self._classify_query_type(query)

        # Calculate complexity
        complexity_score = self._calculate_complexity(query)

        # Extract entities
        entities = self._extract_entities(query)

        # Determine intent
        intent = self._determine_intent(query, context)

        # Generate sub-queries if complex
        sub_queries = []
        if complexity_score > 0.7:
            sub_queries = self._decompose_query(query)

        # Generate rewritten queries
        rewritten_queries = self._generate_rewritten_queries(query, context)

        return QueryAnalysisResult(
            original_query=query,
            query_type=query_type,
            complexity_score=complexity_score,
            sub_queries=sub_queries,
            intent=intent,
            entities=entities,
            rewritten_queries=rewritten_queries,
            metadata={
                "word_count": len(query.split()),
                "has_questions": "?" in query,
                "has_conjunctions": bool(
                    re.search(r"\b(and|or|but|however|therefore)\b", query.lower())
                ),
                "context_available": context is not None,
            },
        )

    def decompose_query(self, query: str) -> QueryDecompositionResult:
        """
        Decompose complex query into simpler sub-queries

        Args:
            query: Complex query to decompose

        Returns:
            Decomposition result with dependencies
        """
        # Identify decomposition points
        decomposition_points = self._find_decomposition_points(query)

        if not decomposition_points:
            return QueryDecompositionResult(
                original_query=query,
                sub_queries=[query],
                dependencies={},
                execution_order=[query],
                confidence_scores={query: 1.0},
            )

        # Split into sub-queries
        sub_queries = self._split_into_sub_queries(query, decomposition_points)

        # Determine dependencies
        dependencies = self._analyze_dependencies(sub_queries)

        # Determine execution order
        execution_order = self._determine_execution_order(sub_queries, dependencies)

        # Calculate confidence scores
        confidence_scores = {
            sq: self._calculate_sub_query_confidence(sq) for sq in sub_queries
        }

        return QueryDecompositionResult(
            original_query=query,
            sub_queries=sub_queries,
            dependencies=dependencies,
            execution_order=execution_order,
            confidence_scores=confidence_scores,
        )

    def expand_query(self, query: str, top_k: int = 5) -> List[str]:
        """
        Generate query expansions for better retrieval

        Args:
            query: Original query
            top_k: Number of expansions to generate

        Returns:
            List of expanded queries
        """
        expansions = [query]  # Always include original

        # Synonym expansion
        synonym_expansions = self._expand_with_synonyms(query)
        expansions.extend(synonym_expansions[: top_k // 2])

        # Related term expansion
        related_expansions = self._expand_with_related_terms(query)
        expansions.extend(related_expansions[: top_k // 2])

        # Remove duplicates and limit
        unique_expansions = list(dict.fromkeys(expansions))
        return unique_expansions[:top_k]

    def rewrite_query(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Rewrite query for better retrieval performance

        Args:
            query: Original query
            context: Conversation context

        Returns:
            Rewritten query
        """
        # Remove noise words
        cleaned_query = self._clean_query(query)

        # Add context if available
        if context:
            contextual_query = self._add_context_to_query(cleaned_query, context)
            return contextual_query

        # Apply general improvements
        improved_query = self._improve_query_structure(cleaned_query)

        return improved_query

    def _classify_query_type(self, query: str) -> str:
        """Classify query type based on patterns"""
        query_lower = query.lower()

        for query_type, pattern in self.query_patterns.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                return query_type

        return "general"

    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)"""
        factors = {
            "length": min(
                len(query.split()) / 20, 1.0
            ),  # Longer queries are more complex
            "conjunctions": len(
                re.findall(r"\b(and|or|but|however|therefore|because)\b", query.lower())
            ),
            "questions": len(re.findall(r"\?", query)),
            "entities": len(self._extract_entities(query)),
            "temporal": (
                1.0
                if re.search(r"\b(before|after|during|when|timeline)\b", query.lower())
                else 0.0
            ),
            "comparative": (
                1.0
                if re.search(r"\b(compare|versus|vs|better|worse)\b", query.lower())
                else 0.0
            ),
        }

        # Weighted combination
        weights = {
            "length": 0.2,
            "conjunctions": 0.2,
            "questions": 0.15,
            "entities": 0.15,
            "temporal": 0.15,
            "comparative": 0.15,
        }

        complexity = sum(factors[key] * weights[key] for key in factors)
        return min(complexity, 1.0)

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        # Simple entity extraction (can be enhanced with NER models)
        entities = []

        # Look for capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word)

        # Look for quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)

        return list(set(entities))  # Remove duplicates

    def _determine_intent(self, query: str, context: Optional[List[str]]) -> str:
        """Determine user intent"""
        intents = {
            "information": ["what", "who", "when", "where", "how", "why", "explain"],
            "comparison": ["compare", "versus", "vs", "better", "worse", "difference"],
            "calculation": ["calculate", "compute", "how many", "how much"],
            "definition": ["define", "meaning", "what is"],
            "procedure": ["how to", "steps", "guide", "tutorial"],
            "recommendation": ["recommend", "suggest", "best", "worst"],
            "verification": ["true", "false", "correct", "verify", "confirm"],
        }

        query_lower = query.lower()

        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent

        return "general"

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into simpler parts"""
        # Split on conjunctions and complex structures
        separators = (
            r"\b(and|or|but|however|therefore|because|then|after|before|while)\b"
        )

        parts = re.split(separators, query, flags=re.IGNORECASE)
        sub_queries = []

        current_query = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part.lower() in [
                "and",
                "or",
                "but",
                "however",
                "therefore",
                "because",
                "then",
                "after",
                "before",
                "while",
            ]:
                if current_query:
                    sub_queries.append(current_query.strip())
                    current_query = ""
            else:
                current_query += " " + part if current_query else part

        if current_query:
            sub_queries.append(current_query.strip())

        # Filter out very short sub-queries
        sub_queries = [sq for sq in sub_queries if len(sq.split()) >= 3]

        return sub_queries if len(sub_queries) > 1 else [query]

    def _find_decomposition_points(self, query: str) -> List[Tuple[int, str]]:
        """Find points where query can be decomposed"""
        decomposition_points = []

        # Look for conjunctions
        conjunctions = ["and", "or", "but", "however", "therefore", "because", "then"]
        words = query.split()

        for i, word in enumerate(words):
            if word.lower() in conjunctions:
                start_pos = query.find(word)
                decomposition_points.append((start_pos, word))

        return decomposition_points

    def _split_into_sub_queries(
        self, query: str, points: List[Tuple[int, str]]
    ) -> List[str]:
        """Split query at decomposition points"""
        if not points:
            return [query]

        sub_queries = []
        start = 0

        for pos, connector in points:
            sub_query = query[start:pos].strip()
            if sub_query:
                sub_queries.append(sub_query)
            start = pos + len(connector)

        # Add remaining part
        remaining = query[start:].strip()
        if remaining:
            sub_queries.append(remaining)

        return [sq for sq in sub_queries if len(sq.split()) >= 2]

    def _analyze_dependencies(self, sub_queries: List[str]) -> Dict[str, List[str]]:
        """Analyze dependencies between sub-queries"""
        dependencies = defaultdict(list)

        for i, sq1 in enumerate(sub_queries):
            for j, sq2 in enumerate(sub_queries):
                if i != j and self._queries_related(sq1, sq2):
                    dependencies[sq1].append(sq2)

        return dict(dependencies)

    def _determine_execution_order(
        self, sub_queries: List[str], dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """Determine optimal execution order"""
        # Simple topological sort
        executed = set()
        order = []

        def can_execute(query: str) -> bool:
            deps = dependencies.get(query, [])
            return all(dep in executed for dep in deps)

        while len(order) < len(sub_queries):
            for query in sub_queries:
                if query not in executed and can_execute(query):
                    order.append(query)
                    executed.add(query)
                    break

        return order

    def _calculate_sub_query_confidence(self, sub_query: str) -> float:
        """Calculate confidence score for sub-query"""
        # Simple confidence based on length and clarity
        words = sub_query.split()
        length_score = min(len(words) / 10, 1.0)  # Prefer moderate length
        clarity_score = (
            1.0
            if "?" in sub_query
            or any(
                w in sub_query.lower()
                for w in ["what", "how", "why", "when", "where", "who"]
            )
            else 0.7
        )

        return (length_score + clarity_score) / 2

    def _queries_related(self, q1: str, q2: str) -> bool:
        """Check if two queries are related"""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / union > 0.2 if union > 0 else False

    def _generate_rewritten_queries(
        self, query: str, context: Optional[List[str]]
    ) -> List[str]:
        """Generate rewritten versions of the query"""
        rewrites = [query]  # Original always included

        # Add context-aware rewrites
        if context:
            contextual_rewrites = self._add_context_to_query(query, context)
            if contextual_rewrites != query:
                rewrites.append(contextual_rewrites)

        # Add simplified version
        simplified = self._simplify_query(query)
        if simplified != query:
            rewrites.append(simplified)

        # Add expanded version
        expanded = self._expand_query_generally(query)
        if expanded != query:
            rewrites.append(expanded)

        return list(set(rewrites))  # Remove duplicates

    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms"""
        # Simple synonym expansion (can be enhanced with WordNet)
        synonyms = {
            "good": ["excellent", "great", "outstanding"],
            "bad": ["poor", "terrible", "awful"],
            "fast": ["quick", "rapid", "speedy"],
            "slow": ["sluggish", "gradual", "leisurely"],
            "big": ["large", "huge", "massive"],
            "small": ["tiny", "little", "miniature"],
        }

        expansions = []
        words = query.split()

        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in synonyms:
                for synonym in synonyms[word_lower][:2]:  # Limit to 2 synonyms
                    new_words = words.copy()
                    new_words[i] = synonym
                    expansions.append(" ".join(new_words))

        return expansions

    def _expand_with_related_terms(self, query: str) -> List[str]:
        """Expand query with related terms"""
        # Domain-specific expansions
        expansions = []

        if "machine learning" in query.lower():
            expansions.append(query.replace("machine learning", "ML"))
            expansions.append(
                query.replace("machine learning", "artificial intelligence")
            )

        if "climate change" in query.lower():
            expansions.append(query.replace("climate change", "global warming"))

        if "programming" in query.lower():
            expansions.append(query.replace("programming", "coding"))
            expansions.append(query.replace("programming", "software development"))

        return expansions

    def _clean_query(self, query: str) -> str:
        """Clean query by removing noise"""
        # Remove extra whitespace
        cleaned = " ".join(query.split())

        # Remove redundant words
        redundant = ["please", "can you", "could you", "i want to know"]
        for word in redundant:
            cleaned = cleaned.replace(word, "")

        # Fix spacing around punctuation
        cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)

        return cleaned.strip()

    def _add_context_to_query(self, query: str, context: List[str]) -> str:
        """Add contextual information to query"""
        if not context:
            return query

        # Extract key entities from context
        context_text = " ".join(context)
        context_entities = self._extract_entities(context_text)

        # Add relevant entities to query
        query_entities = self._extract_entities(query)
        new_entities = [e for e in context_entities if e not in query_entities]

        if new_entities:
            enhanced_query = query + " " + " ".join(new_entities[:2])  # Limit to 2
            return enhanced_query

        return query

    def _improve_query_structure(self, query: str) -> str:
        """Improve query structure for better retrieval"""
        # Add question words if missing
        if not query.startswith(("What", "How", "Why", "When", "Where", "Who")):
            if "is" in query.lower() or "are" in query.lower():
                query = "What " + query
            elif "do" in query.lower() or "does" in query.lower():
                query = "How " + query

        return query

    def _simplify_query(self, query: str) -> str:
        """Simplify complex query"""
        # Remove complex clauses
        sentences = re.split(r"[.!?]", query)
        if len(sentences) > 1:
            return sentences[0].strip()

        # Remove parenthetical expressions
        simplified = re.sub(r"\([^)]*\)", "", query)

        return simplified.strip()

    def _expand_query_generally(self, query: str) -> str:
        """Generally expand query with additional context"""
        # Add common related terms
        expansions = {
            "AI": "artificial intelligence machine learning",
            "ML": "machine learning AI",
            "programming": "coding software development",
            "database": "data storage management",
            "network": "internet connectivity communication",
        }

        query_lower = query.lower()
        for term, expansion in expansions.items():
            if term in query_lower and expansion not in query_lower:
                return query + " " + expansion

        return query


class ConversationalQueryProcessor:
    """
    Handle multi-turn conversations and context

    Implements conversation-aware query processing
    """

    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 5

    def add_to_history(self, user_query: str, system_response: str):
        """Add interaction to conversation history"""
        self.conversation_history.append(
            {"query": user_query, "response": system_response, "timestamp": time.time()}
        )

        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

    def get_conversation_context(self) -> List[str]:
        """Get recent conversation context"""
        return [
            f"Q: {item['query']}\nA: {item['response']}"
            for item in self.conversation_history[-3:]
        ]

    def is_follow_up_query(self, query: str) -> bool:
        """Check if query is a follow-up"""
        follow_up_indicators = [
            "that",
            "it",
            "this",
            "those",
            "these",
            "what about",
            "and",
            "also",
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in follow_up_indicators)

    def resolve_pronouns(self, query: str) -> str:
        """Resolve pronouns in follow-up queries"""
        if not self.is_follow_up_query(query):
            return query

        # Simple pronoun resolution based on last response
        if self.conversation_history:
            last_response = self.conversation_history[-1]["response"]

            # Replace pronouns with entities from last response
            entities = self._extract_entities(last_response)
            if entities:
                # Replace "it" or "that" with main entity
                query = re.sub(r"\bit\b", entities[0], query, flags=re.IGNORECASE)
                query = re.sub(r"\bthat\b", entities[0], query, flags=re.IGNORECASE)

        return query

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        # Simple extraction of capitalized words
        words = text.split()
        entities = [word for word in words if word[0].isupper() and len(word) > 2]
        return list(set(entities))
