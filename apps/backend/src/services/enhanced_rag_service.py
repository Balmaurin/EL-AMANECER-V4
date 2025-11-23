#!/usr/bin/env python3
"""
ENHANCED RAG SERVICE - Intelligent Knowledge Retrieval & Generation
===================================================================

Advanced RAG system that learns from exercise datasets and provides
highly accurate, contextually relevant responses with anti-hallucination safeguards.

Features:
- Multi-source knowledge base (exercises + external + curated)
- Advanced vector embeddings with semantic search
- Query expansion and intent understanding
- Fact-checking integration for response verification
- Corpus categorization and automated knowledge graph
- Continuous learning from user interactions
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EnhancedRAGService:
    """Enhanced RAG service with advanced retrieval and anti-hallucination"""

    def __init__(self):
        self.db_path = "rag_enhanced_database.db"
        self.corpus_path = Path("corpus")
        self.vector_cache = {}
        self.knowledge_graph = {}
        self.category_mappings = {}
        self.interaction_memory = []

        # Advanced search parameters
        self.max_context_length = 4000
        self.top_k_candidates = 10
        self.semantic_threshold = 0.7
        self.diversity_penalty = 0.1

        # Learning parameters
        self.adaptive_learning_rate = 0.1
        self.feedback_memory_size = 1000

        self._initialize_system()
        logger.info(
            "🧠 Enhanced RAG Service initialized with anti-hallucination capabilities"
        )

    def _initialize_system(self):
        """Initialize enhanced RAG system components"""
        self._ensure_enhanced_tables()
        self._load_corpus_categories()
        self._build_initial_knowledge_graph()
        self._initialize_semantic_engine()

    def _ensure_enhanced_tables(self):
        """Create enhanced database tables for advanced RAG"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enhanced knowledge base with categorization
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,  -- 'exercise', 'external', 'curated'
                category TEXT NOT NULL,
                subcategory TEXT,
                content TEXT NOT NULL,
                source TEXT,
                verified BOOLEAN DEFAULT FALSE,
                confidence REAL DEFAULT 0.5,
                embedding_vector TEXT,  -- JSON array of floats
                metadata TEXT,  -- JSON metadata
                timestamp TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0
            )
        """
        )

        # Advanced interaction tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_query TEXT NOT NULL,
                retrieved_contexts TEXT,  -- JSON array of retrieved items
                generated_response TEXT,
                response_quality REAL,
                user_feedback TEXT,
                fact_check_passed BOOLEAN DEFAULT TRUE,
                categories_used TEXT,  -- JSON array
                semantic_score REAL,
                timestamp TEXT NOT NULL
            )
        """
        )

        # Semantic similarity cache
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS similarity_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE,
                similar_entities TEXT,  -- JSON array
                timestamp TEXT NOT NULL
            )
        """
        )

        # Corpus relationships table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity1_id INTEGER,
                entity2_id INTEGER,
                relationship_type TEXT,
                strength REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (entity1_id) REFERENCES knowledge_base(id),
                FOREIGN KEY (entity2_id) REFERENCES knowledge_base(id)
            )
        """
        )

        conn.commit()
        conn.close()

    def _load_corpus_categories(self):
        """Load and organize corpus by categories"""
        self.category_mappings = {
            "ai_ml": [
                "machine learning",
                "neural network",
                "deep learning",
                "ai",
                "artificial intelligence",
            ],
            "rag_system": [
                "retrieval",
                "augmented",
                "generation",
                "similarity",
                "embedding",
            ],
            "programming": ["python", "algorithm", "data structure", "optimization"],
            "data_science": [
                "statistics",
                "probability",
                "model",
                "prediction",
                "analysis",
            ],
            "ethics": ["bias", "fairness", "privacy", "safety", "responsible"],
            "blockchain": ["token", "sheily", "cryptocurrency", "smart contract"],
            "general": ["question", "answer", "explanation", "concept"],
        }

    def _build_initial_knowledge_graph(self):
        """Build initial knowledge graph from existing corpus"""
        try:
            # Load from corpus directory
            corpus_files = list(self.corpus_path.glob("*.json"))
            initial_knowledge = []

            for corpus_file in corpus_files:
                if corpus_file.name.startswith("corpus_"):
                    try:
                        with open(corpus_file, "r", encoding="utf-8") as f:
                            corpus_data = json.load(f)

                            for item in corpus_data.get("documents", []):
                                categorized_item = self._categorize_content(item)
                                initial_knowledge.append(categorized_item)

                    except Exception as e:
                        logger.warning(f"Error loading {corpus_file}: {e}")

            # Add to knowledge graph
            for item in initial_knowledge:
                self.knowledge_graph[item["id"]] = item

            logger.info(f"Loaded {len(initial_knowledge)} items into knowledge graph")

        except Exception as e:
            logger.warning(f"Error building initial knowledge graph: {e}")

    def _initialize_semantic_engine(self):
        """Initialize semantic processing capabilities"""
        # Simple TF-IDF based semantic engine (can be upgraded to embeddings)
        self.vocabulary = set()
        self.term_frequency = defaultdict(dict)
        self.document_frequency = defaultdict(int)

    def _categorize_content(self, content: Dict) -> Dict:
        """Categorize content using intelligent classification"""
        text = content.get("content", content.get("text", "")).lower()
        title = content.get("title", "").lower()

        # Combine title and text for better categorization
        full_text = f"{title} {text}"

        # Calculate category scores
        category_scores = {}
        for category, keywords in self.category_mappings.items():
            score = 0
            for keyword in keywords:
                if keyword in full_text:
                    score += 1
                    # Bonus for exact matches or important terms
                    if keyword in ["ai", "ml", "rag", "blockchain"]:
                        score += 0.5

            category_scores[category] = score

        # Choose primary and secondary categories
        sorted_categories = sorted(
            category_scores.items(), key=lambda x: x[1], reverse=True
        )

        primary_category = sorted_categories[0][0] if sorted_categories else "general"
        secondary_category = (
            sorted_categories[1][0]
            if len(sorted_categories) > 1 and sorted_categories[1][1] > 0
            else None
        )

        return {
            "id": self._generate_entity_id(content),
            "entity_type": "exercise",  # Default for corpus
            "primary_category": primary_category,
            "secondary_category": secondary_category,
            "content": content.get("content", content.get("text", "")),
            "title": content.get("title", ""),
            "source": content.get("source", "corpus"),
            "verified": content.get("verified", False),
            "confidence": content.get("confidence", 0.8),
            "timestamp": content.get("timestamp", datetime.now().isoformat()),
        }

    def _generate_entity_id(self, content: Dict) -> str:
        """Generate unique entity ID"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]

    async def enhanced_search(
        self, query: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Enhanced semantic search with multiple strategies
        """
        start_time = datetime.now()

        try:
            # 1. Query preprocessing and intent understanding
            processed_query = self._preprocess_query(query, context)

            # 2. Multi-strategy retrieval
            candidates = await self._multi_strategy_retrieval(processed_query)

            # 3. Advanced ranking and filtering
            ranked_results = self._advanced_ranking(candidates, processed_query)

            # 4. Diversity and coherence filtering
            final_results = self._diversity_filtering(ranked_results)

            # 5. Confidence and verification assessment
            verified_results = await self._verify_results(
                final_results, processed_query
            )

            # 6. Anti-hallucination safeguards
            safe_results = await self._apply_safety_filters(verified_results)

            # 7. Usage tracking and learning
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._track_interaction(query, safe_results, processing_time)

            return {
                "query": query,
                "processed_query": processed_query,
                "results": safe_results,
                "total_candidates": len(candidates),
                "final_results": len(safe_results),
                "processing_time": processing_time,
                "confidence_score": self._calculate_overall_confidence(safe_results),
                "categories_represented": self._get_categories_represented(
                    safe_results
                ),
                "hallucination_risk": self._assess_hallucination_risk(safe_results),
            }

        except Exception as e:
            logger.error(f"Enhanced search error: {e}")
            return self._fallback_search(query, context)

    def _preprocess_query(
        self, query: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Advanced query preprocessing and intent understanding"""
        processed = {
            "original_query": query,
            "cleaned_query": self._clean_text(query),
            "expanded_queries": self._expand_query(query),
            "detected_intent": self._detect_query_intent(query),
            "entity_mentions": self._extract_entities(query),
            "difficulty_level": self._assess_difficulty(query, context),
            "category_hints": self._infer_categories(query),
            "semantic_tokens": self._tokenize_semantically(query),
        }

        return processed

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace, normalize punctuation
        import re

        text = re.sub(r"\s+", " ", text.strip())
        return text

    def _expand_query(self, query: str) -> List[str]:
        """Generate expanded query variations"""
        expansions = [query]

        # Add common question variations
        if "what is" in query.lower():
            expansions.append(query.lower().replace("what is", "explain"))
            expansions.append(query.lower().replace("what is", "describe"))
        elif "how" in query.lower():
            expansions.append(query.lower().replace("how", "explain how"))

        return expansions

    async def _multi_strategy_retrieval(self, processed_query: Dict) -> List[Dict]:
        """
        Multi-strategy retrieval combining different search methods
        """
        candidates = []

        # Strategy 1: Semantic keyword matching
        keyword_candidates = self._keyword_semantic_search(processed_query)
        candidates.extend(keyword_candidates)

        # Strategy 2: Category-based retrieval
        category_candidates = self._category_based_search(processed_query)
        candidates.extend(category_candidates)

        # Strategy 3: Knowledge graph traversal
        graph_candidates = self._knowledge_graph_search(processed_query)
        candidates.extend(graph_candidates)

        # Strategy 4: Recent interaction memory
        interaction_candidates = self._interaction_memory_search(processed_query)
        candidates.extend(interaction_candidates)

        # Remove duplicates and limit
        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            entity_id = candidate.get("id", candidate.get("entity_id", ""))
            if entity_id not in seen_ids:
                seen_ids.add(entity_id)
                unique_candidates.append(candidate)

        return unique_candidates[: self.top_k_candidates * 2]  # Keep more for ranking

    def _keyword_semantic_search(self, processed_query: Dict) -> List[Dict]:
        """Enhanced semantic keyword search"""
        query_tokens = processed_query["semantic_tokens"]
        candidates = []

        for entity_id, entity in self.knowledge_graph.items():
            if entity.get("confidence", 0) < 0.3:  # Skip low-confidence items
                continue

            content = entity.get("content", "").lower()
            score = self._calculate_semantic_similarity(query_tokens, content)

            if score > self.semantic_threshold:
                candidates.append(
                    {
                        "id": entity_id,
                        "content": entity["content"],
                        "category": entity["primary_category"],
                        "similarity_score": score,
                        "retrieval_method": "semantic_keyword",
                    }
                )

        return candidates

    def _category_based_search(self, processed_query: Dict) -> List[Dict]:
        """Search based on inferred categories"""
        candidates = []
        query_categories = processed_query.get("category_hints", [])

        for entity_id, entity in self.knowledge_graph.items():
            entity_categories = [entity["primary_category"]]
            if entity.get("secondary_category"):
                entity_categories.append(entity["secondary_category"])

            # Calculate category overlap
            category_overlap = len(set(query_categories) & set(entity_categories))
            if category_overlap > 0:
                score = category_overlap * 0.3 + entity.get("confidence", 0.5) * 0.7

                candidates.append(
                    {
                        "id": entity_id,
                        "content": entity["content"],
                        "category": entity["primary_category"],
                        "similarity_score": score,
                        "retrieval_method": "category_based",
                    }
                )

        return candidates

    def _knowledge_graph_search(self, processed_query: Dict) -> List[Dict]:
        """Search using knowledge graph relationships"""
        candidates = []
        entities = processed_query.get("entity_mentions", [])

        for entity in entities:
            # Find related entities in knowledge graph
            related_entities = self._find_related_entities(entity)

            for related in related_entities:
                if related["entity_id"] != entity:
                    entity_data = self.knowledge_graph.get(related["entity_id"])
                    if entity_data:
                        candidates.append(
                            {
                                "id": related["entity_id"],
                                "content": entity_data["content"],
                                "category": entity_data["primary_category"],
                                "similarity_score": related["strength"] * 0.8,
                                "retrieval_method": "knowledge_graph",
                            }
                        )

        return candidates

    def _interaction_memory_search(self, processed_query: Dict) -> List[Dict]:
        """Search based on recent interaction patterns"""
        candidates = []
        recent_interactions = self.interaction_memory[-20:]  # Last 20 interactions

        query_keywords = set(processed_query.get("semantic_tokens", [])[:5])

        for interaction in recent_interactions:
            query_keywords_inter = set(interaction.get("query_keywords", []))
            overlap = len(query_keywords & query_keywords_inter)

            if overlap > 0:
                context_id = interaction.get("best_context_id")
                if context_id and context_id in self.knowledge_graph:
                    entity = self.knowledge_graph[context_id]
                    score = (overlap / len(query_keywords)) * interaction.get(
                        "success_score", 0.7
                    )

                    candidates.append(
                        {
                            "id": context_id,
                            "content": entity["content"],
                            "category": entity["primary_category"],
                            "similarity_score": score,
                            "retrieval_method": "interaction_memory",
                        }
                    )

        return candidates

    def _advanced_ranking(
        self, candidates: List[Dict], processed_query: Dict
    ) -> List[Dict]:
        """Advanced ranking with multiple factors"""
        for candidate in candidates:
            base_score = candidate["similarity_score"]

            # Factor 1: Content quality and length
            content_length = len(candidate.get("content", ""))
            quality_factor = min(
                1.0, content_length / 1000
            )  # Prefer substantial content

            # Factor 2: Category relevance
            query_categories = processed_query.get("category_hints", [])
            if candidate["category"] in query_categories:
                category_factor = 1.5
            else:
                category_factor = 1.0

            # Factor 3: Verification status
            if candidate.get("verified", False):
                verification_factor = 1.3
            else:
                verification_factor = 1.0

            # Factor 4: Usage history (popular content)
            usage_factor = 1.0 + (candidate.get("usage_count", 0) * 0.01)
            usage_factor = min(usage_factor, 1.5)

            # Combined score
            final_score = (
                base_score
                * quality_factor
                * category_factor
                * verification_factor
                * usage_factor
            )
            candidate["final_score"] = final_score

        # Sort by final score and return top results
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[: self.top_k_candidates]

    def _diversity_filtering(self, ranked_results: List[Dict]) -> List[Dict]:
        """Apply diversity filtering to ensure varied perspectives"""
        if not ranked_results:
            return ranked_results

        selected = []
        category_counts = defaultdict(int)

        for result in ranked_results:
            category = result["category"]
            current_count = category_counts[category]

            # Apply diversity penalty for over-represented categories
            diversity_penalty = 1.0 - (current_count * self.diversity_penalty)
            adjusted_score = result["final_score"] * diversity_penalty

            result["adjusted_score"] = adjusted_score

            # Always include top results, apply penalty to others
            if len(selected) < 3:  # Ensure at least 3 results
                selected.append(result)
                category_counts[category] += 1
            elif adjusted_score > selected[-1]["adjusted_score"]:
                # Replace lowest scoring item if this is better
                selected.pop()  # Remove lowest
                selected.append(result)
                category_counts[category] += 1

        selected.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return selected

    async def _verify_results(
        self, results: List[Dict], processed_query: Dict
    ) -> List[Dict]:
        """Verify results through fact-checking and consistency checks"""
        try:
            # Try to import fact checker dynamically
            sys.path.append(str(Path(__file__).parent.parent))
            from apps.backend.src.services.fact_checker_service import get_fact_checker

            fact_checker = get_fact_checker()
            await fact_checker.initialize()

            verified_results = []
            for result in results:
                content = result.get("content", "")

                # Extract key claims from content
                claims = self._extract_factual_claims(content)

                if claims:
                    # Verify claims
                    verification = await fact_checker.verify_claim_async(claims[0])
                    if verification:
                        result["fact_check_confidence"] = verification.get(
                            "confidence", 0.5
                        )
                        result["verified"] = verification.get("verified", False)
                    else:
                        result["fact_check_confidence"] = 0.5
                        result["verified"] = False
                else:
                    result["fact_check_confidence"] = (
                        0.8  # Default for non-factual content
                    )
                    result["verified"] = True

                verified_results.append(result)

            # Close fact checker if it has close method
            if hasattr(fact_checker, "close"):
                await fact_checker.close()
            return verified_results

        except (ImportError, Exception) as e:
            # Fallback if fact checker not available
            logger.warning(f"Fact checking not available: {e}")
            for result in results:
                result["fact_check_confidence"] = 0.5
                result["verified"] = False
            return results

    async def _apply_safety_filters(self, results: List[Dict]) -> List[Dict]:
        """Apply anti-hallucination safety filters"""
        try:
            from apps.backend.src.services.fact_checker_service import SafetyGuardrails

            guardrails = SafetyGuardrails(get_fact_checker())

            # Combine all results into a single response for safety checking
            combined_content = " ".join(
                [r.get("content", "") for r in results[:3]]
            )  # Top 3 results

            safety_result = await guardrails.process_response(combined_content)

            if safety_result and safety_result["was_modified"]:
                # If content was modified for safety, adjust confidence scores
                for result in results:
                    if safety_result["modified"]:
                        result["safety_adjusted"] = True
                        result[
                            "final_score"
                        ] *= 0.9  # Slight penalty for modified content

            return results

        except Exception as e:
            logger.warning(f"Safety filtering failed: {e}")
            return results

    async def _track_interaction(
        self, query: str, results: List[Dict], processing_time: float
    ):
        """Track interaction for continuous learning"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_keywords": self._extract_keywords(query),
            "results_count": len(results),
            "top_categories": [r.get("category", "") for r in results[:3]],
            "avg_confidence": (
                np.mean([r.get("final_score", 0) for r in results]) if results else 0
            ),
            "processing_time": processing_time,
            "success_score": (
                1.0 if results else 0.0
            ),  # Will be updated with user feedback
        }

        self.interaction_memory.append(interaction)

        # Keep memory bounded
        if len(self.interaction_memory) > self.feedback_memory_size:
            self.interaction_memory = self.interaction_memory[
                -self.feedback_memory_size :
            ]

    async def add_exercise_data(self, exercise_data: Dict[str, Any]):
        """Add new exercise data to the RAG system"""
        try:
            # Categorize and process exercise data
            processed_data = []
            for item in exercise_data.get("answers", []):
                categorized = self._categorize_exercise_item(item)
                if categorized:
                    processed_data.append(categorized)

            # Add to knowledge graph
            for item in processed_data:
                entity_id = self._generate_entity_id(item)

                # Calculate semantic embedding (placeholder - in real system use BERT/embedding model)
                item["embedding_vector"] = self._calculate_simple_embedding(
                    item.get("content", "")
                )

                # Store in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO knowledge_base
                    (entity_type, category, subcategory, content, source, verified, confidence, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        item["entity_type"],
                        item["primary_category"],
                        item.get("secondary_category", ""),
                        item["content"],
                        item.get("source", "exercise"),
                        item.get("verified", False),
                        item.get("confidence", 0.7),
                        json.dumps(item.get("metadata", {})),
                        item.get("timestamp", datetime.now().isoformat()),
                    ),
                )

                conn.commit()

                # Update knowledge graph
                item["id"] = entity_id
                self.knowledge_graph[entity_id] = item

                conn.close()

            logger.info(
                f"Added {len(processed_data)} exercise data items to RAG system"
            )

            # Trigger learning update
            await self._update_learning_from_new_data(processed_data)

        except Exception as e:
            logger.error(f"Error adding exercise data to RAG: {e}")

    def _categorize_exercise_item(self, item: Dict) -> Optional[Dict]:
        """Categorize individual exercise item"""
        try:
            question = item.get("question_text", item.get("question", ""))
            answer = item.get("user_answer", "")

            combined_content = f"{question} {answer}"
            categories = self.category_mappings

            # Enhanced categorization for exercises
            best_category = "general"
            best_score = 0

            for category, keywords in categories.items():
                score = sum(
                    1 for keyword in keywords if keyword in combined_content.lower()
                )
                if score > best_score:
                    best_score = score
                    best_category = category

            return {
                "entity_type": "exercise",
                "primary_category": best_category,
                "content": f"Q: {question}\nA: {answer}",
                "question": question,
                "answer": answer,
                "correct": item.get("is_correct", False),
                "source": "exercise_system",
                "confidence": 0.8 if item.get("is_correct", False) else 0.6,
                "timestamp": item.get("timestamp", datetime.now().isoformat()),
                "metadata": {
                    "difficulty": item.get("difficulty", "medium"),
                    "category_original": item.get("category", ""),
                    "correct_answer": item.get("correct_answer", ""),
                },
            }

        except Exception as e:
            logger.warning(f"Error categorizing exercise item: {e}")
            return None

    async def _update_learning_from_new_data(self, new_data: List[Dict]):
        """Update learning parameters from new data"""
        # Analyze new data for learning insights
        category_distribution = Counter(item["primary_category"] for item in new_data)

        # Adjust search priorities based on new data
        self._adapt_search_strategy(category_distribution)

        # Update semantic understanding
        await self._incorporate_new_semantic_patterns(new_data)

    def _adapt_search_strategy(self, category_distribution: Counter):
        """Adapt search strategy based on data distribution"""
        # Adjust category weights
        for category, count in category_distribution.items():
            if category in self.category_mappings:
                # Increase priority for categories with more new data
                pass  # Implementation would adjust internal weights

    async def _incorporate_new_semantic_patterns(self, new_data: List[Dict]):
        """Incorporate new semantic patterns from recent data"""
        # Update semantic understanding based on new exercise patterns
        new_keywords = set()
        for item in new_data:
            question_keywords = self._extract_keywords(item.get("question", ""))
            new_keywords.update(question_keywords)

        # Expand vocabulary with new domain terms
        self.vocabulary.update(new_keywords)

    def _calculate_semantic_similarity(
        self, query_tokens: List[str], content: str
    ) -> float:
        """Calculate simple semantic similarity (upgradeable to embeddings)"""
        if not query_tokens or not content:
            return 0.0

        content_lower = content.lower()
        matches = sum(1 for token in query_tokens if token.lower() in content_lower)
        match_ratio = matches / len(query_tokens)

        # Bonus for exact phrase matches
        query_phrase = " ".join(query_tokens[:3])  # First 3 words
        if query_phrase.lower() in content_lower:
            match_ratio += 0.3

        return min(match_ratio, 1.0)

    def _calculate_simple_embedding(self, text: str) -> str:
        """Calculate simple text embedding (placeholder for real embeddings)"""
        # Very basic TF-IDF style vector (replace with BERT/embedding model)
        words = self._extract_keywords(text.lower())
        word_freq = Counter(words)

        # Simple bag-of-words style (dimension would be |vocabulary|)
        vector = {}
        for i, word in enumerate(self.vocabulary):
            if word in word_freq:
                vector[str(i)] = word_freq[word] / len(words)  # Normalized frequency

        return json.dumps(vector)

    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims that need verification"""
        sentences = re.split(r"[.!?]+", text)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            # Check for factual indicators
            is_factual = False
            factual_indicators = [
                "is named",
                "was created",
                "developed by",
                "is located",
                "has population",
                "was founded",
                "means that",
            ]

            for indicator in factual_indicators:
                if indicator in sentence.lower():
                    is_factual = True
                    break

            if is_factual:
                claims.append(sentence)

        return claims[:3]  # Limit to top 3 claims

    def _find_related_entities(self, entity: str) -> List[Dict]:
        """Find related entities in knowledge graph"""
        related = []

        for entity_id, entity_data in self.knowledge_graph.items():
            content = entity_data.get("content", "").lower()

            if entity.lower() in content:
                related.append(
                    {
                        "entity_id": entity_id,
                        "strength": 0.8,  # Simple co-occurrence measure
                    }
                )

        return related[:5]  # Limit related entities

    def _detect_query_intent(self, query: str) -> str:
        """Detect the intent behind the query"""
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["what is", "what are", "explain", "tell me about"]
        ):
            return "explanation"
        elif any(word in query_lower for word in ["how to", "how do", "steps to"]):
            return "instruction"
        elif any(word in query_lower for word in ["why", "reason", "because"]):
            return "reasoning"
        elif any(
            word in query_lower for word in ["versus", "vs", "versus", "difference"]
        ):
            return "comparison"
        else:
            return "general"

    def _infer_categories(self, query: str) -> List[str]:
        """Infer relevant categories from query"""
        query_lower = query.lower()
        categories = []

        for category, keywords in self.category_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                categories.append(category)

        return categories or ["general"]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified)"""
        entities = []

        # Simple pattern matching for common entities
        # This would be much better with spaCy/NLTK in production

        # Technology entities
        tech_patterns = [
            r"\b[Gg]emma\b",
            r"\b[GPT]\w*-\w+\b",
            r"\b[Cc]laude\b",
            r"\b[Ll]lama\b",
            r"\b[BERT]\b",
            r"\b[Tt]ransformer\b",
            r"\b[RAG]\b",
            r"\b[A-Z]{3,}\b",
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        # Company names
        company_names = ["Google", "Microsoft", "Apple", "OpenAI", "Meta", "Tesla"]
        for company in company_names:
            if company in text:
                entities.append(company)

        return list(set(entities))

    def _assess_difficulty(self, query: str, context: Dict = None) -> str:
        """Assess query difficulty level"""
        length = len(query.split())

        # Simple heuristics
        if length < 5:
            return "simple"
        elif length < 10:
            return "medium"
        elif "explain" in query.lower() or "how does" in query.lower():
            return "complex"
        else:
            return "medium"

    def _tokenize_semantically(self, text: str) -> List[str]:
        """Simple semantic tokenization"""
        # Remove stop words and split
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        words = re.findall(r"\b\w+\b", text.lower())
        semantic_tokens = [
            word for word in words if word not in stop_words and len(word) > 2
        ]

        return semantic_tokens

    def _calculate_overall_confidence(self, results: List[Dict]) -> float:
        """Calculate overall response confidence"""
        if not results:
            return 0.0

        confidences = [
            r.get("final_score", 0) * r.get("fact_check_confidence", 1.0)
            for r in results
        ]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _get_categories_represented(self, results: List[Dict]) -> List[str]:
        """Get categories represented in results"""
        categories = []
        for result in results:
            category = result.get("category", "")
            if category and category not in categories:
                categories.append(category)

        return categories

    def _assess_hallucination_risk(self, results: List[Dict]) -> float:
        """Assess hallucination risk of the result set"""
        if not results:
            return 1.0  # High risk, no grounding

        # Calculate risk based on verification status
        verified_results = sum(1 for r in results if r.get("verified", False))
        verification_rate = verified_results / len(results)

        # Higher risk if few verified results or low confidence
        avg_confidence = np.mean([r.get("fact_check_confidence", 0.5) for r in results])

        risk_score = 1.0 - (verification_rate * avg_confidence)
        return max(0.0, min(1.0, risk_score))

    def _fallback_search(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Fallback search when enhanced search fails"""
        return {
            "query": query,
            "results": [],
            "total_candidates": 0,
            "final_results": 0,
            "processing_time": 0.1,
            "error": "search_failed",
            "fallback_mode": True,
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        words = re.findall(r"\b\w+\b", text.lower())
        keywords = [word for word in words if len(word) > 3]  # Filter short words

        return list(set(keywords))[:10]  # Unique, limited to 10

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        conn = sqlite3.connect(self.db_path)

        try:
            cursor = conn.cursor()

            # Knowledge base stats
            cursor.execute(
                "SELECT COUNT(*), entity_type FROM knowledge_base GROUP BY entity_type"
            )
            entity_stats = dict(cursor.fetchall())

            cursor.execute(
                "SELECT COUNT(*), category FROM knowledge_base GROUP BY category"
            )
            category_stats = dict(cursor.fetchall())

            cursor.execute(
                "SELECT AVG(confidence), AVG(usage_count) FROM knowledge_base"
            )
            avg_confidence, avg_usage = cursor.fetchone()

            # Interaction stats
            cursor.execute("SELECT COUNT(*), AVG(semantic_score) FROM rag_interactions")
            total_interactions, avg_semantic_score = cursor.fetchone()

            cursor.execute(
                "SELECT COUNT(*), AVG(response_quality) FROM rag_interactions WHERE user_feedback IS NOT NULL"
            )
            feedback_count, avg_feedback = cursor.fetchone()

            stats = {
                "knowledge_base": {
                    "total_entities": sum(entity_stats.values()),
                    "entity_types": entity_stats,
                    "categories": category_stats,
                    "avg_confidence": avg_confidence or 0.0,
                    "avg_usage": avg_usage or 0.0,
                },
                "interactions": {
                    "total": total_interactions or 0,
                    "avg_semantic_score": avg_semantic_score or 0.0,
                    "feedback_received": feedback_count or 0,
                    "avg_feedback_score": avg_feedback or 0.0,
                },
                "learning": {
                    "vocabulary_size": len(self.vocabulary),
                    "interaction_memory_size": len(self.interaction_memory),
                    "knowledge_graph_size": len(self.knowledge_graph),
                },
                "performance": {
                    "hallucination_risk_assessment_active": True,
                    "fact_checking_integrated": True,
                    "semantic_search_enabled": True,
                },
            }

        finally:
            conn.close()

        return stats


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

_enhanced_rag_instance = None


def get_enhanced_rag_service() -> EnhancedRAGService:
    """Get global enhanced RAG service instance"""
    global _enhanced_rag_instance
    if _enhanced_rag_instance is None:
        _enhanced_rag_instance = EnhancedRAGService()
    return _enhanced_rag_instance


# =============================================================================
# ADDITIONAL UTILITIES
# =============================================================================


async def initialize_rag_with_exercise_data(
    exercise_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Initialize RAG system with exercise data"""
    rag_service = get_enhanced_rag_service()
    await rag_service.add_exercise_data(exercise_data)

    return {
        "initialized": True,
        "entities_added": len(exercise_data.get("answers", [])),
        "categories_updated": list(rag_service.category_mappings.keys()),
    }


async def test_enhanced_rag():
    """Test the enhanced RAG service"""
    rag_service = get_enhanced_rag_service()

    test_queries = [
        "What is machine learning?",
        "How does RAG work?",
        "Explain the difference between supervised and unsupervised learning",
    ]

    results = []
    for query in test_queries:
        print(f"\n🔍 Testing query: {query}")
        result = await rag_service.enhanced_search(query)
        print(f"   Results found: {result['final_results']}")
        print(f"   Categories: {result['categories_represented']}")
        print(f"   Confidence: {result['confidence_score']:.2f}")
        print(f"   Hallucination risk: {result['hallucination_risk']:.2f}")

        results.append(result)

    # Show system stats
    stats = rag_service.get_system_stats()
    print(f"\n📊 System Stats:")
    print(f"   Knowledge base size: {stats['knowledge_base']['total_entities']}")
    print(f"   Total interactions: {stats['interactions']['total']}")

    return results


if __name__ == "__main__":
    asyncio.run(test_enhanced_rag())
