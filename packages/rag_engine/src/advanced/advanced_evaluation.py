#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced RAG Evaluation Metrics
Based on RAGAS, FactScore, and other evaluation frameworks

Implements comprehensive evaluation suite:
- RAGAS metrics (Faithfulness, Context Relevancy, Answer Relevancy, Answer Correctness)
- FactScore for factual precision
- ROUGE, BLEU, BERTScore for text quality
- Custom metrics for RAG-specific evaluation
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


@dataclass
class RAGASResult:
    """RAGAS evaluation result"""

    faithfulness: float
    context_relevancy: float
    answer_relevancy: float
    answer_correctness: float
    overall_score: float
    metadata: Dict[str, Any]


@dataclass
class FactScoreResult:
    """FactScore evaluation result"""

    atomic_facts: List[str]
    fact_scores: List[float]
    overall_score: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class TextQualityResult:
    """Text quality evaluation result"""

    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    bleu_score: float
    bert_score: float
    semantic_similarity: float


@dataclass
class ComprehensiveEvaluationResult:
    """Complete evaluation result"""

    ragas: RAGASResult
    factscore: FactScoreResult
    text_quality: TextQualityResult
    custom_metrics: Dict[str, float]
    overall_score: float
    processing_time: float


class RAGASEvaluator:
    """
    RAGAS (Retrieval-Augmented Generation Assessment) Evaluator

    Implements the four main RAGAS metrics:
    - Faithfulness: Does the answer contradict any retrieved context?
    - Context Relevancy: How relevant is the retrieved context to the question?
    - Answer Relevancy: How relevant is the answer to the question?
    - Answer Correctness: How correct is the answer given the ground truth?
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize RAGAS evaluator

        Args:
            model_name: Model for semantic evaluation
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.sentence_encoder = None

        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                print("Warning: Could not load evaluation models")

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate faithfulness: Does the answer contradict retrieved contexts?

        Args:
            answer: Generated answer
            contexts: Retrieved context documents

        Returns:
            Faithfulness score (0-1)
        """
        if not contexts:
            return 0.0

        # Break answer into claims
        claims = self._extract_claims(answer)

        faithful_claims = 0
        total_claims = len(claims)

        if total_claims == 0:
            return 1.0  # No claims to contradict

        combined_context = " ".join(contexts)

        for claim in claims:
            if self._claim_supported_by_context(claim, combined_context):
                faithful_claims += 1

        return faithful_claims / total_claims

    def evaluate_context_relevancy(self, question: str, contexts: List[str]) -> float:
        """
        Evaluate context relevancy: How relevant is context to question?

        Args:
            question: User question
            contexts: Retrieved context documents

        Returns:
            Context relevancy score (0-1)
        """
        if not contexts:
            return 0.0

        relevant_sentences = 0
        total_sentences = 0

        for context in contexts:
            sentences = self._split_into_sentences(context)
            total_sentences += len(sentences)

            for sentence in sentences:
                if self._sentence_relevant_to_question(sentence, question):
                    relevant_sentences += 1

        return relevant_sentences / total_sentences if total_sentences > 0 else 0.0

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate answer relevancy: How relevant is answer to question?

        Args:
            question: User question
            answer: Generated answer

        Returns:
            Answer relevancy score (0-1)
        """
        # Generate synthetic questions from answer
        synthetic_questions = self._generate_synthetic_questions(answer)

        if not synthetic_questions:
            return 0.0

        relevant_questions = 0
        for syn_question in synthetic_questions:
            if self._questions_similar(syn_question, question):
                relevant_questions += 1

        return relevant_questions / len(synthetic_questions)

    def evaluate_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Evaluate answer correctness against ground truth

        Args:
            answer: Generated answer
            ground_truth: Ground truth answer

        Returns:
            Correctness score (0-1)
        """
        # Factual correctness
        factual_score = self._evaluate_factual_correctness(answer, ground_truth)

        # Semantic similarity
        semantic_score = self._evaluate_semantic_similarity(answer, ground_truth)

        # Combine scores
        return (factual_score + semantic_score) / 2

    def evaluate_all(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> RAGASResult:
        """
        Evaluate all RAGAS metrics

        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Ground truth answer

        Returns:
            Complete RAGAS evaluation
        """
        faithfulness = self.evaluate_faithfulness(answer, contexts)
        context_relevancy = self.evaluate_context_relevancy(question, contexts)
        answer_relevancy = self.evaluate_answer_relevancy(question, answer)

        answer_correctness = 0.0
        if ground_truth:
            answer_correctness = self.evaluate_answer_correctness(answer, ground_truth)

        # Overall score (weighted average)
        overall_score = (
            0.25 * faithfulness
            + 0.25 * context_relevancy
            + 0.25 * answer_relevancy
            + 0.25 * answer_correctness
        )

        return RAGASResult(
            faithfulness=faithfulness,
            context_relevancy=context_relevancy,
            answer_relevancy=answer_relevancy,
            answer_correctness=answer_correctness,
            overall_score=overall_score,
            metadata={
                "num_contexts": len(contexts),
                "answer_length": len(answer.split()),
                "question_length": len(question.split()),
            },
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract atomic claims from text"""
        # Simple claim extraction based on sentences
        sentences = self._split_into_sentences(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _claim_supported_by_context(self, claim: str, context: str) -> bool:
        """Check if claim is supported by context"""
        # Simple keyword overlap check
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())

        overlap = len(claim_words.intersection(context_words))
        return overlap / len(claim_words) > 0.3 if claim_words else False

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        return re.split(r"[.!?]+", text)

    def _sentence_relevant_to_question(self, sentence: str, question: str) -> bool:
        """Check if sentence is relevant to question"""
        sentence_words = set(sentence.lower().split())
        question_words = set(question.lower().split())

        overlap = len(sentence_words.intersection(question_words))
        return overlap > 0

    def _generate_synthetic_questions(self, answer: str) -> List[str]:
        """Generate synthetic questions from answer"""
        # Simple question generation
        sentences = self._split_into_sentences(answer)
        questions = []

        for sentence in sentences:
            if sentence.strip():
                # Convert statement to question
                question = sentence.strip()
                if not question.endswith("?"):
                    question = f"What is {question.lower()}?"
                questions.append(question)

        return questions[:3]  # Limit to 3 questions

    def _questions_similar(self, q1: str, q2: str) -> bool:
        """Check if two questions are similar"""
        q1_words = set(q1.lower().split())
        q2_words = set(q2.lower().split())

        overlap = len(q1_words.intersection(q2_words))
        union = len(q1_words.union(q2_words))

        return overlap / union > 0.5 if union > 0 else False

    def _evaluate_factual_correctness(self, answer: str, ground_truth: str) -> float:
        """Evaluate factual correctness"""
        # Simple fact extraction and comparison
        answer_facts = self._extract_facts(answer)
        truth_facts = self._extract_facts(ground_truth)

        if not truth_facts:
            return 1.0

        correct_facts = 0
        for truth_fact in truth_facts:
            for answer_fact in answer_facts:
                if self._facts_match(truth_fact, answer_fact):
                    correct_facts += 1
                    break

        return correct_facts / len(truth_facts)

    def _evaluate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Evaluate semantic similarity between texts"""
        if self.sentence_encoder:
            try:
                embeddings = self.sentence_encoder.encode([text1, text2])
                similarity = self._cosine_similarity(embeddings[0], embeddings[1])
                return max(0, min(1, similarity))  # Clamp to [0,1]
            except:
                pass

        # Fallback to simple overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text"""
        sentences = self._split_into_sentences(text)
        return [s.strip() for s in sentences if self._is_factual_statement(s)]

    def _is_factual_statement(self, sentence: str) -> bool:
        """Check if sentence appears to be a factual statement"""
        # Simple heuristics for factual statements
        sentence = sentence.lower().strip()
        return len(sentence) > 10 and not any(
            word in sentence for word in ["maybe", "perhaps", "i think"]
        )

    def _facts_match(self, fact1: str, fact2: str) -> bool:
        """Check if two facts match"""
        return self._evaluate_semantic_similarity(fact1, fact2) > 0.8

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class FactScoreEvaluator:
    """
    FactScore Evaluator for fine-grained factual precision

    Breaks down responses into atomic facts and verifies each against
    reliable sources (simulated here with context matching)
    """

    def __init__(self):
        self.nlp_model = None
        if TRANSFORMERS_AVAILABLE:
            try:
                from transformers import pipeline

                self.nlp_model = pipeline(
                    "text-classification", model="microsoft/DialoGPT-small"
                )
            except:
                pass

    def evaluate_facts(
        self, answer: str, contexts: List[str], ground_truth: Optional[str] = None
    ) -> FactScoreResult:
        """
        Evaluate factual precision using FactScore methodology

        Args:
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Ground truth for comparison

        Returns:
            FactScore evaluation result
        """
        # Extract atomic facts from answer
        atomic_facts = self._extract_atomic_facts(answer)

        if not atomic_facts:
            return FactScoreResult(
                atomic_facts=[],
                fact_scores=[],
                overall_score=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
            )

        # Score each fact
        fact_scores = []
        combined_context = " ".join(contexts)

        for fact in atomic_facts:
            score = self._score_atomic_fact(fact, combined_context, ground_truth)
            fact_scores.append(score)

        # Calculate aggregate metrics
        overall_score = sum(fact_scores) / len(fact_scores)

        # Precision, Recall, F1 against ground truth
        precision, recall, f1 = self._calculate_precision_recall_f1(
            atomic_facts, fact_scores, ground_truth
        )

        return FactScoreResult(
            atomic_facts=atomic_facts,
            fact_scores=fact_scores,
            overall_score=overall_score,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )

    def _extract_atomic_facts(self, text: str) -> List[str]:
        """Extract atomic facts from text"""
        sentences = re.split(r"[.!?]+", text)
        atomic_facts = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            # Break down complex sentences into atomic facts
            facts = self._decompose_sentence(sentence)
            atomic_facts.extend(facts)

        return atomic_facts

    def _decompose_sentence(self, sentence: str) -> List[str]:
        """Decompose complex sentence into atomic facts"""
        # Simple decomposition based on clauses
        facts = []

        # Split on conjunctions
        parts = re.split(r"\s+(and|or|but|however|therefore)\s+", sentence)

        if len(parts) > 1:
            for part in parts:
                if len(part.strip()) > 10:
                    facts.append(part.strip())
        else:
            facts.append(sentence)

        return facts

    def _score_atomic_fact(
        self, fact: str, context: str, ground_truth: Optional[str] = None
    ) -> float:
        """
        Score an atomic fact against context and ground truth

        Returns score from 0-1 where 1 is fully supported
        """
        context_score = self._fact_supported_by_context(fact, context)

        if ground_truth:
            truth_score = self._fact_supported_by_context(fact, ground_truth)
            # Combine context and truth scores
            return (context_score + truth_score) / 2
        else:
            return context_score

    def _fact_supported_by_context(self, fact: str, context: str) -> float:
        """Check if fact is supported by context"""
        if not context:
            return 0.0

        # Multiple verification methods
        lexical_overlap = self._lexical_overlap_score(fact, context)
        semantic_similarity = self._semantic_similarity_score(fact, context)
        entailment_score = self._entailment_score(fact, context)

        # Weighted combination
        return (
            0.4 * lexical_overlap + 0.4 * semantic_similarity + 0.2 * entailment_score
        )

    def _lexical_overlap_score(self, fact: str, context: str) -> float:
        """Calculate lexical overlap score"""
        fact_words = set(fact.lower().split())
        context_words = set(context.lower().split())

        if not fact_words:
            return 0.0

        overlap = len(fact_words.intersection(context_words))
        return overlap / len(fact_words)

    def _semantic_similarity_score(self, fact: str, context: str) -> float:
        """Calculate semantic similarity score"""
        # Simple semantic matching based on word embeddings
        fact_tokens = fact.lower().split()
        context_lower = context.lower()

        matches = 0
        for token in fact_tokens:
            if token in context_lower:
                matches += 1

        return matches / len(fact_tokens) if fact_tokens else 0.0

    def _entailment_score(self, fact: str, context: str) -> float:
        """Calculate entailment score (simplified)"""
        # Check if key entities and relations are present
        fact_lower = fact.lower()
        context_lower = context.lower()

        # Look for key phrases
        key_phrases = self._extract_key_phrases(fact)
        matches = sum(1 for phrase in key_phrases if phrase in context_lower)

        return matches / len(key_phrases) if key_phrases else 0.0

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple key phrase extraction
        words = text.split()
        phrases = []

        # Unigrams and bigrams
        for i in range(len(words)):
            phrases.append(words[i].lower())
            if i < len(words) - 1:
                phrases.append(f"{words[i]} {words[i+1]}".lower())

        return phrases

    def _calculate_precision_recall_f1(
        self, facts: List[str], fact_scores: List[float], ground_truth: Optional[str]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1 against ground truth"""
        if not ground_truth:
            return 0.0, 0.0, 0.0

        truth_facts = self._extract_atomic_facts(ground_truth)

        if not truth_facts:
            return 0.0, 0.0, 0.0

        # Count true positives (correctly identified facts)
        tp = sum(1 for score in fact_scores if score > 0.5)

        # Precision: fraction of predicted facts that are correct
        precision = tp / len(facts) if facts else 0.0

        # Recall: fraction of true facts that were identified
        # (simplified: assume we found some correct facts)
        recall = min(tp / len(truth_facts), 1.0) if truth_facts else 0.0

        # F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1


class TextQualityEvaluator:
    """
    Text Quality Evaluator using ROUGE, BLEU, BERTScore
    """

    def __init__(self):
        self.rouge_scorer = None
        self.sentence_encoder = None

        if NLTK_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
            except:
                pass

        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                pass

    def evaluate_quality(
        self, generated_text: str, reference_text: str
    ) -> TextQualityResult:
        """
        Evaluate text quality using multiple metrics

        Args:
            generated_text: Generated answer
            reference_text: Reference/gold answer

        Returns:
            Text quality evaluation
        """
        # ROUGE scores
        rouge1_f1 = 0.0
        rouge2_f1 = 0.0
        rougeL_f1 = 0.0

        if self.rouge_scorer:
            try:
                scores = self.rouge_scorer.score(reference_text, generated_text)
                rouge1_f1 = scores["rouge1"].fmeasure
                rouge2_f1 = scores["rouge2"].fmeasure
                rougeL_f1 = scores["rougeL"].fmeasure
            except:
                pass

        # BLEU score
        bleu_score = 0.0
        if NLTK_AVAILABLE:
            try:
                smoothing = SmoothingFunction().method4
                reference_tokens = [reference_text.split()]
                generated_tokens = generated_text.split()
                bleu_score = sentence_bleu(
                    reference_tokens, generated_tokens, smoothing_function=smoothing
                )
            except:
                pass

        # BERTScore (simplified as semantic similarity)
        bert_score = self._calculate_semantic_similarity(generated_text, reference_text)

        # Overall semantic similarity
        semantic_similarity = bert_score

        return TextQualityResult(
            rouge1_f1=rouge1_f1,
            rouge2_f1=rouge2_f1,
            rougeL_f1=rougeL_f1,
            bleu_score=bleu_score,
            bert_score=bert_score,
            semantic_similarity=semantic_similarity,
        )

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        if self.sentence_encoder:
            try:
                embeddings = self.sentence_encoder.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return max(0, min(1, similarity))
            except:
                pass

        # Fallback to simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


class ComprehensiveRAGEvaluator:
    """
    Comprehensive RAG Evaluator combining all metrics
    """

    def __init__(self):
        self.ragas_evaluator = RAGASEvaluator()
        self.factscore_evaluator = FactScoreEvaluator()
        self.quality_evaluator = TextQualityEvaluator()

    def evaluate_comprehensive(
        self,
        question: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> ComprehensiveEvaluationResult:
        """
        Perform comprehensive RAG evaluation

        Args:
            question: User question
            generated_answer: Generated answer
            retrieved_contexts: Retrieved context documents
            ground_truth: Ground truth answer

        Returns:
            Comprehensive evaluation result
        """
        import time

        start_time = time.time()

        # RAGAS evaluation
        ragas_result = self.ragas_evaluator.evaluate_all(
            question, generated_answer, retrieved_contexts, ground_truth
        )

        # FactScore evaluation
        factscore_result = self.factscore_evaluator.evaluate_facts(
            generated_answer, retrieved_contexts, ground_truth
        )

        # Text quality evaluation
        text_quality_result = TextQualityResult(
            rouge1_f1=0.0,
            rouge2_f1=0.0,
            rougeL_f1=0.0,
            bleu_score=0.0,
            bert_score=0.0,
            semantic_similarity=0.0,
        )

        if ground_truth:
            text_quality_result = self.quality_evaluator.evaluate_quality(
                generated_answer, ground_truth
            )

        # Custom metrics
        custom_metrics = self._calculate_custom_metrics(
            question, generated_answer, retrieved_contexts, ground_truth
        )

        # Overall score (weighted combination)
        overall_score = self._calculate_overall_score(
            ragas_result, factscore_result, text_quality_result, custom_metrics
        )

        processing_time = time.time() - start_time

        return ComprehensiveEvaluationResult(
            ragas=ragas_result,
            factscore=factscore_result,
            text_quality=text_quality_result,
            custom_metrics=custom_metrics,
            overall_score=overall_score,
            processing_time=processing_time,
        )

    def _calculate_custom_metrics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str],
    ) -> Dict[str, float]:
        """Calculate custom RAG-specific metrics"""
        metrics = {}

        # Context utilization efficiency
        if contexts:
            total_context_words = sum(len(ctx.split()) for ctx in contexts)
            answer_words = len(answer.split())
            metrics["context_utilization"] = (
                min(1.0, answer_words / total_context_words)
                if total_context_words > 0
                else 0.0
            )

        # Answer conciseness
        question_words = len(question.split())
        answer_words = len(answer.split())
        metrics["answer_conciseness"] = (
            min(1.0, question_words / answer_words) if answer_words > 0 else 0.0
        )

        # Factual density
        sentences = re.split(r"[.!?]+", answer)
        factual_sentences = sum(1 for s in sentences if self._is_factual_sentence(s))
        metrics["factual_density"] = (
            factual_sentences / len(sentences) if sentences else 0.0
        )

        # Novelty (information not in question)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        novel_words = answer_words - question_words
        metrics["novelty"] = (
            len(novel_words) / len(answer_words) if answer_words else 0.0
        )

        return metrics

    def _is_factual_sentence(self, sentence: str) -> bool:
        """Check if sentence appears factual"""
        sentence = sentence.lower().strip()
        if len(sentence) < 5:
            return False

        # Check for factual indicators
        factual_indicators = [
            "is",
            "was",
            "are",
            "were",
            "has",
            "have",
            "had",
            "does",
            "do",
            "did",
        ]
        words = sentence.split()

        return any(indicator in words for indicator in factual_indicators)

    def _calculate_overall_score(
        self,
        ragas: RAGASResult,
        factscore: FactScoreResult,
        quality: TextQualityResult,
        custom: Dict[str, float],
    ) -> float:
        """Calculate overall evaluation score"""
        # Weighted combination of all metrics
        weights = {"ragas": 0.4, "factscore": 0.3, "quality": 0.2, "custom": 0.1}

        ragas_score = ragas.overall_score
        factscore_score = factscore.overall_score
        quality_score = (quality.rougeL_f1 + quality.bert_score) / 2
        custom_score = sum(custom.values()) / len(custom) if custom else 0.0

        overall = (
            weights["ragas"] * ragas_score
            + weights["factscore"] * factscore_score
            + weights["quality"] * quality_score
            + weights["custom"] * custom_score
        )

        return max(0.0, min(1.0, overall))  # Clamp to [0,1]

    def get_evaluation_summary(
        self, results: List[ComprehensiveEvaluationResult]
    ) -> Dict[str, Any]:
        """Get summary statistics across multiple evaluations"""
        if not results:
            return {}

        summary = {
            "num_evaluations": len(results),
            "avg_overall_score": sum(r.overall_score for r in results) / len(results),
            "avg_ragas_score": sum(r.ragas.overall_score for r in results)
            / len(results),
            "avg_factscore": sum(r.factscore.overall_score for r in results)
            / len(results),
            "avg_faithfulness": sum(r.ragas.faithfulness for r in results)
            / len(results),
            "avg_context_relevancy": sum(r.ragas.context_relevancy for r in results)
            / len(results),
            "avg_answer_relevancy": sum(r.ragas.answer_relevancy for r in results)
            / len(results),
            "avg_answer_correctness": sum(r.ragas.answer_correctness for r in results)
            / len(results),
            "avg_rougeL": sum(r.text_quality.rougeL_f1 for r in results) / len(results),
            "avg_bleu": sum(r.text_quality.bleu_score for r in results) / len(results),
            "avg_bert_score": sum(r.text_quality.bert_score for r in results)
            / len(results),
            "avg_processing_time": sum(r.processing_time for r in results)
            / len(results),
        }

        return summary
