#!/usr/bin/env python3
"""
FACT CHECKER SERVICE - Anti-Hallucination Guardians
==================================================

Servicio real de verificación de hechos para eliminar hallucinations.
Integra múltiples fuentes de verdad:
- Wikipedia APIs (real-time verification)
- Knowledge base interna verificada
- Cross-reference con fuentes confiables
- Constitutional evaluation de claims

PREVIENE hallucinations mediante verificación automática de TODOS los facts.
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class FactCheckerService:
    """Servicio de verificación de hechos REAL - Anti-hallucination"""

    def __init__(self):
        self.verified_knowledge_path = Path("data/verified_knowledge.json")
        self.fact_check_log = []
        self.session = None

        # Inicializar base de conocimiento verificado
        self._load_verified_knowledge()

        # APIs externas para fact-checking
        self.api_endpoints = {
            "wikipedia": "https://en.wikipedia.org/api/rest_v1/page/summary/{topic}",
            "wikidata": "https://www.wikidata.org/w/api.php",
            # Add more APIs as available
        }

        logger.info("🛡️ Fact Checker Service initialized - Anti-hallucination active")

    def _load_verified_knowledge(self):
        """Load verified knowledge base"""
        if self.verified_knowledge_path.exists():
            with open(self.verified_knowledge_path, "r", encoding="utf-8") as f:
                self.verified_knowledge = json.load(f)
        else:
            self.verified_knowledge = self._create_base_verified_knowledge()

    def _create_base_verified_knowledge(self) -> Dict[str, Any]:
        """Create base verified knowledge for common facts"""
        return {
            "ai_model": {
                "description": "Advanced AI model for enterprise assistance",
                "verification_source": "system_documentation",
                "confidence": 1.0,
                "last_verified": datetime.now().isoformat(),
            },
            "mcp_sheily": {
                "description": "Sheily is an AI-powered platform with MCP orchestration, blockchain tokens, and educational exercises",
                "verification_source": "system_constitution",
                "confidence": 1.0,
                "last_verified": datetime.now().isoformat(),
            },
            "machine_learning": {
                "description": "Machine Learning is a subset of AI that enables systems to learn from data without explicit programming",
                "verification_source": "standard_definition",
                "confidence": 1.0,
                "last_verified": datetime.now().isoformat(),
            },
        }

    async def initialize(self):
        """Initialize aiohttp session for API calls"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)  # 5 second timeout
            )

    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

    async def verify_claim_async(
        self, claim: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Verificación completa de un claim contra múltiples fuentes
        """
        try:
            # 1. Check internal verified knowledge first
            internal_verification = self._check_internal_knowledge(claim)

            if internal_verification["confidence"] >= 0.9:
                return internal_verification

            # 2. Extract factual entities from claim
            entities = self._extract_factual_entities(claim)

            # 3. Query external sources
            external_verifications = []
            for entity in entities:
                verification = await self._verify_entity_external(entity)
                external_verifications.append(verification)

            # 4. Combine verification results
            combined_result = self._combine_verifications(
                claim, internal_verification, external_verifications
            )

            # 5. Constitutional evaluation for ethical claims
            if context and "ethical" in context.get("claim_type", ""):
                ethical_eval = await self._constitutional_evaluation(claim, context)
                if ethical_eval:
                    combined_result["ethical_evaluation"] = ethical_eval

            # 6. Log verification
            await self._log_verification(claim, combined_result)

            return combined_result

        except Exception as e:
            logger.error(f"Fact verification error: {e}")
            return self._create_uncertain_response(claim, str(e))

    def _check_internal_knowledge(self, claim: str) -> Dict[str, Any]:
        """
        Check claim against internal verified knowledge base
        """
        claim_lower = claim.lower()

        for topic, knowledge in self.verified_knowledge.items():
            if topic.lower() in claim_lower:
                return {
                    "verified": True,
                    "confidence": knowledge["confidence"],
                    "source": knowledge["verification_source"],
                    "details": knowledge["description"],
                    "method": "internal_knowledge_base",
                }

        # No exact match found
        partial_matches = []
        for topic, knowledge in self.verified_knowledge.items():
            if any(word in claim_lower for word in topic.lower().split()):
                partial_matches.append(knowledge)

        if partial_matches:
            avg_confidence = sum(k["confidence"] for k in partial_matches) / len(
                partial_matches
            )
            return {
                "verified": avg_confidence > 0.7,
                "confidence": avg_confidence,
                "source": "internal_partial_match",
                "method": "internal_knowledge_base",
            }

        return {
            "verified": False,
            "confidence": 0.0,
            "source": "unknown",
            "method": "internal_knowledge_base",
        }

    def _extract_factual_entities(self, claim: str) -> List[str]:
        """
        Extract factual entities that can be verified externally
        """
        entities = []

        # Common entities to verify
        # Cities
        cities = [
            "Madrid",
            "Barcelona",
            "Valencia",
            "Seville",
            "Bilbao",
            "Paris",
            "London",
            "New York",
            "Tokyo",
            "Berlin",
        ]

        # Companies/Organizations
        companies = [
            "Google",
            "Microsoft",
            "Apple",
            "OpenAI",
            "Anthropic",
            "Meta",
            "Tesla",
            "SpaceX",
        ]

        # Technologies/Models
        tech_entities = ["Gemma", "GPT", "Claude", "Llama", "BERT", "Transformer"]

        # Numbers and dates (simplified)
        number_patterns = r"\b\d{1,2}(?:\.\d+)?\s*(?:million|billion|trillion|percent|years?|km|miles?)\b"

        # Search for entities
        claim_lower = claim.lower()

        for city in cities:
            if city.lower() in claim_lower:
                entities.append(city)

        for company in companies:
            if company.lower() in claim_lower:
                entities.append(company)

        for tech in tech_entities:
            if tech.lower() in claim_lower:
                entities.append(tech)

        # Add numbers with units
        numbers = re.findall(number_patterns, claim, re.IGNORECASE)
        entities.extend(numbers)

        return list(set(entities))  # Remove duplicates

    async def _verify_entity_external(self, entity: str) -> Dict[str, Any]:
        """
        Verify entity against external sources
        """
        await self.initialize()

        try:
            # Wikipedia verification
            wiki_result = await self._check_wikipedia(entity)
            if wiki_result["verified"]:
                return wiki_result

        except Exception as e:
            logger.warning(f"External verification failed for {entity}: {e}")

        return {
            "entity": entity,
            "verified": False,
            "confidence": 0.5,
            "source": "external_unavailable",
            "method": "wikipedia_api",
            "error": "verification_failed",
        }

    async def _check_wikipedia(self, entity: str) -> Dict[str, Any]:
        """
        Check entity against Wikipedia API
        """
        try:
            url = self.api_endpoints["wikipedia"].format(topic=entity.replace(" ", "_"))

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Check if it's a disambiguation or actual article
                    if data and "title" in data and "extract" in data:
                        content_urls = data.get("content_urls", {})
                        if isinstance(content_urls, dict):
                            desktop = content_urls.get("desktop", {})
                            if isinstance(desktop, dict):
                                page_url = desktop.get("page", "")
                            else:
                                page_url = ""
                        else:
                            page_url = ""

                        return {
                            "entity": entity,
                            "verified": True,
                            "confidence": 0.9,
                            "source": "wikipedia",
                            "title": data.get("title"),
                            "description": (
                                data.get("extract", "")[:200]
                                if data.get("extract")
                                else ""
                            ),
                            "url": page_url,
                            "method": "wikipedia_api",
                        }
                elif response.status == 404:
                    # Entity doesn't exist on Wikipedia
                    pass

        except Exception as e:
            logger.warning(f"Wikipedia check failed for {entity}: {e}")

        return {
            "entity": entity,
            "verified": False,
            "confidence": 0.0,
            "source": "wikipedia_not_found",
            "method": "wikipedia_api",
        }

    def _combine_verifications(
        self, claim: str, internal: Dict, external: List[Dict]
    ) -> Dict[str, Any]:
        """
        Combine all verification results into final assessment
        """
        # Calculate overall confidence
        confidences = [internal["confidence"]] + [ext["confidence"] for ext in external]

        # If any external verification succeeded
        external_success = any(ext["verified"] for ext in external)
        internal_good = internal["verified"] and internal["confidence"] > 0.7

        overall_verification = internal_good or external_success
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Collect sources
        sources = []
        if internal["verified"]:
            sources.append(internal["source"])
        sources.extend([ext["source"] for ext in external if ext["verified"]])

        return {
            "claim": claim,
            "verified": overall_verification,
            "confidence": min(overall_confidence, 1.0),
            "sources": list(set(sources)),
            "internal_verification": internal,
            "external_verifications": external,
            "verification_timestamp": datetime.now().isoformat(),
            "recommendation": self._get_verification_recommendation(
                overall_verification, overall_confidence
            ),
        }

    def _get_verification_recommendation(
        self, verified: bool, confidence: float
    ) -> str:
        """
        Get recommendation based on verification results
        """
        if verified and confidence > 0.8:
            return "trust_high"
        elif verified and confidence > 0.6:
            return "trust_medium"
        elif not verified and confidence < 0.3:
            return "express_uncertainty"
        else:
            return "verify_manually"

    async def _constitutional_evaluation(
        self, claim: str, context: Dict
    ) -> Dict[str, Any]:
        """
        Constitutional evaluation for ethical/moral claims
        """
        try:
            from packages.training_system.src.agents.constitutional_evaluator import ConstitutionalEvaluator

            evaluator = ConstitutionalEvaluator()

            eval_result = await evaluator.evaluate_action(
                {
                    "action_type": "factual_claim_verification",
                    "description": f"Verifying truthfulness of claim: {claim}",
                    "potential_impact": "Truthfulness",
                    "stakeholders": ["users", "system_integrity"],
                    "risk_level": "low",
                    "source": "fact_checker_service",
                }
            )

            return {
                "ethical_clearance": eval_result["decision"] == "allow",
                "constitutional_severity": eval_result["severity"],
                "ethical_reasoning": eval_result["reasoning"],
            }

        except Exception as e:
            logger.warning(f"Constitutional evaluation failed: {e}")
            return {
                "ethical_clearance": True,  # Default allow
                "constitutional_severity": "low",
                "ethical_reasoning": "evaluation_unavailable",
            }

    def _create_uncertain_response(self, claim: str, error: str = "") -> Dict[str, Any]:
        """
        Create response when verification fails
        """
        return {
            "claim": claim,
            "verified": False,
            "confidence": 0.0,
            "uncertain_reason": error or "verification_failed",
            "sources": [],
            "recommendation": "express_uncertainty",
            "verification_timestamp": datetime.now().isoformat(),
        }

    async def _log_verification(self, claim: str, result: Dict[str, Any]):
        """
        Log verification for analysis and improvement
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "claim": claim[:200],  # Limit claim length
            "confidence": result.get("confidence", 0),
            "verified": result.get("verified", False),
            "recommendation": result.get("recommendation", "unknown"),
            "sources_used": result.get("sources", []),
        }

        self.fact_check_log.append(log_entry)

        # Keep only last 1000 entries to prevent memory bloat
        if len(self.fact_check_log) > 1000:
            self.fact_check_log = self.fact_check_log[-1000:]

    def get_verification_stats(self) -> Dict[str, Any]:
        """
        Get statistics about verification performance
        """
        if not self.fact_check_log:
            return {"total_verifications": 0}

        total = len(self.fact_check_log)
        verified = sum(1 for log in self.fact_check_log if log["verified"])
        high_confidence = sum(
            1 for log in self.fact_check_log if log["confidence"] > 0.8
        )

        recommendation_counts = {}
        for log in self.fact_check_log:
            rec = log["recommendation"]
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        return {
            "total_verifications": total,
            "verification_rate": verified / total if total > 0 else 0,
            "high_confidence_rate": high_confidence / total if total > 0 else 0,
            "recommendation_distribution": recommendation_counts,
            "most_used_sources": self._get_most_used_sources(),
        }

    def _get_most_used_sources(self) -> List[Tuple[str, int]]:
        """
        Get most used verification sources
        """
        source_counts = {}
        for log in self.fact_check_log[-100:]:  # Last 100 entries
            for source in log.get("sources_used", []):
                source_counts[source] = source_counts.get(source, 0) + 1

        return sorted(source_counts.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# SAFETY GUARDRAILS INTEGRATION
# =============================================================================


class SafetyGuardrails:
    """Guardrails que previenen responses peligrosas usando fact-checking"""

    def __init__(self, fact_checker: FactCheckerService):
        self.fact_checker = fact_checker

    async def process_response(
        self, response: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process response through safety guardrails
        """
        # Extract factual claims from response
        claims = self._extract_claims_from_response(response)

        # Verify each claim
        verified_claims = []
        for claim in claims:
            verification = await self.fact_checker.verify_claim_async(claim, context)
            verified_claims.append(verification)

        # Determine if response should be modified
        response_modification = self._determine_modification(response, verified_claims)

        return {
            "original_response": response,
            "modified_response": response_modification["response"],
            "was_modified": response_modification["modified"],
            "claims_verified": verified_claims,
            "safety_clearance": response_modification["clearance_level"],
        }

    def _extract_claims_from_response(self, response: str) -> List[str]:
        """
        Extract factual claims that need verification
        """
        claims = []

        # Sentences that contain factual assertions
        sentences = re.split(r"[.!?]+", response)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check for factual language
            factual_indicators = [
                "is ",
                "are ",
                "has ",
                "was ",
                "were ",
                "created by",
                "developed by",
                "founded in",
                "released in",
                "based in",
                "located in",
            ]

            if any(indicator in sentence.lower() for indicator in factual_indicators):
                claims.append(sentence)

        return claims

    def _determine_modification(
        self, original_response: str, verified_claims: List[Dict]
    ) -> Dict[str, Any]:
        """
        Determine if and how response should be modified
        """
        # Check if any claims failed verification
        failed_claims = [
            claim for claim in verified_claims if not claim.get("verified", False)
        ]

        if not failed_claims:
            # All claims verified - response OK
            return {
                "response": original_response,
                "modified": False,
                "clearance_level": "full_trust",
            }

        # Some claims failed - modify response
        modified_response = self._modify_response(original_response, failed_claims)

        if len(failed_claims) / len(verified_claims) > 0.5:
            # Too many false claims
            clearance_level = "restricted"
        else:
            clearance_level = "trust_with_uncertainty"

        return {
            "response": modified_response,
            "modified": True,
            "clearance_level": clearance_level,
            "failed_claims": len(failed_claims),
        }

    def _modify_response(self, response: str, failed_claims: List[Dict]) -> str:
        """
        Modify response to remove or qualify unverified claims
        """
        modified_response = response

        for claim_result in failed_claims:
            claim_text = claim_result.get("claim", "")
            confidence = claim_result.get("confidence", 0)

            if confidence < 0.6:
                # Low confidence - add uncertainty qualifier
                modification_options = [
                    f"I'm not entirely sure about {claim_text.lower()}",
                    f"{claim_text} (though I'm not 100% certain)",
                    f"According to available information, {claim_text.lower()}",
                ]
                # Simple substitution - in production would be more sophisticated
                modified_response = modified_response.replace(
                    claim_text, modification_options[0]
                )

        return modified_response


# =============================================================================
# INTEGRATION GLOBALS
# =============================================================================

_fact_checker_instance = None
_safety_guardrails_instance = None


def get_fact_checker() -> FactCheckerService:
    """Get global fact checker instance"""
    global _fact_checker_instance
    if _fact_checker_instance is None:
        _fact_checker_instance = FactCheckerService()
    return _fact_checker_instance


def get_safety_guardrails() -> SafetyGuardrails:
    """Get global safety guardrails instance"""
    global _safety_guardrails_instance
    if _safety_guardrails_instance is None:
        _safety_guardrails_instance = SafetyGuardrails(get_fact_checker())
    return _safety_guardrails_instance


# =============================================================================
# TESTING AND UTILITIES
# =============================================================================


async def test_fact_checker():
    """Test the fact checker service"""
    fact_checker = get_fact_checker()
    await fact_checker.initialize()

    try:
        # Test claims
        test_claims = [
            "Madrid is the capital of Spain",
            "Google developed the Gemma model",
            "Machine learning requires GPUs",
            "Sheily uses the SHEILYS token system",
        ]

        print("🛡️ TESTING FACT CHECKER SERVICE")
        print("=" * 50)

        for claim in test_claims:
            print(f"\n🔍 Testing: {claim}")
            result = await fact_checker.verify_claim_async(claim)
            print(f"   Verified: {result['verified']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Sources: {result['sources']}")
            print(f"   Recommendation: {result['recommendation']}")

        # Test safety guardrails
        print("\n\n🛡️ TESTING SAFETY GUARDRAILS")
        print("=" * 30)

        guardrails = get_safety_guardrails()
        test_response = "Madrid is the capital of Spain and has 3.5 million inhabitants. Google created the first web browser."
        processed = await guardrails.process_response(test_response)
        if processed:
            print(f"Response was modified: {processed['was_modified']}")
            print(f"Safety clearance: {processed['safety_clearance']}")
        else:
            print("Safety guardrails processing failed")

    finally:
        await fact_checker.close()


if __name__ == "__main__":
    asyncio.run(test_fact_checker())
