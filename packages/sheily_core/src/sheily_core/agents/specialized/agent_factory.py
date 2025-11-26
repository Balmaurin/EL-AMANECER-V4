#!/usr/bin/env python3
"""
Specialized Agent Factory - Auto-generate 51+ Enterprise Agents
===============================================================

Generates production-ready specialized agents using templates and domain knowledge.
Each agent is customized for its specific domain with real capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..base.enhanced_base import AgentCapability, EnhancedBaseMCPAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentSpec:
    """Specification for specialized agent"""

    agent_id: str
    agent_name: str
    domain: str
    capabilities: List[AgentCapability]
    specialized_skills: List[str]
    default_timeout: int = 300
    max_concurrent: int = 10
    description: str = ""


# ============================================================================
# DOMAIN SPECIFICATIONS
# ============================================================================

FINANCE_AGENTS = [
    AgentSpec(
        agent_id="financial_analysis_agent",
        agent_name="Financial Analysis Agent",
        domain="finance",
        capabilities=[
            AgentCapability.FINANCIAL_ANALYSIS,
            AgentCapability.ANALYSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=[
            "financial_modeling",
            "ratio_analysis",
            "trend_analysis",
            "forecasting",
        ],
        description="Analyzes financial statements, performs ratio analysis, and forecasts financial trends",
    ),
    AgentSpec(
        agent_id="risk_management_agent",
        agent_name="Risk Management Agent",
        domain="finance",
        capabilities=[
            AgentCapability.RISK_MANAGEMENT,
            AgentCapability.ANALYSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=[
            "var_calculation",
            "stress_testing",
            "scenario_analysis",
            "risk_metrics",
        ],
        description="Evaluates financial risks, calculates VaR, performs stress testing",
    ),
    AgentSpec(
        agent_id="trading_strategy_agent",
        agent_name="Trading Strategy Agent",
        domain="finance",
        capabilities=[
            AgentCapability.FINANCIAL_ANALYSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
            AgentCapability.REAL_TIME_PROCESSING,
        ],
        specialized_skills=[
            "algorithmic_trading",
            "backtesting",
            "signal_generation",
            "portfolio_rebalancing",
        ],
        description="Develops and backtests trading strategies, generates trading signals",
    ),
    AgentSpec(
        agent_id="portfolio_optimization_agent",
        agent_name="Portfolio Optimization Agent",
        domain="finance",
        capabilities=[
            AgentCapability.FINANCIAL_ANALYSIS,
            AgentCapability.SYSTEM_OPTIMIZATION,
        ],
        specialized_skills=[
            "markowitz_optimization",
            "black_litterman",
            "risk_parity",
            "asset_allocation",
        ],
        description="Optimizes investment portfolios using modern portfolio theory",
    ),
    AgentSpec(
        agent_id="credit_scoring_agent",
        agent_name="Credit Scoring Agent",
        domain="finance",
        capabilities=[
            AgentCapability.FINANCIAL_ANALYSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
            AgentCapability.ANOMALY_DETECTION,
        ],
        specialized_skills=[
            "credit_assessment",
            "default_prediction",
            "scorecard_modeling",
        ],
        description="Assesses creditworthiness and predicts default probability",
    ),
    AgentSpec(
        agent_id="fraud_detection_agent",
        agent_name="Fraud Detection Agent",
        domain="finance",
        capabilities=[
            AgentCapability.ANOMALY_DETECTION,
            AgentCapability.PREDICTIVE_ANALYTICS,
            AgentCapability.REAL_TIME_PROCESSING,
        ],
        specialized_skills=[
            "transaction_monitoring",
            "pattern_recognition",
            "fraud_scoring",
        ],
        description="Detects fraudulent transactions and suspicious patterns in financial data",
    ),
    AgentSpec(
        agent_id="wealth_management_agent",
        agent_name="Wealth Management Agent",
        domain="finance",
        capabilities=[
            AgentCapability.FINANCIAL_ANALYSIS,
            AgentCapability.STRATEGIC,
            AgentCapability.RECOMMENDATION,
        ],
        specialized_skills=["wealth_planning", "tax_optimization", "estate_planning"],
        description="Provides personalized wealth management and investment advice",
    ),
    AgentSpec(
        agent_id="market_analysis_agent",
        agent_name="Market Analysis Agent",
        domain="finance",
        capabilities=[
            AgentCapability.MARKET_RESEARCH,
            AgentCapability.ANALYSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=[
            "market_sentiment",
            "technical_analysis",
            "fundamental_analysis",
        ],
        description="Analyzes financial markets and provides market insights",
    ),
    AgentSpec(
        agent_id="derivatives_pricing_agent",
        agent_name="Derivatives Pricing Agent",
        domain="finance",
        capabilities=[
            AgentCapability.FINANCIAL_ANALYSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=[
            "option_pricing",
            "greeks_calculation",
            "volatility_modeling",
        ],
        description="Prices derivatives and calculates option Greeks",
    ),
    AgentSpec(
        agent_id="regulatory_reporting_agent",
        agent_name="Regulatory Reporting Agent",
        domain="finance",
        capabilities=[
            AgentCapability.FINANCIAL_ANALYSIS,
            AgentCapability.SECURITY_AUDIT,
        ],
        specialized_skills=[
            "basel_reporting",
            "solvency_reporting",
            "regulatory_compliance",
        ],
        description="Generates regulatory reports and ensures financial compliance",
    ),
]

SECURITY_AGENTS = [
    AgentSpec(
        agent_id="cybersecurity_agent",
        agent_name="Cybersecurity Agent",
        domain="security",
        capabilities=[
            AgentCapability.SECURITY_AUDIT,
            AgentCapability.THREAT_DETECTION,
            AgentCapability.TECHNICAL,
        ],
        specialized_skills=[
            "vulnerability_scanning",
            "penetration_testing",
            "security_assessment",
        ],
        description="Performs comprehensive security audits and vulnerability assessments",
    ),
    AgentSpec(
        agent_id="threat_detection_agent",
        agent_name="Threat Detection Agent",
        domain="security",
        capabilities=[
            AgentCapability.THREAT_DETECTION,
            AgentCapability.ANOMALY_DETECTION,
            AgentCapability.REAL_TIME_PROCESSING,
        ],
        specialized_skills=[
            "intrusion_detection",
            "malware_analysis",
            "behavior_analytics",
        ],
        description="Detects and analyzes security threats in real-time",
    ),
    AgentSpec(
        agent_id="compliance_agent",
        agent_name="Compliance Agent",
        domain="security",
        capabilities=[AgentCapability.SECURITY_AUDIT, AgentCapability.ANALYSIS],
        specialized_skills=["gdpr_compliance", "sox_compliance", "pci_dss", "hipaa"],
        description="Ensures compliance with security regulations and standards",
    ),
    AgentSpec(
        agent_id="incident_response_agent",
        agent_name="Incident Response Agent",
        domain="security",
        capabilities=[AgentCapability.THREAT_DETECTION, AgentCapability.STRATEGIC],
        specialized_skills=["incident_handling", "forensics", "recovery_planning"],
        description="Manages security incidents and coordinates response efforts",
    ),
    AgentSpec(
        agent_id="vulnerability_scanning_agent",
        agent_name="Vulnerability Scanning Agent",
        domain="security",
        capabilities=[
            AgentCapability.SECURITY_AUDIT,
            AgentCapability.TECHNICAL,
            AgentCapability.ANOMALY_DETECTION,
        ],
        specialized_skills=[
            "vulnerability_assessment",
            "security_scanning",
            "exploit_detection",
        ],
        description="Scans systems for security vulnerabilities and weaknesses",
    ),
    AgentSpec(
        agent_id="identity_access_agent",
        agent_name="Identity & Access Management Agent",
        domain="security",
        capabilities=[AgentCapability.SECURITY_AUDIT, AgentCapability.TECHNICAL],
        specialized_skills=[
            "access_control",
            "authentication",
            "authorization_policies",
        ],
        description="Manages identity and access control policies",
    ),
    AgentSpec(
        agent_id="network_security_agent",
        agent_name="Network Security Agent",
        domain="security",
        capabilities=[
            AgentCapability.SECURITY_AUDIT,
            AgentCapability.REAL_TIME_PROCESSING,
            AgentCapability.ANOMALY_DETECTION,
        ],
        specialized_skills=[
            "firewall_management",
            "network_monitoring",
            "intrusion_prevention",
        ],
        description="Monitors and secures network infrastructure",
    ),
    AgentSpec(
        agent_id="data_privacy_agent",
        agent_name="Data Privacy Agent",
        domain="security",
        capabilities=[AgentCapability.SECURITY_AUDIT, AgentCapability.LEGAL_ANALYSIS],
        specialized_skills=["gdpr_compliance", "data_protection", "privacy_assessment"],
        description="Ensures data privacy compliance and protection",
    ),
]

HEALTHCARE_AGENTS = [
    AgentSpec(
        agent_id="medical_diagnosis_agent",
        agent_name="Medical Diagnosis Agent",
        domain="healthcare",
        capabilities=[
            AgentCapability.MEDICAL_DIAGNOSIS,
            AgentCapability.ANALYSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=[
            "symptom_analysis",
            "differential_diagnosis",
            "diagnostic_imaging",
        ],
        description="Analyzes symptoms and medical data to assist in diagnosis",
    ),
    AgentSpec(
        agent_id="treatment_planning_agent",
        agent_name="Treatment Planning Agent",
        domain="healthcare",
        capabilities=[AgentCapability.TREATMENT_PLANNING, AgentCapability.STRATEGIC],
        specialized_skills=[
            "treatment_protocols",
            "drug_interactions",
            "personalized_medicine",
        ],
        description="Develops personalized treatment plans based on patient data",
    ),
    AgentSpec(
        agent_id="patient_monitoring_agent",
        agent_name="Patient Monitoring Agent",
        domain="healthcare",
        capabilities=[
            AgentCapability.MEDICAL_DIAGNOSIS,
            AgentCapability.REAL_TIME_PROCESSING,
            AgentCapability.ANOMALY_DETECTION,
        ],
        specialized_skills=[
            "vital_signs_monitoring",
            "alert_generation",
            "trend_analysis",
        ],
        description="Monitors patient vital signs and generates alerts",
    ),
    AgentSpec(
        agent_id="clinical_research_agent",
        agent_name="Clinical Research Agent",
        domain="healthcare",
        capabilities=[AgentCapability.ANALYSIS, AgentCapability.DATA_SCIENCE],
        specialized_skills=["clinical_trials", "biostatistics", "evidence_synthesis"],
        description="Analyzes clinical trial data and synthesizes medical evidence",
    ),
    AgentSpec(
        agent_id="radiology_analysis_agent",
        agent_name="Radiology Analysis Agent",
        domain="healthcare",
        capabilities=[
            AgentCapability.MEDICAL_DIAGNOSIS,
            AgentCapability.COMPUTER_VISION,
            AgentCapability.ANOMALY_DETECTION,
        ],
        specialized_skills=[
            "image_analysis",
            "tumor_detection",
            "fracture_identification",
        ],
        description="Analyzes medical images for diagnostic purposes",
    ),
    AgentSpec(
        agent_id="drug_interaction_agent",
        agent_name="Drug Interaction Agent",
        domain="healthcare",
        capabilities=[
            AgentCapability.MEDICAL_DIAGNOSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=["pharmacology", "drug_interactions", "dosage_optimization"],
        description="Identifies drug interactions and optimizes medication regimens",
    ),
]

EDUCATION_AGENTS = [
    AgentSpec(
        agent_id="educational_content_agent",
        agent_name="Educational Content Agent",
        domain="education",
        capabilities=[
            AgentCapability.EDUCATIONAL_DESIGN,
            AgentCapability.CONTENT_CREATION,
            AgentCapability.CREATIVE,
        ],
        specialized_skills=[
            "curriculum_design",
            "learning_objectives",
            "content_adaptation",
        ],
        description="Creates and adapts educational content for different learning styles",
    ),
    AgentSpec(
        agent_id="tutoring_agent",
        agent_name="Intelligent Tutoring Agent",
        domain="education",
        capabilities=[
            AgentCapability.TUTORING,
            AgentCapability.COMMUNICATION,
            AgentCapability.NLP_ADVANCED,
        ],
        specialized_skills=[
            "personalized_tutoring",
            "socratic_method",
            "feedback_generation",
        ],
        description="Provides personalized one-on-one tutoring and feedback",
    ),
    AgentSpec(
        agent_id="assessment_agent",
        agent_name="Assessment Agent",
        domain="education",
        capabilities=[AgentCapability.EDUCATIONAL_DESIGN, AgentCapability.ANALYSIS],
        specialized_skills=[
            "test_generation",
            "automated_grading",
            "learning_analytics",
        ],
        description="Generates assessments and analyzes learning outcomes",
    ),
    AgentSpec(
        agent_id="learning_path_agent",
        agent_name="Learning Path Agent",
        domain="education",
        capabilities=[
            AgentCapability.EDUCATIONAL_DESIGN,
            AgentCapability.RECOMMENDATION,
            AgentCapability.STRATEGIC,
        ],
        specialized_skills=[
            "adaptive_learning",
            "competency_mapping",
            "progress_tracking",
        ],
        description="Designs personalized learning paths based on student progress",
    ),
    AgentSpec(
        agent_id="language_learning_agent",
        agent_name="Language Learning Agent",
        domain="education",
        capabilities=[
            AgentCapability.TUTORING,
            AgentCapability.NLP_ADVANCED,
            AgentCapability.COMMUNICATION,
        ],
        specialized_skills=[
            "language_instruction",
            "pronunciation_feedback",
            "grammar_correction",
        ],
        description="Provides personalized language learning and practice",
    ),
]

ENGINEERING_AGENTS = [
    AgentSpec(
        agent_id="systems_engineering_agent",
        agent_name="Systems Engineering Agent",
        domain="engineering",
        capabilities=[
            AgentCapability.ENGINEERING_DESIGN,
            AgentCapability.TECHNICAL,
            AgentCapability.ANALYSIS,
        ],
        specialized_skills=[
            "system_architecture",
            "requirements_analysis",
            "integration_testing",
        ],
        description="Designs complex systems and manages engineering requirements",
    ),
    AgentSpec(
        agent_id="design_automation_agent",
        agent_name="Design Automation Agent",
        domain="engineering",
        capabilities=[
            AgentCapability.ENGINEERING_DESIGN,
            AgentCapability.SYSTEM_OPTIMIZATION,
        ],
        specialized_skills=[
            "cad_automation",
            "generative_design",
            "topology_optimization",
        ],
        description="Automates engineering design processes and optimizes designs",
    ),
    AgentSpec(
        agent_id="testing_validation_agent",
        agent_name="Testing & Validation Agent",
        domain="engineering",
        capabilities=[AgentCapability.TECHNICAL, AgentCapability.ANALYSIS],
        specialized_skills=[
            "test_automation",
            "validation_protocols",
            "quality_assurance",
        ],
        description="Automates testing and validates engineering specifications",
    ),
    AgentSpec(
        agent_id="simulation_agent",
        agent_name="Simulation Agent",
        domain="engineering",
        capabilities=[
            AgentCapability.ENGINEERING_DESIGN,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=["fem_analysis", "cfd_simulation", "multiphysics_modeling"],
        description="Performs engineering simulations and finite element analysis",
    ),
    AgentSpec(
        agent_id="quality_assurance_agent",
        agent_name="Quality Assurance Agent",
        domain="engineering",
        capabilities=[
            AgentCapability.TECHNICAL,
            AgentCapability.ANALYSIS,
            AgentCapability.ANOMALY_DETECTION,
        ],
        specialized_skills=["quality_control", "defect_detection", "six_sigma"],
        description="Ensures quality standards and detects manufacturing defects",
    ),
    AgentSpec(
        agent_id="maintenance_prediction_agent",
        agent_name="Predictive Maintenance Agent",
        domain="engineering",
        capabilities=[
            AgentCapability.PREDICTIVE_ANALYTICS,
            AgentCapability.ANOMALY_DETECTION,
            AgentCapability.REAL_TIME_PROCESSING,
        ],
        specialized_skills=[
            "failure_prediction",
            "condition_monitoring",
            "maintenance_scheduling",
        ],
        description="Predicts equipment failures and optimizes maintenance schedules",
    ),
]

BUSINESS_AGENTS = [
    AgentSpec(
        agent_id="business_intelligence_agent",
        agent_name="Business Intelligence Agent",
        domain="business",
        capabilities=[
            AgentCapability.BUSINESS_INTELLIGENCE,
            AgentCapability.ANALYSIS,
            AgentCapability.DATA_SCIENCE,
        ],
        specialized_skills=["kpi_analysis", "dashboard_generation", "reporting"],
        description="Analyzes business metrics and generates intelligence reports",
    ),
    AgentSpec(
        agent_id="market_research_agent",
        agent_name="Market Research Agent",
        domain="business",
        capabilities=[AgentCapability.MARKET_RESEARCH, AgentCapability.ANALYSIS],
        specialized_skills=[
            "competitive_analysis",
            "market_segmentation",
            "consumer_insights",
        ],
        description="Conducts market research and competitive analysis",
    ),
    AgentSpec(
        agent_id="strategy_planning_agent",
        agent_name="Strategy Planning Agent",
        domain="business",
        capabilities=[AgentCapability.STRATEGIC, AgentCapability.BUSINESS_INTELLIGENCE],
        specialized_skills=["strategic_planning", "scenario_modeling", "swot_analysis"],
        description="Develops business strategies and strategic plans",
    ),
    AgentSpec(
        agent_id="operations_optimization_agent",
        agent_name="Operations Optimization Agent",
        domain="business",
        capabilities=[
            AgentCapability.SYSTEM_OPTIMIZATION,
            AgentCapability.BUSINESS_INTELLIGENCE,
        ],
        specialized_skills=["process_optimization", "supply_chain", "lean_six_sigma"],
        description="Optimizes business operations and processes",
    ),
    AgentSpec(
        agent_id="customer_insights_agent",
        agent_name="Customer Insights Agent",
        domain="business",
        capabilities=[
            AgentCapability.BUSINESS_INTELLIGENCE,
            AgentCapability.DATA_SCIENCE,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=[
            "customer_segmentation",
            "churn_prediction",
            "lifetime_value",
        ],
        description="Analyzes customer data and provides actionable insights",
    ),
    AgentSpec(
        agent_id="supply_chain_agent",
        agent_name="Supply Chain Optimization Agent",
        domain="business",
        capabilities=[
            AgentCapability.SYSTEM_OPTIMIZATION,
            AgentCapability.PREDICTIVE_ANALYTICS,
            AgentCapability.BUSINESS_INTELLIGENCE,
        ],
        specialized_skills=[
            "inventory_optimization",
            "demand_forecasting",
            "logistics_planning",
        ],
        description="Optimizes supply chain operations and logistics",
    ),
    AgentSpec(
        agent_id="hr_analytics_agent",
        agent_name="HR Analytics Agent",
        domain="business",
        capabilities=[
            AgentCapability.BUSINESS_INTELLIGENCE,
            AgentCapability.DATA_SCIENCE,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=[
            "talent_analytics",
            "retention_prediction",
            "performance_assessment",
        ],
        description="Analyzes HR data and provides workforce insights",
    ),
]

CREATIVE_AGENTS = [
    AgentSpec(
        agent_id="content_creation_agent",
        agent_name="Content Creation Agent",
        domain="creative",
        capabilities=[
            AgentCapability.CONTENT_CREATION,
            AgentCapability.CREATIVE,
            AgentCapability.NLP_ADVANCED,
        ],
        specialized_skills=["copywriting", "storytelling", "content_strategy"],
        description="Creates engaging content for various media and purposes",
    ),
    AgentSpec(
        agent_id="design_agent",
        agent_name="Design Agent",
        domain="creative",
        capabilities=[AgentCapability.CREATIVE, AgentCapability.CONTENT_CREATION],
        specialized_skills=["graphic_design", "ui_ux_design", "brand_identity"],
        description="Creates visual designs and user interfaces",
    ),
    AgentSpec(
        agent_id="marketing_agent",
        agent_name="Marketing Agent",
        domain="creative",
        capabilities=[
            AgentCapability.CONTENT_CREATION,
            AgentCapability.STRATEGIC,
            AgentCapability.MARKET_RESEARCH,
        ],
        specialized_skills=["campaign_planning", "seo_optimization", "social_media"],
        description="Plans and executes marketing campaigns",
    ),
    AgentSpec(
        agent_id="video_production_agent",
        agent_name="Video Production Agent",
        domain="creative",
        capabilities=[
            AgentCapability.CREATIVE,
            AgentCapability.CONTENT_CREATION,
            AgentCapability.COMPUTER_VISION,
        ],
        specialized_skills=["video_editing", "animation", "storyboarding"],
        description="Produces and edits video content",
    ),
    AgentSpec(
        agent_id="brand_strategy_agent",
        agent_name="Brand Strategy Agent",
        domain="creative",
        capabilities=[
            AgentCapability.STRATEGIC,
            AgentCapability.CREATIVE,
            AgentCapability.MARKET_RESEARCH,
        ],
        specialized_skills=[
            "brand_positioning",
            "brand_identity",
            "competitive_analysis",
        ],
        description="Develops brand strategies and positioning",
    ),
]

ADDITIONAL_AGENTS = [
    AgentSpec(
        agent_id="legal_analysis_agent",
        agent_name="Legal Analysis Agent",
        domain="legal",
        capabilities=[
            AgentCapability.LEGAL_ANALYSIS,
            AgentCapability.ANALYSIS,
            AgentCapability.NLP_ADVANCED,
        ],
        specialized_skills=[
            "contract_analysis",
            "legal_research",
            "compliance_checking",
        ],
        description="Analyzes legal documents and provides legal insights",
    ),
    AgentSpec(
        agent_id="environmental_science_agent",
        agent_name="Environmental Science Agent",
        domain="environment",
        capabilities=[
            AgentCapability.ANALYSIS,
            AgentCapability.DATA_SCIENCE,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=[
            "climate_modeling",
            "environmental_impact",
            "sustainability_analysis",
        ],
        description="Analyzes environmental data and assesses sustainability",
    ),
    AgentSpec(
        agent_id="data_science_agent",
        agent_name="Data Science Agent",
        domain="data_science",
        capabilities=[
            AgentCapability.DATA_SCIENCE,
            AgentCapability.ANALYSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=["ml_modeling", "feature_engineering", "data_visualization"],
        description="Performs advanced data science and machine learning tasks",
    ),
    AgentSpec(
        agent_id="communication_strategy_agent",
        agent_name="Communication Strategy Agent",
        domain="communication",
        capabilities=[
            AgentCapability.COMMUNICATION,
            AgentCapability.STRATEGIC,
            AgentCapability.CONTENT_CREATION,
        ],
        specialized_skills=[
            "communication_planning",
            "stakeholder_engagement",
            "crisis_communication",
        ],
        description="Develops communication strategies and manages stakeholder relations",
    ),
    AgentSpec(
        agent_id="innovation_research_agent",
        agent_name="Innovation Research Agent",
        domain="innovation",
        capabilities=[
            AgentCapability.ANALYSIS,
            AgentCapability.STRATEGIC,
            AgentCapability.MARKET_RESEARCH,
        ],
        specialized_skills=[
            "trend_analysis",
            "technology_scouting",
            "innovation_management",
        ],
        description="Researches emerging trends and manages innovation processes",
    ),
    AgentSpec(
        agent_id="nlp_processing_agent",
        agent_name="NLP Processing Agent",
        domain="data_science",
        capabilities=[
            AgentCapability.NLP_ADVANCED,
            AgentCapability.DATA_SCIENCE,
            AgentCapability.ANALYSIS,
        ],
        specialized_skills=[
            "text_classification",
            "sentiment_analysis",
            "entity_extraction",
        ],
        description="Performs advanced natural language processing tasks",
    ),
    AgentSpec(
        agent_id="computer_vision_agent",
        agent_name="Computer Vision Agent",
        domain="data_science",
        capabilities=[
            AgentCapability.COMPUTER_VISION,
            AgentCapability.DATA_SCIENCE,
            AgentCapability.ANOMALY_DETECTION,
        ],
        specialized_skills=[
            "image_classification",
            "object_detection",
            "facial_recognition",
        ],
        description="Analyzes images and videos using computer vision",
    ),
    AgentSpec(
        agent_id="time_series_agent",
        agent_name="Time Series Analysis Agent",
        domain="data_science",
        capabilities=[
            AgentCapability.PREDICTIVE_ANALYTICS,
            AgentCapability.DATA_SCIENCE,
            AgentCapability.ANALYSIS,
        ],
        specialized_skills=["forecasting", "anomaly_detection", "seasonality_analysis"],
        description="Analyzes time series data and generates forecasts",
    ),
    AgentSpec(
        agent_id="iot_analytics_agent",
        agent_name="IoT Analytics Agent",
        domain="engineering",
        capabilities=[
            AgentCapability.REAL_TIME_PROCESSING,
            AgentCapability.DATA_SCIENCE,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ],
        specialized_skills=[
            "sensor_data_analysis",
            "edge_computing",
            "device_monitoring",
        ],
        description="Analyzes IoT sensor data in real-time",
    ),
    AgentSpec(
        agent_id="sustainability_agent",
        agent_name="Sustainability Agent",
        domain="environment",
        capabilities=[
            AgentCapability.ANALYSIS,
            AgentCapability.PREDICTIVE_ANALYTICS,
            AgentCapability.STRATEGIC,
        ],
        specialized_skills=[
            "carbon_footprint",
            "esg_analysis",
            "sustainable_practices",
        ],
        description="Assesses and optimizes sustainability initiatives",
    ),
]

# Combine all agent specs
ALL_AGENT_SPECS = (
    FINANCE_AGENTS
    + SECURITY_AGENTS
    + HEALTHCARE_AGENTS
    + EDUCATION_AGENTS
    + ENGINEERING_AGENTS
    + BUSINESS_AGENTS
    + CREATIVE_AGENTS
    + ADDITIONAL_AGENTS
)


class SpecializedAgentFactory:
    """
    Factory for creating specialized agents
    """

    @staticmethod
    def create_agent(spec_or_name) -> Optional[EnhancedBaseMCPAgent]:
        """
        Create a specialized agent from specification or name

        Args:
            spec_or_name: Either an AgentSpec object or a string (agent_id)

        Returns:
            Created agent or None if not found
        """
        # Import the specialized agent class
        from .template_agent import TemplateSpecializedAgent

        # If it's already a spec, use it directly
        if isinstance(spec_or_name, AgentSpec):
            spec = spec_or_name
        else:
            # Find spec by agent_id
            spec = SpecializedAgentFactory.get_agent_spec(spec_or_name)
            if not spec:
                logger.error(f"Agent specification not found: {spec_or_name}")
                return None

        agent = TemplateSpecializedAgent(
            agent_id=spec.agent_id,
            agent_name=spec.agent_name,
            capabilities=spec.capabilities,
            domain=spec.domain,
            specialized_skills=spec.specialized_skills,
            max_concurrent_tasks=spec.max_concurrent,
            description=spec.description,
        )

        logger.info(f"Created specialized agent: {spec.agent_name} ({spec.domain})")
        return agent

    @staticmethod
    async def create_all_agents() -> Dict[str, EnhancedBaseMCPAgent]:
        """Create all 51+ specialized agents"""
        agents = {}

        for spec in ALL_AGENT_SPECS:
            agent = SpecializedAgentFactory.create_agent(spec)
            agents[spec.agent_id] = agent

        logger.info(f"Created {len(agents)} specialized agents across all domains")
        return agents

    @staticmethod
    def get_agents_by_domain(domain: str) -> List[AgentSpec]:
        """Get all agent specifications for a domain"""
        return [spec for spec in ALL_AGENT_SPECS if spec.domain == domain]

    @staticmethod
    def get_agent_spec(agent_id: str) -> Optional[AgentSpec]:
        """Get specification for specific agent"""
        for spec in ALL_AGENT_SPECS:
            if spec.agent_id == agent_id:
                return spec
        return None

    @staticmethod
    def list_all_domains() -> List[str]:
        """List all available domains"""
        domains = set(spec.domain for spec in ALL_AGENT_SPECS)
        return sorted(list(domains))

    @staticmethod
    def get_domain_summary() -> Dict[str, int]:
        """Get summary of agents per domain"""
        summary = {}
        for spec in ALL_AGENT_SPECS:
            summary[spec.domain] = summary.get(spec.domain, 0) + 1
        return summary


__all__ = [
    "AgentSpec",
    "SpecializedAgentFactory",
    "ALL_AGENT_SPECS",
    "FINANCE_AGENTS",
    "SECURITY_AGENTS",
    "HEALTHCARE_AGENTS",
    "EDUCATION_AGENTS",
    "ENGINEERING_AGENTS",
    "BUSINESS_AGENTS",
    "CREATIVE_AGENTS",
    "ADDITIONAL_AGENTS",
]
