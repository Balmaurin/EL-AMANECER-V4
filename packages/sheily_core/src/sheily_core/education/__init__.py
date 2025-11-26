"""
Sistema Educativo Web3 Completo para Sheily AI
Integración de token economy, NFTs, gamification y analytics educativos
basado en investigación validada de 8 documentos académicos.

Características principales:
- Token Economy Educativa con SHEILYS
- NFT Credentials verificables
- Gamification Engine (raffle tickets, learn-to-earn)
- Analytics y Governance educativo
- Integración LMS completa
"""

from .educational_analytics import EducationalAnalytics, get_educational_analytics
from .gamification_engine import GamificationEngine, get_gamification_engine
from .governance_system import EducationalGovernance, get_educational_governance
from .lms_integration import LMSIntegration, get_lms_integration
from .master_education_system import MasterEducationSystem, get_master_education_system
from .nft_credentials import NFTEducationCredentials, get_nft_credentials
from .token_economy import EducationalTokenEconomy, get_educational_token_economy

__all__ = [
    # Componentes principales
    "EducationalTokenEconomy",
    "NFTEducationCredentials",
    "GamificationEngine",
    "EducationalAnalytics",
    "LMSIntegration",
    "EducationalGovernance",
    "MasterEducationSystem",
    # Funciones de acceso
    "get_educational_token_economy",
    "get_nft_credentials",
    "get_gamification_engine",
    "get_educational_analytics",
    "get_lms_integration",
    "get_educational_governance",
    "get_master_education_system",
]

__version__ = "1.0.0"
__author__ = "Sheily AI Education System"
__description__ = "Sistema educativo Web3 completo integrado en Sheily AI"
