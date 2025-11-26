"""
Sistema de Gobernanza Educativa para Sheily AI
Gobierno participativo del sistema educativo basado en tokens
Basado en investigación: QCoin governance, REAL8 community governance

Características:
- Gobernanza basada en tokens SHEILYS
- Votaciones democráticas
- Políticas educativas transparentes
- Auditoría completa de decisiones
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProposalType(Enum):
    """Tipos de propuestas de gobernanza"""

    EDUCATIONAL_POLICY = "educational_policy"
    REWARD_SYSTEM_UPDATE = "reward_system_update"
    CHALLENGE_CREATION = "challenge_creation"
    PLATFORM_INTEGRATION = "platform_integration"
    GOVERNANCE_RULE_CHANGE = "governance_rule_change"
    BUDGET_ALLOCATION = "budget_allocation"


class ProposalStatus(Enum):
    """Estados de propuestas"""

    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


class VoteType(Enum):
    """Tipos de voto"""

    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"


@dataclass
class GovernanceProposal:
    """Propuesta de gobernanza educativa"""

    proposal_id: str
    title: str
    description: str
    proposal_type: ProposalType
    proposer_id: str
    created_at: datetime
    voting_starts_at: datetime
    voting_ends_at: datetime
    execution_deadline: Optional[datetime] = None

    # Contenido específico
    content: Dict[str, Any] = field(default_factory=dict)

    # Estado y resultados
    status: ProposalStatus = ProposalStatus.DRAFT
    votes_for: int = 0
    votes_against: int = 0
    votes_abstain: int = 0
    total_voting_power: float = 0

    # Resultados
    executed_at: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None

    @property
    def total_votes(self) -> int:
        """Total de votos emitidos"""
        return self.votes_for + self.votes_against + self.votes_abstain

    @property
    def approval_rate(self) -> float:
        """Tasa de aprobación (votos a favor / votos totales)"""
        if self.total_votes == 0:
            return 0.0
        return self.votes_for / self.total_votes

    @property
    def quorum_reached(self) -> bool:
        """Verificar si se alcanzó quorum mínimo"""
        # Quorum: al menos 10% del poder de voto total debe participar
        return self.total_voting_power >= 0.1  # Simplificado

    @property
    def is_passed(self) -> bool:
        """Verificar si la propuesta fue aprobada"""
        return (
            self.approval_rate >= 0.5  # Mayoría simple
            and self.quorum_reached  # Quorum alcanzado
            and datetime.now() > self.voting_ends_at
        )

    @property
    def can_execute(self) -> bool:
        """Verificar si la propuesta puede ejecutarse"""
        return self.status == ProposalStatus.PASSED and (
            not self.execution_deadline or datetime.now() <= self.execution_deadline
        )


@dataclass
class GovernanceVote:
    """Voto en una propuesta de gobernanza"""

    vote_id: str
    proposal_id: str
    voter_id: str
    vote_type: VoteType
    voting_power: float
    voted_at: datetime
    rationale: Optional[str] = None


class EducationalGovernance:
    """
    Sistema de gobernanza educativa
    Gestiona propuestas, votaciones y políticas del sistema educativo
    """

    def __init__(self):
        self.proposals: Dict[str, GovernanceProposal] = {}
        self.votes: Dict[str, List[GovernanceVote]] = {}  # proposal_id -> [votes]
        self.policies: Dict[str, Dict[str, Any]] = {}

        # Configuración de gobernanza
        self.voting_period_days = 7
        self.execution_delay_days = 2
        self.minimum_quorum = 0.1  # 10%

        # Políticas por defecto
        self._initialize_default_policies()

        logger.info("🏛️ Educational Governance system initialized")

    def _initialize_default_policies(self):
        """Inicializar políticas de gobernanza por defecto"""
        self.policies = {
            "reward_system": {
                "max_daily_sheilys": 100,
                "quality_multiplier_cap": 2.0,
                "engagement_bonus_max": 50,
                "updated_at": datetime.now().isoformat(),
            },
            "challenge_creation": {
                "max_active_challenges": 20,
                "min_challenge_duration_days": 1,
                "max_challenge_duration_days": 90,
                "updated_at": datetime.now().isoformat(),
            },
            "nft_credentials": {
                "max_credentials_per_user": 100,
                "verification_required": True,
                "revocation_allowed": True,
                "updated_at": datetime.now().isoformat(),
            },
            "governance": {
                "voting_period_days": self.voting_period_days,
                "execution_delay_days": self.execution_delay_days,
                "minimum_quorum": self.minimum_quorum,
                "updated_at": datetime.now().isoformat(),
            },
        }

    async def create_proposal(
        self,
        proposer_id: str,
        title: str,
        description: str,
        proposal_type: ProposalType,
        content: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Crear una nueva propuesta de gobernanza
        """
        try:
            proposal_id = (
                f"proposal_{int(datetime.now().timestamp())}_{hash(title) % 10000}"
            )

            # Calcular fechas
            created_at = datetime.now()
            voting_starts_at = created_at + timedelta(days=1)  # 24h para revisión
            voting_ends_at = voting_starts_at + timedelta(days=self.voting_period_days)
            execution_deadline = voting_ends_at + timedelta(
                days=self.execution_delay_days
            )

            proposal = GovernanceProposal(
                proposal_id=proposal_id,
                title=title,
                description=description,
                proposal_type=proposal_type,
                proposer_id=proposer_id,
                created_at=created_at,
                voting_starts_at=voting_starts_at,
                voting_ends_at=voting_ends_at,
                execution_deadline=execution_deadline,
                content=content,
                status=ProposalStatus.ACTIVE,
            )

            self.proposals[proposal_id] = proposal
            self.votes[proposal_id] = []

            logger.info(f"📋 Created governance proposal: {proposal_id} - {title}")

            return {
                "success": True,
                "proposal_id": proposal_id,
                "title": title,
                "voting_starts_at": voting_starts_at.isoformat(),
                "voting_ends_at": voting_ends_at.isoformat(),
                "status": "active",
            }

        except Exception as e:
            logger.error(f"Error creating proposal: {e}")
            return {"success": False, "error": str(e)}

    async def cast_vote(
        self,
        voter_id: str,
        proposal_id: str,
        vote_type: VoteType,
        voting_power: float,
        rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Emitir voto en una propuesta
        """
        try:
            if proposal_id not in self.proposals:
                return {"success": False, "error": "Proposal not found"}

            proposal = self.proposals[proposal_id]

            # Verificar que la votación esté activa
            now = datetime.now()
            if now < proposal.voting_starts_at or now > proposal.voting_ends_at:
                return {
                    "success": False,
                    "error": "Voting not active for this proposal",
                }

            # Verificar que no haya votado antes
            existing_vote = next(
                (v for v in self.votes[proposal_id] if v.voter_id == voter_id), None
            )
            if existing_vote:
                return {"success": False, "error": "Already voted on this proposal"}

            # Crear voto
            vote_id = f"vote_{proposal_id}_{voter_id}_{int(now.timestamp())}"
            vote = GovernanceVote(
                vote_id=vote_id,
                proposal_id=proposal_id,
                voter_id=voter_id,
                vote_type=vote_type,
                voting_power=voting_power,
                voted_at=now,
                rationale=rationale,
            )

            # Registrar voto
            self.votes[proposal_id].append(vote)

            # Actualizar contadores de la propuesta
            if vote_type == VoteType.FOR:
                proposal.votes_for += 1
            elif vote_type == VoteType.AGAINST:
                proposal.votes_against += 1
            else:  # ABSTAIN
                proposal.votes_abstain += 1

            proposal.total_voting_power += voting_power

            logger.info(
                f"🗳️ Vote cast: {voter_id} voted {vote_type.value} on {proposal_id}"
            )

            return {
                "success": True,
                "vote_id": vote_id,
                "proposal_id": proposal_id,
                "vote_type": vote_type.value,
                "voting_power": voting_power,
            }

        except Exception as e:
            logger.error(f"Error casting vote: {e}")
            return {"success": False, "error": str(e)}

    async def execute_proposal(
        self, proposal_id: str, executor_id: str
    ) -> Dict[str, Any]:
        """
        Ejecutar una propuesta aprobada
        """
        try:
            if proposal_id not in self.proposals:
                return {"success": False, "error": "Proposal not found"}

            proposal = self.proposals[proposal_id]

            # Verificar que puede ejecutarse
            if not proposal.can_execute:
                return {"success": False, "error": "Proposal cannot be executed"}

            # Ejecutar según tipo de propuesta
            execution_result = await self._execute_proposal_content(proposal)

            # Actualizar estado
            proposal.status = ProposalStatus.EXECUTED
            proposal.executed_at = datetime.now()
            proposal.execution_result = execution_result

            logger.info(f"⚡ Executed proposal: {proposal_id} - {proposal.title}")

            return {
                "success": True,
                "proposal_id": proposal_id,
                "execution_result": execution_result,
                "executed_at": proposal.executed_at.isoformat(),
            }

        except Exception as e:
            logger.error(f"Error executing proposal: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_proposal_content(
        self, proposal: GovernanceProposal
    ) -> Dict[str, Any]:
        """
        Ejecutar el contenido específico de la propuesta
        """
        try:
            if proposal.proposal_type == ProposalType.EDUCATIONAL_POLICY:
                # Actualizar política educativa
                policy_key = proposal.content.get("policy_key")
                new_values = proposal.content.get("new_values", {})

                if policy_key in self.policies:
                    self.policies[policy_key].update(new_values)
                    self.policies[policy_key]["updated_at"] = datetime.now().isoformat()

                    return {
                        "action": "policy_updated",
                        "policy_key": policy_key,
                        "changes": new_values,
                    }

            elif proposal.proposal_type == ProposalType.REWARD_SYSTEM_UPDATE:
                # Actualizar sistema de recompensas
                # En producción: llamar al sistema de token economy
                return {"action": "reward_system_updated", "changes": proposal.content}

            elif proposal.proposal_type == ProposalType.CHALLENGE_CREATION:
                # Crear nuevo challenge
                # En producción: llamar al gamification engine
                return {
                    "action": "challenge_created",
                    "challenge_data": proposal.content,
                }

            elif proposal.proposal_type == ProposalType.GOVERNANCE_RULE_CHANGE:
                # Cambiar reglas de gobernanza
                rule_changes = proposal.content.get("rule_changes", {})
                for rule, value in rule_changes.items():
                    if hasattr(self, rule):
                        setattr(self, rule, value)

                return {"action": "governance_rules_updated", "changes": rule_changes}

            # Propuesta genérica
            return {"action": "generic_execution", "content": proposal.content}

        except Exception as e:
            logger.error(f"Error executing proposal content: {e}")
            return {"error": str(e), "content": proposal.content}

    async def get_proposal_status(self, proposal_id: str) -> Dict[str, Any]:
        """
        Obtener estado detallado de una propuesta
        """
        try:
            if proposal_id not in self.proposals:
                return {"error": "Proposal not found"}

            proposal = self.proposals[proposal_id]
            votes = self.votes.get(proposal_id, [])

            return {
                "proposal_id": proposal_id,
                "title": proposal.title,
                "description": proposal.description,
                "status": proposal.status.value,
                "proposal_type": proposal.proposal_type.value,
                "created_at": proposal.created_at.isoformat(),
                "voting_starts_at": proposal.voting_starts_at.isoformat(),
                "voting_ends_at": proposal.voting_ends_at.isoformat(),
                "votes_for": proposal.votes_for,
                "votes_against": proposal.votes_against,
                "votes_abstain": proposal.votes_abstain,
                "total_votes": proposal.total_votes,
                "approval_rate": proposal.approval_rate,
                "quorum_reached": proposal.quorum_reached,
                "is_passed": proposal.is_passed,
                "can_execute": proposal.can_execute,
                "total_voting_power": proposal.total_voting_power,
                "execution_result": proposal.execution_result,
                "recent_votes": [
                    {
                        "voter_id": vote.voter_id,
                        "vote_type": vote.vote_type.value,
                        "voting_power": vote.voting_power,
                        "voted_at": vote.voted_at.isoformat(),
                    }
                    for vote in votes[-10:]  # Últimos 10 votos
                ],
            }

        except Exception as e:
            logger.error(f"Error getting proposal status: {e}")
            return {"error": str(e)}

    async def get_active_proposals(self) -> List[Dict[str, Any]]:
        """
        Obtener propuestas activas para votación
        """
        try:
            active_proposals = []
            now = datetime.now()

            for proposal in self.proposals.values():
                if (
                    proposal.status == ProposalStatus.ACTIVE
                    and proposal.voting_starts_at <= now <= proposal.voting_ends_at
                ):

                    active_proposals.append(
                        {
                            "proposal_id": proposal.proposal_id,
                            "title": proposal.title,
                            "description": (
                                proposal.description[:200] + "..."
                                if len(proposal.description) > 200
                                else proposal.description
                            ),
                            "proposal_type": proposal.proposal_type.value,
                            "voting_ends_at": proposal.voting_ends_at.isoformat(),
                            "votes_for": proposal.votes_for,
                            "votes_against": proposal.votes_against,
                            "total_votes": proposal.total_votes,
                            "time_remaining_hours": max(
                                0,
                                int(
                                    (proposal.voting_ends_at - now).total_seconds()
                                    / 3600
                                ),
                            ),
                        }
                    )

            return active_proposals

        except Exception as e:
            logger.error(f"Error getting active proposals: {e}")
            return []

    async def get_governance_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema de gobernanza
        """
        try:
            total_proposals = len(self.proposals)
            active_proposals = sum(
                1 for p in self.proposals.values() if p.status == ProposalStatus.ACTIVE
            )
            passed_proposals = sum(
                1 for p in self.proposals.values() if p.status == ProposalStatus.PASSED
            )
            executed_proposals = sum(
                1
                for p in self.proposals.values()
                if p.status == ProposalStatus.EXECUTED
            )

            total_votes = sum(len(votes) for votes in self.votes.values())

            # Estadísticas por tipo
            type_distribution = {}
            for proposal in self.proposals.values():
                prop_type = proposal.proposal_type.value
                if prop_type not in type_distribution:
                    type_distribution[prop_type] = 0
                type_distribution[prop_type] += 1

            return {
                "total_proposals": total_proposals,
                "active_proposals": active_proposals,
                "passed_proposals": passed_proposals,
                "executed_proposals": executed_proposals,
                "total_votes": total_votes,
                "proposal_types": type_distribution,
                "execution_rate": (executed_proposals / max(passed_proposals, 1)) * 100,
                "participation_rate": (
                    (total_votes / max(total_proposals, 1))
                    if total_proposals > 0
                    else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error getting governance stats: {e}")
            return {"error": str(e)}

    async def get_user_governance_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener perfil de gobernanza de un usuario
        """
        try:
            # Propuestas creadas
            created_proposals = [
                p.proposal_id
                for p in self.proposals.values()
                if p.proposer_id == user_id
            ]

            # Votos emitidos
            user_votes = []
            for proposal_id, votes in self.votes.items():
                for vote in votes:
                    if vote.voter_id == user_id:
                        user_votes.append(
                            {
                                "proposal_id": proposal_id,
                                "vote_type": vote.vote_type.value,
                                "voting_power": vote.voting_power,
                                "voted_at": vote.voted_at.isoformat(),
                            }
                        )

            # Estadísticas de participación
            total_votes = len(user_votes)
            proposals_supported = sum(1 for v in user_votes if v["vote_type"] == "for")
            proposals_opposed = sum(
                1 for v in user_votes if v["vote_type"] == "against"
            )

            return {
                "user_id": user_id,
                "proposals_created": len(created_proposals),
                "total_votes": total_votes,
                "proposals_supported": proposals_supported,
                "proposals_opposed": proposals_opposed,
                "participation_rate": (total_votes / max(len(self.proposals), 1)) * 100,
                "recent_activity": user_votes[-5:] if user_votes else [],
            }

        except Exception as e:
            logger.error(f"Error getting user governance profile: {e}")
            return {"error": str(e)}


# Instancia global (singleton)
_educational_governance: Optional[EducationalGovernance] = None


def get_educational_governance() -> EducationalGovernance:
    """Obtener instancia singleton del sistema de gobernanza educativa"""
    global _educational_governance
    if _educational_governance is None:
        _educational_governance = EducationalGovernance()
    return _educational_governance
