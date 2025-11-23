"""
Gamification Engine para Sheily AI
Implementa sistema de gamificación educativa con raffle tickets y learn-to-earn
Basado en investigación: Raffle ticket system, Token economy pedagógica, REAL8

Características:
- Raffle tickets para engagement asíncrono
- Sistema de challenges y quests
- Leaderboards y achievements
- Integración con token economy
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChallengeType(Enum):
    """Tipos de challenges educativos"""

    DAILY_PARTICIPATION = "daily_participation"
    WEEKLY_STREAK = "weekly_streak"
    PEER_REVIEW_MASTER = "peer_review_master"
    QUIZ_CHAMPION = "quiz_champion"
    DISCUSSION_STAR = "discussion_star"
    TUTORING_HERO = "tutoring_hero"
    CREATIVE_CONTRIBUTOR = "creative_contributor"
    CONSISTENCY_CHAMPION = "consistency_champion"


class RaffleTicketType(Enum):
    """Tipos de boletos de rifa"""

    STANDARD = "standard"  # 1 ticket
    PREMIUM = "premium"  # 3 tickets
    GOLD = "gold"  # 5 tickets
    PLATINUM = "platinum"  # 10 tickets


@dataclass
class RaffleTicket:
    """Estructura de boleto de rifa"""

    ticket_id: str
    user_id: str
    ticket_type: RaffleTicketType
    earned_for: str  # Actividad que generó el ticket
    earned_at: datetime
    is_used: bool = False
    used_at: Optional[datetime] = None

    @property
    def value(self) -> int:
        """Valor del ticket en unidades de rifa"""
        values = {
            RaffleTicketType.STANDARD: 1,
            RaffleTicketType.PREMIUM: 3,
            RaffleTicketType.GOLD: 5,
            RaffleTicketType.PLATINUM: 10,
        }
        return values[self.ticket_type]


@dataclass
class EducationalChallenge:
    """Challenge educativo con recompensas"""

    challenge_id: str
    name: str
    description: str
    challenge_type: ChallengeType
    requirements: Dict[str, Any]
    rewards: Dict[str, Any]  # tickets, badges, special_items
    duration_days: int
    start_date: datetime
    end_date: datetime
    is_active: bool = True
    participants: List[str] = field(default_factory=list)
    completions: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # user_id -> completion_data

    @property
    def is_expired(self) -> bool:
        """Verificar si el challenge ha expirado"""
        return datetime.now() > self.end_date

    @property
    def days_remaining(self) -> int:
        """Días restantes para completar el challenge"""
        if self.is_expired:
            return 0
        return max(0, (self.end_date - datetime.now()).days)


@dataclass
class RafflePrize:
    """Premio de rifa"""

    prize_id: str
    name: str
    description: str
    value: str  # Descripción del valor
    quantity: int
    remaining: int
    category: str  # educational, entertainment, merchandise, etc.
    sponsor: Optional[str] = None


class GamificationEngine:
    """
    Motor de gamificación educativa
    Gestiona challenges, raffle tickets, premios y engagement
    """

    def __init__(self):
        self.tickets: Dict[str, RaffleTicket] = {}  # ticket_id -> ticket
        self.user_tickets: Dict[str, List[str]] = {}  # user_id -> [ticket_ids]
        self.challenges: Dict[str, EducationalChallenge] = {}
        self.prizes: Dict[str, RafflePrize] = {}
        self.raffle_history: List[Dict[str, Any]] = []

        # Sistema de challenges activos
        self._initialize_default_challenges()

        # Sistema de premios
        self._initialize_default_prizes()

        logger.info("🎮 Gamification Engine initialized")

    def _initialize_default_challenges(self):
        """Inicializar challenges por defecto"""
        now = datetime.now()

        challenges_data = [
            {
                "id": "daily_login_7",
                "name": "Daily Learner Streak",
                "description": "Log in and complete at least one learning activity for 7 consecutive days",
                "type": ChallengeType.WEEKLY_STREAK,
                "requirements": {"consecutive_days": 7, "min_activities_per_day": 1},
                "rewards": {
                    "tickets": RaffleTicketType.PREMIUM,
                    "badge": "consistency_master",
                },
                "duration": 7,
            },
            {
                "id": "peer_review_5",
                "name": "Peer Review Champion",
                "description": "Provide constructive feedback to 5 different peers",
                "type": ChallengeType.PEER_REVIEW_MASTER,
                "requirements": {"reviews_required": 5, "quality_threshold": 0.8},
                "rewards": {"tickets": RaffleTicketType.GOLD, "badge": "peer_mentor"},
                "duration": 14,
            },
            {
                "id": "quiz_perfect_3",
                "name": "Quiz Master",
                "description": "Achieve perfect scores (100%) on 3 different quizzes",
                "type": ChallengeType.QUIZ_CHAMPION,
                "requirements": {"perfect_scores_required": 3, "min_score": 100},
                "rewards": {
                    "tickets": RaffleTicketType.PLATINUM,
                    "badge": "quiz_wizard",
                },
                "duration": 21,
            },
            {
                "id": "discussion_starter_10",
                "name": "Discussion Catalyst",
                "description": "Start 10 thoughtful discussion threads that receive at least 5 responses each",
                "type": ChallengeType.DISCUSSION_STAR,
                "requirements": {"threads_required": 10, "min_responses_per_thread": 5},
                "rewards": {
                    "tickets": RaffleTicketType.GOLD,
                    "badge": "community_builder",
                },
                "duration": 30,
            },
        ]

        for challenge_data in challenges_data:
            challenge = EducationalChallenge(
                challenge_id=challenge_data["id"],
                name=challenge_data["name"],
                description=challenge_data["description"],
                challenge_type=challenge_data["type"],
                requirements=challenge_data["requirements"],
                rewards=challenge_data["rewards"],
                duration_days=challenge_data["duration"],
                start_date=now,
                end_date=now + timedelta(days=challenge_data["duration"]),
            )
            self.challenges[challenge.challenge_id] = challenge

    def _initialize_default_prizes(self):
        """Inicializar premios por defecto"""
        prizes_data = [
            {
                "id": "amazon_gift_25",
                "name": "Amazon Gift Card $25",
                "description": "Tarjeta de regalo de Amazon por $25",
                "value": "$25",
                "quantity": 5,
                "category": "merchandise",
                "sponsor": "Amazon",
            },
            {
                "id": "course_access_premium",
                "name": "Premium Course Access",
                "description": "Acceso gratuito a un curso premium por 3 meses",
                "value": "$99",
                "quantity": 3,
                "category": "educational",
                "sponsor": "Sheily AI",
            },
            {
                "id": "wireless_headphones",
                "name": "Wireless Headphones",
                "description": "Audífonos inalámbricos premium para estudio",
                "value": "$150",
                "quantity": 2,
                "category": "technology",
                "sponsor": "AudioTech",
            },
            {
                "id": "book_voucher_50",
                "name": "Book Store Voucher $50",
                "description": "Vale para comprar libros en librería online",
                "value": "$50",
                "quantity": 8,
                "category": "educational",
                "sponsor": "BookStore",
            },
            {
                "id": "online_course_bundle",
                "name": "Online Learning Bundle",
                "description": "Acceso a bundle de cursos online por 6 meses",
                "value": "$200",
                "quantity": 2,
                "category": "educational",
                "sponsor": "EduPlatform",
            },
        ]

        for prize_data in prizes_data:
            prize = RafflePrize(
                prize_id=prize_data["id"],
                name=prize_data["name"],
                description=prize_data["description"],
                value=prize_data["value"],
                quantity=prize_data["quantity"],
                remaining=prize_data["quantity"],
                category=prize_data["category"],
                sponsor=prize_data["sponsor"],
            )
            self.prizes[prize.prize_id] = prize

    async def earn_raffle_ticket(
        self,
        user_id: str,
        activity: str,
        ticket_type: RaffleTicketType = RaffleTicketType.STANDARD,
    ) -> Dict[str, Any]:
        """
        Otorgar boleto de rifa por actividad educativa
        """
        try:
            # Generar ticket único
            ticket_id = f"ticket_{user_id}_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"

            ticket = RaffleTicket(
                ticket_id=ticket_id,
                user_id=user_id,
                ticket_type=ticket_type,
                earned_for=activity,
                earned_at=datetime.now(),
            )

            # Almacenar ticket
            self.tickets[ticket_id] = ticket

            # Agregar a colección del usuario
            if user_id not in self.user_tickets:
                self.user_tickets[user_id] = []
            self.user_tickets[user_id].append(ticket_id)

            # Verificar si completa algún challenge
            await self._check_challenge_completion(user_id, activity)

            logger.info(
                f"🎫 Raffle ticket earned: {ticket_id} for {user_id} - {activity}"
            )

            return {
                "success": True,
                "ticket_id": ticket_id,
                "ticket_type": ticket_type.value,
                "value": ticket.value,
                "total_user_tickets": len(self.user_tickets.get(user_id, [])),
            }

        except Exception as e:
            logger.error(f"Error earning raffle ticket: {e}")
            return {"success": False, "error": str(e)}

    async def _check_challenge_completion(self, user_id: str, activity: str):
        """
        Verificar si una actividad completa algún challenge
        """
        for challenge in self.challenges.values():
            if not challenge.is_active or challenge.is_expired:
                continue

            # Agregar usuario como participante si no está
            if user_id not in challenge.participants:
                challenge.participants.append(user_id)

            # Verificar completion basado en tipo de challenge
            completion_data = await self._evaluate_challenge_progress(
                user_id, challenge, activity
            )

            if completion_data["completed"]:
                challenge.completions[user_id] = completion_data

                # Otorgar recompensas del challenge
                await self._grant_challenge_rewards(user_id, challenge)

                logger.info(f"🏆 Challenge completed: {challenge.name} by {user_id}")

    async def _evaluate_challenge_progress(
        self, user_id: str, challenge: EducationalChallenge, activity: str
    ) -> Dict[str, Any]:
        """
        Evaluar progreso de challenge basado en actividad
        """
        # Implementación simplificada - en producción sería más sofisticada
        if challenge.challenge_type == ChallengeType.DAILY_PARTICIPATION:
            # Contar actividades diarias
            daily_count = 1  # Simulado
            required_days = challenge.requirements.get("consecutive_days", 7)
            return {
                "completed": daily_count >= required_days,
                "progress": min(daily_count / required_days, 1.0),
                "current_count": daily_count,
            }

        elif challenge.challenge_type == ChallengeType.PEER_REVIEW_MASTER:
            # Contar reviews de peers
            review_count = 1  # Simulado
            required_reviews = challenge.requirements.get("reviews_required", 5)
            return {
                "completed": review_count >= required_reviews,
                "progress": min(review_count / required_reviews, 1.0),
                "current_count": review_count,
            }

        elif challenge.challenge_type == ChallengeType.QUIZ_CHAMPION:
            # Contar quizzes perfectos
            perfect_count = 1  # Simulado
            required_perfect = challenge.requirements.get("perfect_scores_required", 3)
            return {
                "completed": perfect_count >= required_perfect,
                "progress": min(perfect_count / required_perfect, 1.0),
                "current_count": perfect_count,
            }

        return {"completed": False, "progress": 0.0}

    async def _grant_challenge_rewards(
        self, user_id: str, challenge: EducationalChallenge
    ):
        """
        Otorgar recompensas por completar challenge
        """
        rewards = challenge.rewards

        if "tickets" in rewards:
            ticket_type = rewards["tickets"]
            await self.earn_raffle_ticket(
                user_id=user_id,
                activity=f"Challenge completion: {challenge.name}",
                ticket_type=ticket_type,
            )

        # TODO: Implementar badges y otros rewards

    async def conduct_raffle(
        self, prize_id: str, num_winners: int = 1
    ) -> Dict[str, Any]:
        """
        Realizar rifa para un premio específico
        """
        try:
            if prize_id not in self.prizes:
                return {"success": False, "error": "Prize not found"}

            prize = self.prizes[prize_id]
            if prize.remaining <= 0:
                return {"success": False, "error": "No prizes remaining"}

            # Obtener todos los tickets no usados
            available_tickets = [
                ticket for ticket in self.tickets.values() if not ticket.is_used
            ]

            if len(available_tickets) < num_winners:
                return {"success": False, "error": "Insufficient tickets for raffle"}

            # Realizar selección aleatoria (con pesos por valor de ticket)
            ticket_weights = [ticket.value for ticket in available_tickets]
            winners = random.choices(
                available_tickets, weights=ticket_weights, k=num_winners
            )

            # Marcar tickets como usados
            for winner_ticket in winners:
                winner_ticket.is_used = True
                winner_ticket.used_at = datetime.now()

            # Reducir cantidad de premios
            prize.remaining -= num_winners

            # Registrar en historial
            raffle_record = {
                "raffle_id": f"raffle_{prize_id}_{int(datetime.now().timestamp())}",
                "prize_id": prize_id,
                "prize_name": prize.name,
                "winners": [
                    {
                        "user_id": ticket.user_id,
                        "ticket_id": ticket.ticket_id,
                        "ticket_value": ticket.value,
                    }
                    for ticket in winners
                ],
                "total_participants": len(
                    set(ticket.user_id for ticket in available_tickets)
                ),
                "total_tickets": len(available_tickets),
                "conducted_at": datetime.now().isoformat(),
            }

            self.raffle_history.append(raffle_record)

            logger.info(
                f"🎉 Raffle conducted: {prize.name} - Winners: {[w['user_id'] for w in raffle_record['winners']]}"
            )

            return {
                "success": True,
                "raffle_id": raffle_record["raffle_id"],
                "prize": {
                    "id": prize.prize_id,
                    "name": prize.name,
                    "value": prize.value,
                },
                "winners": raffle_record["winners"],
                "total_participants": raffle_record["total_participants"],
            }

        except Exception as e:
            logger.error(f"Error conducting raffle: {e}")
            return {"success": False, "error": str(e)}

    async def get_user_gamification_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de gamificación de un usuario
        """
        try:
            user_tickets = self.user_tickets.get(user_id, [])
            active_tickets = [
                ticket_id
                for ticket_id in user_tickets
                if not self.tickets[ticket_id].is_used
            ]

            total_ticket_value = sum(
                self.tickets[ticket_id].value for ticket_id in active_tickets
            )

            # Challenges activos del usuario
            user_challenges = []
            for challenge in self.challenges.values():
                if user_id in challenge.participants:
                    progress = challenge.completions.get(user_id, {"progress": 0.0})
                    user_challenges.append(
                        {
                            "challenge_id": challenge.challenge_id,
                            "name": challenge.name,
                            "progress": progress.get("progress", 0.0),
                            "completed": progress.get("completed", False),
                            "days_remaining": challenge.days_remaining,
                        }
                    )

            # Historial de victorias en rifas
            wins = [
                raffle
                for raffle in self.raffle_history
                if any(winner["user_id"] == user_id for winner in raffle["winners"])
            ]

            return {
                "user_id": user_id,
                "active_tickets": len(active_tickets),
                "total_ticket_value": total_ticket_value,
                "challenges_participating": len(user_challenges),
                "challenges_completed": sum(
                    1 for c in user_challenges if c["completed"]
                ),
                "raffle_wins": len(wins),
                "recent_wins": wins[-3:] if wins else [],  # Últimas 3 victorias
                "active_challenges": user_challenges,
            }

        except Exception as e:
            logger.error(f"Error getting user gamification stats: {e}")
            return {"error": str(e)}

    async def get_system_gamification_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas generales del sistema de gamificación
        """
        try:
            total_tickets = len(self.tickets)
            used_tickets = sum(1 for ticket in self.tickets.values() if ticket.is_used)
            active_tickets = total_tickets - used_tickets

            total_users = len(self.user_tickets)
            total_raffles = len(self.raffle_history)

            # Estadísticas de challenges
            active_challenges = sum(
                1 for c in self.challenges.values() if c.is_active and not c.is_expired
            )
            completed_challenges = sum(
                len(completions)
                for completions in [c.completions for c in self.challenges.values()]
            )

            # Estadísticas de premios
            total_prizes = sum(p.quantity for p in self.prizes.values())
            remaining_prizes = sum(p.remaining for p in self.prizes.values())

            return {
                "total_tickets": total_tickets,
                "active_tickets": active_tickets,
                "used_tickets": used_tickets,
                "total_users": total_users,
                "total_raffles": total_raffles,
                "active_challenges": active_challenges,
                "completed_challenges": completed_challenges,
                "total_prizes": total_prizes,
                "remaining_prizes": remaining_prizes,
                "prize_redemption_rate": (remaining_prizes / max(total_prizes, 1))
                * 100,
            }

        except Exception as e:
            logger.error(f"Error getting system gamification stats: {e}")
            return {"error": str(e)}

    async def create_custom_challenge(
        self,
        name: str,
        description: str,
        challenge_type: ChallengeType,
        requirements: Dict[str, Any],
        rewards: Dict[str, Any],
        duration_days: int,
    ) -> Dict[str, Any]:
        """
        Crear challenge personalizado
        """
        try:
            challenge_id = (
                f"custom_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"
            )

            challenge = EducationalChallenge(
                challenge_id=challenge_id,
                name=name,
                description=description,
                challenge_type=challenge_type,
                requirements=requirements,
                rewards=rewards,
                duration_days=duration_days,
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=duration_days),
            )

            self.challenges[challenge_id] = challenge

            logger.info(f"🎯 Custom challenge created: {challenge_id}")

            return {
                "success": True,
                "challenge_id": challenge_id,
                "name": name,
                "end_date": challenge.end_date.isoformat(),
            }

        except Exception as e:
            logger.error(f"Error creating custom challenge: {e}")
            return {"success": False, "error": str(e)}


# Instancia global (singleton)
_gamification_engine: Optional[GamificationEngine] = None


def get_gamification_engine() -> GamificationEngine:
    """Obtener instancia singleton del motor de gamificación"""
    global _gamification_engine
    if _gamification_engine is None:
        _gamification_engine = GamificationEngine()
    return _gamification_engine
