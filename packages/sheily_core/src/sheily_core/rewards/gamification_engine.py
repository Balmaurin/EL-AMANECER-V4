#!/usr/bin/env python3
"""
SHEILYS Gamification Engine
Motor central de gamificación que integra blockchain, NFTs y aprendizaje

Conecta el sistema educativo con el blockchain SHEILYS para crear
un ecosistema completo de Learn-to-Earn con NFTs y recompensas.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from sheily_core.blockchain.transactions import (
        BlockchainWallet,
        NFTCollection,
        SHEILYSTokenManager,
    )
except ImportError:
    # Fallback para desarrollo si los componentes blockchain no están disponibles
    print("Blockchain components not available, running in educational mode")

    class SHEILYSTokenManager:
        def reward_gamification_action(self, user, reason):
            return True

        def get_balance(self, user):
            return 0.0

        def get_staked_balance(self, user):
            return 0.0

        def create_governance_proposal(self, *args):
            return "mock_proposal"

        def vote_on_proposal(self, *args, **kwargs):
            return True

        def mint_nft(self, collection, owner, metadata):
            return f"mock_nft_{int(time.time())}"

    class NFTCollection:
        ACHIEVEMENT_BADGES = "achievement_badges"
        CREDENTIALS_CERTIFICATES = "credentials_certificates"
        GAMIFICATION_REWARDS = "gamification_rewards"

    class BlockchainWallet:
        def create_wallet(self):
            return "mock_wallet"

        def connect_to_blockchain(self, *args):
            pass

        def get_balance(self):
            return {"sheilys": 0.0}


class GamificationLevel(Enum):
    """Niveles de gamificación"""

    NOVICE = "novice"  # Principiante
    LEARNER = "learner"  # Aprendiz
    EXPERT = "expert"  # Experto
    MASTER = "master"  # Maestro
    SAGE = "sage"  # Sabio
    GRANDMASTER = "grandmaster"  # Gran Maestro


class AchievementType(Enum):
    """Tipos de logros disponibles"""

    EXERCISE_COMPLETION = "exercise_completion"
    ACCURACY_STREAK = "accuracy_streak"
    KNOWLEDGE_MASTERY = "knowledge_mastery"
    SHARING_CONTRIBUTION = "sharing_contribution"
    CONSISTENCY_REWARD = "consistency_reward"
    SPECIAL_CHALLENGE = "special_challenge"


@dataclass
class UserGamificationProfile:
    """Perfil de gamificación de usuario"""

    user_id: str
    level: GamificationLevel
    experience_points: int
    total_exercises_completed: int
    accuracy_rate: float
    current_streak: int
    longest_streak: int
    achievements_unlocked: List[str]
    nft_badges: List[str]
    total_sheilyns_earned: float
    joined_at: float
    last_activity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "level": self.level.value,
            "experience_points": self.experience_points,
            "total_exercises_completed": self.total_exercises_completed,
            "accuracy_rate": self.accuracy_rate,
            "current_streak": self.current_streak,
            "longest_streak": self.longest_streak,
            "achievements_unlocked": self.achievements_unlocked,
            "nft_badges": self.nft_badges,
            "total_sheilyns_earned": self.total_sheilyns_earned,
            "joined_at": self.joined_at,
            "last_activity": self.last_activity,
        }


@dataclass
class GamificationAchievement:
    """Logro de gamificación"""

    achievement_id: str
    name: str
    description: str
    type: AchievementType
    requirement: Dict[str, Any]
    reward_sheilyns: float
    nft_reward: Optional[Dict[str, Any]] = None
    unlocked_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "achievement_id": self.achievement_id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "requirement": self.requirement,
            "reward_sheilyns": self.reward_sheilyns,
            "nft_reward": self.nft_reward,
            "unlocked_at": self.unlocked_at,
        }


class GamificationEngine:
    """
    Motor Central de Gamificación SHEILYS

    Integra todo el ecosistema:
    - Sistema educativo con ejercicios
    - Blockchain SHEILYS para recompensas
    - NFTs para achievements
    - Sistema de niveles y streaks
    - Analytics predictivos de aprendizaje
    """

    def __init__(self, token_manager: SHEILYSTokenManager):
        """Inicializar motor de gamificación"""
        self.token_manager = token_manager

        # Perfiles de usuarios
        self.user_profiles: Dict[str, UserGamificationProfile] = {}

        # Logros disponibles
        self.available_achievements: Dict[str, GamificationAchievement] = {}
        self._initialize_achievements()

        # Sistema de niveles
        self.level_thresholds = {
            GamificationLevel.NOVICE: 0,
            GamificationLevel.LEARNER: 100,
            GamificationLevel.EXPERT: 500,
            GamificationLevel.MASTER: 1500,
            GamificationLevel.SAGE: 3000,
            GamificationLevel.GRANDMASTER: 5000,
        }

        # Conexión con blockchain wallet (opcional)
        self.system_wallet: Optional[BlockchainWallet] = None

        print("🎮 Gamification Engine SHEILYS inicializado")

    def _initialize_achievements(self):
        """Inicializar logros disponibles"""
        achievements_data = [
            # Exercise Completions
            {
                "id": "first_steps",
                "name": "Primeros Pasos",
                "description": "Completa tu primer ejercicio",
                "type": AchievementType.EXERCISE_COMPLETION,
                "requirement": {"exercises_completed": 1},
                "reward_sheilyns": 5.0,
                "nft_type": "achievement_badge",
            },
            {
                "id": "practice_makes_perfect",
                "name": "La Práctica Hace al Maestro",
                "description": "Completa 10 ejercicios",
                "type": AchievementType.EXERCISE_COMPLETION,
                "requirement": {"exercises_completed": 10},
                "reward_sheilyns": 25.0,
                "nft_type": "achievement_badge",
            },
            {
                "id": "knowledge_seeker",
                "name": "Buscador del Conocimiento",
                "description": "Completa 50 ejercicios",
                "type": AchievementType.EXERCISE_COMPLETION,
                "requirement": {"exercises_completed": 50},
                "reward_sheilyns": 150.0,
                "nft_type": "achievement_badge",
            },
            {
                "id": "learning_machine",
                "name": "Máquina de Aprendizaje",
                "description": "Completa 100 ejercicios",
                "type": AchievementType.EXERCISE_COMPLETION,
                "requirement": {"exercises_completed": 100},
                "reward_sheilyns": 500.0,
                "nft_type": "achievement_badge",
            },
            # Accuracy Streaks
            {
                "id": "perfect_start",
                "name": "Inicio Perfecto",
                "description": "Consigue 100% en tu primer ejercicio",
                "type": AchievementType.ACCURACY_STREAK,
                "requirement": {"first_exercise_accuracy": 100.0},
                "reward_sheilyns": 10.0,
                "nft_type": "achievement_badge",
            },
            {
                "id": "streak_master",
                "name": "Maestro de Racha",
                "description": "Mantén una racha de precisión del 90% durante 10 ejercicios",
                "type": AchievementType.ACCURACY_STREAK,
                "requirement": {"streak_length": 10, "min_accuracy": 90.0},
                "reward_sheilyns": 200.0,
                "nft_type": "achievement_badge",
            },
            {
                "id": "precision_expert",
                "name": "Experto en Precisión",
                "description": "Consigue 95% de precisión general",
                "type": AchievementType.KNOWLEDGE_MASTERY,
                "requirement": {"overall_accuracy": 95.0},
                "reward_sheilyns": 300.0,
                "nft_type": "credentials_certificate",
            },
            # Consistency Rewards
            {
                "id": "dedicated_learner",
                "name": "Aprendiz Dedicado",
                "description": "Aprende durante 7 días consecutivos",
                "type": AchievementType.CONSISTENCY_REWARD,
                "requirement": {"consecutive_days": 7},
                "reward_sheilyns": 100.0,
                "nft_type": "achievement_badge",
            },
            {
                "id": "persistent_scholar",
                "name": "Estudioso Persistente",
                "description": "Aprende durante 30 días consecutivos",
                "type": AchievementType.CONSISTENCY_REWARD,
                "requirement": {"consecutive_days": 30},
                "reward_sheilyns": 500.0,
                "nft_type": "achievement_badge",
            },
            # Special Challenges
            {
                "id": "speed_demon",
                "name": "Demonio de la Velocidad",
                "description": "Completa un ejercicio completo en menos de 2 minutos",
                "type": AchievementType.SPECIAL_CHALLENGE,
                "requirement": {"completion_time_seconds": 120},
                "reward_sheilyns": 75.0,
                "nft_type": "gamification_reward",
            },
            {
                "id": "perfect_score_master",
                "name": "Maestro de Puntaje Perfecto",
                "description": "Consigue 100% de precisión en 5 ejercicios consecutivos",
                "type": AchievementType.SPECIAL_CHALLENGE,
                "requirement": {"perfect_scores_streak": 5},
                "reward_sheilyns": 250.0,
                "nft_type": "credentials_certificate",
            },
        ]

        for achievement_data in achievements_data:
            achievement = GamificationAchievement(
                achievement_id=achievement_data["id"],
                name=achievement_data["name"],
                description=achievement_data["description"],
                type=achievement_data["type"],
                requirement=achievement_data["requirement"],
                reward_sheilyns=achievement_data["reward_sheilyns"],
                nft_reward=(
                    {
                        "collection": achievement_data["nft_type"],
                        "metadata": {
                            "name": achievement_data["name"],
                            "description": achievement_data["description"],
                            "achievement_type": achievement_data["type"].value,
                            "unlocked_date": None,
                            "rarity": "common",
                        },
                    }
                    if achievement_data["nft_type"]
                    else None
                ),
            )
            self.available_achievements[achievement.achievement_id] = achievement

    def connect_system_wallet(self, wallet: BlockchainWallet):
        """Conectar con la wallet del sistema para distribuir recompensas"""
        self.system_wallet = wallet
        print("🔗 Gamification Engine conectado con blockchain wallet")

    def register_user(self, user_id: str) -> UserGamificationProfile:
        """
        Registrar nuevo usuario en el sistema de gamificación

        Args:
            user_id: ID único del usuario

        Returns:
            UserGamificationProfile: Perfil creado
        """
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        profile = UserGamificationProfile(
            user_id=user_id,
            level=GamificationLevel.NOVICE,
            experience_points=0,
            total_exercises_completed=0,
            accuracy_rate=0.0,
            current_streak=0,
            longest_streak=0,
            achievements_unlocked=[],
            nft_badges=[],
            total_sheilyns_earned=0.0,
            joined_at=time.time(),
            last_activity=time.time(),
        )

        self.user_profiles[user_id] = profile

        # Mint welcome SHEILYS (simbólico por ahora)
        self._reward_user_tokens(user_id, 1.0, "welcome_bonus")

        print(f"✅ Usuario {user_id} registrado en gamificación")
        return profile

    def process_exercise_completion(
        self, user_id: str, exercise_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesar completación de ejercicio y calcular recompensas

        Args:
            exercise_data: Datos del ejercicio completado

        Returns:
            dict: Resultados del procesamiento con recompensas
        """
        if user_id not in self.user_profiles:
            self.register_user(user_id)

        profile = self.user_profiles[user_id]
        profile.last_activity = time.time()

        # Actualizar estadísticas del usuario
        correct_answers = exercise_data.get("correct", 0)
        total_questions = exercise_data.get("total", 0)
        accuracy = (
            (correct_answers / total_questions * 100) if total_questions > 0 else 0
        )

        # Calcular experience points
        base_xp = correct_answers * 10  # 10 XP por respuesta correcta
        accuracy_bonus = int(base_xp * (accuracy / 100))  # Bonus por precisión
        total_xp = base_xp + accuracy_bonus

        profile.experience_points += total_xp
        profile.total_exercises_completed += 1

        # Actualizar racha de precisión
        if accuracy >= 70.0:  # Consideramos buena una precisión del 70%+
            profile.current_streak += 1
            if profile.current_streak > profile.longest_streak:
                profile.longest_streak = profile.current_streak
        else:
            profile.current_streak = 0

        # Actualizar tasa de precisión general
        total_accuracy_sum = profile.accuracy_rate * (
            profile.total_exercises_completed - 1
        )
        profile.accuracy_rate = (
            total_accuracy_sum + accuracy
        ) / profile.total_exercises_completed

        # Actualizar nivel
        new_level = self._calculate_user_level(profile.experience_points)
        level_up = new_level != profile.level
        if level_up:
            old_level = profile.level
            profile.level = new_level
            print(
                f"🎉 Usuario {user_id} subió de nivel: {old_level.value} → {new_level.value}"
            )

        # Calcular recompensas SHEILYS
        base_reward = correct_answers * 3.0  # 3 SHEILYS por respuesta correcta
        streak_bonus = profile.current_streak * 0.5  # Bonus por racha
        accuracy_bonus_reward = base_reward * (accuracy / 100)  # Bonus por precisión
        total_sheilyns_reward = base_reward + streak_bonus + accuracy_bonus_reward

        # Reward level up bonus
        if level_up:
            level_up_bonus = self._get_level_up_bonus(new_level)
            total_sheilyns_reward += level_up_bonus

        # Otorgar recompensas
        self._reward_user_tokens(user_id, total_sheilyns_reward, "exercise_completion")

        # Verificar achievements desbloqueados
        new_achievements = self._check_achievement_unlocks(user_id, exercise_data)

        # Actualizar perfil
        profile.total_sheilyns_earned += total_sheilyns_reward

        result = {
            "experience_gained": total_xp,
            "sheilyns_earned": total_sheilyns_reward,
            "accuracy_percentage": accuracy,
            "current_streak": profile.current_streak,
            "new_level": level_up,
            "level_up_from": profile.level.value if level_up else None,
            "level_up_to": new_level.value if level_up else None,
            "new_achievements": new_achievements,
            "current_level": profile.level.value,
            "total_experience": profile.experience_points,
            "total_exercises": profile.total_exercises_completed,
            "overall_accuracy": profile.accuracy_rate,
        }

        return result

    def _calculate_user_level(self, experience_points: int) -> GamificationLevel:
        """Calcular nivel basado en puntos de experiencia"""
        for level, threshold in reversed(list(self.level_thresholds.items())):
            if experience_points >= threshold:
                return level
        return GamificationLevel.NOVICE

    def _get_level_up_bonus(self, level: GamificationLevel) -> float:
        """Obtener bonus de SHEILYS por subir de nivel"""
        bonuses = {
            GamificationLevel.LEARNER: 10.0,
            GamificationLevel.EXPERT: 25.0,
            GamificationLevel.MASTER: 50.0,
            GamificationLevel.SAGE: 100.0,
            GamificationLevel.GRANDMASTER: 250.0,
        }
        return bonuses.get(level, 0.0)

    def _reward_user_tokens(self, user_id: str, amount: float, reason: str):
        """Recompensar SHEILYS a usuario"""
        try:
            # Usar el token manager para mint y transfer
            success = self.token_manager.reward_gamification_action(user_id, reason)

            if success:
                # Actualizar estadísticas del perfil
                if user_id in self.user_profiles:
                    self.user_profiles[user_id].total_sheilyns_earned += amount

                print(f"💰 {amount} SHEILYS recompensados a {user_id} por: {reason}")
            else:
                print(f"⚠️ Error recompensando {amount} SHEILYS a {user_id}")

        except Exception as e:
            print(f"Error en recompensa: {e}")

    def _check_achievement_unlocks(
        self, user_id: str, exercise_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Verificar qué achievements han sido desbloqueados

        Returns:
            List[Dict]: Lista de achievements desbloqueados
        """
        profile = self.user_profiles[user_id]
        new_achievements = []

        for achievement in self.available_achievements.values():
            if achievement.achievement_id in profile.achievements_unlocked:
                continue  # Ya desbloqueado

            if self._check_achievement_requirement(profile, achievement, exercise_data):
                # Desbloquear achievement
                achievement.unlocked_at = time.time()
                profile.achievements_unlocked.append(achievement.achievement_id)

                # Otorgar recompensas
                self._reward_user_tokens(
                    user_id,
                    achievement.reward_sheilyns,
                    f"achievement_{achievement.achievement_id}",
                )

                # Mint NFT si corresponde
                if achievement.nft_reward:
                    self._mint_achievement_nft(user_id, achievement)

                new_achievements.append(
                    {
                        "achievement_id": achievement.achievement_id,
                        "name": achievement.name,
                        "description": achievement.description,
                        "sheilyns_reward": achievement.reward_sheilyns,
                        "nft_minted": achievement.nft_reward is not None,
                    }
                )

        return new_achievements

    def _check_achievement_requirement(
        self,
        profile: UserGamificationProfile,
        achievement: GamificationAchievement,
        exercise_data: Dict[str, Any],
    ) -> bool:
        """Verificar si se cumple el requisito de un achievement"""
        req = achievement.requirement

        if achievement.type == AchievementType.EXERCISE_COMPLETION:
            return profile.total_exercises_completed >= req.get(
                "exercises_completed", 0
            )

        elif achievement.type == AchievementType.KNOWLEDGE_MASTERY:
            return profile.accuracy_rate >= req.get("overall_accuracy", 0)

        elif achievement.type == AchievementType.ACCURACY_STREAK:
            if "first_exercise_accuracy" in req:
                return (
                    profile.total_exercises_completed == 1
                    and exercise_data.get("accuracy", 0)
                    >= req["first_exercise_accuracy"]
                )
            elif "streak_length" in req:
                return profile.current_streak >= req[
                    "streak_length"
                ] and profile.accuracy_rate >= req.get("min_accuracy", 0)

        elif achievement.type == AchievementType.SPECIAL_CHALLENGE:
            if "completion_time_seconds" in req:
                completion_time = exercise_data.get(
                    "completion_time_seconds", float("inf")
                )
                return completion_time <= req["completion_time_seconds"]
            elif "perfect_scores_streak" in req:
                # Contar respuestas perfectas consecutivas
                return profile.longest_streak >= req["perfect_scores_streak"]

        return False

    def _mint_achievement_nft(self, user_id: str, achievement: GamificationAchievement):
        """Mint NFT para achievement desbloqueado"""
        try:
            if not achievement.nft_reward:
                return

            # Determinar colección NFT
            nft_collection_map = {
                "achievement_badge": NFTCollection.ACHIEVEMENT_BADGES,
                "credentials_certificate": NFTCollection.CREDENTIALS_CERTIFICATES,
                "gamification_reward": NFTCollection.GAMIFICATION_REWARDS,
            }

            collection = nft_collection_map.get(
                achievement.nft_reward["collection"], NFTCollection.ACHIEVEMENT_BADGES
            )

            # Preparar metadata
            metadata = achievement.nft_reward["metadata"].copy()
            metadata["unlocked_date"] = achievement.unlocked_at
            metadata["recipient"] = user_id

            # Mint NFT
            token_id = self.token_manager.mint_nft(collection, user_id, metadata)

            if token_id:
                # Agregar a perfil del usuario
                profile = self.user_profiles[user_id]
                profile.nft_badges.append(token_id)

                print(
                    f"🎨 NFT {token_id} minteado para usuario {user_id} - Achievement: {achievement.name}"
                )

        except Exception as e:
            print(f"Error minting achievement NFT: {e}")

    def get_user_gamification_stats(self, user_id: str) -> Dict[str, Any]:
        """Obtener estadísticas completas de gamificación de usuario"""
        if user_id not in self.user_profiles:
            return {"error": "Usuario no encontrado en gamificación"}

        profile = self.user_profiles[user_id]

        # Calcular progreso hacia siguiente nivel
        current_level_threshold = self.level_thresholds[profile.level]
        next_level = self._get_next_level(profile.level)
        if next_level:
            next_threshold = self.level_thresholds.get(
                next_level, profile.experience_points
            )
            progress_to_next = (
                (
                    (profile.experience_points - current_level_threshold)
                    / (next_threshold - current_level_threshold)
                )
                * 100
                if next_threshold > current_level_threshold
                else 100
            )
        else:
            progress_to_next = 100  # Máximo nivel alcanzado

        return {
            "user_id": profile.user_id,
            "current_level": profile.level.value,
            "next_level": next_level.value if next_level else None,
            "experience_points": profile.experience_points,
            "progress_to_next_level": min(progress_to_next, 100),
            "total_exercises_completed": profile.total_exercises_completed,
            "accuracy_rate": profile.accuracy_rate,
            "current_streak": profile.current_streak,
            "longest_streak": profile.longest_streak,
            "achievements_unlocked": len(profile.achievements_unlocked),
            "nft_badges_owned": len(profile.nft_badges),
            "total_sheilyns_earned": profile.total_sheilyns_earned,
            "member_since": profile.joined_at,
            "last_activity": profile.last_activity,
            "recent_achievements": (
                profile.achievements_unlocked[-5:]
                if profile.achievements_unlocked
                else []
            ),
        }

    def _get_next_level(
        self, current_level: GamificationLevel
    ) -> Optional[GamificationLevel]:
        """Obtener siguiente nivel"""
        levels_order = list(GamificationLevel)
        current_index = levels_order.index(current_level)
        if current_index + 1 < len(levels_order):
            return levels_order[current_index + 1]
        return None

    def get_leaderboard(
        self, category: str = "experience", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtener leaderboard por categoría

        Args:
            category: 'experience', 'accuracy', 'streaks', 'sheilyns'
            limit: Número máximo de resultados

        Returns:
            List[Dict]: Top usuarios en la categoría
        """
        if not self.user_profiles:
            return []

        profiles = list(self.user_profiles.values())

        if category == "experience":
            profiles.sort(key=lambda p: p.experience_points, reverse=True)
            key_name = "experience_points"
        elif category == "accuracy":
            profiles.sort(key=lambda p: p.accuracy_rate, reverse=True)
            key_name = "accuracy_rate"
        elif category == "streaks":
            profiles.sort(key=lambda p: p.longest_streak, reverse=True)
            key_name = "longest_streak"
        elif category == "sheilyns":
            profiles.sort(key=lambda p: p.total_sheilyns_earned, reverse=True)
            key_name = "total_sheilyns_earned"
        else:
            return []

        leaderboard = []
        for i, profile in enumerate(profiles[:limit], 1):
            leaderboard.append(
                {
                    "rank": i,
                    "user_id": profile.user_id,
                    "value": getattr(profile, key_name),
                    "level": profile.level.value,
                }
            )

        return leaderboard

    def get_available_achievements(self) -> List[Dict[str, Any]]:
        """Obtener lista de achievements disponibles"""
        return [
            achievement.to_dict()
            for achievement in self.available_achievements.values()
        ]

    def export_gamification_data(self, user_id: str) -> Dict[str, Any]:
        """Exportar todos los datos de gamificación de un usuario"""
        if user_id not in self.user_profiles:
            return {"error": "Usuario no encontrado"}

        profile = self.user_profiles[user_id]

        return {
            "profile": profile.to_dict(),
            "achievements": [
                self.available_achievements[aid].to_dict()
                for aid in profile.achievements_unlocked
                if aid in self.available_achievements
            ],
            "export_timestamp": time.time(),
        }
