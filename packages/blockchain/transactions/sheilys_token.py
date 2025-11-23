#!/usr/bin/env python3
"""
SHEILYS Token - Token nativo del ecosistema Sheily AI MCP Enterprise
Implementación completa compatible con Solana y Web3

El token SHEILYS facilita:
- Gamificación y recompensas automáticas desde blockchain
- Staking con yield rewards sobre blockchain
- NFT credentials verificables on-chain
- Gobernanza del ecosistema Sheily
- Pagos por servicios con contratos inteligentes
"""

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .sheilys_blockchain import SHEILYSBlockchain, TransactionType, SHEILYSTransaction


class SHEILYSTokenStandard(Enum):
    """Estándares de token SHEILYS soportados"""

    SPL_TOKEN = "spl_token"  # Solana Program Library (SPL)
    ERC20_COMPATIBLE = "erc20_compatible"  # Compatible ERC-20


class NFTCollection(Enum):
    """Colecciones NFT disponibles en SHEILYS"""

    ACHIEVEMENT_BADGES = "achievement_badges"
    CREDENTIALS_CERTIFICATES = "credentials_certificates"
    LEARNING_TRACKS = "learning_tracks"
    GAMIFICATION_REWARDS = "gamification_rewards"
    GOVERNANCE_TOKENS = "governance_tokens"


@dataclass
class SHEILYSTokenMetadata:
    """Metadatos del token SHEILYS según estándar Solana Metaplex"""

    name: str
    symbol: str
    description: str
    image: str
    external_url: str
    attributes: List[Dict[str, Any]]
    properties: Dict[str, Any]

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convertir a formato Metaplex compatible"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "description": self.description,
            "image": self.image,
            "external_url": self.external_url,
            "attributes": self.attributes,
            "properties": self.properties,
        }


@dataclass
class SHEILYSNFT:
    """NFT SHEILYS con metadatos extendidos"""

    token_id: str
    collection: NFTCollection
    owner: str
    metadata: Dict[str, Any]
    minted_at: float
    last_transfer: float
    rarity_score: float
    utility_functions: List[str]

    def calculate_rarity(self) -> float:
        """Calcular rareza del NFT basado en sus atributos"""
        # Implementación simplificada - en producción usar algoritmo más sofisticado
        base_rarity = 1.0
        if self.collection == NFTCollection.ACHIEVEMENT_BADGES:
            base_rarity = 1.2
        elif self.collection == NFTCollection.CREDENTIALS_CERTIFICATES:
            base_rarity = 2.0
        elif self.collection == NFTCollection.GOVERNANCE_TOKENS:
            base_rarity = 3.0

        # Factor por edad (NFTs más antiguos son más raros)
        age_factor = min(1.5, (time.time() - self.minted_at) / (365 * 24 * 3600))
        return base_rarity * age_factor


class SHEILYSTokenManager:
    """
    SHEILYS Token Manager - Gestión completa del ecosistema token SHEILYS

    Funcionalidades:
    - Minting automático por IA
    - Rewards por gamificación
    - Staking con yield rewards
    - NFT minting y gestión
    - Gobierno del ecosistema
    """

    def __init__(self, blockchain: Optional[SHEILYSBlockchain] = None):
        """Inicializar token manager"""

        # Conectar con blockchain SHEILYS
        self.blockchain = blockchain or SHEILYSBlockchain()

        # Balances de tokens (SPL compatible)
        self.token_balances: Dict[str, float] = {}

        # Balances de staked tokens
        self.staked_balances: Dict[str, float] = {}

        # Colecciones NFT
        self.nft_collections: Dict[NFTCollection, List[SHEILYSNFT]] = {}
        self.nft_ownership: Dict[str, List[str]] = {}  # owner -> [token_ids]

        # Metadata del token principal
        self.token_metadata = SHEILYSTokenMetadata(
            name="SHEILYS AI",
            symbol="SHEILYS",
            description="Token nativo del ecosistema Sheily AI MCP Enterprise - Learn to Earn",
            image="https://sheily.ai/images/sheilys-token.png",
            external_url="https://sheily.ai",
            attributes=[
                {"trait_type": "Type", "value": "Educational Token"},
                {"trait_type": "Network", "value": "Solana"},
                {"trait_type": "Standard", "value": "SPL Token"},
                {"trait_type": "Features", "value": "Staking, NFTs, Governance"},
            ],
            properties={
                "educational_token": True,
                "learn_to_earn": True,
                "governance_enabled": True,
                "nft_credentials": True,
                "deflationary": True,
                "staking_enabled": True,
                "max_supply": 1000000000,
                "decimals": 9,
            },
        )

        # Sistema de rewards por gamificación
        self.reward_rates = {
            "exercise_completion": 3.0,  # SHEILYS por ejercicio correcto
            "dataset_generation": 10.0,  # SHEILYS por dataset generado
            "knowledge_sharing": 5.0,  # SHEILYS por compartir conocimiento
            "achievements": 25.0,  # SHEILYS por logros especiales
            "level_up": 50.0,  # SHEILYS por subir de nivel
        }

        # Sistema de staking
        self.staking_pools = {
            "validator_pool": {"apy": 12.0, "min_stake": 1000.0},
            "community_pool": {"apy": 8.0, "min_stake": 100.0},
            "education_pool": {"apy": 15.0, "min_stake": 500.0},
        }

        # Gobernanza
        self.proposals: Dict[str, Dict[str, Any]] = {}
        self.voting_power: Dict[str, float] = {}  # Basado en balance + stake

        # Inicializar colecciones NFT
        self._initialize_nft_collections()

    def _initialize_nft_collections(self):
        """Inicializar colecciones NFT por defecto"""
        for collection in NFTCollection:
            self.nft_collections[collection] = []

    def get_balance(self, address: str) -> float:
        """Obtener balance de SHEILYS de una dirección"""
        return self.token_balances.get(address, 0.0)

    def get_staked_balance(self, address: str) -> float:
        """Obtener balance staked de una dirección"""
        return self.staked_balances.get(address, 0.0)

    def mint_tokens(
        self, to_address: str, amount: float, reason: str = "minting"
    ) -> bool:
        """
        Mint tokens SHEILYS (controlado por sistema IA)

        Args:
            to_address: Dirección destinataria
            amount: Cantidad a mintear
            reason: Reason para transparencia

        Returns:
            bool: True si el mint fue exitoso
        """
        try:
            # Verificar límite de supply
            current_supply = sum(self.token_balances.values()) + sum(
                self.staked_balances.values()
            )
            if current_supply + amount > self.token_metadata.properties["max_supply"]:
                return False

            # Mint tokens
            self.token_balances[to_address] = self.get_balance(to_address) + amount

            # Registrar en blockchain
            tx = SHEILYSTransaction(
                transaction_id=f"mint_{to_address}_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                sender="sheily_system_minter",
                receiver=to_address,
                amount=amount,
                transaction_type=TransactionType.REWARD,
                timestamp=time.time(),
                signature=f"system_mint_signature_{reason}",
                metadata={
                    "mint_reason": reason,
                    "type": "token_mint",
                    "supply_inflation": amount,
                },
            )
            tx_success = self.blockchain.add_transaction(tx)

            return tx_success

        except Exception as e:
            print(f"Error minting tokens: {e}")
            return False

    def transfer_tokens(
        self, from_address: str, to_address: str, amount: float
    ) -> bool:
        """
        Transferir tokens SHEILYS entre direcciones

        Returns:
            bool: True si la transferencia fue exitosa
        """
        try:
            available_balance = self.get_balance(from_address)
            if available_balance < amount:
                return False

            # Ejecutar transferencia
            self.token_balances[from_address] -= amount
            self.token_balances[to_address] = self.get_balance(to_address) + amount

            # Registrar transacción en blockchain
            tx_id = f"transfer_{from_address}_{to_address}_{int(time.time())}"
            tx_hash = hashlib.sha256(
                f"{tx_id}{from_address}{to_address}{amount}{time.time()}".encode()
            ).hexdigest()

            self.blockchain.add_transaction(
                SHEILYSTransaction(
                    transaction_id=tx_hash,
                    sender=from_address,
                    receiver=to_address,
                    amount=amount,
                    transaction_type=TransactionType.TRANSFER,
                    timestamp=time.time(),
                    signature=f"{from_address}_transfer_sig",
                    metadata={
                        "transfer_type": "direct_transfer",
                        "gas_used": 0.001,  # Gas simbólico
                    },
                )
            )

            # Auto-burn (quemado automático del 1% para deflación)
            burn_amount = amount * 0.01
            if burn_amount > 0:
                self.token_balances[from_address] -= burn_amount

            return True

        except Exception as e:
            print(f"Error transferring tokens: {e}")
            return False

    def stake_tokens(
        self, address: str, amount: float, pool_name: str = "community_pool"
    ) -> bool:
        """
        Stake tokens para ganar rewards

        Args:
            address: Dirección que hace stake
            amount: Cantidad a stakear
            pool_name: Pool de staking

        Returns:
            bool: True si el stake fue exitoso
        """
        try:
            if pool_name not in self.staking_pools:
                return False

            pool_config = self.staking_pools[pool_name]
            if amount < pool_config["min_stake"]:
                return False

            available_balance = self.get_balance(address)
            if available_balance < amount:
                return False

            # Mover a stake
            self.token_balances[address] -= amount
            self.staked_balances[address] = self.get_staked_balance(address) + amount

            # Registrar en blockchain
            self.blockchain.stake_tokens(address, amount)

            return True

        except Exception as e:
            print(f"Error staking tokens: {e}")
            return False

    def claim_staking_rewards(self, address: str) -> float:
        """
        Claim staking rewards acumulados

        Returns:
            float: Cantidad de rewards claimed
        """
        try:
            staked_amount = self.get_staked_balance(address)
            if staked_amount == 0:
                return 0.0

            # Calcular rewards basado en APY promedio (implementación simplificada)
            avg_apy = 10.0  # 10% APY promedio
            daily_reward = staked_amount * (avg_apy / 100) / 365
            claimed_amount = daily_reward * 30  # Simular 30 días

            # Mint rewards
            self.mint_tokens(address, claimed_amount, f"staking_rewards_{address}")

            return claimed_amount

        except Exception as e:
            print(f"Error claiming staking rewards: {e}")
            return 0.0

    def reward_gamification_action(self, user_address: str, action_type: str) -> float:
        """
        Reward tokens por acciones de gamificación

        Args:
            user_address: Usuario que recibe reward
            action_type: Tipo de acción ('exercise_completion', etc.)

        Returns:
            float: Cantidad de tokens rewarded
        """
        if action_type not in self.reward_rates:
            return 0.0

        reward_amount = self.reward_rates[action_type]

        # Mint reward tokens
        success = self.mint_tokens(
            user_address, reward_amount, f"gamification_{action_type}"
        )

        return reward_amount if success else 0.0

    def mint_nft(
        self, collection: NFTCollection, owner: str, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Mint un nuevo NFT SHEILYS

        Args:
            collection: Colección del NFT
            owner: Propietario del NFT
            metadata: Metadata del NFT

        Returns:
            str: Token ID del NFT minteado, o None si falló
        """
        try:
            # Generar token ID único
            token_id = f"{collection.value}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

            # Crear NFT
            nft = SHEILYSNFT(
                token_id=token_id,
                collection=collection,
                owner=owner,
                metadata=metadata,
                minted_at=time.time(),
                last_transfer=time.time(),
                rarity_score=0.0,  # Se calcula después
                utility_functions=self._get_nft_utility_functions(collection),
            )

            # Calcular rareza
            nft.rarity_score = nft.calculate_rarity()

            # Agregar a colección
            self.nft_collections[collection].append(nft)

            # Actualizar ownership
            if owner not in self.nft_ownership:
                self.nft_ownership[owner] = []
            self.nft_ownership[owner].append(token_id)

            # Registrar en blockchain
            self.blockchain.add_transaction(
                {
                    "transaction_id": f"nft_mint_{token_id}",
                    "sender": "sheily_nft_minter",
                    "receiver": owner,
                    "amount": 1,  # NFTs son únicos
                    "transaction_type": TransactionType.NFT_MINT,
                    "timestamp": time.time(),
                    "signature": f"nft_mint_signature_{token_id}",
                    "metadata": {
                        "nft_token_id": token_id,
                        "collection": collection.value,
                        "rarity": nft.rarity_score,
                    },
                }
            )

            return token_id

        except Exception as e:
            print(f"Error minting NFT: {e}")
            return None

    def _get_nft_utility_functions(self, collection: NFTCollection) -> List[str]:
        """Obtener funciones de utilidad de un NFT basado en su colección"""
        utilities = {
            NFTCollection.ACHIEVEMENT_BADGES: [
                "access_premium_features",
                "vote_governance",
                "exclusive_community",
            ],
            NFTCollection.CREDENTIALS_CERTIFICATES: [
                "verify_credentials",
                "access_certified_content",
                "professional_networking",
            ],
            NFTCollection.LEARNING_TRACKS: [
                "unlock_advance_content",
                "certificate_verification",
                "career_credits",
            ],
            NFTCollection.GAMIFICATION_REWARDS: [
                "boosted_rewards",
                "exclusive_events",
                "leaderboard_bonus",
            ],
            NFTCollection.GOVERNANCE_TOKENS: [
                "voting_power_boost",
                "proposal_creation",
                "delegate_voting",
            ],
        }
        return utilities.get(collection, [])

    def transfer_nft(self, token_id: str, from_address: str, to_address: str) -> bool:
        """
        Transferir NFT entre direcciones

        Returns:
            bool: True si la transferencia fue exitosa
        """
        try:
            # Encontrar NFT
            nft = None
            collection = None
            for coll_name, nfts in self.nft_collections.items():
                for n in nfts:
                    if n.token_id == token_id:
                        nft = n
                        collection = coll_name
                        break
                if nft:
                    break

            if not nft or nft.owner != from_address:
                return False

            # Actualizar propietario
            nft.owner = to_address
            nft.last_transfer = time.time()

            # Actualizar ownership tracking
            if from_address in self.nft_ownership:
                self.nft_ownership[from_address].remove(token_id)
            if to_address not in self.nft_ownership:
                self.nft_ownership[to_address] = []
            self.nft_ownership[to_address].append(token_id)

            # Registrar en blockchain (corregir bug cuando collection es None)
            collection_str = collection.value if collection else "unknown"
            self.blockchain.add_transaction(
                {
                    "transaction_id": f"nft_transfer_{token_id}_{from_address}_{to_address}",
                    "sender": from_address,
                    "receiver": to_address,
                    "amount": 1,
                    "transaction_type": TransactionType.NFT_TRANSFER,
                    "timestamp": time.time(),
                    "signature": f"nft_transfer_signature_{token_id}",
                    "metadata": {
                        "nft_token_id": token_id,
                        "collection": collection_str,
                        "rarity": nft.rarity_score,
                    },
                }
            )

            return True

        except Exception as e:
            print(f"Error transferring NFT: {e}")
            return False

    def get_user_nfts(self, address: str) -> List[Dict[str, Any]]:
        """Obtener NFTs de un usuario"""
        user_nfts = []
        for collection, nfts in self.nft_collections.items():
            for nft in nfts:
                if nft.owner == address:
                    user_nfts.append(
                        {
                            "token_id": nft.token_id,
                            "collection": collection.value,
                            "metadata": nft.metadata,
                            "rarity_score": nft.rarity_score,
                            "minted_at": nft.minted_at,
                            "utility_functions": nft.utility_functions,
                        }
                    )
        return user_nfts

    def create_governance_proposal(
        self, proposer: str, title: str, description: str, voting_period_days: int = 7
    ) -> str:
        """
        Crear propuesta de gobernanza

        Returns:
            str: ID de la propuesta
        """
        proposal_id = (
            f"prop_{int(time.time())}_{hashlib.sha256(title.encode()).hexdigest()[:8]}"
        )

        voting_power = self.get_balance(proposer) + self.get_staked_balance(proposer)

        self.proposals[proposal_id] = {
            "id": proposal_id,
            "proposer": proposer,
            "title": title,
            "description": description,
            "created_at": time.time(),
            "voting_ends": time.time() + (voting_period_days * 24 * 3600),
            "total_votes": 0,
            "votes_for": 0,
            "votes_against": 0,
            "status": "active",
        }

        return proposal_id

    def vote_on_proposal(self, proposal_id: str, voter: str, votes_for: bool) -> bool:
        """
        Votar en una propuesta de gobernanza

        Args:
            proposal_id: ID de la propuesta
            voter: Dirección del votante
            votes_for: True para votar a favor, False para votar en contra

        Returns:
            bool: True si el voto fue registrado
        """
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]

        if time.time() > proposal["voting_ends"]:
            return False

        # Calcular poder de voto
        voting_power = self.get_balance(voter) + self.get_staked_balance(voter)

        if votes_for:
            proposal["votes_for"] += voting_power
        else:
            proposal["votes_against"] += voting_power

        proposal["total_votes"] += voting_power

        return True

    def get_token_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del token SHEILYS"""
        total_supply = sum(self.token_balances.values()) + sum(
            self.staked_balances.values()
        )
        total_staked = sum(self.staked_balances.values())
        total_holders = len([b for b in self.token_balances.values() if b > 0])
        total_stakers = len([s for s in self.staked_balances.values() if s > 0])

        # Calcular métricas NFT
        total_nfts = sum(len(nfts) for nfts in self.nft_collections.values())

        return {
            "token_metadata": self.token_metadata.to_metadata_dict(),
            "total_supply": total_supply,
            "circulating_supply": total_supply,
            "staked_supply": total_staked,
            "holders_count": total_holders,
            "stakers_count": total_stakers,
            "staking_ratio": total_staked / total_supply if total_supply > 0 else 0,
            "total_nfts": total_nfts,
            "nft_collections": {
                k.value: len(v) for k, v in self.nft_collections.items()
            },
            "active_proposals": len(
                [p for p in self.proposals.values() if p["status"] == "active"]
            ),
            "reward_rates": self.reward_rates,
        }
