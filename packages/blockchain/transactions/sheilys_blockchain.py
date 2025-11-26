#!/usr/bin/env python3
"""
SHEILYS Blockchain Core
Implementación del núcleo blockchain para el token SHEILYS de Sheily AI MCP Enterprise

Características:
- Proof-of-Stake (PoS) optimizado para enterprise
- Minting controlado por sistema IA
- Integración con gamificación y NFTs
- Zero-trust security validation
"""

import hashlib
import json
import secrets
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TransactionType(Enum):
    """Tipos de transacciones soportadas por SHEILYS blockchain"""

    TRANSFER = "transfer"
    STAKE = "stake"
    UNSTAKE = "unstake"
    REWARD = "reward"
    NFT_MINT = "nft_mint"
    NFT_TRANSFER = "nft_transfer"
    GAMIFICATION = "gamification"


class BlockStatus(Enum):
    """Estados de los bloques"""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"


@dataclass
class SHEILYSTransaction:
    """Transacción SHEILYS blockchain"""

    transaction_id: str
    sender: str
    receiver: str
    amount: float
    transaction_type: TransactionType
    timestamp: float
    signature: str
    metadata: Dict[str, Any]
    gas_used: float = 0.0
    block_height: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        data = asdict(self)
        data["transaction_type"] = self.transaction_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SHEILYSTransaction":
        """Crear desde diccionario"""
        data_copy = data.copy()
        data_copy["transaction_type"] = TransactionType(data["transaction_type"])
        return cls(**data_copy)

    def calculate_hash(self) -> str:
        """Calcular hash de la transacción"""
        tx_data = {
            "transaction_id": self.transaction_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "transaction_type": self.transaction_type.value,
            "timestamp": self.timestamp,
            "metadata": json.dumps(self.metadata, sort_keys=True),
        }
        tx_string = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()


@dataclass
class SHEILYSBlock:
    """Bloque SHEILYS blockchain"""

    block_height: int
    previous_hash: str
    timestamp: float
    transactions: List[SHEILYSTransaction]
    validator: str  # Address del validador (PoS)
    block_hash: str = ""
    status: BlockStatus = BlockStatus.PENDING
    difficulty: int = 1  # Para compatibilidad futura con PoW si es necesario
    nonce: str = ""

    def calculate_hash(self) -> str:
        """Calcular hash del bloque (Proof-of-Stake)"""
        block_data = {
            "block_height": self.block_height,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "transactions": [tx.calculate_hash() for tx in self.transactions],
            "validator": self.validator,
            "nonce": self.nonce,
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class SHEILYSBlockchain:
    """
    SHEILYS Blockchain Core - Sistema blockchain enterprise-grade

    Implementa Proof-of-Stake con características optimizadas para Sheily AI MCP:
    - Validación por agentes especializados
    - Minting controlado por IA
    - Integración con gamificación
    - Zero-trust security
    """

    def __init__(self, genesis_validator: str = "sheily_system_validator"):
        """Inicializar blockchain SHEILYS"""
        self.chain: List[SHEILYSBlock] = []
        self.pending_transactions: List[SHEILYSTransaction] = []
        self.stakes: Dict[str, float] = {}  # Address -> stake amount
        self.validators: List[str] = [genesis_validator]
        self.total_supply: float = 0.0
        self.circulating_supply: float = 0.0
        self.block_time: int = 60  # 1 minuto entre bloques
        self.max_block_size: int = 1000  # Máximo 1000 transacciones por bloque
        self.min_stake: float = 1000.0  # Stake mínimo para validar

        # Gamification integration
        self.gamification_contracts: Dict[str, Dict[str, Any]] = {}
        self.nft_contracts: Dict[str, Dict[str, Any]] = {}

        # Initialize with genesis block
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Crear bloque génesis"""
        genesis_tx = SHEILYSTransaction(
            transaction_id="genesis_sheily",
            sender="sheily_system",
            receiver="genesis_fund",
            amount=1000000.0,  # 1 millón SHEILYS iniciales
            transaction_type=TransactionType.REWARD,
            timestamp=time.time(),
            signature="genesis_signature_system",
            metadata={"genesis": True, "purpose": "initial_supply"},
        )

        genesis_block = SHEILYSBlock(
            block_height=0,
            previous_hash="0" * 64,
            timestamp=time.time(),
            transactions=[genesis_tx],
            validator="genesis_validator",
        )

        genesis_block.block_hash = genesis_block.calculate_hash()
        genesis_block.status = BlockStatus.FINALIZED
        genesis_block.nonce = secrets.token_hex(16)

        self.chain.append(genesis_block)
        self.total_supply = genesis_tx.amount
        self.circulating_supply = genesis_tx.amount

    def add_transaction(self, transaction: SHEILYSTransaction) -> bool:
        """
        Agregar transacción al pool de transacciones pendientes

        Returns:
            bool: True si la transacción fue agregada exitosamente
        """
        try:
            # Validar transacción
            if not self._validate_transaction(transaction):
                return False

            # Verificar si ya existe
            if any(
                tx.transaction_id == transaction.transaction_id
                for tx in self.pending_transactions
            ):
                return False

            self.pending_transactions.append(transaction)
            return True

        except Exception as e:
            print(f"Error agregando transacción: {e}")
            return False

    def _validate_transaction(self, transaction: SHEILYSTransaction) -> bool:
        """Validar transacción básica"""
        try:
            # Validar campos requeridos
            if not all(
                [
                    transaction.transaction_id,
                    transaction.sender,
                    transaction.receiver,
                    transaction.amount > 0,
                    transaction.timestamp > 0,
                    transaction.signature,
                ]
            ):
                return False

            # Validar timestamp (no futuro, no muy antiguo)
            current_time = time.time()
            if (
                transaction.timestamp > current_time + 300
            ):  # Máximo 5 minutos en el futuro
                return False
            if (
                transaction.timestamp < current_time - 3600
            ):  # Máximo 1 hora en el pasado
                return False

            # Validar hash
            if transaction.calculate_hash() != transaction.transaction_id:
                return False

            return True

        except Exception:
            return False

    def create_block(self, validator_address: str) -> Optional[SHEILYSBlock]:
        """
        Crear nuevo bloque con transacciones pendientes

        Args:
            validator_address: Address del validador (stake holder)

        Returns:
            SHEILYSBlock: Nuevo bloque o None si no se puede crear
        """
        try:
            # Verificar que el validador tiene stake suficiente
            if self.stakes.get(validator_address, 0) < self.min_stake:
                return None

            # Limpiar transacciones expiradas
            self._clean_expired_transactions()

            # Obtener transacciones para el bloque
            transactions = self.pending_transactions[: self.max_block_size]

            if not transactions:
                return None

            # Crear bloque
            latest_block = self.chain[-1]
            new_block = SHEILYSBlock(
                block_height=latest_block.block_height + 1,
                previous_hash=latest_block.block_hash,
                timestamp=time.time(),
                transactions=transactions,
                validator=validator_address,
            )

            # Proof of Stake - generar nonce aleatorio (simplificado)
            new_block.nonce = secrets.token_hex(16)
            new_block.block_hash = new_block.calculate_hash()

            # Remover transacciones del pool
            for tx in transactions:
                if tx in self.pending_transactions:
                    self.pending_transactions.remove(tx)
                    tx.block_height = new_block.block_height

            return new_block

        except Exception as e:
            print(f"Error creando bloque: {e}")
            return None

    def add_block(self, block: SHEILYSBlock) -> bool:
        """
        Agregar bloque a la cadena

        Args:
            block: Bloque a agregar

        Returns:
            bool: True si el bloque fue agregado exitosamente
        """
        try:
            # Validar bloque
            if not self._validate_block(block):
                return False

            # Agregar a la cadena
            block.status = BlockStatus.CONFIRMED
            self.chain.append(block)

            # Actualizar estado de la blockchain
            self._update_blockchain_state(block)

            return True

        except Exception as e:
            print(f"Error agregando bloque: {e}")
            return False

    def _validate_block(self, block: SHEILYSBlock) -> bool:
        """Validar bloque completo"""
        try:
            if not self.chain:
                return True  # Genesis block ya validado

            latest_block = self.chain[-1]

            # Validar altura del bloque
            if block.block_height != latest_block.block_height + 1:
                return False

            # Validar hash anterior
            if block.previous_hash != latest_block.block_hash:
                return False

            # Validar hash del bloque
            if block.calculate_hash() != block.block_hash:
                return False

            # Validar stake del validador
            if self.stakes.get(block.validator, 0) < self.min_stake:
                return False

            # Validar transacciones
            for tx in block.transactions:
                if not self._validate_transaction(tx):
                    return False

            return True

        except Exception:
            return False

    def _update_blockchain_state(self, block: SHEILYSBlock):
        """Actualizar estado global de la blockchain después de agregar bloque"""
        # Aquí se implementaría lógica de actualización de balances, stakes, etc.

        # Para nuestro caso, simplemente marcamos el bloque como finalizado
        block.status = BlockStatus.FINALIZED

    def stake_tokens(self, address: str, amount: float) -> bool:
        """
        Stake tokens para convertirse en validador

        Args:
            address: Address que hace stake
            amount: Cantidad de SHEILYS a stakear

        Returns:
            bool: True si el stake fue exitoso
        """
        try:
            # Verificar que tiene suficientes tokens (implementación simplificada)
            current_stake = self.stakes.get(address, 0)

            if amount < 0:
                return False

            self.stakes[address] = current_stake + amount

            # Agregar como validador si supera el mínimo
            if (
                self.stakes[address] >= self.min_stake
                and address not in self.validators
            ):
                self.validators.append(address)

            # Crear transacción de stake
            stake_tx = SHEILYSTransaction(
                transaction_id=f"stake_{address}_{int(time.time())}",
                sender=address,
                receiver="stake_contract",
                amount=amount,
                transaction_type=TransactionType.STAKE,
                timestamp=time.time(),
                signature=f"stake_signature_{address}",
                metadata={"stake_type": "validator"},
            )

            self.add_transaction(stake_tx)
            return True

        except Exception as e:
            print(f"Error haciendo stake: {e}")
            return False

    def _clean_expired_transactions(self):
        """Limpiar transacciones expiradas del pool"""
        current_time = time.time()
        max_age = 300  # 5 minutos máximo

        self.pending_transactions = [
            tx
            for tx in self.pending_transactions
            if current_time - tx.timestamp < max_age
        ]

    def get_blockchain_info(self) -> Dict[str, Any]:
        """Obtener información general de la blockchain"""
        return {
            "chain_length": len(self.chain),
            "pending_transactions": len(self.pending_transactions),
            "total_validators": len(self.validators),
            "total_staked": sum(self.stakes.values()),
            "total_supply": self.total_supply,
            "circulating_supply": self.circulating_supply,
            "latest_block": self.chain[-1].block_height if self.chain else 0,
            "block_time": self.block_time,
        }

    def get_block(self, height: int) -> Optional[SHEILYSBlock]:
        """Obtener bloque por altura"""
        if 0 <= height < len(self.chain):
            return self.chain[height]
        return None

    def get_transaction(self, tx_id: str) -> Optional[SHEILYSTransaction]:
        """Buscar transacción por ID en toda la cadena"""
        for block in self.chain:
            for tx in block.transactions:
                if tx.transaction_id == tx_id:
                    return tx
        return None

    # Gamification integration methods
    def register_gamification_contract(
        self, contract_name: str, contract_data: Dict[str, Any]
    ):
        """Registrar contrato de gamificación"""
        self.gamification_contracts[contract_name] = {
            "data": contract_data,
            "created_at": time.time(),
            "active": True,
        }

    def register_nft_contract(self, contract_name: str, contract_data: Dict[str, Any]):
        """Registrar contrato NFT"""
        self.nft_contracts[contract_name] = {
            "data": contract_data,
            "created_at": time.time(),
            "active": True,
            "total_supply": 0,
        }

    def mint_gamification_reward(
        self, player_address: str, reward_type: str, amount: float
    ) -> bool:
        """Mint reward tokens para gamificación"""
        try:
            reward_tx = SHEILYSTransaction(
                transaction_id=f"reward_{player_address}_{reward_type}_{int(time.time())}",
                sender="gamification_contract",
                receiver=player_address,
                amount=amount,
                transaction_type=TransactionType.REWARD,
                timestamp=time.time(),
                signature="gamification_system_signature",
                metadata={
                    "reward_type": reward_type,
                    "source": "gamification",
                    "player": player_address,
                },
            )

            return self.add_transaction(reward_tx)

        except Exception as e:
            print(f"Error minting reward: {e}")
            return False
