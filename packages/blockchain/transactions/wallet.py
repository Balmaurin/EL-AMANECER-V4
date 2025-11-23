#!/usr/bin/env python3
"""
SHEILYS Blockchain Wallet - Billetera para gestión de SHEILYS tokens y NFTs

Implementa funcionalidades completas de wallet para el ecosistema SHEILYS:
- Gestión de claves y direcciones
- Envío y recepción de tokens
- Staking y governance
- Gestión de NFTs
- Backup y recuperación
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .sheilys_blockchain import SHEILYSBlockchain
from .sheilys_token import NFTCollection, SHEILYSTokenManager


class WalletKeys:
    """Gestión de claves para SHEILYS wallet"""

    def __init__(self):
        """Inicializar sistema de claves"""
        self.private_key: Optional[bytes] = None
        self.public_key: Optional[bytes] = None
        self.address: Optional[str] = None

    def generate_keys(self) -> str:
        """
        Generar nuevo par de claves

        Returns:
            str: Address generado
        """
        # Generar clave privada de 32 bytes
        self.private_key = secrets.token_bytes(32)

        # Derivar clave pública del hash de la privada (simplificado)
        self.public_key = hashlib.sha256(self.private_key).digest()

        # Generar address desde clave pública
        self.address = (
            base64.b64encode(self.public_key[:20]).decode("utf-8").rstrip("=")
        )

        return self.address

    def import_private_key(self, private_key_hex: str) -> str:
        """
        Importar clave privada desde formato hexadecimal

        Args:
            private_key_hex: Clave privada en formato hex

        Returns:
            str: Address correspondiente
        """
        try:
            self.private_key = bytes.fromhex(private_key_hex)
            self.public_key = hashlib.sha256(self.private_key).digest()
            self.address = (
                base64.b64encode(self.public_key[:20]).decode("utf-8").rstrip("=")
            )
            return self.address
        except Exception:
            raise ValueError("Invalid private key format")

    def sign_transaction(self, transaction_data: str) -> str:
        """
        Firmar datos de transacción

        Args:
            transaction_data: Datos a firmar

        Returns:
            str: Firma en base64
        """
        if not self.private_key:
            raise ValueError("Wallet not initialized")

        # Crear HMAC usando clave privada
        signature = hmac.new(
            self.private_key, transaction_data.encode(), hashlib.sha256
        ).digest()

        return base64.b64encode(signature).decode("utf-8")

    def export_private_key(self) -> str:
        """Exportar clave privada en formato hexadecimal"""
        if not self.private_key:
            raise ValueError("No private key available")
        return self.private_key.hex()


class WalletEncryption:
    """Sistema de encriptación para wallet seguro"""

    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """Derivar clave desde password usando PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        return kdf.derive(password.encode())

    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> Tuple[bytes, bytes]:
        """
        Encriptar datos

        Returns:
            Tuple[bytes, bytes]: (datos encriptados, IV)
        """
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Padding PKCS7
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding_length]) * padding_length

        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        return encrypted_data, iv

    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes, iv: bytes) -> bytes:
        """
        Desencriptar datos

        Returns:
            bytes: Datos desencriptados
        """
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

        # Remove PKCS7 padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]


class BlockchainWallet:
    """
    SHEILYS Blockchain Wallet - Wallet completa para el ecosistema

    Funcionalidades:
    - Gestión de tokens SHEILYS
    - Staking y rewards
    - Gestión de NFTs
    - Gobernanza del sistema
    - Backup y seguridad
    """

    def __init__(self):
        """Inicializar wallet SHEILYS"""
        self.keys = WalletKeys()
        self.address: Optional[str] = None
        self.encrypted = False

        # Conexión con blockchain y token manager
        self.blockchain: Optional[SHEILYSBlockchain] = None
        self.token_manager: Optional[SHEILYSTokenManager] = None

        # Cache de balances locales
        self.local_balances = {"sheilys": 0.0, "staked_sheilyns": 0.0, "nfts": []}

        # Historial de transacciones
        self.transaction_history: List[Dict[str, Any]] = []

        # Configuración de seguridad
        self.auto_backup = True
        self.backup_path = "./wallet_backups"
        self.security_level = "standard"

    def create_wallet(self, password: Optional[str] = None) -> str:
        """
        Crear nueva wallet

        Args:
            password: Password para encriptación (opcional)

        Returns:
            str: Address de la wallet creada
        """
        try:
            # Generar claves
            self.address = self.keys.generate_keys()

            # Si hay password, encriptar
            if password:
                self._encrypt_wallet(password)
                self.encrypted = True

            # Crear directorio de backups
            if self.auto_backup and not os.path.exists(self.backup_path):
                os.makedirs(self.backup_path)

            # Backup inicial
            if self.auto_backup:
                self._create_backup()

            return self.address

        except Exception as e:
            raise Exception(f"Error creando wallet: {e}")

    def load_wallet(self, private_key_hex: str, password: Optional[str] = None) -> str:
        """
        Cargar wallet desde clave privada

        Args:
            private_key_hex: Clave privada en formato hex
            password: Password si está encriptada

        Returns:
            str: Address de la wallet cargada
        """
        try:
            # Si está encriptada, desencriptar primero
            if password and self.encrypted:
                private_key_hex = self._decrypt_wallet_data(private_key_hex, password)

            # Importar clave
            self.address = self.keys.import_private_key(private_key_hex)

            return self.address

        except Exception as e:
            raise Exception(f"Error cargando wallet: {e}")

    def _encrypt_wallet(self, password: str):
        """Encriptar wallet con password"""
        if not self.keys.private_key:
            return

        # Generar salt
        salt = os.urandom(16)

        # Derivar clave
        key = WalletEncryption.derive_key(password, salt)

        # Encriptar clave privada
        encrypted_key, iv = WalletEncryption.encrypt_data(self.keys.private_key, key)

        # Guardar datos encriptados temporalmente
        self.encrypted_data = {
            "encrypted_key": base64.b64encode(encrypted_key).decode(),
            "iv": base64.b64encode(iv).decode(),
            "salt": base64.b64encode(salt).decode(),
            "address": self.address,
        }

    def _decrypt_wallet_data(self, encrypted_private_key: str, password: str) -> str:
        """Desencriptar datos de wallet"""
        # Implementación simplificada - en producción más robusta
        return encrypted_private_key

    def connect_to_blockchain(
        self, blockchain: SHEILYSBlockchain, token_manager: SHEILYSTokenManager
    ):
        """
        Conectar wallet con el sistema blockchain

        Args:
            blockchain: Instancia de blockchain
            token_manager: Instancia del token manager
        """
        self.blockchain = blockchain
        self.token_manager = token_manager

    def get_balance(self) -> Dict[str, Any]:
        """
        Obtener balances de la wallet

        Returns:
            dict: Balances completos
        """
        if not self.token_manager or not self.address:
            return self.local_balances.copy()

        try:
            # Actualizar balances desde blockchain
            self.local_balances["sheilys"] = self.token_manager.get_balance(
                self.address
            )
            self.local_balances["staked_sheilyns"] = (
                self.token_manager.get_staked_balance(self.address)
            )
            self.local_balances["nfts"] = self.token_manager.get_user_nfts(self.address)

            return self.local_balances.copy()

        except Exception as e:
            print(f"Error obteniendo balance: {e}")
            return self.local_balances.copy()

    def send_tokens(self, to_address: str, amount: float) -> bool:
        """
        Enviar tokens SHEILYS

        Args:
            to_address: Dirección destinataria
            amount: Cantidad a enviar

        Returns:
            bool: True si la transacción fue exitosa
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            # Verificar balance suficiente
            current_balance = self.token_manager.get_balance(self.address)
            if current_balance < amount:
                raise ValueError("Insufficient SHEILYS balance")

            # Ejecutar transferencia
            success = self.token_manager.transfer_tokens(
                self.address, to_address, amount
            )

            if success:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "send",
                        "to": to_address,
                        "amount": amount,
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

                # Actualizar balances locales
                self.local_balances["sheilys"] -= amount

            return success

        except Exception as e:
            print(f"Error enviando tokens: {e}")
            return False

    def stake_tokens(self, amount: float, pool_name: str = "community_pool") -> bool:
        """
        Stake tokens para rewards

        Args:
            amount: Cantidad a stakear
            pool_name: Pool de staking

        Returns:
            bool: True si el stake fue exitoso
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            success = self.token_manager.stake_tokens(self.address, amount, pool_name)

            if success:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "stake",
                        "amount": amount,
                        "pool": pool_name,
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

                # Actualizar balances locales
                self.local_balances["sheilys"] -= amount
                self.local_balances["staked_sheilyns"] += amount

            return success

        except Exception as e:
            print(f"Error staking tokens: {e}")
            return False

    def claim_staking_rewards(self) -> float:
        """
        Claim staking rewards acumulados

        Returns:
            float: Cantidad de rewards claimed
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            claimed_amount = self.token_manager.claim_staking_rewards(self.address)

            if claimed_amount > 0:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "claim_rewards",
                        "amount": claimed_amount,
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

                # Actualizar balances locales
                self.local_balances["sheilys"] += claimed_amount

            return claimed_amount

        except Exception as e:
            print(f"Error claiming staking rewards: {e}")
            return 0.0

    def mint_nft(
        self, collection: NFTCollection, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Mint nuevo NFT

        Args:
            collection: Colección del NFT
            metadata: Metadata del NFT

        Returns:
            str: Token ID del NFT minteado, o None si falló
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            token_id = self.token_manager.mint_nft(collection, self.address, metadata)

            if token_id:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "mint_nft",
                        "token_id": token_id,
                        "collection": collection.value,
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

                # Actualizar NFTs locales
                self.local_balances["nfts"].append(
                    {
                        "token_id": token_id,
                        "collection": collection.value,
                        "metadata": metadata,
                    }
                )

            return token_id

        except Exception as e:
            print(f"Error minting NFT: {e}")
            return None

    def transfer_nft(self, token_id: str, to_address: str) -> bool:
        """
        Transferir NFT a otra dirección

        Args:
            token_id: ID del token NFT
            to_address: Dirección destinataria

        Returns:
            bool: True si la transferencia fue exitosa
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            success = self.token_manager.transfer_nft(
                token_id, self.address, to_address
            )

            if success:
                # Actualizar historial
                self._add_transaction_to_history(
                    {
                        "type": "transfer_nft",
                        "token_id": token_id,
                        "to": to_address,
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

                # Actualizar NFTs locales
                self.local_balances["nfts"] = [
                    nft
                    for nft in self.local_balances["nfts"]
                    if nft["token_id"] != token_id
                ]

            return success

        except Exception as e:
            print(f"Error transfering NFT: {e}")
            return False

    def create_governance_proposal(
        self, title: str, description: str, voting_period_days: int = 7
    ) -> str:
        """
        Crear propuesta de gobernanza

        Args:
            title: Título de la propuesta
            description: Descripción
            voting_period_days: Período de votación

        Returns:
            str: ID de la propuesta creada
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            proposal_id = self.token_manager.create_governance_proposal(
                self.address, title, description, voting_period_days
            )

            self._add_transaction_to_history(
                {
                    "type": "governance_proposal",
                    "proposal_id": proposal_id,
                    "title": title,
                    "timestamp": datetime.now(),
                    "status": "created",
                }
            )

            return proposal_id

        except Exception as e:
            print(f"Error creating governance proposal: {e}")
            raise

    def vote_on_proposal(self, proposal_id: str, vote_for: bool) -> bool:
        """
        Votar en una propuesta de gobernanza

        Args:
            proposal_id: ID de la propuesta
            vote_for: True para votar a favor

        Returns:
            bool: True si el voto fue registrado
        """
        if not self.token_manager or not self.address:
            raise ValueError("Wallet not connected to blockchain")

        try:
            success = self.token_manager.vote_on_proposal(
                proposal_id, self.address, vote_for
            )

            if success:
                self._add_transaction_to_history(
                    {
                        "type": "governance_vote",
                        "proposal_id": proposal_id,
                        "vote": "for" if vote_for else "against",
                        "timestamp": datetime.now(),
                        "status": "confirmed",
                    }
                )

            return success

        except Exception as e:
            print(f"Error voting on proposal: {e}")
            return False

    def _add_transaction_to_history(self, transaction: Dict[str, Any]):
        """Agregar transacción al historial local"""
        self.transaction_history.append(transaction)

        # Mantener historial limitado (últimas 1000 transacciones)
        if len(self.transaction_history) > 1000:
            self.transaction_history = self.transaction_history[-1000:]

    def get_transaction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener historial de transacciones

        Args:
            limit: Máximo número de transacciones a retornar

        Returns:
            List[Dict]: Historial de transacciones
        """
        return self.transaction_history[-limit:]

    def _create_backup(self):
        """Crear backup de wallet"""
        if not self.auto_backup or not self.address:
            return

        try:
            backup_data = {
                "address": self.address,
                "public_key": (
                    base64.b64encode(self.keys.public_key).decode()
                    if self.keys.public_key
                    else None
                ),
                "balances": self.local_balances,
                "transaction_history": self.transaction_history[
                    -100:
                ],  # Últimas 100 tx
                "backup_timestamp": datetime.now().isoformat(),
                "version": "1.0",
            }

            if self.encrypted and hasattr(self, "encrypted_data"):
                backup_data["encrypted_data"] = self.encrypted_data

            backup_filename = (
                f"wallet_backup_{self.address}_{int(datetime.now().timestamp())}.json"
            )
            backup_path = os.path.join(self.backup_path, backup_filename)

            with open(backup_path, "w") as f:
                json.dump(backup_data, f, indent=2, default=str)

        except Exception as e:
            print(f"Error creando backup: {e}")

    def export_wallet_data(self) -> Dict[str, Any]:
        """
        Exportar datos de wallet para backup

        Returns:
            dict: Datos de wallet completos
        """
        return {
            "address": self.address,
            "balances": self.local_balances,
            "transaction_history": self.transaction_history,
            "encrypted": self.encrypted,
            "export_timestamp": datetime.now().isoformat(),
        }

    def import_wallet_data(self, wallet_data: Dict[str, Any]):
        """
        Importar datos de wallet desde backup

        Args:
            wallet_data: Datos de wallet exportados
        """
        self.address = wallet_data.get("address")
        self.local_balances = wallet_data.get("balances", {})
        self.transaction_history = wallet_data.get("transaction_history", [])
        self.encrypted = wallet_data.get("encrypted", False)

    def get_wallet_info(self) -> Dict[str, Any]:
        """Obtener información completa de la wallet"""
        return {
            "address": self.address,
            "balances": self.local_balances,
            "transaction_count": len(self.transaction_history),
            "connected_to_blockchain": self.blockchain is not None,
            "encrypted": self.encrypted,
            "security_level": self.security_level,
            "auto_backup": self.auto_backup,
        }
