#!/usr/bin/env python3
"""
Sistema de Mandatos Criptográficos para Sheily AI
Implementa transacciones seguras y mandatos criptográficos para operaciones críticas
Proporciona garantías de integridad, no-repudio y auditabilidad
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cryptography
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import hmac as crypto_hmac
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa, utils

from sheily_core.agent_quality import evaluate_agent_quality
from sheily_core.agent_tracing import trace_agent_execution
from sheily_core.spiffe_identity import SPIFFERole, SPIFFETrustDomain, spiffe_system

logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DATOS PARA MANDATOS CRIPTOGRÁFICOS
# =============================================================================


class MandateType(Enum):
    """Tipos de mandatos criptográficos"""

    TRANSACTION = "transaction"
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    AUTHORIZATION = "authorization"
    DELEGATION = "delegation"
    REVOCATION = "revocation"


class MandateStatus(Enum):
    """Estados de un mandato"""

    DRAFT = "draft"
    PROPOSED = "proposed"
    SIGNED = "signed"
    EXECUTED = "executed"
    REVOKED = "revoked"
    EXPIRED = "expired"
    REJECTED = "rejected"


class SecurityLevel(Enum):
    """Niveles de seguridad para mandatos"""

    BASIC = "basic"  # Firma simple
    ENHANCED = "enhanced"  # Multi-firma
    CRITICAL = "critical"  # Threshold crypto
    MAXIMUM = "maximum"  # HSM requerido


@dataclass
class CryptographicMandate:
    """Mandato criptográfico principal"""

    mandate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MandateType = MandateType.TRANSACTION
    title: str = ""
    description: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    issuer_spiffe_id: str = ""
    subject_spiffe_ids: List[str] = field(default_factory=list)
    required_signers: List[str] = field(default_factory=list)  # SPIFFE IDs
    signatures: Dict[str, str] = field(
        default_factory=dict
    )  # spiffe_id -> signature_b64
    security_level: SecurityLevel = SecurityLevel.BASIC
    status: MandateStatus = MandateStatus.DRAFT
    validity_period: timedelta = field(default_factory=lambda: timedelta(hours=24))
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    expires_at: datetime = field(
        default_factory=lambda: datetime.now() + timedelta(hours=24)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "mandate_id": self.mandate_id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "issuer_spiffe_id": self.issuer_spiffe_id,
            "subject_spiffe_ids": self.subject_spiffe_ids,
            "required_signers": self.required_signers,
            "signatures": self.signatures,
            "security_level": self.security_level.value,
            "status": self.status.value,
            "validity_period": self.validity_period.total_seconds(),
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "expires_at": self.expires_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MandateSignature:
    """Firma individual de un mandato"""

    signature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mandate_id: str = ""
    signer_spiffe_id: str = ""
    signature_data: str = ""  # Base64 encoded signature
    signing_method: str = "rsa-sha256"
    signed_at: datetime = field(default_factory=datetime.now)
    signature_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "signature_id": self.signature_id,
            "mandate_id": self.mandate_id,
            "signer_spiffe_id": self.signer_spiffe_id,
            "signature_data": self.signature_data,
            "signing_method": self.signing_method,
            "signed_at": self.signed_at.isoformat(),
            "signature_metadata": self.signature_metadata,
        }


@dataclass
class MandateExecution:
    """Ejecución de un mandato"""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mandate_id: str = ""
    executor_spiffe_id: str = ""
    execution_result: Dict[str, Any] = field(default_factory=dict)
    execution_status: str = "pending"  # pending, running, completed, failed
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "execution_id": self.execution_id,
            "mandate_id": self.mandate_id,
            "executor_spiffe_id": self.executor_spiffe_id,
            "execution_result": self.execution_result,
            "execution_status": self.execution_status,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "execution_metadata": self.execution_metadata,
        }


@dataclass
class MandateAuditEvent:
    """Evento de auditoría para mandatos"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mandate_id: str = ""
    event_type: str = ""  # created, signed, executed, revoked, expired
    actor_spiffe_id: str = ""
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "event_id": self.event_id,
            "mandate_id": self.mandate_id,
            "event_type": self.event_type,
            "actor_spiffe_id": self.actor_spiffe_id,
            "action": self.action,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }


# =============================================================================
# MOTOR CRIPTOGRÁFICO
# =============================================================================


class CryptographicEngine:
    """Motor criptográfico para firmas y validaciones"""

    def __init__(self):
        self.key_cache: Dict[str, rsa.RSAPrivateKey] = {}  # spiffe_id -> private_key
        self.public_key_cache: Dict[str, rsa.RSAPublicKey] = (
            {}
        )  # spiffe_id -> public_key

    def generate_keypair(self, spiffe_id: str) -> Tuple[str, str]:
        """Genera un par de claves RSA para un SPIFFE ID"""
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        public_key = private_key.public_key()

        # Serializar claves
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

        # Cachear claves
        self.key_cache[spiffe_id] = private_key
        self.public_key_cache[spiffe_id] = public_key

        return private_pem, public_pem

    def sign_data(self, spiffe_id: str, data: bytes) -> str:
        """Firma datos usando la clave privada del SPIFFE ID"""
        if spiffe_id not in self.key_cache:
            raise ValueError(f"No private key available for {spiffe_id}")

        private_key = self.key_cache[spiffe_id]

        # Crear firma
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        return base64.b64encode(signature).decode()

    def verify_signature(self, spiffe_id: str, data: bytes, signature_b64: str) -> bool:
        """Verifica una firma usando la clave pública del SPIFFE ID"""
        try:
            if spiffe_id not in self.public_key_cache:
                # Intentar obtener clave pública del sistema SPIFFE
                svid = asyncio.run(spiffe_system.get_agent_svid(spiffe_id))
                if svid and svid.public_key:
                    public_key = serialization.load_pem_public_key(
                        svid.public_key.encode(), backend=default_backend()
                    )
                    self.public_key_cache[spiffe_id] = public_key
                else:
                    return False

            public_key = self.public_key_cache[spiffe_id]
            signature = base64.b64decode(signature_b64)

            # Verificar firma
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            return True

        except (InvalidSignature, Exception) as e:
            logger.error(f"Signature verification failed for {spiffe_id}: {e}")
            return False

    def create_mandate_hash(self, mandate: CryptographicMandate) -> str:
        """Crea un hash criptográfico del contenido del mandato"""
        # Crear representación canónica del mandato
        canonical_data = {
            "mandate_id": mandate.mandate_id,
            "type": mandate.type.value,
            "title": mandate.title,
            "description": mandate.description,
            "content": mandate.content,
            "issuer_spiffe_id": mandate.issuer_spiffe_id,
            "subject_spiffe_ids": sorted(mandate.subject_spiffe_ids),
            "required_signers": sorted(mandate.required_signers),
            "security_level": mandate.security_level.value,
            "created_at": mandate.created_at.isoformat(),
            "expires_at": mandate.expires_at.isoformat(),
        }

        # Serializar a JSON canónico
        json_str = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"))

        # Crear hash SHA-256
        hash_obj = hashlib.sha256(json_str.encode())
        return hash_obj.hexdigest()

    def validate_mandate_integrity(self, mandate: CryptographicMandate) -> bool:
        """Valida la integridad de un mandato"""
        # Verificar expiración
        if datetime.now() > mandate.expires_at:
            return False

        # Verificar que todas las firmas requeridas estén presentes
        for signer in mandate.required_signers:
            if signer not in mandate.signatures:
                return False

        # Verificar todas las firmas
        mandate_hash = self.create_mandate_hash(mandate)
        for signer_spiffe_id, signature_b64 in mandate.signatures.items():
            if not self.verify_signature(
                signer_spiffe_id, mandate_hash.encode(), signature_b64
            ):
                return False

        return True


# =============================================================================
# SISTEMA DE MANDATOS CRIPTOGRÁFICOS
# =============================================================================


class CryptographicMandatesSystem:
    """Sistema principal de mandatos criptográficos"""

    def __init__(self):
        self.crypto_engine = CryptographicEngine()
        self.mandates: Dict[str, CryptographicMandate] = {}
        self.executions: Dict[str, MandateExecution] = {}
        self.audit_log: List[MandateAuditEvent] = []
        self.mandate_scheduler = None
        self.is_running = False

    async def start_mandates_system(self):
        """Inicia el sistema de mandatos criptográficos"""
        self.is_running = True

        # Iniciar scheduler de limpieza
        self.mandate_scheduler = asyncio.create_task(self._mandate_maintenance_loop())

        logger.info("Cryptographic Mandates System started")

    async def stop_mandates_system(self):
        """Detiene el sistema de mandatos criptográficos"""
        self.is_running = False

        if self.mandate_scheduler:
            self.mandate_scheduler.cancel()
            try:
                await self.mandate_scheduler
            except asyncio.CancelledError:
                pass

        logger.info("Cryptographic Mandates System stopped")

    async def create_mandate(
        self,
        type: MandateType,
        title: str,
        description: str,
        content: Dict[str, Any],
        issuer_spiffe_id: str,
        subject_spiffe_ids: List[str],
        required_signers: List[str],
        security_level: SecurityLevel = SecurityLevel.BASIC,
    ) -> CryptographicMandate:
        """Crea un nuevo mandato criptográfico"""
        mandate = CryptographicMandate(
            type=type,
            title=title,
            description=description,
            content=content,
            issuer_spiffe_id=issuer_spiffe_id,
            subject_spiffe_ids=subject_spiffe_ids,
            required_signers=required_signers,
            security_level=security_level,
        )

        self.mandates[mandate.mandate_id] = mandate

        # Auditar creación
        self._audit_event(
            mandate.mandate_id, "created", issuer_spiffe_id, "mandate_created"
        )

        logger.info(f"Created cryptographic mandate: {mandate.mandate_id}")
        return mandate

    async def sign_mandate(self, mandate_id: str, signer_spiffe_id: str) -> bool:
        """Firma un mandato criptográfico"""
        if mandate_id not in self.mandates:
            raise ValueError(f"Mandate {mandate_id} not found")

        mandate = self.mandates[mandate_id]

        # Verificar que el firmante esté autorizado
        if signer_spiffe_id not in mandate.required_signers:
            self._audit_event(
                mandate_id, "signature_attempt", signer_spiffe_id, "unauthorized_signer"
            )
            return False

        # Verificar que no esté ya firmado
        if signer_spiffe_id in mandate.signatures:
            return True  # Ya firmado

        # Crear hash del mandato
        mandate_hash = self.crypto_engine.create_mandate_hash(mandate)

        # Firmar el hash
        signature_b64 = self.crypto_engine.sign_data(
            signer_spiffe_id, mandate_hash.encode()
        )

        # Almacenar firma
        mandate.signatures[signer_spiffe_id] = signature_b64

        # Verificar si el mandato está completamente firmado
        if len(mandate.signatures) >= len(mandate.required_signers):
            mandate.status = MandateStatus.SIGNED

        # Auditar firma
        self._audit_event(mandate_id, "signed", signer_spiffe_id, "mandate_signed")

        logger.info(f"Mandate {mandate_id} signed by {signer_spiffe_id}")
        return True

    async def execute_mandate(
        self, mandate_id: str, executor_spiffe_id: str
    ) -> MandateExecution:
        """Ejecuta un mandato criptográfico"""
        if mandate_id not in self.mandates:
            raise ValueError(f"Mandate {mandate_id} not found")

        mandate = self.mandates[mandate_id]

        # Verificar que el mandato esté firmado y válido
        if not self.crypto_engine.validate_mandate_integrity(mandate):
            raise ValueError(f"Mandate {mandate_id} is not valid or not fully signed")

        # Verificar autorización del ejecutor
        if (
            executor_spiffe_id not in mandate.subject_spiffe_ids
            and executor_spiffe_id != mandate.issuer_spiffe_id
        ):
            raise ValueError(
                f"Executor {executor_spiffe_id} not authorized for mandate {mandate_id}"
            )

        # Crear ejecución
        execution = MandateExecution(
            mandate_id=mandate_id,
            executor_spiffe_id=executor_spiffe_id,
            execution_status="running",
        )

        self.executions[execution.execution_id] = execution

        # Ejecutar el mandato (lógica específica por tipo)
        try:
            result = await self._execute_mandate_logic(mandate, executor_spiffe_id)

            execution.execution_result = result
            execution.execution_status = "completed"
            execution.completed_at = datetime.now()

            mandate.status = MandateStatus.EXECUTED
            mandate.executed_at = datetime.now()

        except Exception as e:
            execution.execution_result = {"error": str(e)}
            execution.execution_status = "failed"
            execution.completed_at = datetime.now()

            logger.error(f"Mandate execution failed: {e}")

        # Auditar ejecución
        self._audit_event(
            mandate_id,
            "executed",
            executor_spiffe_id,
            f"mandate_{execution.execution_status}",
        )

        logger.info(
            f"Mandate {mandate_id} executed with status: {execution.execution_status}"
        )
        return execution

    async def _execute_mandate_logic(
        self, mandate: CryptographicMandate, executor_spiffe_id: str
    ) -> Dict[str, Any]:
        """Ejecuta la lógica específica del tipo de mandato"""
        if mandate.type == MandateType.TRANSACTION:
            return await self._execute_transaction_mandate(mandate, executor_spiffe_id)
        elif mandate.type == MandateType.AUTHORIZATION:
            return await self._execute_authorization_mandate(
                mandate, executor_spiffe_id
            )
        elif mandate.type == MandateType.DELEGATION:
            return await self._execute_delegation_mandate(mandate, executor_spiffe_id)
        else:
            # Mandato genérico
            return {
                "mandate_type": mandate.type.value,
                "executed_by": executor_spiffe_id,
                "execution_time": datetime.now().isoformat(),
                "result": "success",
            }

    async def _execute_transaction_mandate(
        self, mandate: CryptographicMandate, executor_spiffe_id: str
    ) -> Dict[str, Any]:
        """Ejecuta un mandato de transacción"""
        content = mandate.content

        # Simular ejecución de transacción
        transaction_result = {
            "transaction_id": str(uuid.uuid4()),
            "amount": content.get("amount", 0),
            "currency": content.get("currency", "USD"),
            "from_account": content.get("from_account"),
            "to_account": content.get("to_account"),
            "executed_at": datetime.now().isoformat(),
            "executor": executor_spiffe_id,
            "status": "completed",
        }

        # Aquí iría la lógica real de ejecución de transacción
        logger.info(f"Executed transaction mandate: {transaction_result}")

        return transaction_result

    async def _execute_authorization_mandate(
        self, mandate: CryptographicMandate, executor_spiffe_id: str
    ) -> Dict[str, Any]:
        """Ejecuta un mandato de autorización"""
        content = mandate.content

        authorization_result = {
            "authorization_id": str(uuid.uuid4()),
            "resource": content.get("resource"),
            "action": content.get("action"),
            "authorized_by": executor_spiffe_id,
            "authorized_at": datetime.now().isoformat(),
            "valid_until": (datetime.now() + timedelta(hours=1)).isoformat(),
            "status": "granted",
        }

        logger.info(f"Executed authorization mandate: {authorization_result}")

        return authorization_result

    async def _execute_delegation_mandate(
        self, mandate: CryptographicMandate, executor_spiffe_id: str
    ) -> Dict[str, Any]:
        """Ejecuta un mandato de delegación"""
        content = mandate.content

        delegation_result = {
            "delegation_id": str(uuid.uuid4()),
            "delegate_from": content.get("delegate_from"),
            "delegate_to": content.get("delegate_to"),
            "permissions": content.get("permissions", []),
            "delegated_by": executor_spiffe_id,
            "delegated_at": datetime.now().isoformat(),
            "valid_until": content.get("valid_until"),
            "status": "active",
        }

        logger.info(f"Executed delegation mandate: {delegation_result}")

        return delegation_result

    async def revoke_mandate(self, mandate_id: str, revoker_spiffe_id: str) -> bool:
        """Revoca un mandato criptográfico"""
        if mandate_id not in self.mandates:
            return False

        mandate = self.mandates[mandate_id]

        # Verificar autorización para revocar
        if revoker_spiffe_id != mandate.issuer_spiffe_id:
            self._audit_event(
                mandate_id,
                "revocation_attempt",
                revoker_spiffe_id,
                "unauthorized_revoker",
            )
            return False

        mandate.status = MandateStatus.REVOKED

        # Auditar revocación
        self._audit_event(mandate_id, "revoked", revoker_spiffe_id, "mandate_revoked")

        logger.info(f"Mandate {mandate_id} revoked by {revoker_spiffe_id}")
        return True

    def validate_mandate(self, mandate_id: str) -> bool:
        """Valida un mandato criptográfico"""
        if mandate_id not in self.mandates:
            return False

        mandate = self.mandates[mandate_id]
        return self.crypto_engine.validate_mandate_integrity(mandate)

    def get_mandate_status(self, mandate_id: str) -> Optional[CryptographicMandate]:
        """Obtiene el estado de un mandato"""
        return self.mandates.get(mandate_id)

    def get_pending_mandates_for_signer(
        self, signer_spiffe_id: str
    ) -> List[CryptographicMandate]:
        """Obtiene mandatos pendientes de firma para un firmante"""
        pending = []

        for mandate in self.mandates.values():
            if (
                mandate.status in [MandateStatus.DRAFT, MandateStatus.PROPOSED]
                and signer_spiffe_id in mandate.required_signers
                and signer_spiffe_id not in mandate.signatures
            ):
                pending.append(mandate)

        return pending

    def get_audit_trail(
        self, mandate_id: str = None, event_type: str = None, limit: int = 100
    ) -> List[MandateAuditEvent]:
        """Obtiene el trail de auditoría filtrado"""
        events = self.audit_log

        if mandate_id:
            events = [e for e in events if e.mandate_id == mandate_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def _audit_event(
        self,
        mandate_id: str,
        event_type: str,
        actor_spiffe_id: str,
        action: str,
        details: Dict[str, Any] = None,
    ):
        """Registra evento de auditoría"""
        event = MandateAuditEvent(
            mandate_id=mandate_id,
            event_type=event_type,
            actor_spiffe_id=actor_spiffe_id,
            action=action,
            details=details or {},
        )

        self.audit_log.append(event)

        # Mantener límite de log (últimos 10000 eventos)
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]

    async def _mandate_maintenance_loop(self):
        """Loop de mantenimiento de mandatos"""
        while self.is_running:
            try:
                # Limpiar mandatos expirados cada hora
                await asyncio.sleep(3600)  # 1 hora

                expired_count = 0
                now = datetime.now()

                for mandate in self.mandates.values():
                    if mandate.expires_at < now and mandate.status not in [
                        MandateStatus.EXECUTED,
                        MandateStatus.REVOKED,
                    ]:
                        mandate.status = MandateStatus.EXPIRED
                        expired_count += 1

                if expired_count > 0:
                    logger.info(f"Expired {expired_count} mandates during maintenance")

            except Exception as e:
                logger.error(f"Error in mandate maintenance loop: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos en caso de error

    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de mandatos"""
        total_mandates = len(self.mandates)
        signed_mandates = len(
            [m for m in self.mandates.values() if m.status == MandateStatus.SIGNED]
        )
        executed_mandates = len(
            [m for m in self.mandates.values() if m.status == MandateStatus.EXECUTED]
        )
        expired_mandates = len(
            [m for m in self.mandates.values() if m.status == MandateStatus.EXPIRED]
        )

        mandate_types = {}
        for mandate in self.mandates.values():
            mtype = mandate.type.value
            mandate_types[mtype] = mandate_types.get(mtype, 0) + 1

        return {
            "total_mandates": total_mandates,
            "signed_mandates": signed_mandates,
            "executed_mandates": executed_mandates,
            "expired_mandates": expired_mandates,
            "mandate_types": mandate_types,
            "total_audit_events": len(self.audit_log),
            "system_running": self.is_running,
        }


# =============================================================================
# INTEGRACIÓN CON SISTEMA EXISTENTE
# =============================================================================

# Instancia global del sistema de mandatos criptográficos
crypto_mandates_system = CryptographicMandatesSystem()


async def initialize_crypto_mandates():
    """Inicializa el sistema de mandatos criptográficos"""
    await crypto_mandates_system.start_mandates_system()


async def create_secure_transaction(
    issuer_spiffe_id: str,
    amount: float,
    from_account: str,
    to_account: str,
    required_approvers: List[str],
) -> str:
    """Crea una transacción segura con mandato criptográfico"""
    mandate = await crypto_mandates_system.create_mandate(
        type=MandateType.TRANSACTION,
        title=f"Secure Transaction: {amount} USD",
        description=f"Transfer {amount} USD from {from_account} to {to_account}",
        content={
            "amount": amount,
            "currency": "USD",
            "from_account": from_account,
            "to_account": to_account,
            "transaction_type": "transfer",
        },
        issuer_spiffe_id=issuer_spiffe_id,
        subject_spiffe_ids=[issuer_spiffe_id],  # El issuer ejecuta
        required_signers=required_approvers,
        security_level=SecurityLevel.ENHANCED,
    )

    return mandate.mandate_id


async def sign_mandate(mandate_id: str, signer_spiffe_id: str) -> bool:
    """Firma un mandato criptográfico"""
    return await crypto_mandates_system.sign_mandate(mandate_id, signer_spiffe_id)


async def execute_mandate(mandate_id: str, executor_spiffe_id: str) -> Dict[str, Any]:
    """Ejecuta un mandato criptográfico"""
    execution = await crypto_mandates_system.execute_mandate(
        mandate_id, executor_spiffe_id
    )
    return execution.execution_result


# =============================================================================
# DEMO Y TESTING
# =============================================================================


async def demo_crypto_mandates():
    """Demostración del sistema de mandatos criptográficos"""
    print("🔐 Inicializando sistema de mandatos criptográficos...")

    await initialize_crypto_mandates()

    # Crear identidades SPIFFE para testing
    await spiffe_system.initialize_spiffe_system()

    issuer_id = await register_spiffe_agent("transaction-issuer", "admin")
    approver1_id = await register_spiffe_agent("approver-1", "worker")
    approver2_id = await register_spiffe_agent("approver-2", "worker")

    print(f"✅ Identidades SPIFFE creadas:")
    print(f"   Issuer: {issuer_id}")
    print(f"   Approver 1: {approver1_id}")
    print(f"   Approver 2: {approver2_id}")

    # Crear una transacción segura
    print("\n💰 Creando transacción segura...")
    mandate_id = await create_secure_transaction(
        issuer_spiffe_id=issuer_id,
        amount=1000.00,
        from_account="account-123",
        to_account="account-456",
        required_approvers=[approver1_id, approver2_id],
    )

    print(f"✅ Transacción creada con mandato: {mandate_id}")

    # Firmar el mandato
    print("\n✍️ Firmando el mandato...")

    sign1 = await sign_mandate(mandate_id, approver1_id)
    sign2 = await sign_mandate(mandate_id, approver2_id)

    print(f"✅ Firma 1 (Approver 1): {sign1}")
    print(f"✅ Firma 2 (Approver 2): {sign2}")

    # Verificar estado del mandato
    mandate = crypto_mandates_system.get_mandate_status(mandate_id)
    if mandate:
        print(f"📊 Estado del mandato: {mandate.status.value}")
        print(f"   Firmas requeridas: {len(mandate.required_signers)}")
        print(f"   Firmas obtenidas: {len(mandate.signatures)}")

    # Ejecutar el mandato
    print("\n⚡ Ejecutando el mandato...")
    try:
        result = await execute_mandate(mandate_id, issuer_id)
        print(f"✅ Mandato ejecutado exitosamente:")
        print(f"   Transaction ID: {result.get('transaction_id')}")
        print(f"   Amount: {result.get('amount')} {result.get('currency')}")
        print(f"   Status: {result.get('status')}")
    except Exception as e:
        print(f"❌ Error ejecutando mandato: {e}")

    # Obtener estadísticas del sistema
    print("\n📈 Estadísticas del sistema:")
    stats = crypto_mandates_system.get_system_stats()

    print(f"   Total mandatos: {stats['total_mandates']}")
    print(f"   Mandatos firmados: {stats['signed_mandates']}")
    print(f"   Mandatos ejecutados: {stats['executed_mandates']}")
    print(f"   Eventos de auditoría: {stats['total_audit_events']}")

    # Mostrar audit trail
    print("\n📋 Últimos eventos de auditoría:")
    audit_events = crypto_mandates_system.get_audit_trail(mandate_id, limit=5)

    for event in audit_events[-3:]:  # Mostrar últimos 3
        print(
            f"   • {event.event_type}: {event.action} - {event.timestamp.strftime('%H:%M:%S')}"
        )

    print("\n🎉 Demo de mandatos criptográficos completada!")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "CryptographicEngine",
    "CryptographicMandatesSystem",
    # Modelos de datos
    "CryptographicMandate",
    "MandateSignature",
    "MandateExecution",
    "MandateAuditEvent",
    "MandateType",
    "MandateStatus",
    "SecurityLevel",
    # Sistema global
    "crypto_mandates_system",
    # Funciones de utilidad
    "initialize_crypto_mandates",
    "create_secure_transaction",
    "sign_mandate",
    "execute_mandate",
    "demo_crypto_mandates",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Cryptographic Mandates System"
