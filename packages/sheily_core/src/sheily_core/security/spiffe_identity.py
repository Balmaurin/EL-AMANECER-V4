#!/usr/bin/env python3
"""
SPIFFE Identity Management para Sheily AI
Implementa el estándar SPIFFE para gestión de identidades en sistemas distribuidos
Proporciona autenticación, autorización y auditoría para agentes autónomos
"""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cryptography
import jwt
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID

from sheily_core.a2a_protocol import a2a_system
from sheily_core.agent_quality import evaluate_agent_quality
from sheily_core.agent_tracing import trace_agent_execution

logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DATOS SPIFFE
# =============================================================================


class SPIFFETrustDomain(Enum):
    """Dominios de confianza SPIFFE"""

    SHEILY_PRODUCTION = "spiffe://sheily.ai/production"
    SHEILY_STAGING = "spiffe://sheily.ai/staging"
    SHEILY_DEVELOPMENT = "spiffe://sheily.ai/development"
    AGENT_FEDERATION = "spiffe://federation.agents"


class SPIFFERole(Enum):
    """Roles SPIFFE para agentes"""

    AGENT_COORDINATOR = "agent_coordinator"
    AGENT_WORKER = "agent_worker"
    AGENT_SPECIALIST = "agent_specialist"
    AGENT_AUDITOR = "agent_auditor"
    AGENT_ADMIN = "agent_admin"
    SYSTEM_SERVICE = "system_service"


class SPIFFEWorkloadType(Enum):
    """Tipos de workload SPIFFE"""

    AGENT_CORE = "agent_core"
    AGENT_TOOL = "agent_tool"
    AGENT_SERVICE = "agent_service"
    SYSTEM_COMPONENT = "system_component"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class SPIFFESVID:
    """SPIFFE Verifiable Identity Document"""

    spiffe_id: str  # spiffe://trust-domain/path
    public_key: str  # PEM encoded public key
    private_key: Optional[str] = None  # PEM encoded private key (solo para issuer)
    certificate_chain: List[str] = field(default_factory=list)  # PEM certificates
    expires_at: datetime = field(
        default_factory=lambda: datetime.now() + timedelta(hours=24)
    )
    issued_at: datetime = field(default_factory=datetime.now)
    issuer_id: str = ""
    subject_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "spiffe_id": self.spiffe_id,
            "public_key": self.public_key,
            "certificate_chain": self.certificate_chain,
            "expires_at": self.expires_at.isoformat(),
            "issued_at": self.issued_at.isoformat(),
            "issuer_id": self.issuer_id,
            "subject_info": self.subject_info,
        }


@dataclass
class SPIFFEEntry:
    """Entrada en el registro SPIFFE"""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spiffe_id: str = ""
    parent_id: str = ""  # SPIFFE ID del workload padre
    selectors: Dict[str, str] = field(default_factory=dict)  # Selectores para matching
    workload_type: SPIFFEWorkloadType = SPIFFEWorkloadType.AGENT_CORE
    role: SPIFFERole = SPIFFERole.AGENT_WORKER
    trust_domain: SPIFFETrustDomain = SPIFFETrustDomain.SHEILY_PRODUCTION
    svid: Optional[SPIFFESVID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, revoked, expired

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "entry_id": self.entry_id,
            "spiffe_id": self.spiffe_id,
            "parent_id": self.parent_id,
            "selectors": self.selectors,
            "workload_type": self.workload_type.value,
            "role": self.role.value,
            "trust_domain": self.trust_domain.value,
            "svid": self.svid.to_dict() if self.svid else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "status": self.status,
        }


@dataclass
class SPIFFEPolicy:
    """Política de autorización SPIFFE"""

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    trust_domain: SPIFFETrustDomain = SPIFFETrustDomain.SHEILY_PRODUCTION
    rules: List[Dict[str, Any]] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "trust_domain": self.trust_domain.value,
            "rules": self.rules,
            "permissions": self.permissions,
            "conditions": self.conditions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
        }


@dataclass
class SPIFFEAuditEvent:
    """Evento de auditoría SPIFFE"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""  # authentication, authorization, issuance, revocation
    spiffe_id: str = ""
    action: str = ""
    resource: str = ""
    result: str = ""  # success, failure, denied
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "spiffe_id": self.spiffe_id,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }


# =============================================================================
# AUTORIDAD CERTIFICADORA SPIFFE
# =============================================================================


class SPIFFEAuthority:
    """Autoridad certificadora SPIFFE"""

    def __init__(self, trust_domain: SPIFFETrustDomain):
        self.trust_domain = trust_domain
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.certificate = self._generate_root_certificate()
        self.issued_certificates: Dict[str, SPIFFESVID] = {}
        self.revoked_certificates: set = set()

    def _generate_root_certificate(self) -> x509.Certificate:
        """Genera certificado raíz para la autoridad"""
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "ES"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Madrid"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Madrid"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Sheily AI"),
                x509.NameAttribute(
                    NameOID.COMMON_NAME, f"SPIFFE Authority - {self.trust_domain.value}"
                ),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(self.public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365 * 10))  # 10 años
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.UniformResourceIdentifier(self.trust_domain.value),
                    ]
                ),
                critical=False,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(self.private_key, hashes.SHA256(), default_backend())
        )

        return cert

    def issue_svid(
        self, spiffe_id: str, public_key_pem: str, subject_info: Dict[str, Any] = None
    ) -> SPIFFESVID:
        """Emite un SVID para un SPIFFE ID"""
        if spiffe_id in self.issued_certificates:
            raise ValueError(f"SPIFFE ID {spiffe_id} already has an issued SVID")

        # Parsear clave pública
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode(), backend=default_backend()
        )

        # Generar certificado
        certificate = self._generate_certificate(
            spiffe_id, public_key, subject_info or {}
        )

        # Crear SVID
        svid = SPIFFESVID(
            spiffe_id=spiffe_id,
            public_key=public_key_pem,
            certificate_chain=[
                certificate.public_bytes(serialization.Encoding.PEM).decode(),
                self.certificate.public_bytes(serialization.Encoding.PEM).decode(),
            ],
            issuer_id=self.trust_domain.value,
            subject_info=subject_info or {},
        )

        self.issued_certificates[spiffe_id] = svid
        logger.info(f"Issued SVID for {spiffe_id}")
        return svid

    def _generate_certificate(
        self, spiffe_id: str, public_key, subject_info: Dict[str, Any]
    ) -> x509.Certificate:
        """Genera certificado para SVID"""
        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "ES"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Sheily AI"),
                x509.NameAttribute(NameOID.COMMON_NAME, spiffe_id),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self.certificate.subject)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(hours=24))  # 24 horas
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.UniformResourceIdentifier(spiffe_id),
                    ]
                ),
                critical=False,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(self.private_key, hashes.SHA256(), default_backend())
        )

        return cert

    def revoke_svid(self, spiffe_id: str):
        """Revoca un SVID"""
        if spiffe_id in self.issued_certificates:
            del self.issued_certificates[spiffe_id]
            self.revoked_certificates.add(spiffe_id)
            logger.info(f"Revoked SVID for {spiffe_id}")

    def validate_svid(self, svid: SPIFFESVID) -> bool:
        """Valida un SVID"""
        # Verificar expiración
        if datetime.now() > svid.expires_at:
            return False

        # Verificar revocación
        if svid.spiffe_id in self.revoked_certificates:
            return False

        # Verificar cadena de certificados
        try:
            # Parsear certificado
            cert_pem = svid.certificate_chain[0]
            cert = x509.load_pem_x509_certificate(cert_pem.encode(), default_backend())

            # Verificar firma con certificado raíz
            self.public_key.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_hash_algorithm,
            )

            # Verificar SPIFFE ID en SAN
            san = cert.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            )
            uris = san.value.get_values_for_type(x509.UniformResourceIdentifier)
            if svid.spiffe_id not in uris:
                return False

            return True

        except Exception as e:
            logger.error(f"SVID validation failed: {e}")
            return False


# =============================================================================
# REGISTRO SPIFFE
# =============================================================================


class SPIFFERegistry:
    """Registro central de identidades SPIFFE"""

    def __init__(self):
        self.entries: Dict[str, SPIFFEEntry] = {}
        self.policies: Dict[str, SPIFFEPolicy] = {}
        self.audit_log: List[SPIFFEAuditEvent] = []
        self.authorities: Dict[SPIFFETrustDomain, SPIFFEAuthority] = {}

    async def initialize_trust_domains(self):
        """Inicializa autoridades para todos los dominios de confianza"""
        for domain in SPIFFETrustDomain:
            self.authorities[domain] = SPIFFEAuthority(domain)
            logger.info(f"Initialized SPIFFE authority for {domain.value}")

    def register_workload(
        self,
        spiffe_id: str,
        workload_type: SPIFFEWorkloadType,
        role: SPIFFERole,
        selectors: Dict[str, str] = None,
        parent_id: str = "",
        metadata: Dict[str, Any] = None,
    ) -> SPIFFEEntry:
        """Registra un nuevo workload en el registro SPIFFE"""
        if spiffe_id in self.entries:
            raise ValueError(f"SPIFFE ID {spiffe_id} already registered")

        # Extraer trust domain del SPIFFE ID
        trust_domain = self._extract_trust_domain(spiffe_id)

        entry = SPIFFEEntry(
            spiffe_id=spiffe_id,
            parent_id=parent_id,
            selectors=selectors or {},
            workload_type=workload_type,
            role=role,
            trust_domain=trust_domain,
            metadata=metadata or {},
        )

        self.entries[spiffe_id] = entry

        # Auditar registro
        self._audit_event("registration", spiffe_id, "workload_registered", "success")

        logger.info(f"Registered workload: {spiffe_id}")
        return entry

    def issue_svid(self, spiffe_id: str, public_key_pem: str) -> SPIFFESVID:
        """Emite SVID para un workload registrado"""
        if spiffe_id not in self.entries:
            raise ValueError(f"SPIFFE ID {spiffe_id} not registered")

        entry = self.entries[spiffe_id]
        authority = self.authorities.get(entry.trust_domain)

        if not authority:
            raise ValueError(
                f"No authority available for trust domain {entry.trust_domain}"
            )

        # Emitir SVID
        svid = authority.issue_svid(spiffe_id, public_key_pem, entry.metadata)

        # Actualizar entrada
        entry.svid = svid
        entry.last_seen = datetime.now()

        # Auditar emisión
        self._audit_event("issuance", spiffe_id, "svid_issued", "success")

        return svid

    def authenticate_request(
        self, spiffe_id: str, token: str = None, certificate: str = None
    ) -> bool:
        """Autentica una solicitud usando identidad SPIFFE"""
        if spiffe_id not in self.entries:
            self._audit_event(
                "authentication",
                spiffe_id,
                "request_authenticated",
                "denied",
                {"reason": "spiffe_id_not_registered"},
            )
            return False

        entry = self.entries[spiffe_id]

        # Verificar que el workload esté activo
        if entry.status != "active":
            self._audit_event(
                "authentication",
                spiffe_id,
                "request_authenticated",
                "denied",
                {"reason": "workload_not_active"},
            )
            return False

        # Verificar SVID si se proporciona
        if certificate and entry.svid:
            authority = self.authorities.get(entry.trust_domain)
            if authority and not authority.validate_svid(entry.svid):
                self._audit_event(
                    "authentication",
                    spiffe_id,
                    "request_authenticated",
                    "denied",
                    {"reason": "invalid_svid"},
                )
                return False

        # Actualizar último visto
        entry.last_seen = datetime.now()

        self._audit_event(
            "authentication", spiffe_id, "request_authenticated", "success"
        )
        return True

    def authorize_action(
        self, spiffe_id: str, action: str, resource: str, context: Dict[str, Any] = None
    ) -> bool:
        """Autoriza una acción basada en políticas SPIFFE"""
        if spiffe_id not in self.entries:
            self._audit_event(
                "authorization",
                spiffe_id,
                action,
                "denied",
                {"resource": resource, "reason": "spiffe_id_not_registered"},
            )
            return False

        entry = self.entries[spiffe_id]

        # Verificar políticas aplicables
        applicable_policies = self._find_applicable_policies(entry)

        for policy in applicable_policies:
            if self._evaluate_policy(policy, entry, action, resource, context or {}):
                self._audit_event(
                    "authorization",
                    spiffe_id,
                    action,
                    "granted",
                    {"resource": resource, "policy": policy.policy_id},
                )
                return True

        self._audit_event(
            "authorization",
            spiffe_id,
            action,
            "denied",
            {"resource": resource, "reason": "no_applicable_policy"},
        )
        return False

    def _find_applicable_policies(self, entry: SPIFFEEntry) -> List[SPIFFEPolicy]:
        """Encuentra políticas aplicables para una entrada"""
        applicable = []

        for policy in self.policies.values():
            if policy.status != "active" or policy.trust_domain != entry.trust_domain:
                continue

            # Verificar reglas de la política
            if self._policy_matches_entry(policy, entry):
                applicable.append(policy)

        return applicable

    def _policy_matches_entry(self, policy: SPIFFEPolicy, entry: SPIFFEEntry) -> bool:
        """Verifica si una política coincide con una entrada"""
        for rule in policy.rules:
            rule_type = rule.get("type")

            if rule_type == "role" and entry.role.value != rule.get("value"):
                return False
            elif rule_type == "workload_type" and entry.workload_type.value != rule.get(
                "value"
            ):
                return False
            elif rule_type == "selector":
                selector_key = rule.get("key")
                selector_value = rule.get("value")
                if entry.selectors.get(selector_key) != selector_value:
                    return False

        return True

    def _evaluate_policy(
        self,
        policy: SPIFFEPolicy,
        entry: SPIFFEEntry,
        action: str,
        resource: str,
        context: Dict[str, Any],
    ) -> bool:
        """Evalúa si una política permite una acción"""
        # Verificar permisos
        if action not in policy.permissions:
            return False

        # Verificar condiciones
        conditions = policy.conditions

        # Condición de tiempo
        if "time_window" in conditions:
            time_window = conditions["time_window"]
            current_hour = datetime.now().hour
            if not (time_window["start"] <= current_hour <= time_window["end"]):
                return False

        # Condición de rol
        if "allowed_roles" in conditions:
            if entry.role.value not in conditions["allowed_roles"]:
                return False

        # Condición de contexto
        if "context_checks" in conditions:
            for check in conditions["context_checks"]:
                context_key = check.get("key")
                expected_value = check.get("value")
                if context.get(context_key) != expected_value:
                    return False

        return True

    def create_policy(
        self,
        name: str,
        description: str,
        trust_domain: SPIFFETrustDomain,
        rules: List[Dict[str, Any]],
        permissions: List[str],
        conditions: Dict[str, Any] = None,
    ) -> SPIFFEPolicy:
        """Crea una nueva política de autorización"""
        policy = SPIFFEPolicy(
            name=name,
            description=description,
            trust_domain=trust_domain,
            rules=rules,
            permissions=permissions,
            conditions=conditions or {},
        )

        self.policies[policy.policy_id] = policy
        logger.info(f"Created policy: {policy.name}")
        return policy

    def _extract_trust_domain(self, spiffe_id: str) -> SPIFFETrustDomain:
        """Extrae el dominio de confianza de un SPIFFE ID"""
        # spiffe://trust-domain/path -> trust-domain
        if spiffe_id.startswith("spiffe://"):
            trust_domain_part = spiffe_id[9:].split("/")[0]
            for domain in SPIFFETrustDomain:
                if trust_domain_part in domain.value:
                    return domain

        return SPIFFETrustDomain.SHEILY_PRODUCTION

    def _audit_event(
        self,
        event_type: str,
        spiffe_id: str,
        action: str,
        result: str,
        details: Dict[str, Any] = None,
    ):
        """Registra evento de auditoría"""
        event = SPIFFEAuditEvent(
            event_type=event_type,
            spiffe_id=spiffe_id,
            action=action,
            result=result,
            details=details or {},
        )

        self.audit_log.append(event)

        # Mantener límite de log (últimos 10000 eventos)
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]

    def get_audit_trail(
        self, spiffe_id: str = None, event_type: str = None, limit: int = 100
    ) -> List[SPIFFEAuditEvent]:
        """Obtiene el trail de auditoría filtrado"""
        events = self.audit_log

        if spiffe_id:
            events = [e for e in events if e.spiffe_id == spiffe_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def get_workload_status(self, spiffe_id: str) -> Optional[SPIFFEEntry]:
        """Obtiene el estado de un workload"""
        return self.entries.get(spiffe_id)


# =============================================================================
# WORKLOAD ATTESTATOR SPIFFE
# =============================================================================


class SPIFFEWorkloadAttestor:
    """Attestor de workloads para SPIFFE"""

    def __init__(self, registry: SPIFFERegistry):
        self.registry = registry
        self.attestation_methods = {
            "kubernetes": self._attest_kubernetes,
            "aws": self._attest_aws,
            "gcp": self._attest_gcp,
            "docker": self._attest_docker,
            "local": self._attest_local,
        }

    async def attest_workload(
        self, attestation_data: Dict[str, Any]
    ) -> Optional[SPIFFEEntry]:
        """Attesta un workload y devuelve su entrada SPIFFE"""
        platform = attestation_data.get("platform", "local")
        attestor = self.attestation_methods.get(platform, self._attest_local)

        try:
            spiffe_id, selectors, metadata = await attestor(attestation_data)

            # Verificar si ya está registrado
            existing_entry = self.registry.get_workload_status(spiffe_id)
            if existing_entry:
                # Actualizar último visto
                existing_entry.last_seen = datetime.now()
                return existing_entry

            # Registrar nuevo workload
            workload_type = self._infer_workload_type(attestation_data)
            role = self._infer_role(attestation_data)

            entry = self.registry.register_workload(
                spiffe_id=spiffe_id,
                workload_type=workload_type,
                role=role,
                selectors=selectors,
                metadata=metadata,
            )

            logger.info(f"Attested workload: {spiffe_id}")
            return entry

        except Exception as e:
            logger.error(f"Workload attestation failed: {e}")
            return None

    async def _attest_kubernetes(
        self, data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Attesta workload de Kubernetes"""
        namespace = data.get("namespace", "default")
        pod_name = data.get("pod_name", "unknown")
        service_account = data.get("service_account", "default")

        spiffe_id = f"spiffe://sheily.ai/production/k8s/{namespace}/sa/{service_account}/pod/{pod_name}"

        selectors = {
            "k8s:namespace": namespace,
            "k8s:pod": pod_name,
            "k8s:serviceaccount": service_account,
        }

        metadata = {
            "platform": "kubernetes",
            "namespace": namespace,
            "pod_name": pod_name,
            "service_account": service_account,
        }

        return spiffe_id, selectors, metadata

    async def _attest_aws(
        self, data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Attesta workload de AWS"""
        account_id = data.get("account_id", "unknown")
        region = data.get("region", "us-east-1")
        instance_id = data.get("instance_id", "unknown")

        spiffe_id = f"spiffe://sheily.ai/production/aws/{account_id}/{region}/instance/{instance_id}"

        selectors = {
            "aws:account": account_id,
            "aws:region": region,
            "aws:instance": instance_id,
        }

        metadata = {
            "platform": "aws",
            "account_id": account_id,
            "region": region,
            "instance_id": instance_id,
        }

        return spiffe_id, selectors, metadata

    async def _attest_gcp(
        self, data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Attesta workload de GCP"""
        project_id = data.get("project_id", "unknown")
        zone = data.get("zone", "us-central1-a")
        instance_name = data.get("instance_name", "unknown")

        spiffe_id = f"spiffe://sheily.ai/production/gcp/{project_id}/{zone}/instance/{instance_name}"

        selectors = {
            "gcp:project": project_id,
            "gcp:zone": zone,
            "gcp:instance": instance_name,
        }

        metadata = {
            "platform": "gcp",
            "project_id": project_id,
            "zone": zone,
            "instance_name": instance_name,
        }

        return spiffe_id, selectors, metadata

    async def _attest_docker(
        self, data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Attesta workload de Docker"""
        container_id = data.get("container_id", "unknown")
        image = data.get("image", "unknown")

        spiffe_id = f"spiffe://sheily.ai/production/docker/container/{container_id}"

        selectors = {"docker:container": container_id, "docker:image": image}

        metadata = {"platform": "docker", "container_id": container_id, "image": image}

        return spiffe_id, selectors, metadata

    async def _attest_local(
        self, data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Attesta workload local (desarrollo/testing)"""
        process_id = data.get("process_id", str(uuid.uuid4())[:8])
        agent_type = data.get("agent_type", "unknown")

        spiffe_id = f"spiffe://sheily.ai/development/local/{agent_type}/{process_id}"

        selectors = {"local:process": process_id, "local:agent_type": agent_type}

        metadata = {
            "platform": "local",
            "process_id": process_id,
            "agent_type": agent_type,
        }

        return spiffe_id, selectors, metadata

    def _infer_workload_type(self, data: Dict[str, Any]) -> SPIFFEWorkloadType:
        """Infier el tipo de workload desde los datos de attestación"""
        platform = data.get("platform", "local")
        agent_type = data.get("agent_type", "")

        if "agent" in agent_type.lower():
            return SPIFFEWorkloadType.AGENT_CORE
        elif "tool" in agent_type.lower():
            return SPIFFEWorkloadType.AGENT_TOOL
        elif "service" in agent_type.lower():
            return SPIFFEWorkloadType.AGENT_SERVICE
        else:
            return SPIFFEWorkloadType.SYSTEM_COMPONENT

    def _infer_role(self, data: Dict[str, Any]) -> SPIFFERole:
        """Infier el rol desde los datos de attestación"""
        agent_type = data.get("agent_type", "").lower()

        if "coordinator" in agent_type:
            return SPIFFERole.AGENT_COORDINATOR
        elif "specialist" in agent_type or "expert" in agent_type:
            return SPIFFERole.AGENT_SPECIALIST
        elif "auditor" in agent_type:
            return SPIFFERole.AGENT_AUDITOR
        elif "admin" in agent_type:
            return SPIFFERole.AGENT_ADMIN
        else:
            return SPIFFERole.AGENT_WORKER


# =============================================================================
# SISTEMA SPIFFE PRINCIPAL
# =============================================================================


class SPIFFESystem:
    """Sistema principal de gestión de identidades SPIFFE"""

    def __init__(self):
        self.registry = SPIFFERegistry()
        self.attestor = SPIFFEWorkloadAttestor(self.registry)
        self.is_initialized = False

    async def initialize_spiffe_system(self):
        """Inicializa el sistema SPIFFE"""
        if not self.is_initialized:
            await self.registry.initialize_trust_domains()
            await self._create_default_policies()
            self.is_initialized = True
            logger.info("SPIFFE System initialized")

    async def _create_default_policies(self):
        """Crea políticas por defecto"""
        # Política para agentes workers
        self.registry.create_policy(
            name="Agent Worker Policy",
            description="Política básica para agentes trabajadores",
            trust_domain=SPIFFETrustDomain.SHEILY_PRODUCTION,
            rules=[
                {"type": "role", "value": "agent_worker"},
                {"type": "workload_type", "value": "agent_core"},
            ],
            permissions=["task:execute", "resource:read", "communication:send"],
            conditions={
                "time_window": {"start": 0, "end": 23},  # Siempre permitido
                "allowed_roles": ["agent_worker", "agent_specialist"],
            },
        )

        # Política para coordinadores
        self.registry.create_policy(
            name="Coordinator Policy",
            description="Política para agentes coordinadores",
            trust_domain=SPIFFETrustDomain.SHEILY_PRODUCTION,
            rules=[{"type": "role", "value": "agent_coordinator"}],
            permissions=[
                "task:assign",
                "task:cancel",
                "agent:manage",
                "resource:write",
                "communication:broadcast",
            ],
        )

        # Política para administradores
        self.registry.create_policy(
            name="Admin Policy",
            description="Política para agentes administradores",
            trust_domain=SPIFFETrustDomain.SHEILY_PRODUCTION,
            rules=[{"type": "role", "value": "agent_admin"}],
            permissions=[
                "system:configure",
                "policy:manage",
                "audit:read",
                "workload:revoke",
            ],
        )

        logger.info("Created default SPIFFE policies")

    async def register_agent(
        self, agent_id: str, agent_type: str = "worker", platform: str = "local"
    ) -> Optional[SPIFFEEntry]:
        """Registra un agente en el sistema SPIFFE"""
        # Preparar datos de attestación
        attestation_data = {
            "platform": platform,
            "agent_type": agent_type,
            "process_id": agent_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Attestar workload
        entry = await self.attestor.attest_workload(attestation_data)

        if entry:
            logger.info(
                f"Registered agent {agent_id} with SPIFFE ID: {entry.spiffe_id}"
            )

        return entry

    async def authenticate_agent(
        self, agent_id: str, spiffe_id: str, token: str = None
    ) -> bool:
        """Autentica un agente usando su identidad SPIFFE"""
        return self.registry.authenticate_request(spiffe_id, token)

    async def authorize_agent_action(
        self, spiffe_id: str, action: str, resource: str, context: Dict[str, Any] = None
    ) -> bool:
        """Autoriza una acción de un agente"""
        return self.registry.authorize_action(
            spiffe_id, action, resource, context or {}
        )

    async def get_agent_svid(self, agent_id: str) -> Optional[SPIFFESVID]:
        """Obtiene el SVID de un agente"""
        # Buscar entrada por agent_id (esto es simplificado)
        for entry in self.registry.entries.values():
            if agent_id in entry.spiffe_id:
                return entry.svid

        return None

    def get_audit_trail(
        self, spiffe_id: str = None, event_type: str = None, limit: int = 100
    ) -> List[SPIFFEAuditEvent]:
        """Obtiene el trail de auditoría"""
        return self.registry.get_audit_trail(spiffe_id, event_type, limit)

    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado del sistema SPIFFE"""
        total_entries = len(self.registry.entries)
        active_entries = len(
            [e for e in self.registry.entries.values() if e.status == "active"]
        )
        total_policies = len(self.registry.policies)
        total_audit_events = len(self.registry.audit_log)

        trust_domains = {}
        for domain in SPIFFETrustDomain:
            domain_entries = [
                e for e in self.registry.entries.values() if e.trust_domain == domain
            ]
            trust_domains[domain.value] = len(domain_entries)

        return {
            "initialized": self.is_initialized,
            "total_entries": total_entries,
            "active_entries": active_entries,
            "total_policies": total_policies,
            "total_audit_events": total_audit_events,
            "trust_domains": trust_domains,
            "authorities_count": len(self.registry.authorities),
        }


# =============================================================================
# INTEGRACIÓN CON SISTEMA EXISTENTE
# =============================================================================

# Instancia global del sistema SPIFFE
spiffe_system = SPIFFESystem()


async def initialize_spiffe():
    """Inicializa el sistema SPIFFE"""
    await spiffe_system.initialize_spiffe_system()


async def register_spiffe_agent(
    agent_id: str, agent_type: str = "worker"
) -> Optional[str]:
    """Registra un agente en SPIFFE y devuelve su SPIFFE ID"""
    entry = await spiffe_system.register_agent(agent_id, agent_type)
    return entry.spiffe_id if entry else None


async def authenticate_spiffe_request(spiffe_id: str, token: str = None) -> bool:
    """Autentica una solicitud usando SPIFFE"""
    return await spiffe_system.authenticate_agent("", spiffe_id, token)


async def authorize_spiffe_action(spiffe_id: str, action: str, resource: str) -> bool:
    """Autoriza una acción usando SPIFFE"""
    return await spiffe_system.authorize_agent_action(spiffe_id, action, resource)


# =============================================================================
# DEMO Y TESTING
# =============================================================================


async def demo_spiffe_system():
    """Demostración del sistema SPIFFE"""
    print("🔐 Inicializando sistema SPIFFE (SPIFFE Identity Management)...")

    await initialize_spiffe()

    # Registrar agentes
    print("\n📝 Registrando agentes en SPIFFE...")

    agent1_id = await register_spiffe_agent("coordinator-agent", "coordinator")
    agent2_id = await register_spiffe_agent("worker-agent-1", "worker")
    agent3_id = await register_spiffe_agent("worker-agent-2", "worker")

    print(f"✅ Agente coordinador registrado: {agent1_id}")
    print(f"✅ Agente worker 1 registrado: {agent2_id}")
    print(f"✅ Agente worker 2 registrado: {agent3_id}")

    # Autenticar agentes
    print("\n🔍 Probando autenticación...")
    auth1 = await authenticate_spiffe_request(agent1_id)
    auth2 = await authenticate_spiffe_request(agent2_id)

    print(f"✅ Coordinador autenticado: {auth1}")
    print(f"✅ Worker autenticado: {auth2}")

    # Autorizar acciones
    print("\n🛡️ Probando autorización...")

    # Coordinador puede asignar tareas
    can_assign = await authorize_spiffe_action(agent1_id, "task:assign", "task:*")
    print(f"✅ Coordinador puede asignar tareas: {can_assign}")

    # Worker puede ejecutar tareas
    can_execute = await authorize_spiffe_action(agent2_id, "task:execute", "task:123")
    print(f"✅ Worker puede ejecutar tareas: {can_execute}")

    # Worker NO puede asignar tareas
    cannot_assign = await authorize_spiffe_action(agent2_id, "task:assign", "task:*")
    print(f"✅ Worker NO puede asignar tareas: {not cannot_assign}")

    # Obtener estado del sistema
    print("\n📊 Estado del sistema SPIFFE:")
    status = spiffe_system.get_system_status()

    print(f"   • Inicializado: {status['initialized']}")
    print(f"   • Total entradas: {status['total_entries']}")
    print(f"   • Entradas activas: {status['active_entries']}")
    print(f"   • Total políticas: {status['total_policies']}")
    print(f"   • Eventos de auditoría: {status['total_audit_events']}")

    # Mostrar audit trail
    print("\n📋 Últimos eventos de auditoría:")
    audit_events = spiffe_system.get_audit_trail(limit=5)

    for event in audit_events[-3:]:  # Mostrar últimos 3
        print(f"   • {event.event_type}: {event.action} - {event.result}")

    print("\n🎉 Demo del sistema SPIFFE completada!")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "SPIFFEAuthority",
    "SPIFFERegistry",
    "SPIFFEWorkloadAttestor",
    "SPIFFESystem",
    # Modelos de datos
    "SPIFFESVID",
    "SPIFFEEntry",
    "SPIFFEPolicy",
    "SPIFFEAuditEvent",
    "SPIFFETrustDomain",
    "SPIFFERole",
    "SPIFFEWorkloadType",
    # Sistema global
    "spiffe_system",
    # Funciones de utilidad
    "initialize_spiffe",
    "register_spiffe_agent",
    "authenticate_spiffe_request",
    "authorize_spiffe_action",
    "demo_spiffe_system",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - SPIFFE Identity Management System"
