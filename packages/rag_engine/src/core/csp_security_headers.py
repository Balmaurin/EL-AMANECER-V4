#!/usr/bin/env python3
"""
Automated CSP Headers - Security Headers Management System
=========================================================

Sistema automatizado para gestión de headers de seguridad HTTP:
- Content Security Policy (CSP) dinámico y adaptable
- Headers de seguridad estándares (HSTS, X-Frame-Options, etc.)
- Análisis de vulnerabilidades basado en CSP violations
- Reportes de seguridad con recomendaciones CSP
- Integración con logs y monitoreo de seguridad
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csp-security-headers")

# Headers de seguridad estándar
DEFAULT_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


class CSPViolation:
    """Clase para representar violaciones CSP"""

    def __init__(self, violation_data: Dict[str, Any]):
        self.document_uri = violation_data.get("document-uri", "")
        self.violated_directive = violation_data.get("violated-directive", "")
        self.effective_directive = violation_data.get("effective-directive", "")
        self.original_policy = violation_data.get("original-policy", "")
        self.blocked_uri = violation_data.get("blocked-uri", "")
        self.status_code = violation_data.get("status-code", 0)
        self.source_file = violation_data.get("source-file", "")
        self.line_number = violation_data.get("line-number", 0)
        self.column_number = violation_data.get("column-number", 0)
        self.timestamp = datetime.now()
        self.user_agent = violation_data.get("user-agent", "")

        # Extraer dominio del blocked_uri
        self.blocked_domain = self._extract_domain(self.blocked_uri)

        # Clasificar tipo de violación
        self.violation_type = self._classify_violation()

    def _extract_domain(self, uri: str) -> str:
        """Extraer dominio de una URI"""
        if not uri or uri in ["inline", "eval", "blob", "data"]:
            return uri

        try:
            # URI simple parsing
            if "://" in uri:
                domain_part = uri.split("://")[1].split("/")[0].split(":")[0]
                return domain_part
            elif uri.startswith("//"):
                return uri[2:].split("/")[0].split(":")[0]
        except:
            pass

        return uri

    def _classify_violation(self) -> str:
        """Clasificar tipo de violación CSP"""
        directive = self.violated_directive

        if "script-src" in directive:
            if self.blocked_uri in ["inline", "eval"]:
                return "inline_scripts"
            elif "cdn" in self.blocked_domain.lower():
                return "cdn_scripts"
            else:
                return "external_scripts"
        elif "style-src" in directive:
            return (
                "external_styles"
                if self.blocked_uri not in ["inline"]
                else "inline_styles"
            )
        elif "img-src" in directive:
            return "images"
        elif "font-src" in directive:
            return "fonts"
        elif "connect-src" in directive:
            return "api_calls"
        elif "frame-src" in directive or "frame-ancestors" in directive:
            return "iframes"
        else:
            return "other"

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            "document_uri": self.document_uri,
            "violated_directive": self.violated_directive,
            "effective_directive": self.effective_directive,
            "original_policy": self.original_policy,
            "blocked_uri": self.blocked_uri,
            "blocked_domain": self.blocked_domain,
            "status_code": self.status_code,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "timestamp": self.timestamp.isoformat(),
            "user_agent": self.user_agent,
            "violation_type": self.violation_type,
        }


class AutomatedCSPManager:
    """Gestor automatizado de Content Security Policy"""

    def __init__(self, csp_data_path: str = "./security/csp_data"):
        self.csp_data_path = Path(csp_data_path)
        self.csp_data_path.mkdir(parents=True, exist_ok=True)

        # Estado CSP
        self.violations_file = self.csp_data_path / "csp_violations.json"
        self.csp_policy_file = self.csp_data_path / "csp_policy.json"
        self.security_headers_file = self.csp_data_path / "security_headers.json"

        # Datos en memoria
        self.violations: List[CSPViolation] = []
        self.current_csp_policy = self._get_default_csp_policy()
        self.security_headers = DEFAULT_SECURITY_HEADERS.copy()

        # Métricas
        self.violation_stats = defaultdict(int)
        self.domain_whitelist: Set[str] = set()
        self.learning_mode = True  # Modo de aprendizaje inicial

        print("🛡️ Automated CSP Manager inicializado")

    def _get_default_csp_policy(self) -> Dict[str, List[str]]:
        """Obtener política CSP por defecto"""
        return {
            "default-src": ["'self'"],
            "script-src": [
                "'self'",
                "'unsafe-inline'",
                "'unsafe-eval'",
            ],  # Inicial agresivo
            "style-src": ["'self'", "'unsafe-inline'"],
            "img-src": ["'self'", "data:", "https:"],
            "font-src": ["'self'", "https:", "data:"],
            "connect-src": ["'self'"],
            "frame-src": ["'none'"],
            "object-src": ["'none'"],
            "base-uri": ["'self'"],
            "form-action": ["'self'"],
        }

    async def report_csp_violation(
        self, violation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reportar una violación CSP"""
        violation = CSPViolation(violation_data)
        self.violations.append(violation)

        # Actualizar estadísticas
        self.violation_stats[violation.violation_type] += 1

        # Guardar en disco
        await self._save_violations()

        logger.info(
            f"🚨 CSP Violation: {violation.violated_directive} - {violation.blocked_uri}"
        )

        # Si estamos en modo aprendizaje, considerar adaptar política
        if self.learning_mode:
            await self._adapt_policy_from_violation(violation)

        return {"status": "reported", "violation_id": len(self.violations) - 1}

    async def _adapt_policy_from_violation(self, violation: CSPViolation):
        """Adaptar política CSP basada en violación reportada"""
        # Solo adaptar para dominios confiables y recursos críticos
        if violation.blocked_domain and violation.blocked_domain not in [
            "inline",
            "eval",
        ]:

            # Verificar si el dominio parece seguro
            if self._is_domain_trustworthy(violation.blocked_domain):
                directive = (
                    violation.effective_directive or violation.violated_directive
                )

                if directive.endswith("-src"):
                    base_directive = directive.replace("-src", "")

                    if self.current_csp_policy.get(directive):
                        # Verificar si ya está permitido
                        if (
                            f"https://{violation.blocked_domain}"
                            not in self.current_csp_policy[directive]
                        ):
                            if (
                                f"*.{violation.blocked_domain.split('.')[-1]}"
                                != violation.blocked_domain
                            ):
                                # Agregar dominio específico
                                self.current_csp_policy[directive].append(
                                    f"https://{violation.blocked_domain}"
                                )
                                logger.info(
                                    f"➕ Adaptado CSP: agregado {violation.blocked_domain} a {directive}"
                                )

                    await self._save_csp_policy()

    def _is_domain_trustworthy(self, domain: str) -> bool:
        """Verificar si un dominio parece confiable"""
        if not domain:
            return False

        # Lista de dominios confiables comunes
        trusted_domains = [
            "fonts.googleapis.com",
            "fonts.gstatic.com",
            "cdn.jsdelivr.net",
            "unpkg.com",
            "code.jquery.com",
            "stackpath.bootstrapcdn.com",
            "maxcdn.bootstrapcdn.com",
            "cdnjs.cloudflare.com",
            "ajax.googleapis.com",
            "apis.google.com",
        ]

        # Verificar dominios whitelist locales
        if domain in self.domain_whitelist:
            return True

        # Verificar dominios conocidos
        if any(trusted in domain for trusted in trusted_domains):
            return True

        # Verificar URLs locales/desarrollo
        if domain in ["localhost", "127.0.0.1", "0.0.0.0"] or domain.endswith(".local"):
            return True

        return False

    def add_trusted_domain(self, domain: str):
        """Agregar dominio a whitelist de confianza"""
        self.domain_whitelist.add(domain)
        logger.info(f"✅ Dominio agregado a whitelist: {domain}")

    def generate_csp_header(self) -> str:
        """Generar header CSP completo basado en política actual"""
        directives = []

        for directive, values in self.current_csp_policy.items():
            if values:
                values_str = " ".join(values)
                directives.append(f"{directive} {values_str}")

        csp_header = "; ".join(directives)

        # Agregar report-uri si está habilitado
        csp_header += "; report-uri /api/csp-report"

        # Agregar report-to para navegadores modernos
        csp_header += "; report-to csp-endpoint"

        return csp_header

    def get_security_headers(self) -> Dict[str, str]:
        """Obtener headers de seguridad completos"""
        headers = self.security_headers.copy()
        headers["Content-Security-Policy"] = self.generate_csp_header()

        # Headers adicionales dinámicos basados en análisis
        headers.update(self._get_dynamic_security_headers())

        return headers

    def _get_dynamic_security_headers(self) -> Dict[str, str]:
        """Generar headers dinámicos basados en análisis de amenazas"""
        dynamic_headers = {}

        # CORS dinámico basado en dominios detectados
        if self.violations:
            allowed_origins = set()
            for violation in self.violations[-100:]:  # Últimas 100 violaciones
                if violation.blocked_domain and self._is_domain_trustworthy(
                    violation.blocked_domain
                ):
                    allowed_origins.add(f"https://{violation.blocked_domain}")

            if allowed_origins:
                dynamic_headers["Access-Control-Allow-Origin"] = ", ".join(
                    sorted(allowed_origins)
                )
                dynamic_headers["Access-Control-Allow-Methods"] = (
                    "GET, POST, PUT, DELETE, OPTIONS"
                )
                dynamic_headers["Access-Control-Allow-Headers"] = (
                    "Content-Type, Authorization, X-CSRF-Token"
                )

        return dynamic_headers

    def harden_csp_policy(self, risk_level: str = "medium"):
        """Endurecer política CSP basado en nivel de riesgo"""
        risk_configs = {
            "low": {
                "remove_unsafe_inline": False,
                "remove_unsafe_eval": False,
                "strict_frame_src": False,
            },
            "medium": {
                "remove_unsafe_inline": True,
                "remove_unsafe_eval": True,
                "strict_frame_src": True,
            },
            "high": {
                "remove_unsafe_inline": True,
                "remove_unsafe_eval": True,
                "strict_frame_src": True,
                "nonce_required": True,
            },
        }

        config = risk_configs.get(risk_level, risk_configs["medium"])

        # Aplicar configuraciones
        if config.get("remove_unsafe_inline"):
            for directive in ["script-src", "style-src"]:
                if directive in self.current_csp_policy:
                    self.current_csp_policy[directive] = [
                        src
                        for src in self.current_csp_policy[directive]
                        if "'unsafe-inline'" not in src
                    ]

        if config.get("remove_unsafe_eval"):
            if "script-src" in self.current_csp_policy:
                self.current_csp_policy["script-src"] = [
                    src
                    for src in self.current_csp_policy["script-src"]
                    if "'unsafe-eval'" not in src
                ]

        if config.get("strict_frame_src"):
            self.current_csp_policy["frame-src"] = ["'none'"]

        if config.get("nonce_required"):
            # Agregar soporte para nonces
            self.current_csp_policy["script-src"].insert(0, "'strict-dynamic'")
            # Nota: Los nonces deben ser generados dinámicamente por request

        logger.info(f"🔒 Política CSP endurecida para nivel de riesgo: {risk_level}")

    async def _save_violations(self):
        """Guardar violaciones en disco"""
        try:
            violations_data = [
                v.to_dict() for v in self.violations[-1000:]
            ]  # Últimas 1000
            with open(self.violations_file, "w", encoding="utf-8") as f:
                json.dump(violations_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando violaciones: {e}")

    async def _save_csp_policy(self):
        """Guardar política CSP en disco"""
        try:
            with open(self.csp_policy_file, "w", encoding="utf-8") as f:
                json.dump(self.current_csp_policy, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando política CSP: {e}")

    async def load_policies(self):
        """Cargar políticas guardadas desde disco"""
        try:
            if self.csp_policy_file.exists():
                with open(self.csp_policy_file, "r", encoding="utf-8") as f:
                    loaded_policy = json.load(f)
                    self.current_csp_policy.update(loaded_policy)
                    logger.info("✅ Política CSP cargada desde disco")

            if self.violations_file.exists():
                with open(self.violations_file, "r", encoding="utf-8") as f:
                    violations_data = json.load(f)
                    for v_data in violations_data[-500:]:  # Cargar últimas 500
                        violation = CSPViolation(v_data)
                        self.violations.append(violation)
                    logger.info(
                        f"✅ {len(self.violations)} violaciones cargadas desde disco"
                    )
        except Exception as e:
            logger.error(f"Error cargando políticas: {e}")

    def get_violation_report(self, days: int = 30) -> Dict[str, Any]:
        """Generar reporte de violaciones CSP"""
        since_date = datetime.now() - timedelta(days=days)
        recent_violations = [v for v in self.violations if v.timestamp >= since_date]

        report = {
            "total_violations": len(recent_violations),
            "violations_by_type": dict(
                Counter(v.violation_type for v in recent_violations)
            ),
            "violations_by_directive": dict(
                Counter(v.violated_directive for v in recent_violations)
            ),
            "top_blocked_domains": dict(
                Counter(
                    v.blocked_domain for v in recent_violations if v.blocked_domain
                ).most_common(10)
            ),
            "violations_timeline": self._get_violations_timeline(recent_violations),
            "recommendations": self._generate_csp_recommendations(recent_violations),
        }

        return report

    def _get_violations_timeline(
        self, violations: List[CSPViolation]
    ) -> Dict[str, int]:
        """Obtener timeline de violaciones por día"""
        timeline = defaultdict(int)

        for violation in violations:
            date_key = violation.timestamp.strftime("%Y-%m-%d")
            timeline[date_key] += 1

        return dict(sorted(timeline.items()))

    def _generate_csp_recommendations(
        self, violations: List[CSPViolation]
    ) -> List[str]:
        """Generar recomendaciones basadas en patrones de violaciones"""
        recommendations = []

        violation_types = Counter(v.violation_type for v in violations)
        directive_violations = Counter(v.violated_directive for v in violations)

        # Recomendaciones basadas en tipos de violaciones
        if violation_types.get("inline_scripts", 0) > 10:
            recommendations.append(
                "Considerar eliminar 'unsafe-inline' de script-src - usar nonces o hashes"
            )
            recommendations.append("Mover scripts inline a archivos externos")

        if violation_types.get("external_scripts", 0) > 20:
            recommendations.append(
                "Revisar y limitar dominios permitidos en script-src"
            )

        if directive_violations.get("style-src", 0) > 15:
            recommendations.append(
                "Considerar eliminar 'unsafe-inline' de style-src - usar CSS externo"
            )

        if violation_types.get("api_calls", 0) > 30:
            recommendations.append(
                "Limitar dominios permitidos en connect-src para llamadas API"
            )

        # Recomendaciones generales
        if len(violations) > 100:
            recommendations.append(
                "Implementar política CSP más restrictiva - considerar usar 'strict-dynamic'"
            )
            recommendations.append(
                "Configurar monitoreo CSP en producción para ajuste continuo"
            )

        if not recommendations:
            recommendations.append(
                "Políticas CSP funcionando bien - monitorear para ajustes menores"
            )

        return recommendations

    def disable_learning_mode(self):
        """Deshabilitar modo de aprendizaje y consolidar política"""
        self.learning_mode = False

        # Endurecer política por defecto
        self.harden_csp_policy("medium")

        logger.info("🔒 Modo de aprendizaje CSP deshabilitado - política endurecida")


class SecurityHeadersMiddleware:
    """Middleware para aplicar headers de seguridad automáticamente"""

    def __init__(self, csp_manager: AutomatedCSPManager):
        self.csp_manager = csp_manager
        self.nonce_cache = {}  # Cache de nonces por sesión/request

    def generate_nonce(self) -> str:
        """Generar nonce criptográfico único"""
        nonce_bytes = os.urandom(16)
        nonce = base64.b64encode(nonce_bytes).decode("utf-8")
        return nonce

    async def apply_security_headers(self, request, response) -> None:
        """Aplicar headers de seguridad a una respuesta HTTP"""
        try:
            # Obtener headers desde el manager
            security_headers = self.csp_manager.get_security_headers()

            # Generar nonce único para esta respuesta
            nonce = self.generate_nonce()

            # Aplicar CSP header con nonce si es necesario
            csp_header = security_headers.get("Content-Security-Policy", "")
            if csp_header:
                # Reemplazar 'strict-dynamic' con nonce
                csp_header = csp_header.replace(
                    "'strict-dynamic'", f"'strict-dynamic' 'nonce-{nonce}'"
                )
                security_headers["Content-Security-Policy"] = csp_header

            # Aplicar headers a la respuesta
            for header_name, header_value in security_headers.items():
                response.headers[header_name] = header_value

            # Agregar nonce a locals para usar en templates
            if hasattr(request, "locals"):
                request.locals["csp_nonce"] = nonce

            logger.debug(
                f"🛡️ Headers de seguridad aplicados - CSP: {(csp_header[:50] + '...') if csp_header else 'None'}"
            )

        except Exception as e:
            logger.error(f"Error aplicando headers de seguridad: {e}")

    async def report_csp_violation_endpoint(self, request) -> Dict[str, Any]:
        """Endpoint para reportar violaciones CSP"""
        try:
            violation_data = await request.json()
            result = await self.csp_manager.report_csp_violation(violation_data)
            return result
        except Exception as e:
            logger.error(f"Error procesando reporte CSP: {e}")
            return {"status": "error", "message": str(e)}


class CSPViolationDashboard:
    """Dashboard para monitoreo de violaciones CSP"""

    def __init__(self, csp_manager: AutomatedCSPManager):
        self.csp_manager = csp_manager

    def generate_security_report(self) -> Dict[str, Any]:
        """Generar reporte completo de seguridad CSP"""
        violation_report = self.csp_manager.get_violation_report(days=30)

        security_score = self._calculate_security_score(violation_report)
        risk_level = self._assess_risk_level(security_score)

        report = {
            "generated_at": datetime.now().isoformat(),
            "security_score": security_score,
            "risk_level": risk_level,
            "csp_effectiveness": self._analyze_csp_effectiveness(),
            "violation_analysis": violation_report,
            "policy_recommendations": self.csp_manager._generate_csp_recommendations(
                self.csp_manager.violations[-100:]
            ),
            "learning_status": {
                "learning_mode_active": self.csp_manager.learning_mode,
                "adaptations_made": len(self.csp_manager.domain_whitelist),
                "policy_complexity": len(self.csp_manager.current_csp_policy),
            },
        }

        return report

    def _calculate_security_score(self, violation_report: Dict[str, Any]) -> float:
        """Calcular score de seguridad basado en métricas"""
        total_violations = violation_report.get("total_violations", 0)

        # Base score de 100
        security_score = 100.0

        # Penalización por violaciones
        if total_violations > 0:
            # -1 punto por cada 10 violaciones
            violation_penalty = min(
                total_violations / 10, 30
            )  # Max 30 puntos de penalización
            security_score -= violation_penalty

        # Bonus por CSP restrictivo
        csp_directives = len(self.csp_manager.current_csp_policy)
        if csp_directives > 8:
            security_score += 5  # Bonus por CSP comprehensive

        # Penalización por CSP permisivo
        permissive_indicators = 0
        for directive, sources in self.csp_manager.current_csp_policy.items():
            if any("'unsafe-inline'" in src for src in sources):
                permissive_indicators += 1
            if any("'unsafe-eval'" in src for src in sources):
                permissive_indicators += 2

        security_score -= permissive_indicators * 2

        return max(0.0, min(100.0, security_score))

    def _assess_risk_level(self, security_score: float) -> str:
        """Evaluar nivel de riesgo basado en score"""
        if security_score >= 90:
            return "very_low"
        elif security_score >= 80:
            return "low"
        elif security_score >= 70:
            return "medium"
        elif security_score >= 60:
            return "high"
        else:
            return "critical"

    def _analyze_csp_effectiveness(self) -> Dict[str, Any]:
        """Analizar efectividad de la política CSP actual"""
        violations = self.csp_manager.violations[-100:]  # Últimas 100

        blocked_resources = sum(1 for v in violations if v.blocked_uri)
        inline_scripts_blocked = sum(
            1 for v in violations if v.violation_type == "inline_scripts"
        )
        external_scripts_blocked = sum(
            1 for v in violations if v.violation_type == "external_scripts"
        )

        effectiveness = {
            "policy_coverage": len(self.csp_manager.current_csp_policy),
            "resources_protected": blocked_resources,
            "risky_inline_scripts_blocked": inline_scripts_blocked,
            "external_resources_filtered": external_scripts_blocked,
            "policy_maturity": (
                "learning" if self.csp_manager.learning_mode else "production"
            ),
            "whitelisted_domains": list(self.csp_manager.domain_whitelist)[
                :10
            ],  # Top 10
        }

        return effectiveness


# ================================
# DEMO Y EJEMPLOS DE USO
# ================================


async def demo_csp_security_system():
    """Demo del sistema de CSP y headers de seguridad"""
    print("🛡️ DEMO: Automated CSP Security Headers System")
    print("=" * 60)

    # Inicializar sistema
    csp_manager = AutomatedCSPManager()

    # Cargar políticas existentes
    await csp_manager.load_policies()

    # Simular algunas violaciones comunes
    sample_violations = [
        {
            "document-uri": "https://example.com/dashboard",
            "violated-directive": "script-src",
            "blocked-uri": "https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js",
            "source-file": "dashboard.html",
            "line-number": 45,
        },
        {
            "document-uri": "https://example.com/app",
            "violated-directive": "style-src",
            "blocked-uri": "https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap",
            "source-file": "app.css",
            "line-number": 12,
        },
        {
            "document-uri": "https://example.com/api-client",
            "violated-directive": "connect-src",
            "blocked-uri": "https://api.example.com/data",
            "source-file": "api-client.js",
            "line-number": 78,
        },
    ]

    # Reportar violaciones y adaptar política
    for violation_data in sample_violations:
        result = await csp_manager.report_csp_violation(violation_data)
        print(
            f"🚨 Violación reportada: {violation_data['violated-directive']} - {violation_data['blocked-uri']}"
        )

    # Agregar dominios confiables
    csp_manager.add_trusted_domain("cdn.jsdelivr.net")
    csp_manager.add_trusted_domain("fonts.googleapis.com")
    csp_manager.add_trusted_domain("api.example.com")

    # Generar CSP header
    csp_header = csp_manager.generate_csp_header()
    print(f"\n📋 CSP Header generado:\n{csp_header}\n")

    # Obtener todos los headers de seguridad
    security_headers = csp_manager.get_security_headers()
    print("🔒 Headers de seguridad aplicados:")
    for header, value in security_headers.items():
        print(f"  • {header}: {value[:80]}{'...' if len(value) > 80 else ''}")

    # Endurecer política
    csp_manager.harden_csp_policy("medium")
    print("\n🔏 Política CSP endurecida para medio riesgo")

    # Generar reporte de seguridad
    dashboard = CSPViolationDashboard(csp_manager)
    security_report = dashboard.generate_security_report()

    print("\n📊 Reporte de Seguridad Final:")
    print(f"  • Security Score: {security_report['security_score']:.1f}/100")
    print(f"  • Risk Level: {security_report['risk_level']}")
    print(
        f"  • CSP Effectiveness: {security_report['csp_effectiveness']['policy_coverage']} directivas"
    )

    violation_report = security_report["violation_analysis"]
    print(f"  • Violations (30 días): {violation_report['total_violations']}")

    if security_report["policy_recommendations"]:
        print("  • Recomendaciones:")
        for rec in security_report["policy_recommendations"][:3]:
            print(f"    - {rec}")

    # Middleware demo
    middleware = SecurityHeadersMiddleware(csp_manager)

    print("\n✅ Demo del sistema CSP completada exitosamente")
    return security_report


if __name__ == "__main__":
    # Demo
    asyncio.run(demo_csp_security_system())
