"""
Sheily MCP Enterprise - Security Scanner
Sistema avanzado de escaneo de vulnerabilidades
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SecurityScanner:
    """Escáner avanzado de vulnerabilidades"""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    async def scan_vulnerabilities(self) -> Dict[str, Any]:
        """Escanear vulnerabilidades de seguridad"""

        results = {
            "vulnerabilities": [],
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "total_scanned": 0,
            "scan_methods": [],
            "errors": [],
        }

        # Method 1: Try to use safety library if installed
        safety_results = await self._scan_with_safety()
        if safety_results:
            results["vulnerabilities"].extend(safety_results["vulnerabilities"])
            results["scan_methods"].append("safety")
            results["errors"].extend(safety_results.get("errors", []))
            self._count_vulnerabilities(results, safety_results["vulnerabilities"])

        # Method 2: Try pip-audit if available
        pip_audit_results = await self._scan_with_pip_audit()
        if pip_audit_results:
            results["vulnerabilities"].extend(pip_audit_results["vulnerabilities"])
            results["scan_methods"].append("pip-audit")
            results["errors"].extend(pip_audit_results.get("errors", []))
            self._count_vulnerabilities(results, pip_audit_results["vulnerabilities"])

        # Method 3: Basic security checks based on package versions
        basic_results = await self._basic_security_check()
        results["vulnerabilities"].extend(basic_results["vulnerabilities"])
        results["scan_methods"].append("basic")
        self._count_vulnerabilities(results, basic_results["vulnerabilities"])

        # Remove duplicates
        unique_vulns = self._deduplicate_vulnerabilities(results["vulnerabilities"])
        results["vulnerabilities"] = unique_vulns
        results["total_scanned"] = len(results["vulnerabilities"])

        logger.info(
            f"Security scan completed: found {len(results['vulnerabilities'])} vulnerabilities "
            f"using {len(results['scan_methods'])} methods"
        )

        return results

    async def _scan_with_safety(self) -> Optional[Dict[str, Any]]:
        """Scan using safety library if available"""

        try:
            # Try to run safety check
            result = await self._run_command(
                [sys.executable, "-m", "safety", "check", "--json"]
            )

            if result["success"]:
                try:
                    safety_data = json.loads(result["output"])

                    vulnerabilities = []
                    for vuln in safety_data.get("vulnerabilities", []):
                        vulnerabilities.append(
                            {
                                "package": vuln.get("package_name", ""),
                                "installed_version": vuln.get("current_version", ""),
                                "vulnerable_version": vuln.get("affected_versions", []),
                                "severity": "high",  # safety doesn't provide severity, assume high
                                "description": vuln.get(
                                    "vulnerability_description", ""
                                ),
                                "cve_id": vuln.get("vulnerability_id", ""),
                                "source": "safety",
                                "advisory": vuln.get("vulnerability_description", ""),
                            }
                        )

                    return {"vulnerabilities": vulnerabilities, "errors": []}

                except json.JSONDecodeError:
                    return {
                        "vulnerabilities": [],
                        "errors": ["Safety returned invalid JSON"],
                    }

        except Exception as e:
            logger.debug(f"Safety scan failed: {e}")

        return None

    async def _scan_with_pip_audit(self) -> Optional[Dict[str, Any]]:
        """Scan using pip-audit if available"""

        try:
            result = await self._run_command(
                [sys.executable, "-m", "pip_audit", "--format=json"]
            )

            if result["success"]:
                try:
                    audit_data = json.loads(result["output"])

                    vulnerabilities = []
                    if isinstance(audit_data, list):
                        for vuln in audit_data:
                            vulnerabilities.append(
                                {
                                    "package": vuln.get("name", ""),
                                    "installed_version": vuln.get("version", ""),
                                    "vulnerable_version": vuln.get("specifiers", []),
                                    "severity": vuln.get("severity", "medium").lower(),
                                    "description": vuln.get("description", ""),
                                    "cve_id": vuln.get("id", ""),
                                    "source": "pip-audit",
                                    "advisory": vuln.get("description", ""),
                                }
                            )

                    return {"vulnerabilities": vulnerabilities, "errors": []}

                except json.JSONDecodeError:
                    return {
                        "vulnerabilities": [],
                        "errors": ["pip-audit returned invalid JSON"],
                    }

        except Exception as e:
            logger.debug(f"pip-audit scan failed: {e}")

        return None

    async def _basic_security_check(self) -> Dict[str, Any]:
        """Basic security checks based on package versions and known issues"""

        vulnerabilities = []

        try:
            # Get installed packages
            result = await self._run_command(
                [sys.executable, "-m", "pip", "list", "--format=json"]
            )

            if result["success"]:
                packages = json.loads(result["output"])

                # Known vulnerable patterns
                vulnerable_patterns = [
                    {
                        "name": "django",
                        "check_versions": lambda v: self._version_less_than(
                            v, "3.2.20"
                        ),
                        "severity": "critical",
                        "description": "Django versions before 3.2.20 have security vulnerabilities",
                    },
                    {
                        "name": "flask",
                        "check_versions": lambda v: self._version_less_than(v, "2.3.0"),
                        "severity": "high",
                        "description": "Flask versions before 2.3.0 have known security issues",
                    },
                    {
                        "name": "requests",
                        "check_versions": lambda v: v.startswith("0.")
                        or v.startswith("1."),
                        "severity": "medium",
                        "description": "Very old requests versions may have vulnerabilities",
                    },
                ]

                for package in packages:
                    pkg_name = package["name"].lower()
                    pkg_version = package["version"]

                    for pattern in vulnerable_patterns:
                        if pkg_name == pattern["name"] and pattern["check_versions"](
                            pkg_version
                        ):
                            vulnerabilities.append(
                                {
                                    "package": package["name"],
                                    "installed_version": pkg_version,
                                    "vulnerable_version": ["old versions"],
                                    "severity": pattern["severity"],
                                    "description": pattern["description"],
                                    "cve_id": "N/A",
                                    "source": "basic_checker",
                                    "advisory": pattern["description"],
                                }
                            )

        except Exception as e:
            logger.debug(f"Basic security check failed: {e}")

        return {"vulnerabilities": vulnerabilities}

    def _version_less_than(self, version: str, target: str) -> bool:
        """Simple version comparison"""

        try:
            # Simple comparison for basic common cases
            v_parts = version.split(".")
            t_parts = target.split(".")

            for i, (v, t) in enumerate(zip(v_parts, t_parts)):
                if int(v) < int(t):
                    return True
                elif int(v) > int(t):
                    return False

            return len(v_parts) < len(t_parts)

        except (ValueError, AttributeError):
            # If parsing fails, assume it might be vulnerable
            return True

    def _count_vulnerabilities(
        self, results: Dict[str, Any], vulnerabilities: List[Dict[str, Any]]
    ):
        """Count vulnerabilities by severity"""

        severity_map = {
            "critical": "critical_count",
            "high": "high_count",
            "medium": "medium_count",
            "low": "low_count",
        }

        for vuln in vulnerabilities:
            severity = vuln.get("severity", "unknown").lower()
            count_key = severity_map.get(severity, "low_count")
            results[count_key] += 1

    def _deduplicate_vulnerabilities(
        self, vulnerabilities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate vulnerabilities"""

        seen = set()
        unique = []

        for vuln in vulnerabilities:
            key = (
                vuln.get("package", "").lower(),
                vuln.get("cve_id", ""),
                vuln.get("source", ""),
            )

            if key not in seen:
                seen.add(key)
                unique.append(vuln)

        return unique

    async def _run_command(self, cmd: List[str], cwd: str = None) -> Dict[str, Any]:
        """Execute command asynchronously"""

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "output": stdout.decode("utf-8", errors="ignore").strip(),
                "error": stderr.decode("utf-8", errors="ignore").strip(),
                "returncode": process.returncode,
            }

        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": f"Command not found: {cmd[0]}",
                "returncode": -1,
            }

        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "returncode": -1}
