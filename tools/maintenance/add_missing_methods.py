#!/usr/bin/env python3
"""
SCRIPT PARA AGREGAR MÉTODOS FALTANTES A LOS AGENTES
====================================================

Este script agrega automáticamente los métodos que faltan
según las pruebas del sistema.
"""

import os
from pathlib import Path


def add_methods_to_agent(agent_file: str, methods_to_add: dict):
    """Agrega métodos faltantes a un agente específico"""
    try:
        with open(agent_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Buscar la clase del agente
        lines = content.split("\n")
        class_start = -1
        for i, line in enumerate(lines):
            if line.startswith("class ") and "Agent" in line:
                class_start = i
                break

        if class_start == -1:
            print(f"❌ No se encontró la clase en {agent_file}")
            return False

        # Encontrar el final de la clase (último método)
        class_end = len(lines) - 1
        indent_level = None

        for i in range(class_start + 1, len(lines)):
            line = lines[i]
            if line.strip().startswith("def ") and not line.startswith("    "):
                # Encontramos el siguiente método a nivel de módulo
                class_end = i - 1
                break
            elif (
                line.strip() and not line.startswith(" ") and not line.startswith("\t")
            ):
                # Línea que no está indentada (fuera de la clase)
                class_end = i - 1
                break

        # Agregar métodos antes del final de la clase
        new_methods = []

        for method_name, method_code in methods_to_add.items():
            # Verificar si el método ya existe
            method_exists = False
            for line in lines[class_start:class_end]:
                if f"def {method_name}(" in line:
                    method_exists = True
                    break

            if not method_exists:
                new_methods.append(f"\n{method_code}")

        if new_methods:
            # Insertar métodos antes del final de la clase
            insert_pos = class_end
            for method in reversed(new_methods):
                lines.insert(insert_pos, method)

            # Escribir archivo actualizado
            with open(agent_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"✅ Agregados {len(new_methods)} métodos a {agent_file}")
            return True
        else:
            print(f"ℹ️ Todos los métodos ya existen en {agent_file}")
            return False

    except Exception as e:
        print(f"❌ Error procesando {agent_file}: {e}")
        return False


def main():
    """Función principal"""
    agents_dir = Path("sheily_core/agents")

    # Métodos a agregar por agente
    methods_map = {
        "docker_manager_agent.py": {
            "list_containers": '''    def list_containers(self) -> Dict[str, Any]:
        """Lista todos los contenedores Docker"""
        try:
            if not self.docker_client:
                return {"error": "Docker client not available", "containers": []}

            containers = self.docker_client.containers.list(all=True)
            container_info = []

            for container in containers:
                container_info.append({
                    "id": container.id[:12],
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "status": container.status,
                    "ports": container.ports
                })

            return {
                "containers": container_info,
                "total": len(container_info),
                "running": len([c for c in container_info if c["status"] == "running"])
            }
        except Exception as e:
            return {"error": str(e), "containers": []}''',
            "list_images": '''    def list_images(self) -> Dict[str, Any]:
        """Lista todas las imágenes Docker"""
        try:
            if not self.docker_client:
                return {"error": "Docker client not available", "images": []}

            images = self.docker_client.images.list()
            image_info = []

            for image in images:
                tags = image.tags if image.tags else ["<none>"]
                image_info.append({
                    "id": image.id[:12],
                    "tags": tags,
                    "size": image.attrs.get("Size", 0),
                    "created": image.attrs.get("Created", "")
                })

            return {
                "images": image_info,
                "total": len(image_info)
            }
        except Exception as e:
            return {"error": str(e), "images": []}''',
        },
        "kubernetes_agent.py": {
            "list_pods": '''    def list_pods(self, namespace: str = "default") -> Dict[str, Any]:
        """Lista todos los pods en un namespace"""
        try:
            if not self.k8s_client:
                return {"error": "Kubernetes client not available", "pods": []}

            pods = self.k8s_client.list_namespaced_pod(namespace)
            pod_info = []

            for pod in pods.items:
                pod_info.append({
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "node": pod.spec.node_name,
                    "containers": len(pod.spec.containers)
                })

            return {
                "pods": pod_info,
                "total": len(pod_info),
                "namespace": namespace
            }
        except Exception as e:
            return {"error": str(e), "pods": []}''',
            "list_services": '''    def list_services(self, namespace: str = "default") -> Dict[str, Any]:
        """Lista todos los servicios en un namespace"""
        try:
            if not self.k8s_client:
                return {"error": "Kubernetes client not available", "services": []}

            services = self.k8s_client.list_namespaced_service(namespace)
            service_info = []

            for svc in services.items:
                service_info.append({
                    "name": svc.metadata.name,
                    "namespace": svc.metadata.namespace,
                    "type": svc.spec.type,
                    "cluster_ip": svc.spec.cluster_ip,
                    "ports": [{"port": p.port, "target_port": p.target_port, "protocol": p.protocol} for p in (svc.spec.ports or [])]
                })

            return {
                "services": service_info,
                "total": len(service_info),
                "namespace": namespace
            }
        except Exception as e:
            return {"error": str(e), "services": []}''',
        },
        "dependency_manager_agent.py": {
            "get_dependency_tree": '''    def get_dependency_tree(self) -> Dict[str, Any]:
        """Obtiene el árbol de dependencias del proyecto"""
        try:
            dependencies = {}

            # Buscar archivos de dependencias
            req_files = list(self.project_root.glob("**/requirements*.txt"))
            req_files.extend(self.project_root.glob("**/Pipfile"))
            req_files.extend(self.project_root.glob("**/pyproject.toml"))

            for req_file in req_files:
                try:
                    with open(req_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    deps = self._parse_dependencies(content, req_file.name)
                    dependencies[str(req_file.relative_to(self.project_root))] = deps

                except Exception as e:
                    print(f"Error reading {req_file}: {e}")

            return {
                "dependency_files": list(dependencies.keys()),
                "dependencies": dependencies,
                "total_files": len(dependencies),
                "total_dependencies": sum(len(deps) for deps in dependencies.values())
            }
        except Exception as e:
            return {"error": str(e), "dependencies": {}}''',
            "scan_security_vulnerabilities": '''    def scan_security_vulnerabilities(self) -> Dict[str, Any]:
        """Escanea vulnerabilidades de seguridad en dependencias"""
        try:
            vulnerabilities = []
            tree = self.get_dependency_tree()

            for file_path, deps in tree.get("dependencies", {}).items():
                for dep_name, dep_version in deps:
                    vuln = self._check_known_vulnerability(dep_name, dep_version)
                    if vuln:
                        vuln["file"] = file_path
                        vulnerabilities.append(vuln)

            return {
                "vulnerabilities_found": len(vulnerabilities),
                "vulnerabilities": vulnerabilities,
                "severity_breakdown": {
                    "critical": len([v for v in vulnerabilities if v.get("severity") == "critical"]),
                    "high": len([v for v in vulnerabilities if v.get("severity") == "high"]),
                    "medium": len([v for v in vulnerabilities if v.get("severity") == "medium"]),
                    "low": len([v for v in vulnerabilities if v.get("severity") == "low"])
                }
            }
        except Exception as e:
            return {"error": str(e), "vulnerabilities_found": 0}''',
        },
        "log_manager_agent.py": {
            "get_recent_logs": '''    def get_recent_logs(self, limit: int = 100) -> Dict[str, Any]:
        """Obtiene logs recientes del sistema"""
        try:
            logs = []

            # Obtener logs de diferentes fuentes
            for source_name, source_config in self.log_sources.items():
                try:
                    source_logs = self._get_logs_from_source(source_name, limit)
                    logs.extend(source_logs)
                except Exception as e:
                    print(f"Error getting logs from {source_name}: {e}")

            # Ordenar por timestamp (más recientes primero)
            logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            logs = logs[:limit]

            return {
                "logs": logs,
                "total": len(logs),
                "sources": list(self.log_sources.keys())
            }
        except Exception as e:
            return {"error": str(e), "logs": []}''',
            "get_log_metrics": '''    def get_log_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de logs"""
        try:
            recent_logs = self.get_recent_logs(1000)
            logs = recent_logs.get("logs", [])

            metrics = {
                "total_logs": len(logs),
                "error_count": len([l for l in logs if l.get("level") == "ERROR"]),
                "warning_count": len([l for l in logs if l.get("level") == "WARNING"]),
                "info_count": len([l for l in logs if l.get("level") == "INFO"]),
                "sources_count": len(self.log_sources),
                "alerts_active": len(self.alerts)
            }

            return metrics
        except Exception as e:
            return {"error": str(e), "total_logs": 0}''',
        },
        "memory_tuning_agent.py": {
            "get_memory_stats": '''    def get_memory_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de memoria del sistema"""
        try:
            import psutil

            memory = psutil.virtual_memory()
            process = psutil.Process()

            stats = {
                "system_memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percentage": memory.percent
                },
                "process_memory": {
                    "rss": process.memory_info().rss,
                    "vms": process.memory_info().vms,
                    "percentage": process.memory_percent()
                },
                "large_objects": self._get_large_objects(),
                "memory_pressure": "high" if memory.percent > 90 else "medium" if memory.percent > 75 else "low"
            }

            return stats
        except Exception as e:
            return {"error": str(e), "system_memory": {}, "process_memory": {}}''',
            "get_optimization_suggestions": '''    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Obtiene sugerencias de optimización de memoria"""
        try:
            stats = self.get_memory_stats()
            suggestions = []

            memory_percent = stats.get("system_memory", {}).get("percentage", 0)

            if memory_percent > 90:
                suggestions.append({
                    "type": "critical",
                    "message": "Memory usage is critically high",
                    "action": "Reduce memory-intensive operations immediately"
                })
            elif memory_percent > 75:
                suggestions.append({
                    "type": "warning",
                    "message": "Memory usage is high",
                    "action": "Consider optimizing memory usage"
                })

            large_objects = stats.get("large_objects", [])
            if len(large_objects) > 10:
                suggestions.append({
                    "type": "info",
                    "message": f"Found {len(large_objects)} large objects",
                    "action": "Review and optimize large object usage"
                })

            return suggestions
        except Exception as e:
            return [{"type": "error", "message": str(e), "action": "Check memory monitoring"}]

    def _get_large_objects(self) -> List[Dict[str, Any]]:
        """Obtiene objetos grandes en memoria"""
        try:
            import gc
            large_objects = []

            for obj in gc.get_objects():
                size = sys.getsizeof(obj)
                if size > self.large_object_threshold:
                    large_objects.append({
                        "type": type(obj).__name__,
                        "size": size,
                        "id": id(obj)
                    })

            return sorted(large_objects, key=lambda x: x["size"], reverse=True)[:20]
        except Exception:
            return []''',
        },
        "testing_agent.py": {
            "list_test_suites": '''    def list_test_suites(self) -> Dict[str, Any]:
        """Lista todas las suites de pruebas disponibles"""
        try:
            suites = []

            for suite_name, suite_info in self.test_suites.items():
                suites.append({
                    "name": suite_name,
                    "test_count": suite_info.get("test_count", 0),
                    "last_run": suite_info.get("last_run"),
                    "status": suite_info.get("status", "unknown")
                })

            return {
                "suites": suites,
                "total_suites": len(suites),
                "total_tests": sum(s["test_count"] for s in suites)
            }
        except Exception as e:
            return {"error": str(e), "suites": []}''',
            "get_test_results": '''    def get_test_results(self) -> Dict[str, Any]:
        """Obtiene resultados de las últimas pruebas ejecutadas"""
        try:
            results = {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "coverage": 0.0,
                "last_run": None,
                "test_suites": []
            }

            for suite_name, suite_info in self.test_suites.items():
                suite_results = suite_info.get("last_results", {})
                results["total_tests"] += suite_results.get("total", 0)
                results["passed"] += suite_results.get("passed", 0)
                results["failed"] += suite_results.get("failed", 0)
                results["skipped"] += suite_results.get("skipped", 0)

                results["test_suites"].append({
                    "name": suite_name,
                    "results": suite_results,
                    "last_run": suite_info.get("last_run")
                })

            if results["total_tests"] > 0:
                results["coverage"] = (results["passed"] / results["total_tests"]) * 100

            return results
        except Exception as e:
            return {"error": str(e), "total_tests": 0}''',
        },
        "documentation_agent.py": {
            "list_documentation": '''    def list_documentation(self) -> Dict[str, Any]:
        """Lista toda la documentación disponible"""
        try:
            docs = []

            # Buscar archivos de documentación
            for ext in ['.md', '.rst', '.txt', '.html']:
                for doc_file in self.docs_dir.glob(f"**/*{ext}"):
                    try:
                        stat = doc_file.stat()
                        docs.append({
                            "path": str(doc_file.relative_to(self.docs_dir)),
                            "name": doc_file.name,
                            "extension": ext,
                            "size": stat.st_size,
                            "modified": stat.st_mtime
                        })
                    except Exception:
                        pass

            return {
                "documentation_files": docs,
                "total_files": len(docs),
                "total_size": sum(d["size"] for d in docs)
            }
        except Exception as e:
            return {"error": str(e), "documentation_files": []}''',
            "generate_api_docs": '''    def generate_api_docs(self) -> Dict[str, Any]:
        """Genera documentación de API automáticamente"""
        try:
            api_docs = {
                "endpoints": [],
                "models": [],
                "generated_at": datetime.now().isoformat()
            }

            # Buscar archivos Python con endpoints
            for py_file in self.source_dir.glob("**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Buscar patrones de endpoints
                    import re
                    endpoint_patterns = [
                        r'@(?:app|router|api)\.(?:get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
                        r'def\s+(\w+)\s*\([^)]*\):\s*["\'](?:GET|POST|PUT|DELETE|PATCH)\s+([^\']+)'
                    ]

                    for pattern in endpoint_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            if isinstance(match, tuple):
                                method, path = match
                            else:
                                path = match
                                method = "unknown"

                            api_docs["endpoints"].append({
                                "path": path,
                                "method": method.upper(),
                                "file": str(py_file.relative_to(self.source_dir))
                            })

                except Exception as e:
                    print(f"Error processing {py_file}: {e}")

            return api_docs
        except Exception as e:
            return {"error": str(e), "endpoints": []}''',
        },
        "security_hardening_agent.py": {
            "scan_security_issues": '''    def scan_security_issues(self) -> Dict[str, Any]:
        """Escanea problemas de seguridad en el sistema"""
        try:
            issues = []

            # Verificar configuraciones de seguridad
            config_issues = self._check_security_configs()
            issues.extend(config_issues)

            # Verificar permisos de archivos
            file_issues = self._check_file_permissions()
            issues.extend(file_issues)

            # Verificar políticas de seguridad
            policy_issues = self._check_security_policies()
            issues.extend(policy_issues)

            return {
                "issues_found": len(issues),
                "issues": issues,
                "severity_breakdown": {
                    "critical": len([i for i in issues if i.get("severity") == "critical"]),
                    "high": len([i for i in issues if i.get("severity") == "high"]),
                    "medium": len([i for i in issues if i.get("severity") == "medium"]),
                    "low": len([i for i in issues if i.get("severity") == "low"])
                }
            }
        except Exception as e:
            return {"error": str(e), "issues_found": 0}''',
            "get_hardening_status": '''    def get_hardening_status(self) -> Dict[str, Any]:
        """Obtiene el estado de hardening de seguridad"""
        try:
            status = {
                "encryption_enabled": self.encryption_enabled,
                "policies_loaded": len(self.security_policies),
                "keys_generated": len(list(self.keys_dir.glob("*"))) if self.keys_dir.exists() else 0,
                "reports_generated": len(list(self.reports_dir.glob("*.json"))) if self.reports_dir.exists() else 0,
                "last_scan": getattr(self, 'last_scan_time', None)
            }

            return status
        except Exception as e:
            return {"error": str(e), "encryption_enabled": False}''',
        },
        "performance_profiling_agent.py": {
            "get_performance_metrics": '''    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento del sistema"""
        try:
            import psutil
            import time

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage('/')

            # Network metrics
            network = psutil.net_io_counters()

            metrics = {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "core_count": cpu_count
                },
                "memory": {
                    "usage_percent": memory.percent,
                    "used_gb": memory.used / (1024**3),
                    "available_gb": memory.available / (1024**3)
                },
                "disk": {
                    "usage_percent": disk.percent,
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                "timestamp": datetime.now().isoformat()
            }

            return metrics
        except Exception as e:
            return {"error": str(e), "cpu": {}, "memory": {}}''',
            "identify_bottlenecks": '''    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identifica cuellos de botella en el rendimiento"""
        try:
            metrics = self.get_performance_metrics()
            bottlenecks = []

            # Check CPU bottleneck
            if metrics.get("cpu", {}).get("usage_percent", 0) > 90:
                bottlenecks.append({
                    "type": "cpu",
                    "severity": "critical",
                    "message": "CPU usage is extremely high",
                    "recommendation": "Optimize CPU-intensive operations or scale resources"
                })

            # Check memory bottleneck
            if metrics.get("memory", {}).get("usage_percent", 0) > 90:
                bottlenecks.append({
                    "type": "memory",
                    "severity": "critical",
                    "message": "Memory usage is critically high",
                    "recommendation": "Reduce memory consumption or increase RAM"
                })

            # Check disk bottleneck
            if metrics.get("disk", {}).get("usage_percent", 0) > 95:
                bottlenecks.append({
                    "type": "disk",
                    "severity": "high",
                    "message": "Disk space is nearly full",
                    "recommendation": "Free up disk space or expand storage"
                })

            return bottlenecks
        except Exception as e:
            return [{"type": "error", "message": str(e), "severity": "unknown"}]''',
        },
        "resource_management_agent.py": {
            "get_resource_usage": '''    def get_resource_usage(self) -> Dict[str, Any]:
        """Obtiene uso de recursos del sistema"""
        try:
            import psutil

            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            usage = {
                "cpu_percent": cpu_usage,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "timestamp": datetime.now().isoformat()
            }

            return usage
        except Exception as e:
            return {"error": str(e), "cpu_percent": 0, "memory_percent": 0}''',
            "generate_optimization_plan": '''    def generate_optimization_plan(self) -> Dict[str, Any]:
        """Genera un plan de optimización de recursos"""
        try:
            usage = self.get_resource_usage()
            plan = {
                "recommendations": [],
                "priority_actions": [],
                "estimated_savings": {}
            }

            # CPU optimization
            if usage.get("cpu_percent", 0) > 80:
                plan["recommendations"].append("Optimize CPU-intensive operations")
                plan["priority_actions"].append("Profile and optimize hot code paths")

            # Memory optimization
            if usage.get("memory_percent", 0) > 80:
                plan["recommendations"].append("Reduce memory usage")
                plan["priority_actions"].append("Implement memory pooling or caching")

            # Disk optimization
            if usage.get("disk_percent", 0) > 85:
                plan["recommendations"].append("Optimize disk usage")
                plan["priority_actions"].append("Implement log rotation and cleanup")

            return plan
        except Exception as e:
            return {"error": str(e), "recommendations": []}''',
        },
        "core_optimization_agent.py": {
            "get_core_metrics": '''    def get_core_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del núcleo del sistema"""
        try:
            import psutil
            import gc

            metrics = {
                "process_count": len(psutil.pids()),
                "thread_count": sum(len(psutil.Process(pid).threads()) for pid in psutil.pids()[:10]),  # Sample
                "garbage_collections": {
                    "gen0": gc.get_count()[0],
                    "gen1": gc.get_count()[1],
                    "gen2": gc.get_count()[2]
                },
                "open_files": len(psutil.Process().open_files()),
                "connections": len(psutil.Process().connections()),
                "timestamp": datetime.now().isoformat()
            }

            return metrics
        except Exception as e:
            return {"error": str(e), "process_count": 0}''',
            "get_optimization_status": '''    def get_optimization_status(self) -> Dict[str, Any]:
        """Obtiene el estado de optimización del núcleo"""
        try:
            metrics = self.get_core_metrics()
            status = {
                "optimization_level": self.optimization_level,
                "numba_enabled": self.numba_available,
                "cython_enabled": self.cython_available,
                "performance_score": self._calculate_performance_score(metrics),
                "last_optimization": getattr(self, 'last_optimization_time', None)
            }

            return status
        except Exception as e:
            return {"error": str(e), "optimization_level": "unknown"}

    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calcula una puntuación de rendimiento"""
        try:
            # Simple scoring based on metrics
            score = 100.0

            # Penalize high process count
            if metrics.get("process_count", 0) > 100:
                score -= 20

            # Penalize high garbage collection
            if metrics.get("garbage_collections", {}).get("gen2", 0) > 1000:
                score -= 15

            return max(0.0, score)
        except Exception:
            return 50.0''',
        },
        "auto_improvement_agent.py": {
            "get_pending_improvements": '''    def get_pending_improvements(self) -> List[Dict[str, Any]]:
        """Obtiene mejoras pendientes identificadas"""
        try:
            improvements = []

            # Simular mejoras pendientes basadas en análisis
            if hasattr(self, 'code_quality_score') and self.code_quality_score < self.quality_threshold:
                improvements.append({
                    "type": "code_quality",
                    "priority": "high",
                    "description": f"Code quality score ({self.code_quality_score:.1f}) below threshold ({self.quality_threshold})",
                    "estimated_effort": "medium",
                    "automated": True
                })

            if hasattr(self, 'performance_score') and self.performance_score < self.performance_threshold:
                improvements.append({
                    "type": "performance",
                    "priority": "medium",
                    "description": f"Performance score ({self.performance_score:.1f}) below threshold ({self.performance_threshold})",
                    "estimated_effort": "high",
                    "automated": False
                })

            return improvements
        except Exception as e:
            return [{"type": "error", "description": str(e), "priority": "unknown"}]''',
            "get_auto_fixes": '''    def get_auto_fixes(self) -> List[Dict[str, Any]]:
        """Obtiene correcciones automáticas aplicadas"""
        try:
            fixes = getattr(self, 'applied_fixes', [])

            return [{
                "fix_id": f"fix_{i}",
                "type": fix.get("type", "unknown"),
                "description": fix.get("description", ""),
                "applied_at": fix.get("timestamp", datetime.now().isoformat()),
                "success": fix.get("success", True)
            } for i, fix in enumerate(fixes)]

        except Exception as e:
            return [{"type": "error", "description": str(e), "success": False}]''',
        },
    }

    # Procesar cada agente
    processed_count = 0
    for agent_file, methods in methods_map.items():
        agent_path = agents_dir / agent_file
        if agent_path.exists():
            if add_methods_to_agent(str(agent_path), methods):
                processed_count += 1

    print(f"\n✅ Procesamiento completado. {processed_count} agentes actualizados.")


if __name__ == "__main__":
    main()
