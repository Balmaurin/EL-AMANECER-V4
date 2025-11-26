"""
Service Registry Endpoints - MCP Integration
Exponer el registro de servicios al MCP Orchestrator
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from apps.backend.src.core.mcp_service_registry import (
    service_registry,
    ServiceCategory,
    ServiceStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/services/manifest")
async def get_service_manifest():
    """
    Obtener manifiesto completo de todos los servicios registrados
    Endpoint principal para que el MCP conozca todos los módulos disponibles
    """
    try:
        manifest = service_registry.export_service_manifest()
        logger.info(f"📋 Service manifest exported: {manifest['stats']['total_services']} services")
        return manifest
    except Exception as e:
        logger.error(f"❌ Error exporting service manifest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export manifest: {str(e)}")


@router.get("/services/list")
async def list_services(
    category: Optional[str] = None,
    status: Optional[str] = None
):
    """
    Listar servicios con filtros opcionales
    """
    try:
        services = service_registry.get_all_services()
        
        # Filtrar por categoría si se especifica
        if category:
            try:
                cat_enum = ServiceCategory(category)
                services = {
                    sid: svc for sid, svc in services.items()
                    if svc.category == cat_enum
                }
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        # Filtrar por estado si se especifica
        if status:
            try:
                status_enum = ServiceStatus(status)
                services = {
                    sid: svc for sid, svc in services.items()
                    if svc.status == status_enum
                }
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        return {
            "services": [
                {
                    "id": sid,
                    "name": svc.name,
                    "category": svc.category.value,
                    "status": svc.status.value,
                    "module_path": svc.module_path,
                    "capabilities": svc.capabilities,
                    "dependencies": list(svc.dependencies),
                    "health_score": svc.health_score
                }
                for sid, svc in services.items()
            ],
            "total": len(services)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error listing services: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list services: {str(e)}")


@router.get("/services/{service_id}")
async def get_service_details(service_id: str):
    """
    Obtener detalles completos de un servicio específico
    """
    try:
        service = service_registry.get_service(service_id)
        if not service:
            raise HTTPException(status_code=404, detail=f"Service not found: {service_id}")
        
        return {
            "id": service_id,
            "name": service.name,
            "category": service.category.value,
            "status": service.status.value,
            "module_path": service.module_path,
            "version": service.version,
            "description": service.description,
            "capabilities": service.capabilities,
            "dependencies": list(service.dependencies),
            "endpoints": [
                {
                    "path": ep.path,
                    "method": ep.method,
                    "description": ep.description,
                    "requires_auth": ep.requires_auth
                }
                for ep in service.endpoints
            ],
            "metadata": service.metadata,
            "health_score": service.health_score,
            "last_health_check": service.last_health_check.isoformat() if service.last_health_check else None,
            "registered_at": service.registered_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting service details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get service: {str(e)}")


@router.get("/services/stats")
async def get_registry_stats():
    """
    Obtener estadísticas del registro de servicios
    """
    try:
        stats = service_registry.get_registry_stats()
        return {
            "registry_stats": stats,
            "message": "Service Registry operational"
        }
    except Exception as e:
        logger.error(f"❌ Error getting registry stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/services/dependencies")
async def get_dependency_graph():
    """
    Obtener grafo de dependencias entre servicios
    """
    try:
        graph = service_registry.get_dependency_graph()
        return {
            "dependency_graph": graph,
            "service_count": len(graph)
        }
    except Exception as e:
        logger.error(f"❌ Error getting dependency graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dependencies: {str(e)}")


@router.post("/services/{service_id}/health-check")
async def health_check_service(service_id: str):
    """
    Ejecutar health check en un servicio específico
    """
    try:
        result = service_registry.health_check_service(service_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Service not found: {service_id}")
        
        service = service_registry.get_service(service_id)
        return {
            "service_id": service_id,
            "status": service.status.value,
            "health_score": service.health_score,
            "last_check": service.last_health_check.isoformat(),
            "healthy": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error checking service health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/services/categories")
async def list_categories():
    """
    Listar todas las categorías de servicios disponibles
    """
    return {
        "categories": [cat.value for cat in ServiceCategory],
        "descriptions": {
            "core": "Core infrastructure services",
            "api": "REST API endpoints",
            "service": "Business logic services",
            "infrastructure": "Lower-level infrastructure"
        }
    }


@router.get("/services/capabilities")
async def list_all_capabilities():
    """
    Listar todas las capabilities disponibles en el sistema
    """
    try:
        services = service_registry.get_all_services()
        all_capabilities = set()
        capability_providers = {}
        
        for service_id, service in services.items():
            for capability in service.capabilities:
                all_capabilities.add(capability)
                if capability not in capability_providers:
                    capability_providers[capability] = []
                capability_providers[capability].append({
                    "service_id": service_id,
                    "service_name": service.name
                })
        
        return {
            "total_capabilities": len(all_capabilities),
            "capabilities": sorted(list(all_capabilities)),
            "providers": capability_providers
        }
    except Exception as e:
        logger.error(f"❌ Error listing capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list capabilities: {str(e)}")


# Export router
__all__ = ["router"]
