"""
Marketplace API - Endpoints para el marketplace de Sheily
"""

from typing import Dict, List, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.backend.src.models.database import get_db, User, Transaction
from apps.backend.src.api.v1.routes.dependencies import get_current_user

router = APIRouter()

class Product(BaseModel):
    id: str
    name: str
    description: str
    price: float
    category: str
    image_url: str

class PurchaseRequest(BaseModel):
    product_id: str

# Hardcoded products for now
PRODUCTS = [
    Product(
        id="prod_agent_finance",
        name="Agente Financiero Premium",
        description="Acceso ilimitado al agente financiero avanzado con análisis de mercado en tiempo real.",
        price=500.0,
        category="agents",
        image_url="/assets/products/finance_agent.png"
    ),
    Product(
        id="prod_agent_security",
        name="Experto en Ciberseguridad",
        description="Auditoría de seguridad automatizada y monitoreo de amenazas 24/7.",
        price=750.0,
        category="agents",
        image_url="/assets/products/security_agent.png"
    ),
    Product(
        id="prod_dataset_medical",
        name="Dataset Médico Especializado",
        description="Dataset curado de 10,000 casos médicos anonimizados para entrenamiento.",
        price=1000.0,
        category="datasets",
        image_url="/assets/products/medical_dataset.png"
    ),
    Product(
        id="prod_analytics_pro",
        name="Analytics Pro Dashboard",
        description="Dashboard avanzado con métricas predictivas y reportes personalizados.",
        price=300.0,
        category="tools",
        image_url="/assets/products/analytics.png"
    ),
    Product(
        id="prod_fine_tuning_slot",
        name="Slot de Fine-Tuning Prioritario",
        description="Prioridad en la cola de entrenamiento para tus modelos personalizados.",
        price=200.0,
        category="services",
        image_url="/assets/products/gpu.png"
    )
]

@router.get("/products", response_model=List[Product])
async def get_products():
    """Listar productos disponibles en el marketplace"""
    return PRODUCTS

@router.post("/purchase", response_model=Dict[str, Any])
async def purchase_product(
    request: PurchaseRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Comprar un producto usando tokens SHEILY"""
    
    # Find product
    product = next((p for p in PRODUCTS if p.id == request.product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    # Check balance
    if current_user.sheily_tokens < product.price:
        raise HTTPException(
            status_code=400, 
            detail=f"Saldo insuficiente. Necesitas {product.price} SHEILY, tienes {current_user.sheily_tokens}"
        )
    
    try:
        # Deduct tokens
        current_user.sheily_tokens -= product.price
        
        # Record transaction
        transaction = Transaction(
            user_id=current_user.id,
            transaction_type="purchase",
            amount=-product.price,
            description=f"Compra de {product.name}",
            status="confirmed",
            confirmed_at=datetime.utcnow(),
            system_metadata={"product_id": product.id, "product_name": product.name}
        )
        db.add(transaction)
        
        # Grant access (simulated logic here - in real app would update permissions/inventory)
        # For now we just log it in metadata or similar
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Has comprado {product.name} exitosamente",
            "new_balance": current_user.sheily_tokens,
            "transaction_id": transaction.id
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando compra: {str(e)}"
        )
