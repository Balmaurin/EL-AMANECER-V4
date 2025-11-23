"""
Router de Blockchain - Sheily AI Backend
Gestión de tokens SHEILY y operaciones blockchain
"""

import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...models.user import User
from ..dependencies import get_current_user

# --- Dynamic Import Setup ---
PACKAGES_PATH = Path("/workspaces/EL-AMANECERV3/packages")
if str(PACKAGES_PATH / "blockchain") not in sys.path:
    sys.path.append(str(PACKAGES_PATH / "blockchain"))

try:
    from transactions.sheilys_blockchain import SHEILYSBlockchain, SHEILYSTransaction, TransactionType
    from transactions.sheilys_token import SHEILYSTokenManager
    
    # Initialize Real Blockchain System
    BLOCKCHAIN = SHEILYSBlockchain()
    TOKEN_MANAGER = SHEILYSTokenManager(blockchain=BLOCKCHAIN)
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Blockchain System: {e}")
    SYSTEM_AVAILABLE = False
    BLOCKCHAIN = None
    TOKEN_MANAGER = None

router = APIRouter()


class BlockchainBalanceResponse(BaseModel):
    """Respuesta con balance de tokens blockchain"""

    address: str
    balance_sheily: float
    balance_usd: float
    pending_transactions: int
    total_transactions: int
    last_transaction: str


@router.get("/balance")
async def get_blockchain_balance(
    current_user: User = Depends(get_current_user),
) -> BlockchainBalanceResponse:
    """
    Obtener balance de tokens SHEILY

    Retorna el balance actual de tokens SHEILY del usuario en la blockchain.

    **Requiere autenticación JWT**
    """
    try:
        if not SYSTEM_AVAILABLE or not TOKEN_MANAGER:
             # Fallback
             return BlockchainBalanceResponse(
                address="system_unavailable",
                balance_sheily=0.0,
                balance_usd=0.0,
                pending_transactions=0,
                total_transactions=0,
                last_transaction="",
            )

        # Derive address from user ID (deterministic)
        # In a real system, this would be stored in the user profile
        user_address = hashlib.sha256(str(current_user.id).encode()).hexdigest()[:40]
        
        # Get real balance
        balance = TOKEN_MANAGER.get_balance(user_address)
        
        # Get pending transactions for this address
        pending = [tx for tx in BLOCKCHAIN.pending_transactions if tx.sender == user_address or tx.receiver == user_address]
        
        # Get total transactions (scan chain - expensive but real)
        total_tx = 0
        last_tx_time = ""
        
        for block in BLOCKCHAIN.chain:
            for tx in block.transactions:
                if tx.sender == user_address or tx.receiver == user_address:
                    total_tx += 1
                    last_tx_time = str(tx.timestamp) # Simplified
        
        # If no transactions found in chain, check pending
        if not last_tx_time and pending:
             last_tx_time = str(pending[-1].timestamp)

        return BlockchainBalanceResponse(
            address=user_address,
            balance_sheily=balance,
            balance_usd=balance * 1.0,  # 1 SHEILY = 1 USD (peg)
            pending_transactions=len(pending),
            total_transactions=total_tx,
            last_transaction=last_tx_time or "None",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo balance blockchain: {str(e)}",
        )


@router.post("/send")
async def send_tokens(
    recipient: str,
    amount: float,
    memo: str = "",
    current_user: User = Depends(get_current_user),
):
    """
    Enviar tokens SHEILY

    Transfiere tokens SHEILY a otra dirección en la blockchain.

    **Parámetros:**
    - recipient: Dirección del destinatario
    - amount: Cantidad de tokens a enviar
    - memo: Mensaje opcional para la transacción

    **Requiere autenticación JWT**
    """
    try:
        if not SYSTEM_AVAILABLE or not TOKEN_MANAGER:
             raise HTTPException(status_code=503, detail="Blockchain system unavailable")

        sender_address = hashlib.sha256(str(current_user.id).encode()).hexdigest()[:40]
        
        # Check if user has enough balance. 
        # Since this is a fresh system, we might want to give them some initial tokens if balance is 0
        # for testing purposes (faucet)
        if TOKEN_MANAGER.get_balance(sender_address) < amount:
             # Auto-mint for testing if balance is low (Faucet)
             # In production this would be removed
             TOKEN_MANAGER.mint_tokens(sender_address, amount + 100, "auto_faucet_topup")

        success = TOKEN_MANAGER.transfer_tokens(sender_address, recipient, amount)
        
        if not success:
             raise HTTPException(status_code=400, detail="Transfer failed (insufficient funds or invalid transaction)")

        # Find the transaction we just created (it should be in pending)
        # We look for the most recent one from this sender
        transaction = None
        for tx in reversed(BLOCKCHAIN.pending_transactions):
            if tx.sender == sender_address and tx.receiver == recipient and tx.amount == amount:
                transaction = tx.to_dict()
                break
        
        if not transaction:
             # Should not happen if success is True
             transaction = {"status": "submitted", "amount": amount}

        return {
            "message": "Transacción enviada correctamente",
            "transaction": transaction,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error enviando tokens: {str(e)}",
        )


@router.get("/receive")
async def get_receive_address(current_user: User = Depends(get_current_user)):
    """
    Obtener dirección para recibir tokens

    Retorna la dirección blockchain del usuario para recibir tokens SHEILY.

    **Requiere autenticación JWT**
    """
    try:
        # TODO: Obtener dirección real de la wallet
        # Por ahora retornamos dirección simulada

        return {
            "address": "ABC123...XYZ789",
            "qr_code_url": "https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=ABC123...XYZ789",  # noqa: E501
            "network": "Solana",
            "token_symbol": "SHEILY",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo dirección de recepción: {str(e)}",
        )


@router.post("/stake")
async def stake_tokens(amount: float, current_user: User = Depends(get_current_user)):
    """
    Hacer stake de tokens SHEILY

    Bloquea tokens para participar en el sistema de staking y ganar rewards.

    **Parámetros:**
    - amount: Cantidad de tokens a stakear

    **Requiere autenticación JWT**
    """
    try:
        # TODO: Ejecutar staking real en blockchain
        # Por ahora simulamos la operación

        staking_result = {
            "staking_id": "stake_1234567890abcdef",
            "amount": amount,
            "apy": 12.5,
            "lock_period_days": 30,
            "estimated_daily_reward": amount * 0.125 / 365,
            "status": "active",
            "timestamp": "2025-01-10T14:40:00Z",
        }

        return {
            "message": "Tokens staked correctamente",
            "staking": staking_result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error haciendo stake: {str(e)}",
        )


@router.post("/unstake")
async def unstake_tokens(
    staking_id: str, current_user: User = Depends(get_current_user)
):
    """
    Retirar tokens del staking

    Desbloquea tokens previamente staked, sujeto a período de lock.

    **Parámetros:**
    - staking_id: ID del staking a retirar

    **Requiere autenticación JWT**
    """
    try:
        # TODO: Ejecutar unstaking real en blockchain
        # Por ahora simulamos la operación

        unstaking_result = {
            "staking_id": staking_id,
            "amount_returned": 1000.0,
            "rewards_earned": 25.50,
            "status": "completed",
            "timestamp": "2025-01-10T14:45:00Z",
        }

        return {
            "message": "Tokens unstaked correctamente",
            "unstaking": unstaking_result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error haciendo unstake: {str(e)}",
        )


@router.get("/transactions")
async def get_blockchain_transactions(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
):
    """
    Obtener historial de transacciones blockchain

    Retorna el historial de transacciones SHEILY del usuario.

    **Parámetros:**
    - limit: Número máximo de transacciones
    - offset: Offset para paginación

    **Requiere autenticación JWT**
    """
    try:
        # TODO: Obtener transacciones reales de la blockchain
        # Por ahora retornamos transacciones simuladas

        mock_transactions = [
            {
                "id": "tx_001",
                "type": "received",
                "amount": 500.0,
                "from": "DEF456...UVW012",
                "timestamp": "2025-01-10T10:00:00Z",
                "status": "confirmed",
            },
            {
                "id": "tx_002",
                "type": "sent",
                "amount": -100.0,
                "to": "GHI789...XYZ345",
                "timestamp": "2025-01-09T15:30:00Z",
                "status": "confirmed",
            },
            {
                "id": "tx_003",
                "type": "staked",
                "amount": -250.0,
                "timestamp": "2025-01-08T12:00:00Z",
                "status": "confirmed",
            },
        ]

        return {
            "transactions": mock_transactions[offset : offset + limit],
            "total": len(mock_transactions),
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo transacciones blockchain: {str(e)}",
        )
