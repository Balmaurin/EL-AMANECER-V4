# 📡 DOCUMENTACIÓN DE API REST - EL-AMANECERV3

## 🌐 URL Base

```
Desarrollo: http://localhost:8080
Producción: https://api.elamanecerv3.com
```

**Prefijo de API:** `/api/v1`

---

## 🔐 Autenticación

Todos los endpoints (excepto `/auth/register` y `/auth/login`) requieren un token JWT en el header:

```http
Authorization: Bearer <jwt_token>
```

---

## 📑 Tabla de Contenidos

1. [Autenticación](#-autenticación)
2. [Chat e IA](#-chat-e-ia)
3. [Datasets y Entrenamiento](#-datasets-y-entrenamiento)
4. [Usuarios](#-usuarios)
5. [Sistema](#-sistema)
6. [Uploads](#-uploads)
7. [Vault (Caja Fuerte)](#-vault-caja-fuerte)
8. [WebSocket](#-websocket)

---

## 🔑 Autenticación

### Registrar Usuario

```http
POST /api/v1/auth/register
```

**Body:**
```json
{
  "email": "usuario@example.com",
  "password": "password_seguro123",
  "username": "usuario123"
}
```

**Response 201:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "uuid-1234",
    "email": "usuario@example.com",
    "username": "usuario123"
  }
}
```

---

### Iniciar Sesión

```http
POST /api/v1/auth/login
```

**Body:**
```json
{
  "email": "usuario@example.com",
  "password": "password_seguro123"
}
```

**Response 200:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

### Refrescar Token

```http
POST /api/v1/auth/refresh
```

**Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

---

### Obtener Perfil Actual

```http
GET /api/v1/auth/me
```

**Headers:**
```
Authorization: Bearer <token>
```

**Response 200:**
```json
{
  "id": "uuid-1234",
  "email": "usuario@example.com",
  "username": "usuario123",
  "created_at": "2025-01-01T00:00:00Z",
  "sheilys_balance": 1000,
  "level": 5,
  "xp": 2500
}
```

---

## 💬 Chat e IA

### Enviar Mensaje

```http
POST /api/v1/chat
```

**Body:**
```json
{
  "message": "¿Qué es la conciencia artificial?",
  "conversation_id": "uuid-optional",
  "use_rag": true,
  "model": "llama-3.2-3b"
}
```

**Response 200:**
```json
{
  "response": "La conciencia artificial es...",
  "conversation_id": "uuid-5678",
  "sources": [
    {
      "title": "Meta-Cognition System",
      "relevance": 0.95,
      "chunk": "El sistema de meta-cognición permite..."
    }
  ],
  "tokens_used": 450,
  "sheilys_earned": 5
}
```

---

### Listar Conversaciones

```http
GET /api/v1/chat/conversations
```

**Query Params:**
- `limit` (int, default: 20)
- `offset` (int, default: 0)

**Response 200:**
```json
{
  "conversations": [
    {
      "id": "uuid-5678",
      "title": "Conversación sobre IA",
      "last_message": "La conciencia artificial es...",
      "created_at": "2025-01-15T10:30:00Z",
      "message_count": 15
    }
  ],
  "total": 50
}
```

---

### Obtener Conversación

```http
GET /api/v1/chat/conversation/{conversation_id}
```

**Response 200:**
```json
{
  "id": "uuid-5678",
  "title": "Conversación sobre IA",
  "messages": [
    {
      "role": "user",
      "content": "¿Qué es la conciencia artificial?",
      "timestamp": "2025-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "La conciencia artificial es...",
      "timestamp": "2025-01-15T10:30:05Z"
    }
  ]
}
```

---

### Eliminar Conversación

```http
DELETE /api/v1/chat/conversation/{conversation_id}
```

**Response 204:** No Content

---

### Buscar con RAG

```http
POST /api/v1/chat/rag/search
```

**Body:**
```json
{
  "query": "blockchain SHEILYS",
  "top_k": 5,
  "min_similarity": 0.7
}
```

**Response 200:**
```json
{
  "results": [
    {
      "content": "SHEILYS Blockchain Core...",
      "source": "packages/blockchain/sheilys_blockchain.py",
      "similarity": 0.92,
      "metadata": {
        "type": "code",
        "language": "python"
      }
    }
  ]
}
```

---

### Ejecutar Agente Especializado

```http
POST /api/v1/chat/agents/{agent_id}/execute
```

**Agentes Disponibles:**
- `finance` - Análisis financiero
- `quantitative` - Trading cuantitativo
- `research` - Investigación científica

**Body:**
```json
{
  "task": "risk_assessment",
  "parameters": {
    "portfolio": {
      "AAPL": 0.4,
      "GOOGL": 0.3,
      "MSFT": 0.3
    },
    "confidence_level": 0.95
  }
}
```

**Response 200:**
```json
{
  "agent_id": "finance",
  "task_id": "task-uuid-9012",
  "status": "completed",
  "result": {
    "var_95": 0.052,
    "sharpe_ratio": 1.45,
    "recommendation": "HOLD",
    "risk_level": "MEDIUM"
  },
  "execution_time_ms": 1200
}
```

---

## 📊 Datasets y Entrenamiento

### Crear Dataset

```http
POST /api/v1/datasets
```

**Body (multipart/form-data):**
```
file: <archivo.jsonl>
name: "Dataset Personalizado"
description: "Conversaciones de dominio específico"
```

**Response 201:**
```json
{
  "dataset_id": "dataset-uuid-3456",
  "name": "Dataset Personalizado",
  "size_bytes": 524288,
  "num_examples": 1000,
  "status": "processing"
}
```

---

### Listar Datasets

```http
GET /api/v1/datasets
```

**Response 200:**
```json
{
  "datasets": [
    {
      "id": "dataset-uuid-3456",
      "name": "Dataset Personalizado",
      "num_examples": 1000,
      "created_at": "2025-01-15T12:00:00Z"
    }
  ]
}
```

---

### Iniciar Entrenamiento LoRA

```http
POST /api/v1/datasets/train
```

**Body:**
```json
{
  "dataset_id": "dataset-uuid-3456",
  "model_name": "llama-3.2-3b",
  "lora_config": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj"]
  },
  "training_config": {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4
  }
}
```

**Response 202:**
```json
{
  "job_id": "job-uuid-7890",
  "status": "queued",
  "estimated_time_minutes": 45,
  "message": "Entrenamiento iniciado"
}
```

---

### Ver Estado de Entrenamiento

```http
GET /api/v1/datasets/train/jobs/{job_id}
```

**Response 200:**
```json
{
  "job_id": "job-uuid-7890",
  "status": "training",
  "progress": 0.65,
  "current_epoch": 2,
  "total_epochs": 3,
  "loss": 0.342,
  "metrics": {
    "perplexity": 12.5,
    "tokens_per_second": 1200
  },
  "started_at": "2025-01-15T13:00:00Z",
  "estimated_completion": "2025-01-15T13:45:00Z"
}
```

---

### Cancelar Entrenamiento

```http
DELETE /api/v1/datasets/train/jobs/{job_id}
```

**Response 200:**
```json
{
  "message": "Entrenamiento cancelado",
  "final_checkpoint": "models/checkpoint-epoch-2"
}
```

---

## 👤 Usuarios

### Obtener Perfil

```http
GET /api/v1/users/profile
```

**Response 200:**
```json
{
  "id": "uuid-1234",
  "username": "usuario123",
  "email": "usuario@example.com",
  "sheilys_balance": 1000,
  "level": 5,
  "xp": 2500,
  "next_level_xp": 3000,
  "nfts_owned": [
    {
      "id": "nft-uuid-1111",
      "type": "course_completion",
      "title": "Introducción a IA",
      "issued_at": "2025-01-10T00:00:00Z"
    }
  ],
  "achievements": [
    "first_chat",
    "100_messages",
    "model_trained"
  ]
}
```

---

### Actualizar Perfil

```http
PUT /api/v1/users/profile
```

**Body:**
```json
{
  "username": "nuevo_nombre",
  "bio": "Desarrollador de IA",
  "avatar_url": "https://cdn.example.com/avatar.png"
}
```

---

### Obtener Balance de Tokens

```http
GET /api/v1/users/tokens
```

**Response 200:**
```json
{
  "sheilys_balance": 1000,
  "pending_rewards": 50,
  "lifetime_earned": 5000,
  "lifetime_spent": 4000
}
```

---

### Comprar Tokens

```http
POST /api/v1/users/purchase
```

**Body:**
```json
{
  "amount": 1000,
  "payment_method": "stripe",
  "payment_token": "tok_xxx"
}
```

**Response 200:**
```json
{
  "transaction_id": "tx-uuid-2222",
  "amount": 1000,
  "new_balance": 2000,
  "receipt_url": "https://receipts.example.com/tx-uuid-2222"
}
```

---

### Ver Historial de Transacciones

```http
GET /api/v1/users/transactions
```

**Query Params:**
- `limit` (int, default: 50)
- `type` (string: "all", "earned", "spent")

**Response 200:**
```json
{
  "transactions": [
    {
      "id": "tx-uuid-3333",
      "type": "earned",
      "amount": 10,
      "reason": "Chat de calidad",
      "timestamp": "2025-01-15T14:00:00Z"
    },
    {
      "id": "tx-uuid-4444",
      "type": "spent",
      "amount": -50,
      "reason": "Compra de curso",
      "timestamp": "2025-01-14T10:00:00Z"
    }
  ],
  "total": 150
}
```

---

## ⚙️ Sistema

### Obtener Estadísticas del Sistema

```http
GET /api/v1/system/stats
```

**Response 200:**
```json
{
  "uptime_seconds": 3600000,
  "cpu_usage_percent": 45.2,
  "memory_usage_mb": 8192,
  "memory_total_mb": 16384,
  "gpu_available": true,
  "gpu_usage_percent": 75.3,
  "disk_usage_gb": 120,
  "disk_total_gb": 500,
  "active_users": 42,
  "total_conversations": 15000,
  "total_tokens_minted": 1000000
}
```

---

### Verificar Estado de Servicios

```http
GET /api/v1/system/status
```

**Response 200:**
```json
{
  "overall": "healthy",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "rag_engine": "healthy",
    "llm_service": "healthy",
    "blockchain": "degraded"
  },
  "version": "3.0.0",
  "environment": "production"
}
```

---

### Obtener Información del Sistema

```http
GET /api/v1/system/info
```

**Response 200:**
```json
{
  "name": "EL-AMANECERV3",
  "version": "3.0.0",
  "python_version": "3.10.12",
  "architecture": "x86_64",
  "os": "Linux",
  "models_loaded": [
    "llama-3.2-3b-instruct",
    "sentence-transformers/all-MiniLM-L6-v2"
  ],
  "adapters_loaded": [
    "finance-v1",
    "research-v2"
  ]
}
```

---

### Ejecutar Limpieza de Mantenimiento

```http
POST /api/v1/system/maintenance/cleanup
```

**Response 200:**
```json
{
  "message": "Limpieza completada",
  "cleaned": {
    "temp_files_mb": 250,
    "old_logs_mb": 100,
    "expired_sessions": 50
  }
}
```

---

### Reiniciar Servicio

```http
POST /api/v1/system/restart/service/{service_name}
```

**Servicios Disponibles:**
- `rag_engine`
- `llm_service`
- `cache`

**Response 200:**
```json
{
  "message": "Servicio 'rag_engine' reiniciado exitosamente",
  "status": "running"
}
```

---

## 📤 Uploads

### Subir Archivos

```http
POST /api/v1/uploads/files
```

**Body (multipart/form-data):**
```
files: <archivo1.pdf>
files: <archivo2.txt>
process_rag: true
```

**Response 200:**
```json
{
  "uploads": [
    {
      "filename": "archivo1.pdf",
      "size_bytes": 1048576,
      "status": "success",
      "chunks_created": 45,
      "indexed": true
    },
    {
      "filename": "archivo2.txt",
      "size_bytes": 2048,
      "status": "success",
      "chunks_created": 3,
      "indexed": true
    }
  ]
}
```

---

### Obtener Estadísticas de Uploads

```http
GET /api/v1/uploads/stats
```

**Response 200:**
```json
{
  "total_files": 150,
  "total_size_mb": 500,
  "indexed_chunks": 5000,
  "file_types": {
    "pdf": 80,
    "txt": 50,
    "md": 20
  }
}
```

---

### Eliminar Archivo

```http
DELETE /api/v1/uploads/files/{filename}
```

**Response 204:** No Content

---

## 🔒 Vault (Caja Fuerte)

### Autenticar en Vault

```http
POST /api/v1/vault/auth
```

**Body:**
```json
{
  "password": "vault_password"
}
```

**Response 200:**
```json
{
  "vault_token": "vault-token-xxx",
  "expires_in": 3600
}
```

---

### Listar Items en Vault

```http
GET /api/v1/vault/items
```

**Headers:**
```
X-Vault-Token: vault-token-xxx
```

**Response 200:**
```json
{
  "items": [
    {
      "id": "item-uuid-5555",
      "name": "API Key OpenAI",
      "type": "secret",
      "created_at": "2025-01-10T00:00:00Z"
    }
  ]
}
```

---

### Guardar Item en Vault

```http
POST /api/v1/vault/items
```

**Body:**
```json
{
  "name": "API Key OpenA I",
  "type": "secret",
  "value": "sk-xxxxxxxxxxxx",
  "metadata": {
    "service": "openai",
    "environment": "production"
  }
}
```

**Response 201:**
```json
{
  "id": "item-uuid-5555",
  "message": "Item guardado exitosamente"
}
```

---

### Eliminar Item de Vault

```http
DELETE /api/v1/vault/items/{item_id}
```

**Response 204:** No Content

---

## 🌐 WebSocket

### Conexión WebSocket

```javascript
// Cliente JavaScript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'bearer_token_here'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Mensaje recibido:', data);
};
```

### Enviar Mensaje por WebSocket

```javascript
ws.send(JSON.stringify({
  type: 'chat',
  message: '¿Qué es la conciencia artificial?',
  conversation_id: 'uuid-optional'
}));
```

### Respuesta por WebSocket

```json
{
  "type": "chat_response",
  "message": "La conciencia artificial es...",
  "conversation_id": "uuid-5678",
  "timestamp": "2025-01-15T15:00:00Z"
}
```

---

## 📊 Códigos de Estado HTTP

| Código | Significado |
|--------|-------------|
| 200 | OK - Solicitud exitosa |
| 201 | Created - Recurso creado |
| 202 | Accepted - Solicitud aceptada (procesamiento asíncrono) |
| 204 | No Content - Éxito sin contenido de respuesta |
| 400 | Bad Request - Solicitud mal formada |
| 401 | Unauthorized - No autenticado |
| 403 | Forbidden - Sin permisos |
| 404 | Not Found - Recurso no encontrado |
| 429 | Too Many Requests - Rate limit excedido |
| 500 | Internal Server Error - Error del servidor |
| 503 | Service Unavailable - Servicio temporalmente no disponible |

---

## ⚡ Rate Limiting

- **Límite por defecto:** 100 requests/minuto por usuario
- **Límite para carga de archivos:** 10 requests/minuto
- **Límite para entrenamiento:** 5 requests/hora

**Header de respuesta:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1642363200
```

---

## 🔍 Ejemplos de Uso con cURL

### Chat Básico
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hola, ¿cómo estás?"}'
```

### Entrenar Modelo
```bash
curl -X POST http://localhost:8080/api/v1/datasets/train \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "dataset-uuid-3456",
    "model_name": "llama-3.2-3b",
    "lora_config": {"r": 16}
  }'
```

### Subir Archivo
```bash
curl -X POST http://localhost:8080/api/v1/uploads/files \
  -H "Authorization: Bearer <token>" \
  -F "files=@documento.pdf" \
  -F "process_rag=true"
```

---

## 📚 SDKs y Clientes

### Python
```python
from elamanecerv3 import Client

client = Client(api_key="your_api_key")
response = client.chat("¿Qué es la conciencia artificial?")
print(response.message)
```

### JavaScript/TypeScript
```typescript
import { ElAmanecerV3 } from '@elamanecerv3/sdk';

const client = new ElAmanecerV3({ apiKey: 'your_api_key' });
const response = await client.chat('¿Qué es la conciencia artificial?');
console.log(response.message);
```

---

## 🔗 Enlaces Útiles

- [Guía de Despliegue](DEPLOYMENT_GUIDE.md)
- [Arquitectura Visual](ARQUITECTURA_DIAGRAMA.md)
- [Documentación Completa](ANALISIS_PROYECTO_COMPLETO.md)
- [README Principal](README.md)

---

*Documentación generada automáticamente. Última actualización: 2025-01-23* 📡
