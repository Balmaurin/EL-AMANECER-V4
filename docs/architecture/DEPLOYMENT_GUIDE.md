# 🚀 GUÍA DE DESPLIEGUE - EL-AMANECERV3

## 📋 Tabla de Contenidos

1. [Requisitos del Sistema](#-requisitos-del-sistema)
2. [Despliegue Local (Desarrollo)](#%EF%B8%8F-despliegue-local-desarrollo)
3. [Despliegue Docker](#-despliegue-docker)
4. [Despliegue en Producción](#-despliegue-en-producción)
5. [Configuración de Base de Datos](#-configuración-de-base-de-datos)
6. [Variables de Entorno](#-variables-de-entorno)
7. [Monitoreo y Logs](#-monitoreo-y-logs)
8. [Seguridad](#-seguridad)
9. [Backup y Recuperación](#-backup-y-recuperación)
10. [Troubleshooting](#-troubleshooting)

---

## 💻 Requisitos del Sistema

### Mínimos (Desarrollo)
```
CPU: 4 núcleos
RAM: 16 GB
Almacenamiento: 100 GB SSD
SO: Windows 10/11, Ubuntu 20.04+, macOS 12+
Python: 3.10+
```

### Recomendados (Producción)
```
CPU: 16 núcleos
RAM: 64 GB
Almacenamiento: 500 GB NVMe SSD
GPU: NVIDIA RTX 3090 o superior (24GB VRAM)
SO: Ubuntu 22.04 LTS
Python: 3.10
PostgreSQL: 15+
Redis: 7+
```

---

## 🖥️ Despliegue Local (Desarrollo)

### Paso 1: Clonar Repositorio

```bash
git clone https://github.com/yourusername/EL-AMANECERV3.git
cd EL-AMANECERV3-main
```

### Paso 2: Crear Entorno Virtual

```bash
# Windows
python -m venv .v env
.venv\Scripts\activate

# Linux/Mac
python3.10 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
# Dependencias base
pip install --upgrade pip
pip install -r requirements.txt

# Dependencias opcionales (GPU, experimental)
pip install -r requirements-gpu.txt  # Si tienes GPU NVIDIA
pip install -r requirements-experimental.txt  # Motores cuánticos, etc.
```

### Paso 4: Configurar Variables de Entorno

```bash
# Copiar plantilla
cp .env.example .env

# Editar con tu editor favorito
nano .env  # o vim, code, notepad++
```

**Configuración mínima:**
```env
# Base de Datos
DATABASE_URL=sqlite:///./data/sheily.db

# Seguridad
SECRET_KEY=tu_clave_secreta_super_segura_cambiar_esto
JWT_SECRET_KEY=otra_clave_secreta_diferente

# LLM Local
LLAMA_MODEL_PATH=./models/Llama-3.2-3B-Instruct-f16.gguf

# Opcional: APIs externas
OPENAI_API_KEY=sk-xxxx  # Si quieres usar GPT-4
```

### Paso 5: Inicializar Base de Datos

```bash
# Crear tablas
python -m config.database.migrate_db

# Cargar datos de ejemplo (opcional)
python tools/development/seed_database.py
```

### Paso 6: Descargar Modelo LLM Local

```bash
# Descargar Llama 3.2 3B (si no lo tienes)
mkdir -p models
cd models

# Opción 1: Desde Hugging Face
python -m tools.downloaders.download_llama \
  --model "Llama-3.2-3B-Instruct" \
  --format "gguf"

# Opción 2: URL directa (ejemplo)
wget https://huggingface.co/.../Llama-3.2-3B-Instruct-f16.gguf
```

### Paso 7: Iniciar Sistema

```bash
# Backend API
python start_backend.py

# En otra terminal: Frontend
python start_frontend.py

# O ambos a la vez:
python start_system.py
```

**Acceder:**
- Backend API: http://localhost:8080
- Frontend Web: http://localhost:8000
- Documentación API: http://localhost:8080/docs

---

## 🐳 Despliegue Docker

### Opción 1: Docker Compose (Recomendado)

```bash
# Construir imágenes
docker-compose build

# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: elamanecerv3
      POSTGRES_USER: sheily
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  backend:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      - DATABASE_URL=postgresql://sheily:${DB_PASSWORD}@postgres:5432/elamanecerv3
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    ports:
      - "8080:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8000:8000"
    depends_on:
      - backend

volumes:
  postgres_data:
  redis_data:
```

### Opción 2: Docker Manual

```bash
# Construir imagen
docker build -t elamanecerv3:latest .

# Ejecutar contenedor
docker run -d \
  --name elamanecerv3 \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e DATABASE_URL=sqlite:///./data/sheily.db \
  elamanecerv3:latest
```

---

## 🌐 Despliegue en Producción

### Arquitectura Recomendada

```
Internet
   │
┌──▼─────────┐
│ CloudFlare │  (CDN + WAF)
│   / Nginx  │
└──┬─────────┘
   │
┌──▼──────────────┐
│ Load Balancer   │
│ (HAProxy/Nginx) │
└──┬──────────────┘
   │
   ├─────────┬─────────┬─────────┐
   │         │         │         │
┌──▼──┐  ┌──▼──┐  ┌──▼──┐  ┌──▼──┐
│API 1│  │API 2│  │API 3│  │API 4│
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │         │         │         │
   └─────────┴─────────┴─────────┘
              │
      ┌───────┴────────┐
      │                │
   ┌──▼────┐     ┌────▼───┐
   │Postgres│     │ Redis  │
   │Cluster │     │Cluster │
   └────────┘     └────────┘
```

### Paso 1: Configurar Servidor

```bash
# Ubuntu 22.04
sudo apt update && sudo apt upgrade -y

# Instalar dependencias del sistema
sudo apt install -y \
  python3.10 python3.10-venv python3.10-dev \
  postgresql-client redis-tools \
  nginx certbot python3-certbot-nginx \
  build-essential git curl wget

# Crear usuario de aplicación
sudo useradd -m -s /bin/bash sheily
sudo su - sheily
```

### Paso 2: Clonar y Configurar

```bash
cd /home/sheily
git clone https://github.com/yourusername/EL-AMANECERV3.git app
cd app

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gunicorn uvicorn[standard]
```

### Paso 3: Configurar PostgreSQL

```bash
# Conectar a PostgreSQL
sudo -u postgres psql

# Crear base de datos y usuario
CREATE DATABASE elamanecerv3;
CREATE USER sheily WITH ENCRYPTED PASSWORD 'tu_password_seguro';
GRANT ALL PRIVILEGES ON DATABASE elamanecerv3 TO sheily;
\q

# Ejecutar migraciones
export DATABASE_URL="postgresql://sheily:tu_password@localhost:5432/elamanecerv3"
python -m config.database.migrate_db
```

### Paso 4: Configurar Supervisor (Process Manager)

```bash
sudo apt install supervisor

# Crear configuración
sudo nano /etc/supervisor/conf.d/elamanecerv3.conf
```

**Contenido:**
```ini
[program:elamanecerv3-backend]
command=/home/sheily/app/.venv/bin/gunicorn \
  apps.backend.src.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8080 \
  --access-logfile /var/log/supervisor/elamanecerv3-access.log \
  --error-logfile /var/log/supervisor/elamanecerv3-error.log
directory=/home/sheily/app
user=sheily
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
```

```bash
# Recargar supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start elamanecerv3-backend

# Ver estado
sudo supervisorctl status
```

### Paso 5: Configurar Nginx como Reverse Proxy

```bash
sudo nano /etc/nginx/sites-available/elamanecerv3
```

**Contenido:**
```nginx
# Rate Limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# Upstream Backend
upstream backend {
    least_conn;
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    # Si tienes múltiples instancias:
    # server 127.0.0.1:8081 max_fails=3 fail_timeout=30s;
    # server 127.0.0.1:8082 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.elamanecerv3.com;

    # Redirigir a HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.elamanecerv3.com;

    # Certificados SSL (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/api.elamanecerv3.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.elamanecerv3.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Logs
    access_log /var/log/nginx/elamanecerv3-access.log;
    error_log /var/log/nginx/elamanecerv3-error.log;

    # Max Upload Size
    client_max_body_size 100M;

    # API Endpoints
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_Timeout 60s;
        proxy_read_timeout 300s;
    }

    # WebSocket
    location /ws {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Documentación API
    location /docs {
        proxy_pass http://backend;
        proxy_set_header Host $host;
    }

    # Health Check
    location /health {
        access_log off;
        proxy_pass http://backend;
    }
}
```

```bash
# Habilitar sitio
sudo ln -s /etc/nginx/sites-available/elamanecerv3 /etc/nginx/sites-enabled/

# Verificar configuración
sudo nginx -t

# Recargar Nginx
sudo systemctl reload nginx
```

### Paso 6: Obtener Certificado SSL

```bash
sudo certbot --nginx -d api.elamanecerv3.com

# Renovación automática (ya configurada por Certbot)
sudo certbot renew --dry-run
```

---

## 🗄️ Configuración de Base de Datos

### PostgreSQL Optimizado para Producción

```sql
-- postgresql.conf (ajustar según tu hardware)

# Memoria
shared_buffers = 16GB
effective_cache_size = 48GB
maintenance_work_mem = 2GB
work_mem = 512MB

# Checkpoints
checkpoint_completion_target = 0.9
wal_buffers = 16MB
max_wal_size = 4GB
min_wal_size = 1GB

# Planner
random_page_cost = 1.1
effective_io_concurrency = 200

# Workers
max_worker_processes = 16
max_parallel_workers_per_gather = 4
max_parallel_workers = 16

# Logging
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d.log'
log_rotation_age = 1d
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_min_duration_statement = 1000  # Log consultas > 1s
```

### Índices Recomendados

```sql
-- Índices para mejorar rendimiento
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_timestamp ON transactions(created_at DESC);
```

### Backup Automático (pg_dump)

```bash
#!/bin/bash
# /home/sheily/scripts/backup_postgres.sh

BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="elamanecerv3"

# Crear backup
pg_dump -U sheily -h localhost $DB_NAME | gzip > "$BACKUP_DIR/backup_$DATE.sql.gz"

# Eliminar backups antiguos (> 7 días)
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete

echo "Backup completado: backup_$DATE.sql.gz"
```

**Cron:**
```bash
# Ejecutar backup diario a las 2 AM
crontab -e

0 2 * * * /home/sheily/scripts/backup_postgres.sh >> /var/log/backup.log 2>&1
```

---

## 🔐 Variables de Entorno

### Archivo `.env` Completo

```env
# ===== BÁSICO =====
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# ===== SEGURIDAD =====
SECRET_KEY=tu_clave_secreta_256_bits_cambiar_esto
JWT_SECRET_KEY=otra_clave_diferente_256_bits
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# ===== BASE DE DATOS =====
DATABASE_URL=postgresql://sheily:password@localhost:5432/elamanecerv3
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# ===== REDIS =====
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# ===== LLM LOCAL =====
LLAMA_MODEL_PATH=./models/Llama-3.2-3B-Instruct-f16.gguf
LLAMA_N_CTX=4096
LLAMA_N_GPU_LAYERS=35  # Para GPU
LLAMA_N_THREADS=8

# ===== RAG / EMBEDDINGS =====
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma_memory
FAISS_INDEX_PATH=./data/faiss_index

# ===== APIS EXTERNAS (Opcional) =====
OPENAI_API_KEY=sk-xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx

# ===== BLOCKCHAIN =====
BLOCKCHAIN_NETWORK=mainnet  # o testnet
RPC_URL=https://api.mainnet-beta.solana.com

# ===== MONITOREO =====
PROMETHEUS_PORT=9090
GRAFANA_ENABLE=true

# ===== EMAIL (Notificaciones) =====
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=notificaciones@elamanecerv3.com
SMTP_PASSWORD=app_password
SMTP_FROM=noreply@elamanecerv3.com

# ===== ALMACENAMIENTO =====
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=100

# ===== RATE LIMITING =====
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_TRAINING_PER_HOUR=5
```

---

## 📊 Monitoreo y Logs

### Prometheus + Grafana

**docker-compose.monitoring.yml:**
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123

volumes:
  prometheus_data:
  grafana_data:
```

### Logs Centralizados (ELK Stack)

```bash
# Instalar Filebeat
wget https://artifacts.elastic.co/downloads/beats/filebeat/filebeat-8.12.0-amd64.deb
sudo dpkg -i filebeat-8.12.0-amd64.deb

# Configurar para enviar a Elasticsearch
sudo nano /etc/filebeat/filebeat.yml
```

---

## 🔒 Seguridad

### Checklist de Seguridad Producción

- ✅ HTTPS habilitado (TLS 1.2+)
- ✅ Firewall configurado (UFW/iptables)
- ✅ SSH solo con keys (password disabled)
- ✅ Rate limiting habilitado
- ✅ WAF configurado (CloudFlare/ModSecurity)
- ✅ Secrets en variables de entorno (nunca en código)
- ✅ Backups automáticos configurados
- ✅ Logging habilitado
- ✅ Monitoreo activo
- ✅ Updates automáticos de seguridad

### Firewall (UFW)

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

---

## 💾 Backup y Recuperación

### Script de Backup Completo

```bash
#!/bin/bash
# backup_full.sh

BACKUP_ROOT="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 1. Backup PostgreSQL
pg_dump elamanecerv3 | gzip > "$BACKUP_ROOT/db_$DATE.sql.gz"

# 2. Backup Archivos de Usuario
tar -czf "$BACKUP_ROOT/uploads_$DATE.tar.gz" ./data/uploads

# 3. Backup Modelos/Adaptadores
tar -czf "$BACKUP_ROOT/models_$DATE.tar.gz" ./models ./data/lora_adapters

# 4. Backup Configuración
tar -czf "$BACKUP_ROOT/config_$DATE.tar.gz" ./config .env

# 5. Sincronizar a S3/Cloud
# aws s3 sync $BACKUP_ROOT s3://elamanecerv3-backups/

echo "Backup completo: $DATE"
```

---

## 🐛 Troubleshooting

### Problema: API no responde

**Diagnóstico:**
```bash
# Ver logs
sudo supervisorctl tail -f elamanecerv3-backend stderr

# Verificar proceso
sudo supervisorctl status

# Verificar puertos
sudo netstat -tulpn | grep 8080

# Verificar Nginx
sudo nginx -t
sudo systemctl status nginx
```

### Problema: GPU no detectada

**Solución:**
```bash
# Verificar drivers NVIDIA
nvidia-smi

# Verificar PyTorch con GPU
python -c "import torch; print(torch.cuda.is_available())"

# Reinstalar CUDA toolkit si es necesario
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Problema: Base de datos lenta

**Solución:**
```sql
-- Analizar consultas lentas
SELECT pid, query, state, query_start
FROM pg_stat_activity
WHERE state != 'idle'
AND query_duration > interval '5 seconds';

-- Reindexar tablas
REINDEX DATABASE elamanecerv3;

-- Vacuum
VACUUM ANALYZE;
```

---

## 📞 Soporte

Para problemas de despliegue, contactar al equipo técnico o abrir un issue en el repositorio.

---

*Guía actualizada: Noviembre 2025* 🚀
