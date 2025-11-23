#!/bin/bash
# =====================================
# SCRIPT DE CONFIGURACIÓN DEL ENTORNO
# =====================================

set -e

echo "🚀 Configurando entorno EL-AMANECERV3..."

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Detectar directorio raíz del proyecto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}📁 Directorio del proyecto: $PROJECT_ROOT${NC}"

# ====== 1. CONFIGURAR .ENV ======
echo -e "\n${YELLOW}1️⃣ Configurando variables de entorno...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Creado .env desde .env.example${NC}"
    echo -e "${YELLOW}⚠️  Edita .env y configura tus API keys${NC}"
else
    echo -e "${GREEN}✓ Archivo .env ya existe${NC}"
fi

# ====== 2. CONFIGURAR PYTHONPATH ======
echo -e "\n${YELLOW}2️⃣ Configurando PYTHONPATH...${NC}"

# Crear archivo de activación
cat > activate_project.sh << 'EOF'
#!/bin/bash
# Activar entorno del proyecto

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT/apps:$PROJECT_ROOT/packages:$PYTHONPATH"

# Cargar variables de entorno
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "✅ Entorno activado"
echo "   PYTHONPATH: $PYTHONPATH"
echo "   PROJECT_ROOT: $PROJECT_ROOT"
EOF

chmod +x activate_project.sh
echo -e "${GREEN}✓ Creado activate_project.sh${NC}"
echo -e "${YELLOW}   Usa: source ./activate_project.sh${NC}"

# Activar entorno actual
export PYTHONPATH="$PROJECT_ROOT/apps:$PROJECT_ROOT/packages:$PYTHONPATH"

# ====== 3. CREAR DIRECTORIOS ======
echo -e "\n${YELLOW}3️⃣ Creando directorios de datos...${NC}"
mkdir -p data/{uploads,cache,logs,temp}
mkdir -p data/models/{checkpoints,production}
mkdir -p data/embeddings
mkdir -p data/corpus
mkdir -p data/datasets
echo -e "${GREEN}✓ Directorios de datos creados${NC}"

# ====== 4. VERIFICAR POETRY ======
echo -e "\n${YELLOW}4️⃣ Verificando Poetry...${NC}"
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}⚠️  Poetry no instalado. Instalando...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi
echo -e "${GREEN}✓ Poetry: $(poetry --version)${NC}"

# ====== 5. INSTALAR DEPENDENCIAS PYTHON ======
echo -e "\n${YELLOW}5️⃣ Instalando dependencias Python...${NC}"
poetry config virtualenvs.in-project true
poetry install --no-root
echo -e "${GREEN}✓ Dependencias Python instaladas${NC}"

# ====== 6. VERIFICAR NODE/NPM ======
echo -e "\n${YELLOW}6️⃣ Verificando Node.js...${NC}"
if command -v node &> /dev/null; then
    echo -e "${GREEN}✓ Node.js: $(node --version)${NC}"
    echo -e "${GREEN}✓ npm: $(npm --version)${NC}"
    
    # Instalar dependencias del frontend
    if [ -f apps/frontend/package.json ]; then
        echo -e "${YELLOW}   Instalando dependencias del frontend...${NC}"
        cd apps/frontend
        npm install
        cd "$PROJECT_ROOT"
        echo -e "${GREEN}✓ Dependencias del frontend instaladas${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Node.js no encontrado (opcional para frontend)${NC}"
fi

# ====== 7. VERIFICAR DOCKER ======
echo -e "\n${YELLOW}7️⃣ Verificando Docker...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker: $(docker --version)${NC}"
    if command -v docker-compose &> /dev/null; then
        echo -e "${GREEN}✓ Docker Compose: $(docker-compose --version)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Docker no encontrado (opcional para infraestructura)${NC}"
fi

# ====== RESUMEN ======
echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ ¡ENTORNO CONFIGURADO EXITOSAMENTE!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}📋 Próximos pasos:${NC}"
echo -e "   1. Edita ${YELLOW}.env${NC} con tus configuraciones"
echo -e "   2. Activa el entorno: ${YELLOW}source ./activate_project.sh${NC}"
echo -e "   3. Inicia servicios: ${YELLOW}make docker-up${NC}"
echo -e "   4. Ejecuta tests: ${YELLOW}make test${NC}"
echo -e "   5. Inicia backend: ${YELLOW}make dev-backend${NC}"
echo -e "   6. Inicia frontend: ${YELLOW}make dev-frontend${NC}"

echo -e "\n${YELLOW}📚 Comandos útiles:${NC}"
echo -e "   ${YELLOW}make help${NC}     - Ver todos los comandos disponibles"
echo -e "   ${YELLOW}make lint${NC}     - Verificar código"
echo -e "   ${YELLOW}make format${NC}   - Formatear código"
echo -e "   ${YELLOW}make test${NC}     - Ejecutar tests"

echo ""
