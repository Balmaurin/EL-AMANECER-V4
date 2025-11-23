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
