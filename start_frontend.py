#!/usr/bin/env python3
"""
Servidor Frontend Simple para EL-AMANECERV3
===========================================
Sirve la interfaz web estática en el puerto 8000.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
# Ruta al directorio donde está sheily-web.html
WEB_DIR = Path("apps/frontend/public/public").absolute()

class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Si piden la raíz, servir sheily-web.html
        if self.path == "/":
            self.path = "/sheily-web.html"
        return super().do_GET()

if __name__ == "__main__":
    if not WEB_DIR.exists():
        print(f"❌ Error: No se encuentra el directorio {WEB_DIR}")
        exit(1)
        
    os.chdir(WEB_DIR)
    
    print(f"🌐 Iniciando Servidor Frontend en http://localhost:{PORT}")
    print(f"📂 Sirviendo desde: {WEB_DIR}")
    print("Presiona Ctrl+C para detener.")
    
    with socketserver.TCPServer(("", PORT), SPAHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Servidor detenido.")
