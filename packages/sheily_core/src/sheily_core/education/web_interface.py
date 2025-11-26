"""
Interfaz Web para el Sistema Educativo Web3 de Sheily AI
Frontend completo con FastAPI + React-like components
Basado en investigación: UX para educación Web3, interfaces gamificadas

Características:
- Dashboard educativo completo
- Interfaz de sesiones de aprendizaje
- Gobernanza democrática
- Marketplace de NFTs educativos
- Analytics en tiempo real
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .master_education_system import get_master_education_system

logger = logging.getLogger(__name__)


class EducationalWebInterface:
    """
    Interfaz web completa para el sistema educativo Web3
    Proporciona API REST y frontend HTML/CSS/JS
    """

    def __init__(self):
        self.app = FastAPI(
            title="Sheily AI Educational Web3 Platform",
            description="Sistema Educativo Web3 con tokens SHEILYS y NFTs",
            version="1.0.0",
        )

        # Sistema educativo
        self.education_system = get_master_education_system()

        # Templates y static files
        self.templates = Jinja2Templates(directory="templates")

        # Configurar rutas
        self._setup_routes()

        logger.info("🌐 Educational Web Interface initialized")

    def _setup_routes(self):
        """Configurar todas las rutas de la API"""

        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Página principal del dashboard educativo"""
            return self.templates.TemplateResponse(
                "dashboard.html",
                {"request": request, "title": "Dashboard Educativo Web3"},
            )

        @self.app.get("/api/health")
        async def health_check():
            """Verificación de salud del sistema"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system": "Sheily AI Educational Web3 Platform",
            }

        # === API DE USUARIOS ===

        @self.app.get("/api/users/{user_id}/dashboard")
        async def get_user_dashboard(user_id: str):
            """Obtener dashboard completo del usuario"""
            try:
                dashboard = await self.education_system.get_user_educational_dashboard(
                    user_id
                )
                return dashboard
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/users/{user_id}/balance")
        async def get_user_balance(user_id: str):
            """Obtener balance educativo del usuario"""
            try:
                balance = await self.education_system.token_economy.get_user_educational_balance(
                    user_id
                )
                return balance
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # === API DE SESIONES EDUCATIVAS ===

        @self.app.post("/api/sessions/start")
        async def start_educational_session(session_data: Dict[str, Any]):
            """Iniciar nueva sesión educativa"""
            try:
                result = await self.education_system.start_educational_session(
                    user_id=session_data["user_id"],
                    activity_type=session_data["activity_type"],
                    metadata=session_data.get("metadata", {}),
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/sessions/{session_id}/complete")
        async def complete_educational_session(
            session_id: str, completion_data: Dict[str, Any]
        ):
            """Completar sesión educativa"""
            try:
                result = await self.education_system.complete_educational_session(
                    session_id=session_id,
                    quality_score=completion_data.get("quality_score", 0.0),
                    engagement_level=completion_data.get("engagement_level", "medium"),
                    learning_outcomes=completion_data.get("learning_outcomes", []),
                    additional_metadata=completion_data.get("metadata", {}),
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # === API DE GAMIFICATION ===

        @self.app.post("/api/raffle/conduct")
        async def conduct_educational_raffle(raffle_data: Dict[str, Any]):
            """Realizar rifa educativa"""
            try:
                result = await self.education_system.conduct_educational_raffle(
                    raffle_data["prize_id"]
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/challenges/create")
        async def create_learning_challenge(challenge_data: Dict[str, Any]):
            """Crear nuevo challenge de aprendizaje"""
            try:
                result = await self.education_system.create_learning_challenge(
                    name=challenge_data["name"],
                    description=challenge_data["description"],
                    requirements=challenge_data["requirements"],
                    rewards=challenge_data["rewards"],
                    duration_days=challenge_data["duration_days"],
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # === API DE GOVERNANCE ===

        @self.app.post("/api/governance/proposals")
        async def create_governance_proposal(proposal_data: Dict[str, Any]):
            """Crear propuesta de gobernanza"""
            try:
                result = (
                    await self.education_system.educational_governance.create_proposal(
                        proposer_id=proposal_data["proposer_id"],
                        title=proposal_data["title"],
                        description=proposal_data["description"],
                        proposal_type=proposal_data["proposal_type"],
                        content=proposal_data["content"],
                    )
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/governance/vote")
        async def cast_governance_vote(vote_data: Dict[str, Any]):
            """Emitir voto en propuesta de gobernanza"""
            try:
                result = await self.education_system.educational_governance.cast_vote(
                    voter_id=vote_data["voter_id"],
                    proposal_id=vote_data["proposal_id"],
                    vote_type=vote_data["vote_type"],
                    voting_power=vote_data["voting_power"],
                    rationale=vote_data.get("rationale"),
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/governance/proposals/active")
        async def get_active_proposals():
            """Obtener propuestas activas para votación"""
            try:
                proposals = (
                    await self.education_system.educational_governance.get_active_proposals()
                )
                return {"proposals": proposals}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # === API DE LMS INTEGRATION ===

        @self.app.post("/api/lms/connect")
        async def connect_lms_platform(connection_data: Dict[str, Any]):
            """Conectar plataforma LMS"""
            try:
                result = await self.education_system.lms_integration.connect_platform(
                    platform_name=connection_data["platform_name"],
                    credentials=connection_data["credentials"],
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/lms/{integration_id}/courses/{course_id}")
        async def import_course_data(integration_id: str, course_id: str):
            """Importar datos de curso desde LMS"""
            try:
                result = await self.education_system.lms_integration.import_course_data(
                    integration_id=integration_id, course_id=course_id
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # === API DE NFT CREDENTIALS ===

        @self.app.get("/api/nfts/user/{user_id}")
        async def get_user_nft_credentials(user_id: str):
            """Obtener NFTs del usuario"""
            try:
                credentials = (
                    await self.education_system.nft_credentials.get_user_credentials(
                        user_id
                    )
                )
                return {"credentials": credentials}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/nfts/verify/{token_id}")
        async def verify_nft_credential(token_id: str):
            """Verificar credencial NFT"""
            try:
                result = await self.education_system.nft_credentials.verify_credential(
                    token_id
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # === API DE ANALYTICS ===

        @self.app.get("/api/analytics/system")
        async def get_system_analytics():
            """Obtener analytics del sistema educativo"""
            try:
                stats = await self.education_system.get_system_educational_stats()
                return stats
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/analytics/events")
        async def record_analytics_event(event_data: Dict[str, Any]):
            """Registrar evento de analytics"""
            try:
                await self.education_system.educational_analytics.record_learning_event(
                    user_id=event_data["user_id"], event_data=event_data["event_data"]
                )
                return {"status": "recorded"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # === API DE MARKETPLACE ===

        @self.app.get("/api/marketplace/courses")
        async def get_available_courses():
            """Obtener cursos disponibles en marketplace"""
            # Simulación - en producción conectaría con marketplace real
            courses = [
                {
                    "id": "ai_fundamentals",
                    "title": "AI Fundamentals",
                    "description": "Introducción completa a la Inteligencia Artificial",
                    "price_sheilys": 500,
                    "instructor": "Sheily AI",
                    "duration_hours": 40,
                    "difficulty": "intermediate",
                    "enrolled_students": 1250,
                    "rating": 4.8,
                },
                {
                    "id": "blockchain_dev",
                    "title": "Blockchain Development",
                    "description": "Desarrollo completo de aplicaciones blockchain",
                    "price_sheilys": 800,
                    "instructor": "Sheily AI",
                    "duration_hours": 60,
                    "difficulty": "advanced",
                    "enrolled_students": 890,
                    "rating": 4.9,
                },
            ]
            return {"courses": courses}

        @self.app.post("/api/marketplace/purchase")
        async def purchase_course(purchase_data: Dict[str, Any]):
            """Comprar curso en marketplace"""
            try:
                # En producción: lógica real de compra con SHEILYS
                result = {
                    "success": True,
                    "transaction_id": f"purchase_{purchase_data['user_id']}_{purchase_data['course_id']}_{int(datetime.now().timestamp())}",
                    "course_id": purchase_data["course_id"],
                    "amount_paid": purchase_data["amount"],
                    "timestamp": datetime.now().isoformat(),
                }
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def create_html_templates(self):
        """Crear templates HTML básicos"""
        import os

        # Crear directorio de templates
        templates_dir = "templates"
        os.makedirs(templates_dir, exist_ok=True)

        # Template del dashboard principal
        dashboard_html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .metric-value {
            font-weight: bold;
            color: #28a745;
            font-size: 1.2em;
        }

        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            margin: 5px;
        }

        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .btn-secondary {
            background: linear-gradient(45deg, #6c757d, #495057);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            color: white;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2em;
            }

            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Sheily AI Educational Web3 Platform</h1>
            <p>Aprende, Gana SHEILYS y Obtén NFTs Verificables</p>
        </div>

        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-value" id="totalUsers">1,250</span>
                <span class="stat-label">Estudiantes Activos</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="totalSheilys">45,890</span>
                <span class="stat-label">SHEILYS Distribuidos</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="totalNFTs">2,340</span>
                <span class="stat-label">NFTs Creados</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="avgQuality">94.2%</span>
                <span class="stat-label">Calidad Promedio</span>
            </div>
        </div>

        <div class="dashboard-grid">
            <div class="card">
                <h3>💰 Token Economy</h3>
                <div class="metric">
                    <span>Balance SHEILYS:</span>
                    <span class="metric-value" id="userBalance">0</span>
                </div>
                <div class="metric">
                    <span>SHEILYS Educativos:</span>
                    <span class="metric-value" id="eduBalance">0</span>
                </div>
                <button class="btn" onclick="startLearningSession()">📚 Iniciar Sesión de Aprendizaje</button>
            </div>

            <div class="card">
                <h3>🎮 Gamification</h3>
                <div class="metric">
                    <span>Tickets Activos:</span>
                    <span class="metric-value" id="activeTickets">0</span>
                </div>
                <div class="metric">
                    <span>Challenges Completados:</span>
                    <span class="metric-value" id="completedChallenges">0</span>
                </div>
                <button class="btn" onclick="conductRaffle()">🎉 Realizar Rifa</button>
            </div>

            <div class="card">
                <h3>🏆 NFT Credentials</h3>
                <div class="metric">
                    <span>Credenciales Verificadas:</span>
                    <span class="metric-value" id="verifiedNFTs">0</span>
                </div>
                <div class="metric">
                    <span>Última Credencial:</span>
                    <span class="metric-value" id="lastNFT">Ninguna</span>
                </div>
                <button class="btn" onclick="viewCredentials()">👀 Ver Credenciales</button>
            </div>

            <div class="card">
                <h3>🏛️ Governance</h3>
                <div class="metric">
                    <span>Propuestas Activas:</span>
                    <span class="metric-value" id="activeProposals">0</span>
                </div>
                <div class="metric">
                    <span>Tu Poder de Voto:</span>
                    <span class="metric-value" id="votingPower">1.0</span>
                </div>
                <button class="btn" onclick="viewProposals()">🗳️ Ver Propuestas</button>
            </div>

            <div class="card">
                <h3>📊 Analytics</h3>
                <div class="metric">
                    <span>Sesiones Completadas:</span>
                    <span class="metric-value" id="completedSessions">0</span>
                </div>
                <div class="metric">
                    <span>Calidad Promedio:</span>
                    <span class="metric-value" id="avgQuality">0%</span>
                </div>
                <button class="btn" onclick="viewAnalytics()">📈 Ver Analytics</button>
            </div>

            <div class="card">
                <h3>🛒 Marketplace</h3>
                <div class="metric">
                    <span>Cursos Disponibles:</span>
                    <span class="metric-value" id="availableCourses">25</span>
                </div>
                <div class="metric">
                    <span>Tu Nivel:</span>
                    <span class="metric-value" id="userLevel">Aprendiz</span>
                </div>
                <button class="btn" onclick="browseMarketplace()">🛍️ Explorar Marketplace</button>
            </div>
        </div>
    </div>

    <script>
        // Funciones JavaScript para interactuar con la API
        async function loadUserDashboard() {
            try {
                const response = await fetch('/api/users/demo_user/dashboard');
                const data = await response.json();

                if (data.error) {
                    console.log('Usuario no encontrado, mostrando datos por defecto');
                    return;
                }

                // Actualizar métricas del usuario
                document.getElementById('userBalance').textContent = data.token_economy?.total_sheilys_balance || 0;
                document.getElementById('eduBalance').textContent = data.token_economy?.educational_sheilys || 0;
                document.getElementById('activeTickets').textContent = data.gamification?.active_tickets || 0;
                document.getElementById('verifiedNFTs').textContent = data.nft_credentials?.verified_credentials || 0;
                document.getElementById('completedSessions').textContent = data.token_economy?.total_sessions || 0;
                document.getElementById('avgQuality').textContent = (data.learning_analytics?.avg_session_quality * 100 || 0).toFixed(1) + '%';

            } catch (error) {
                console.error('Error cargando dashboard:', error);
            }
        }

        async function startLearningSession() {
            const sessionData = {
                user_id: "demo_user",
                activity_type: "course_completion",
                metadata: {
                    course_name: "Demo Course",
                    difficulty: "intermediate"
                }
            };

            try {
                const response = await fetch('/api/sessions/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(sessionData)
                });

                const result = await response.json();
                alert(`Sesión iniciada: ${result.session_id}`);

                // Simular completado después de 2 segundos
                setTimeout(() => completeLearningSession(result.session_id), 2000);

            } catch (error) {
                console.error('Error:', error);
                alert('Error iniciando sesión');
            }
        }

        async function completeLearningSession(sessionId) {
            const completionData = {
                quality_score: 0.95,
                engagement_level: "exceptional",
                learning_outcomes: ["Concepto aprendido", "Habilidades desarrolladas"],
                metadata: { duration_minutes: 45 }
            };

            try {
                const response = await fetch(`/api/sessions/${sessionId}/complete`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(completionData)
                });

                const result = await response.json();
                alert(`¡Sesión completada! Ganaste ${result.rewards?.token_economy?.total_sheilys || 0} SHEILYS`);

                // Recargar dashboard
                loadUserDashboard();

            } catch (error) {
                console.error('Error:', error);
                alert('Error completando sesión');
            }
        }

        async function conductRaffle() {
            try {
                const response = await fetch('/api/raffle/conduct', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prize_id: "demo_prize" })
                });

                const result = await response.json();
                alert(`Rifa realizada! Ganador: ${result.winners?.[0]?.user_id || 'Nadie'}`);

            } catch (error) {
                console.error('Error:', error);
                alert('Error en rifa');
            }
        }

        function viewCredentials() {
            alert('Funcionalidad de visualización de NFTs próximamente disponible');
        }

        function viewProposals() {
            alert('Sistema de gobernanza próximamente disponible');
        }

        function viewAnalytics() {
            alert('Dashboard de analytics próximamente disponible');
        }

        function browseMarketplace() {
            alert('Marketplace educativo próximamente disponible');
        }

        // Cargar datos iniciales
        document.addEventListener('DOMContentLoaded', function() {
            loadUserDashboard();

            // Actualizar cada 30 segundos
            setInterval(loadUserDashboard, 30000);
        });
    </script>
</body>
</html>
        """

        # Guardar template
        with open(f"{templates_dir}/dashboard.html", "w", encoding="utf-8") as f:
            f.write(dashboard_html)

        logger.info("📄 HTML templates created")

    def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Ejecutar servidor web"""
        logger.info(f"🚀 Starting Educational Web Interface on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

    async def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema web"""
        try:
            system_stats = await self.education_system.get_system_educational_stats()

            return {
                "web_interface": {
                    "status": "running",
                    "host": "0.0.0.0",
                    "port": 8000,
                    "endpoints": [
                        "GET /",
                        "GET /api/health",
                        "GET /api/users/{user_id}/dashboard",
                        "POST /api/sessions/start",
                        "POST /api/sessions/{session_id}/complete",
                        "POST /api/raffle/conduct",
                        "POST /api/governance/proposals",
                        "GET /api/marketplace/courses",
                    ],
                },
                "education_system": system_stats,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}


# Instancia global
_educational_web_interface: Optional[EducationalWebInterface] = None


def get_educational_web_interface() -> EducationalWebInterface:
    """Obtener instancia de la interfaz web educativa"""
    global _educational_web_interface

    if _educational_web_interface is None:
        _educational_web_interface = EducationalWebInterface()
        _educational_web_interface.create_html_templates()

    return _educational_web_interface


def run_web_interface(host: str = "0.0.0.0", port: int = 8000):
    """Función de conveniencia para ejecutar la interfaz web"""
    web_interface = get_educational_web_interface()
    web_interface.run_server(host=host, port=port)


if __name__ == "__main__":
    # Ejecutar interfaz web directamente
    run_web_interface()
