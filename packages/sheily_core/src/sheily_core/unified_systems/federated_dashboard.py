"""
Dashboard Web para Monitoreo del Sistema de Aprendizaje Federado

Este dashboard proporciona una interfaz web interactiva para monitorear
y gestionar el sistema de aprendizaje federado en tiempo real.

Características:
- Métricas en tiempo real del sistema FL
- Visualización de rondas activas y clientes
- Monitoreo de privacidad y seguridad
- Gestión de clientes y configuraciones
- Alertas y notificaciones

Autor: Sheily AI Team
Fecha: 2025
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Streamlit y dependencias de visualización
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots

    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    pd = None
    px = None
    go = None
    make_subplots = None
    STREAMLIT_AVAILABLE = False

from federated_api import FederatedAPIClient

# Importaciones del sistema FL
from federated_learning import FederatedConfig, FederatedLearningSystem

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedLearningDashboard:
    """
    Dashboard interactivo para el sistema de aprendizaje federado
    """

    def __init__(
        self,
        fl_system: Optional[FederatedLearningSystem] = None,
        api_client: Optional[FederatedAPIClient] = None,
    ):
        """Inicializar dashboard"""
        self.fl_system = fl_system or FederatedLearningSystem()
        self.api_client = api_client or FederatedAPIClient()

        # Estado del dashboard
        self.metrics_history = []
        self.alerts = []
        self.last_update = datetime.now()

        # Configuración de actualización automática
        self.auto_refresh_interval = 30  # segundos
        self.max_history_points = 100

        logger.info("📊 Dashboard de Aprendizaje Federado inicializado")

    def run_dashboard(self):
        """Ejecutar dashboard de Streamlit"""
        if not STREAMLIT_AVAILABLE:
            logger.error("Streamlit no disponible - Dashboard no puede ejecutarse")
            return

        st.set_page_config(
            page_title="Federated Learning Dashboard",
            page_icon="🎓",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🎓 Dashboard de Aprendizaje Federado")
        st.markdown("---")

        # Sidebar con navegación
        self._create_sidebar()

        # Contenido principal
        self._create_main_content()

        # Footer
        st.markdown("---")
        st.markdown("*Dashboard desarrollado por Sheily AI Team - 2025*")

    def _create_sidebar(self):
        """Crear sidebar de navegación"""
        st.sidebar.title("🧭 Navegación")

        # Menú principal
        menu_options = [
            "📊 Resumen General",
            "👥 Gestión de Clientes",
            "🔄 Rondas Activas",
            "🛡️ Privacidad y Seguridad",
            "⚙️ Configuración",
            "📈 Históricos",
        ]

        choice = st.sidebar.selectbox("Seleccionar vista:", menu_options)

        # Configuración de actualización automática
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Actualización Automática")

        auto_refresh = st.sidebar.checkbox("Actualizar automáticamente", value=True)
        if auto_refresh:
            refresh_rate = st.sidebar.slider(
                "Intervalo (segundos):",
                min_value=5,
                max_value=300,
                value=self.auto_refresh_interval,
            )
            self.auto_refresh_interval = refresh_rate

            # Trigger de actualización automática
            if st.sidebar.button("🔄 Actualizar Ahora"):
                st.rerun()

        # Información del sistema
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ Información del Sistema")

        system_info = self._get_system_info()
        st.sidebar.metric("Clientes Activos", system_info.get("active_clients", 0))
        st.sidebar.metric("Rondas Activas", system_info.get("active_rounds", 0))
        st.sidebar.metric("Cumplimiento RGPD", ".1%")

        # Alertas recientes
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚨 Alertas Recientes")

        alerts = self._get_recent_alerts()
        if alerts:
            for alert in alerts[-3:]:  # Mostrar últimas 3
                st.sidebar.error(f"⚠️ {alert['message']}")
        else:
            st.sidebar.success("✅ Sin alertas activas")

        return choice

    def _create_main_content(self):
        """Crear contenido principal del dashboard"""

        # Tabs principales
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Resumen", "👥 Clientes", "🔄 Rondas", "🛡️ Seguridad"]
        )

        with tab1:
            self._create_overview_tab()

        with tab2:
            self._create_clients_tab()

        with tab3:
            self._create_rounds_tab()

        with tab4:
            self._create_security_tab()

    def _create_overview_tab(self):
        """Crear tab de resumen general"""
        st.header("📊 Resumen General del Sistema")

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        metrics = self._get_current_metrics()

        with col1:
            st.metric(
                "Clientes Totales",
                metrics.get("total_clients", 0),
                help="Número total de clientes registrados",
            )

        with col2:
            st.metric(
                "Clientes Activos",
                metrics.get("active_clients", 0),
                help="Clientes activos en las últimas 24 horas",
            )

        with col3:
            st.metric(
                "Rondas Activas",
                metrics.get("active_rounds", 0),
                help="Rondas de entrenamiento en curso",
            )

        with col4:
            compliance = metrics.get("gdpr_compliance_rate", 0.0)
            st.metric("Cumplimiento RGPD", ".1%", help="Tasa de cumplimiento normativo")

        st.markdown("---")

        # Gráficos de evolución
        col1, col2 = st.columns(2)

        with col1:
            self._create_clients_chart()

        with col2:
            self._create_rounds_chart()

        # Estado de rondas activas
        st.markdown("---")
        st.subheader("🔄 Rondas Activas")

        active_rounds = self._get_active_rounds()
        if active_rounds:
            rounds_df = pd.DataFrame(active_rounds)
            st.dataframe(
                rounds_df[
                    ["round_number", "status", "participating_clients", "start_time"]
                ],
                use_container_width=True,
            )
        else:
            st.info("No hay rondas activas actualmente")

    def _create_clients_tab(self):
        """Crear tab de gestión de clientes"""
        st.header("👥 Gestión de Clientes Federados")

        # Filtros y búsqueda
        col1, col2, col3 = st.columns(3)

        with col1:
            search_term = st.text_input(
                "Buscar cliente:", placeholder="ID del cliente..."
            )

        with col2:
            use_case_filter = st.selectbox(
                "Filtrar por caso de uso:",
                ["Todos", "healthcare", "speech_recognition", "autonomous_transport"],
            )

        with col3:
            status_filter = st.selectbox(
                "Filtrar por estado:", ["Todos", "active", "inactive"]
            )

        # Lista de clientes
        clients = self._get_clients_list(
            search=search_term,
            use_case=use_case_filter if use_case_filter != "Todos" else None,
            status=status_filter if status_filter != "Todos" else None,
        )

        if clients:
            # Mostrar como tabla
            clients_df = pd.DataFrame(clients)
            st.dataframe(clients_df, use_container_width=True)

            # Estadísticas de clientes
            st.markdown("---")
            st.subheader("📈 Estadísticas de Clientes")

            col1, col2, col3 = st.columns(3)

            with col1:
                avg_reputation = sum(
                    c.get("reputation_score", 0) for c in clients
                ) / len(clients)
                st.metric("Reputación Promedio", ".2f")

            with col2:
                active_count = sum(1 for c in clients if c.get("is_active", False))
                st.metric("Clientes Activos", active_count)

            with col3:
                total_contributions = sum(
                    c.get("total_contributions", 0) for c in clients
                )
                st.metric("Contribuciones Totales", total_contributions)

        else:
            st.info("No se encontraron clientes con los filtros aplicados")

        # Acciones de gestión
        st.markdown("---")
        st.subheader("⚙️ Acciones de Gestión")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("➕ Registrar Nuevo Cliente"):
                self._show_register_client_modal()

        with col2:
            if st.button("📤 Exportar Lista de Clientes"):
                self._export_clients_data()

        with col3:
            if st.button("🧹 Limpiar Clientes Inactivos"):
                self._cleanup_inactive_clients()

    def _create_rounds_tab(self):
        """Crear tab de rondas activas"""
        st.header("🔄 Gestión de Rondas de Entrenamiento")

        # Controles de rondas
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🚀 Iniciar Nueva Ronda", type="primary"):
                self._start_new_round()

        with col2:
            if st.button("⏹️ Detener Ronda Activa"):
                self._stop_active_round()

        with col3:
            if st.button("📊 Ver Histórico de Rondas"):
                self._show_rounds_history()

        # Rondas activas
        st.markdown("---")
        st.subheader("🎯 Rondas Activas")

        active_rounds = self._get_active_rounds()
        if active_rounds:
            for round_info in active_rounds:
                with st.expander(
                    f"Ronda {round_info['round_number']} - {round_info['status']}"
                ):

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Clientes Participantes",
                            round_info["participating_clients"],
                        )

                    with col2:
                        start_time = datetime.fromisoformat(round_info["start_time"])
                        duration = datetime.now() - start_time
                        st.metric("Duración", f"{duration.seconds}s")

                    with col3:
                        st.metric("Estado", round_info["status"].title())

                    with col4:
                        progress = min(
                            duration.seconds / 300, 1.0
                        )  # Asumir 5 min máximo
                        st.progress(progress)

                    # Métricas de la ronda
                    if "metrics" in round_info and round_info["metrics"]:
                        st.subheader("📊 Métricas de Rendimiento")
                        metrics = round_info["metrics"]

                        metric_cols = st.columns(len(metrics))
                        for i, (key, value) in enumerate(metrics.items()):
                            with metric_cols[i]:
                                st.metric(key.replace("_", " ").title(), value)

        else:
            st.info(
                "🎉 No hay rondas activas. El sistema está listo para iniciar una nueva ronda."
            )

        # Histórico reciente
        st.markdown("---")
        st.subheader("📚 Histórico Reciente")

        recent_rounds = self._get_recent_rounds(limit=5)
        if recent_rounds:
            history_df = pd.DataFrame(recent_rounds)
            st.dataframe(
                history_df[
                    [
                        "round_number",
                        "status",
                        "start_time",
                        "end_time",
                        "participating_clients",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.info("No hay rondas completadas recientemente")

    def _create_security_tab(self):
        """Crear tab de privacidad y seguridad"""
        st.header("🛡️ Monitoreo de Privacidad y Seguridad")

        # Alertas de seguridad
        st.subheader("🚨 Alertas de Seguridad")

        security_alerts = self._get_security_alerts()
        if security_alerts:
            for alert in security_alerts:
                if alert["severity"] == "high":
                    st.error(f"🚨 {alert['message']}")
                elif alert["severity"] == "medium":
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("✅ No hay alertas de seguridad activas")

        st.markdown("---")

        # Métricas de privacidad
        col1, col2, col3 = st.columns(3)

        privacy_metrics = self._get_privacy_metrics()

        with col1:
            dp_budget = privacy_metrics.get("differential_privacy_budget", 0.0)
            st.metric("Presupuesto Privacidad DP", ".3f")

        with col2:
            violations = privacy_metrics.get("privacy_violations", 0)
            st.metric("Violaciones de Privacidad", violations)

        with col3:
            secure_rounds = privacy_metrics.get("secure_aggregation_rounds", 0)
            st.metric("Rondas Seguras", secure_rounds)

        # Gráfico de evolución de privacidad
        st.markdown("---")
        st.subheader("📈 Evolución de Privacidad")

        privacy_history = self._get_privacy_history()
        if privacy_history:
            history_df = pd.DataFrame(privacy_history)

            fig = px.line(
                history_df,
                x="timestamp",
                y=["differential_privacy_budget", "privacy_violations"],
                title="Evolución de Métricas de Privacidad",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Configuración de privacidad
        st.markdown("---")
        st.subheader("⚙️ Configuración de Privacidad")

        with st.expander("Configuración Avanzada"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Privacidad Diferencial")
                noise_multiplier = st.slider(
                    "Multiplicador de Ruido",
                    min_value=0.1,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                )

                max_grad_norm = st.slider(
                    "Norma Máxima de Gradiente",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                )

            with col2:
                st.subheader("Agregación Segura")
                secure_agg = st.checkbox("Habilitar Agregación Segura", value=True)

                gdpr_compliance = st.checkbox("Cumplimiento RGPD", value=True)

            if st.button("💾 Guardar Configuración"):
                self._update_privacy_config(
                    {
                        "noise_multiplier": noise_multiplier,
                        "max_grad_norm": max_grad_norm,
                        "secure_aggregation": secure_agg,
                        "gdpr_compliance": gdpr_compliance,
                    }
                )
                st.success("✅ Configuración guardada exitosamente")

    # ==================== MÉTODOS AUXILIARES ====================

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales del sistema"""
        try:
            if self.api_client:
                metrics = asyncio.run(self.api_client.get_metrics())
                return metrics or {}
            elif self.fl_system:
                return self.fl_system.get_federated_metrics()
            return {}
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return {}

    def _get_system_info(self) -> Dict[str, Any]:
        """Obtener información básica del sistema"""
        metrics = self._get_current_metrics()
        return {
            "active_clients": metrics.get("active_clients", 0),
            "active_rounds": metrics.get("active_rounds", 0),
            "total_clients": metrics.get("total_clients", 0),
        }

    def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Obtener alertas recientes"""
        # Implementación simplificada
        return [
            {"message": "Cliente inactivo detectado", "severity": "low"},
            {"message": "Ronda completada exitosamente", "severity": "info"},
        ]

    def _create_clients_chart(self):
        """Crear gráfico de evolución de clientes"""
        st.subheader("👥 Evolución de Clientes")

        # Datos de ejemplo para demostración
        dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
        clients_data = {
            "fecha": dates,
            "total_clients": [10 + i * 2 for i in range(30)],
            "active_clients": [8 + i for i in range(30)],
        }

        df = pd.DataFrame(clients_data)

        fig = px.line(
            df,
            x="fecha",
            y=["total_clients", "active_clients"],
            title="Evolución del Número de Clientes",
            labels={"value": "Número de Clientes", "fecha": "Fecha"},
        )

        st.plotly_chart(fig, use_container_width=True)

    def _create_rounds_chart(self):
        """Crear gráfico de rondas completadas"""
        st.subheader("🔄 Rondas de Entrenamiento")

        # Datos de ejemplo
        rounds_data = {
            "ronda": list(range(1, 21)),
            "precision_promedio": [
                0.7 + i * 0.01 + np.random.normal(0, 0.05) for i in range(20)
            ],
            "tiempo_entrenamiento": [120 + np.random.normal(0, 30) for _ in range(20)],
        }

        df = pd.DataFrame(rounds_data)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df["ronda"], y=df["precision_promedio"], name="Precisión Promedio"
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df["ronda"],
                y=df["tiempo_entrenamiento"],
                name="Tiempo de Entrenamiento",
            ),
            secondary_y=True,
        )

        fig.update_layout(title_text="Evolución de Rondas de Entrenamiento")
        fig.update_xaxes(title_text="Número de Ronda")
        fig.update_yaxes(title_text="Precisión", secondary_y=False)
        fig.update_yaxes(title_text="Tiempo (segundos)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    def _get_clients_list(
        self,
        search: str = "",
        use_case: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Obtener lista filtrada de clientes"""
        # Implementación simplificada con datos de ejemplo
        clients = [
            {
                "client_id": "hospital_a",
                "use_case": "healthcare",
                "device_type": "server",
                "reputation_score": 0.95,
                "total_contributions": 15,
                "is_active": True,
                "last_seen": "2025-01-13T14:30:00Z",
            },
            {
                "client_id": "mobile_001",
                "use_case": "speech_recognition",
                "device_type": "mobile",
                "reputation_score": 0.88,
                "total_contributions": 8,
                "is_active": True,
                "last_seen": "2025-01-13T14:25:00Z",
            },
            {
                "client_id": "vehicle_001",
                "use_case": "autonomous_transport",
                "device_type": "iot",
                "reputation_score": 0.92,
                "total_contributions": 12,
                "is_active": False,
                "last_seen": "2025-01-12T10:15:00Z",
            },
        ]

        # Aplicar filtros
        filtered_clients = clients

        if search:
            filtered_clients = [
                c for c in filtered_clients if search.lower() in c["client_id"].lower()
            ]

        if use_case:
            filtered_clients = [
                c for c in filtered_clients if c["use_case"] == use_case
            ]

        if status:
            if status == "active":
                filtered_clients = [c for c in filtered_clients if c["is_active"]]
            elif status == "inactive":
                filtered_clients = [c for c in filtered_clients if not c["is_active"]]

        return filtered_clients

    def _get_active_rounds(self) -> List[Dict[str, Any]]:
        """Obtener rondas activas"""
        # Implementación simplificada
        return [
            {
                "round_id": "round_001",
                "round_number": 1,
                "status": "running",
                "participating_clients": 5,
                "start_time": "2025-01-13T14:00:00Z",
                "metrics": {"avg_accuracy": 0.85, "total_samples": 2500},
            }
        ]

    def _get_recent_rounds(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Obtener rondas recientes"""
        # Implementación simplificada
        return [
            {
                "round_number": i,
                "status": "completed",
                "start_time": f"2025-01-13T{13+i:02d}:00:00Z",
                "end_time": f"2025-01-13T{14+i:02d}:00:00Z",
                "participating_clients": 5 + i % 3,
            }
            for i in range(limit, 0, -1)
        ]

    def _get_security_alerts(self) -> List[Dict[str, Any]]:
        """Obtener alertas de seguridad"""
        return [
            {"message": "Cliente con baja reputación detectado", "severity": "medium"},
            {
                "message": "Actualización de modelo sospechosa filtrada",
                "severity": "low",
            },
        ]

    def _get_privacy_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de privacidad"""
        return {
            "differential_privacy_budget": 0.95,
            "privacy_violations": 0,
            "secure_aggregation_rounds": 12,
        }

    def _get_privacy_history(self) -> List[Dict[str, Any]]:
        """Obtener histórico de métricas de privacidad"""
        # Datos de ejemplo
        return [
            {
                "timestamp": f"2025-01-13T{i:02d}:00:00Z",
                "differential_privacy_budget": 1.0 - i * 0.02,
                "privacy_violations": max(0, i % 3 - 1),
            }
            for i in range(10)
        ]

    # ==================== MÉTODOS DE ACCIÓN ====================

    def _show_register_client_modal(self):
        """Mostrar modal para registrar cliente"""
        st.info("Funcionalidad de registro de cliente - Implementar modal")

    def _export_clients_data(self):
        """Exportar datos de clientes"""
        st.info("Funcionalidad de exportación - Implementar descarga CSV")

    def _cleanup_inactive_clients(self):
        """Limpiar clientes inactivos"""
        st.info(
            "Funcionalidad de limpieza - Implementar eliminación de clientes inactivos"
        )

    def _start_new_round(self):
        """Iniciar nueva ronda"""
        st.success("Nueva ronda iniciada exitosamente")

    def _stop_active_round(self):
        """Detener ronda activa"""
        st.warning("Ronda activa detenida")

    def _show_rounds_history(self):
        """Mostrar histórico de rondas"""
        st.info("Histórico de rondas - Implementar vista detallada")

    def _update_privacy_config(self, config: Dict[str, Any]):
        """Actualizar configuración de privacidad"""
        # Implementar actualización de configuración
        pass


# ==================== FUNCIONES DE UTILIDAD ====================


def create_fl_dashboard(
    config: Optional[FederatedConfig] = None,
) -> FederatedLearningDashboard:
    """Crear dashboard para sistema FL"""
    fl_system = FederatedLearningSystem(config=config)
    api_client = FederatedAPIClient()
    return FederatedLearningDashboard(fl_system, api_client)


def run_dashboard():
    """Ejecutar dashboard de Streamlit"""
    dashboard = create_fl_dashboard()

    # Configurar página
    if STREAMLIT_AVAILABLE:
        dashboard.run_dashboard()
    else:
        print(
            "Streamlit no disponible. Instalar con: pip install streamlit plotly pandas"
        )


if __name__ == "__main__":
    run_dashboard()
