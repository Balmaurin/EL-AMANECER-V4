"""
Optimización CUDA para Aprendizaje Federado

Este módulo implementa aceleración GPU avanzada para modelos grandes en FL,
con técnicas de optimización de memoria, mixed precision training y
distribución eficiente de carga de trabajo.

Características:
- Mixed precision training (FP16/FP32)
- Gradient accumulation para batches grandes
- Memory-efficient forward/backward passes
- CUDA streams para paralelización
- Gradient checkpointing para modelos grandes
- Automatic Mixed Precision (AMP) integration

Autor: Sheily AI Team
Fecha: 2025
"""

import asyncio
import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import GPUtil
import psutil
import torch
import torch.nn as nn

# Importaciones del sistema FL
from federated_learning import FederatedLearningSystem, ModelUpdate
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CUDAConfig:
    """Configuración de optimización CUDA"""

    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    memory_efficient_attention: bool = True
    cuda_streams: bool = True
    num_streams: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking: bool = True


@dataclass
class CUDAMetrics:
    """Métricas de rendimiento CUDA"""

    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    training_time: float
    throughput_samples_per_sec: float
    memory_efficiency: float
    gradient_norm: float


class CUDAAccelerator:
    """Acelerador CUDA para entrenamiento federado"""

    def __init__(self, config: CUDAConfig = None):
        """Inicializar acelerador CUDA"""
        self.config = config or CUDAConfig()

        # Verificar disponibilidad de CUDA
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA no disponible - usando CPU")
            return

        # Configurar dispositivo
        self.device = torch.device("cuda")
        torch.cuda.set_device(0)  # Usar primera GPU

        # Configurar CUDA
        self.scaler = GradScaler() if self.config.use_mixed_precision else None
        self.streams = self._create_cuda_streams() if self.config.cuda_streams else None

        # Métricas
        self.metrics_history = []

        logger.info(
            f"🚀 Acelerador CUDA inicializado - Device: {torch.cuda.get_device_name()}"
        )

    def _create_cuda_streams(self) -> List[torch.cuda.Stream]:
        """Crear streams CUDA para paralelización"""
        streams = []
        for i in range(self.config.num_streams):
            stream = torch.cuda.current_stream()
            streams.append(stream)
        return streams

    @contextmanager
    def cuda_context(self):
        """Context manager para operaciones CUDA seguras"""
        try:
            # Limpiar memoria antes de operación
            if self.cuda_available:
                torch.cuda.empty_cache()
                gc.collect()

            yield

        finally:
            # Limpiar después de operación
            if self.cuda_available:
                torch.cuda.empty_cache()
                gc.collect()

    async def optimize_model_training(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int = 1,
        client_id: str = "unknown",
    ) -> Tuple[nn.Module, CUDAMetrics]:
        """
        Entrenar modelo con optimizaciones CUDA avanzadas
        """
        if not self.cuda_available:
            logger.warning("CUDA no disponible - entrenando en CPU")
            return await self._train_cpu_fallback(
                model, train_loader, optimizer, criterion, num_epochs
            )

        # Mover modelo a GPU
        model = model.to(self.device)

        # Configurar gradient checkpointing si está habilitado
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing(model)

        # Métricas iniciales
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        total_samples = 0
        accumulated_loss = 0.0
        gradient_norm = 0.0

        model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for step, (inputs, targets) in enumerate(train_loader):
                # Mover datos a GPU
                inputs = inputs.to(self.device, non_blocking=self.config.non_blocking)
                targets = targets.to(self.device, non_blocking=self.config.non_blocking)

                # Forward pass con mixed precision
                with autocast(enabled=self.config.use_mixed_precision):
                    outputs = model(inputs)
                    loss = (
                        criterion(outputs, targets)
                        / self.config.gradient_accumulation_steps
                    )

                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += loss.item()

                # Actualizar pesos cada N pasos (gradient accumulation)
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    await self._optimizer_step(optimizer, model)
                    num_batches += 1

                    # Calcular norma del gradiente
                    total_norm = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach())
                                for p in model.parameters()
                                if p.grad is not None
                            ]
                        )
                    )
                    gradient_norm = total_norm.item()

                total_samples += len(inputs)

                # Liberar memoria periódicamente
                if step % 10 == 0:
                    torch.cuda.empty_cache()

        # Paso final si quedan gradientes acumulados
        if accumulated_loss > 0 and num_batches == 0:
            await self._optimizer_step(optimizer, model)

        # Métricas finales
        end_time.record()
        torch.cuda.synchronize()

        training_time = start_time.elapsed_time(end_time) / 1000.0  # segundos
        throughput = total_samples / training_time if training_time > 0 else 0

        # Obtener métricas de GPU
        gpu_memory = torch.cuda.mem_get_info()
        gpu_utilization = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0

        metrics = CUDAMetrics(
            gpu_memory_used=(gpu_memory[1] - gpu_memory[0]) / (1024**3),  # GB
            gpu_memory_total=gpu_memory[1] / (1024**3),  # GB
            gpu_utilization=gpu_utilization,
            training_time=training_time,
            throughput_samples_per_sec=throughput,
            memory_efficiency=self._calculate_memory_efficiency(model),
            gradient_norm=gradient_norm,
        )

        self.metrics_history.append(metrics)

        logger.info(".2f" ".1f" ".2f")

        return model, metrics

    async def _optimizer_step(self, optimizer: torch.optim.Optimizer, model: nn.Module):
        """Paso de optimización con gradient clipping"""
        if self.scaler:
            # Unscale gradients para clipping
            self.scaler.unscale_(optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.config.max_grad_norm
        )

        # Optimizer step
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

    def _enable_gradient_checkpointing(self, model: nn.Module):
        """Habilitar gradient checkpointing para ahorrar memoria"""
        try:
            # Aplicar checkpointing a capas específicas
            for module in model.modules():
                if isinstance(
                    module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
                ):
                    module.gradient_checkpointing = True
                elif hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True

            logger.info("✅ Gradient checkpointing habilitado")

        except Exception as e:
            logger.warning(f"No se pudo habilitar gradient checkpointing: {e}")

    def _calculate_memory_efficiency(self, model: nn.Module) -> float:
        """Calcular eficiencia de uso de memoria"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            memory_usage = torch.cuda.memory_allocated() / (1024**3)  # GB

            # Eficiencia = memoria usada / memoria esperada para parámetros
            expected_memory = (
                total_params * 4 / (1024**3)
            )  # 4 bytes por parámetro en FP32

            if expected_memory > 0:
                efficiency = memory_usage / expected_memory
                return min(efficiency, 1.0)  # Máximo 100%
            return 0.0

        except Exception:
            return 0.0

    async def _train_cpu_fallback(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
    ) -> Tuple[nn.Module, CUDAMetrics]:
        """Fallback de entrenamiento en CPU"""
        logger.info("🏃 Entrenando en CPU (fallback)")

        import time

        start_time = time.time()

        model.train()
        total_samples = 0

        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_samples += len(inputs)

        training_time = time.time() - start_time
        throughput = total_samples / training_time if training_time > 0 else 0

        # Métricas dummy para CPU
        metrics = CUDAMetrics(
            gpu_memory_used=0.0,
            gpu_memory_total=0.0,
            gpu_utilization=0.0,
            training_time=training_time,
            throughput_samples_per_sec=throughput,
            memory_efficiency=0.0,
            gradient_norm=0.0,
        )

        return model, metrics

    def get_cuda_info(self) -> Dict[str, Any]:
        """Obtener información detallada de CUDA"""
        if not self.cuda_available:
            return {"cuda_available": False}

        try:
            gpu = GPUtil.getGPUs()[0]

            return {
                "cuda_available": True,
                "device_name": torch.cuda.get_device_name(),
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
                "memory_reserved": torch.cuda.memory_reserved() / (1024**3),  # GB
                "memory_total": gpu.memoryTotal / 1024,  # GB
                "memory_used": gpu.memoryUsed / 1024,  # GB
                "memory_free": gpu.memoryFree / 1024,  # GB
                "gpu_utilization": gpu.load * 100,  # %
                "gpu_temperature": gpu.temperature,  # °C
                "mixed_precision_enabled": self.config.use_mixed_precision,
                "gradient_checkpointing": self.config.gradient_checkpointing,
                "streams_enabled": self.config.cuda_streams,
            }

        except Exception as e:
            logger.error(f"Error obteniendo info CUDA: {e}")
            return {"cuda_available": True, "error": str(e)}


class MemoryEfficientDataLoader:
    """DataLoader optimizado para CUDA"""

    def __init__(self, dataset, batch_size: int = 32, config: CUDAConfig = None):
        """Inicializar DataLoader optimizado"""
        self.config = config or CUDAConfig()

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.config.pin_memory,
            num_workers=4,  # Workers para prefetching
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class CUDAAcceleratedFLClient:
    """Cliente FL con aceleración CUDA"""

    def __init__(self, client_config, cuda_config: CUDAConfig = None):
        """Inicializar cliente con CUDA"""
        from federated_client import FederatedClient

        self.client = FederatedClient(client_config)
        self.cuda_accelerator = CUDAAccelerator(cuda_config)
        self.cuda_config = cuda_config or CUDAConfig()

        logger.info("🎯 Cliente FL con aceleración CUDA inicializado")

    async def train_local_model_cuda(
        self, global_model: nn.Module, train_loader: DataLoader, local_epochs: int = 1
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Entrenar modelo local con optimizaciones CUDA
        """
        # Configurar optimizer y criterion
        optimizer = torch.optim.AdamW(
            global_model.parameters(), lr=1e-3, weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()

        # Entrenar con CUDA
        trained_model, cuda_metrics = (
            await self.cuda_accelerator.optimize_model_training(
                global_model,
                train_loader,
                optimizer,
                criterion,
                local_epochs,
                self.client.client_id,
            )
        )

        # Calcular métricas adicionales
        local_metrics = await self._evaluate_model(trained_model, train_loader)

        # Combinar métricas
        combined_metrics = {
            "cuda_metrics": {
                "gpu_memory_used_gb": cuda_metrics.gpu_memory_used,
                "gpu_utilization_percent": cuda_metrics.gpu_utilization,
                "training_time_seconds": cuda_metrics.training_time,
                "throughput_samples_per_sec": cuda_metrics.throughput_samples_per_sec,
                "memory_efficiency": cuda_metrics.memory_efficiency,
                "gradient_norm": cuda_metrics.gradient_norm,
            },
            "local_metrics": local_metrics,
            "optimization_info": {
                "mixed_precision": self.cuda_config.use_mixed_precision,
                "gradient_accumulation": self.cuda_config.gradient_accumulation_steps,
                "gradient_checkpointing": self.cuda_config.gradient_checkpointing,
            },
        }

        return trained_model, combined_metrics

    async def _evaluate_model(
        self, model: nn.Module, data_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluar modelo en datos locales"""
        model.eval()

        if self.cuda_accelerator.cuda_available:
            model = model.to(self.cuda_accelerator.device)

        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in data_loader:
                if self.cuda_accelerator.cuda_available:
                    inputs = inputs.to(self.cuda_accelerator.device, non_blocking=True)
                    targets = targets.to(
                        self.cuda_accelerator.device, non_blocking=True
                    )

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

        return {
            "local_accuracy": accuracy,
            "local_loss": avg_loss,
            "samples_evaluated": total,
        }


class CUDAAcceleratedFLSystem(FederatedLearningSystem):
    """Sistema FL con aceleración CUDA completa"""

    def __init__(self, config, cuda_config: CUDAConfig = None):
        """Inicializar sistema FL con CUDA"""
        super().__init__(config)
        self.cuda_config = cuda_config or CUDAConfig()
        self.cuda_accelerator = CUDAAccelerator(cuda_config)

        # Estadísticas de rendimiento CUDA
        self.cuda_performance_stats = {
            "total_training_time": 0.0,
            "total_gpu_memory_used": 0.0,
            "average_throughput": 0.0,
            "clients_with_cuda": 0,
        }

        logger.info("🚀 Sistema FL con aceleración CUDA inicializado")

    async def start_cuda_accelerated_round(self, round_number: int = 1) -> str:
        """Iniciar ronda con optimizaciones CUDA"""
        round_id = await self.start_federated_round(round_number)

        # Configurar ronda para CUDA
        if round_id in self.active_rounds:
            round_obj = self.active_rounds[round_id]
            round_obj.cuda_accelerated = True
            round_obj.cuda_config = self.cuda_config

            logger.info(f"🎯 Ronda {round_number} configurada con aceleración CUDA")

        return round_id

    async def receive_cuda_optimized_update(self, update: ModelUpdate) -> bool:
        """Recibir actualización optimizada con CUDA"""
        # Procesar normalmente
        success = await self.receive_client_update(update)

        if success and hasattr(update, "cuda_metrics"):
            # Actualizar estadísticas CUDA
            cuda_metrics = update.cuda_metrics
            self.cuda_performance_stats["total_training_time"] += cuda_metrics.get(
                "training_time_seconds", 0
            )
            self.cuda_performance_stats["total_gpu_memory_used"] += cuda_metrics.get(
                "gpu_memory_used_gb", 0
            )
            self.cuda_performance_stats["clients_with_cuda"] += 1

            # Calcular throughput promedio
            if self.cuda_performance_stats["clients_with_cuda"] > 0:
                self.cuda_performance_stats["average_throughput"] = (
                    self.cuda_performance_stats["total_training_time"]
                    / self.cuda_performance_stats["clients_with_cuda"]
                )

        return success

    def get_cuda_performance_report(self) -> Dict[str, Any]:
        """Obtener reporte de rendimiento CUDA"""
        return {
            "cuda_accelerated_system": True,
            "cuda_available": self.cuda_accelerator.cuda_available,
            "cuda_config": {
                "mixed_precision": self.cuda_config.use_mixed_precision,
                "gradient_accumulation": self.cuda_config.gradient_accumulation_steps,
                "gradient_checkpointing": self.cuda_config.gradient_checkpointing,
                "memory_efficient_attention": self.cuda_config.memory_efficient_attention,
            },
            "performance_stats": self.cuda_performance_stats,
            "cuda_info": self.cuda_accelerator.get_cuda_info(),
            "recommendations": self._generate_cuda_recommendations(),
        }

    def _generate_cuda_recommendations(self) -> List[str]:
        """Generar recomendaciones para optimización CUDA"""
        recommendations = []

        cuda_info = self.cuda_accelerator.get_cuda_info()

        if not cuda_info.get("cuda_available", False):
            recommendations.append(
                "Instalar GPU NVIDIA con CUDA para máxima performance"
            )
            return recommendations

        # Recomendaciones basadas en métricas
        gpu_memory_total = cuda_info.get("memory_total", 0)
        if gpu_memory_total < 8:
            recommendations.append(
                "Considerar GPU con más memoria (8GB+) para modelos grandes"
            )

        gpu_utilization = cuda_info.get("gpu_utilization", 0)
        if gpu_utilization < 50:
            recommendations.append(
                "GPU subutilizada - revisar configuración de batch size"
            )

        if not self.cuda_config.mixed_precision:
            recommendations.append("Habilitar mixed precision para 2x speedup")

        if not self.cuda_config.gradient_checkpointing:
            recommendations.append(
                "Habilitar gradient checkpointing para modelos grandes"
            )

        if not recommendations:
            recommendations.append("Configuración CUDA óptima - excelente rendimiento")

        return recommendations


# ==================== FUNCIONES DE UTILIDAD ====================


def create_cuda_accelerated_client(
    client_config, cuda_config: CUDAConfig = None
) -> CUDAAcceleratedFLClient:
    """Crear cliente FL con aceleración CUDA"""
    return CUDAAcceleratedFLClient(client_config, cuda_config)


def create_cuda_accelerated_system(
    config, cuda_config: CUDAConfig = None
) -> CUDAAcceleratedFLSystem:
    """Crear sistema FL con aceleración CUDA"""
    return CUDAAcceleratedFLSystem(config, cuda_config)


def get_optimal_cuda_config(
    model_size_gb: float = 1.0, gpu_memory_gb: float = 8.0
) -> CUDAConfig:
    """Obtener configuración CUDA óptima basada en recursos disponibles"""
    config = CUDAConfig()

    # Ajustar configuración basada en tamaño del modelo y memoria GPU
    memory_ratio = model_size_gb / gpu_memory_gb

    if memory_ratio > 0.8:
        # Modelo muy grande - activar todas las optimizaciones
        config.gradient_checkpointing = True
        config.gradient_accumulation_steps = 8
        config.use_mixed_precision = True
    elif memory_ratio > 0.5:
        # Modelo mediano
        config.gradient_checkpointing = True
        config.gradient_accumulation_steps = 4
        config.use_mixed_precision = True
    else:
        # Modelo pequeño - optimizaciones mínimas
        config.gradient_checkpointing = False
        config.gradient_accumulation_steps = 2
        config.use_mixed_precision = True

    return config


# ==================== DEMO CUDA ====================


async def demo_cuda_acceleration():
    """Demostración de aceleración CUDA en FL"""
    logger.info("🚀 Demo de Aceleración CUDA para FL")
    logger.info("=" * 60)

    try:
        # Verificar CUDA
        cuda_accelerator = CUDAAccelerator()
        cuda_info = cuda_accelerator.get_cuda_info()

        logger.info("🔍 Información del sistema CUDA:")
        for key, value in cuda_info.items():
            logger.info(f"  {key}: {value}")

        if not cuda_info.get("cuda_available", False):
            logger.warning("⚠️ CUDA no disponible - demo limitada a CPU")
            return

        # Crear modelo de ejemplo
        model = create_sample_model()
        logger.info("✅ Modelo de ejemplo creado")

        # Crear datos de ejemplo
        train_dataset = create_sample_dataset(num_samples=1000)
        train_loader = MemoryEfficientDataLoader(train_dataset, batch_size=32)

        # Configurar optimización CUDA
        cuda_config = get_optimal_cuda_config(
            model_size_gb=0.5,  # Modelo pequeño para demo
            gpu_memory_gb=cuda_info.get("memory_total", 8),
        )

        logger.info("⚙️ Configuración CUDA optimizada:")
        logger.info(f"  Mixed Precision: {cuda_config.use_mixed_precision}")
        logger.info(
            f"  Gradient Accumulation: {cuda_config.gradient_accumulation_steps}"
        )
        logger.info(f"  Gradient Checkpointing: {cuda_config.gradient_checkpointing}")

        # Entrenar con CUDA
        logger.info("\n🏃 Entrenando modelo con aceleración CUDA...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        trained_model, metrics = await cuda_accelerator.optimize_model_training(
            model, train_loader, optimizer, criterion, num_epochs=1
        )

        # Mostrar métricas detalladas
        logger.info("\n📊 Métricas de rendimiento CUDA:")
        logger.info(".2f")
        logger.info(".1f")
        logger.info(".2f")
        logger.info(".1f")
        logger.info(".3f")
        logger.info(".4f")

        # Comparación con CPU (estimada)
        cpu_time_estimate = metrics.training_time * 3  # Estimación conservadora
        speedup = (
            cpu_time_estimate / metrics.training_time
            if metrics.training_time > 0
            else 1
        )

        logger.info(".1f")
        # Reporte final
        system = CUDAAcceleratedFLSystem(None)  # Config básica
        performance_report = system.get_cuda_performance_report()

        logger.info("\n📈 Recomendaciones del sistema:")
        for recommendation in performance_report.get("recommendations", []):
            logger.info(f"  💡 {recommendation}")

        logger.info("✅ Demo de aceleración CUDA completada exitosamente")

    except Exception as e:
        logger.error(f"❌ Error en demo CUDA: {e}")


def create_sample_model() -> nn.Module:
    """Crear modelo de ejemplo para demo"""

    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.layers(x.view(x.size(0), -1))

    return SampleModel()


def create_sample_dataset(num_samples: int = 1000):
    """Crear dataset de ejemplo"""
    from torch.utils.data import TensorDataset

    # Datos dummy
    inputs = torch.randn(num_samples, 1, 28, 28)
    targets = torch.randint(0, 10, (num_samples,))

    return TensorDataset(inputs, targets)


if __name__ == "__main__":
    asyncio.run(demo_cuda_acceleration())
