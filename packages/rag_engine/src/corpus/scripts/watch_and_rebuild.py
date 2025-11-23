import os
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md", ".jsonl"}


def _snapshot(dir_path: Path):
    return {
        (p.name, int(p.stat().st_mtime))
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    }


def _validate_path_security(input_path: str, param_name: str) -> Path:
    """
    Valida y sanitiza rutas para evitar vulnerabilidades de path traversal.

    Args:
        input_path: Ruta a validar
        param_name: Nombre del parámetro para mensajes de error

    Returns:
        Path object sanitizado y validado

    Raises:
        ValueError: Si la ruta contiene caracteres peligrosos o path traversal
        PermissionError: Si la ruta está fuera del directorio del proyecto
    """
    if not input_path or not isinstance(input_path, str):
        raise ValueError(f"{param_name} no puede estar vacío")

    # ✅ SEGURIDAD: Resolver rutas absolutas para evitar ataques
    path_obj = Path(input_path).resolve()

    # ✅ SEGURIDAD: Detectar y prevenir path traversal
    if ".." in input_path or input_path.startswith("/"):
        raise ValueError(
            f"Ataque de path traversal detectado en {param_name}: {input_path}"
        )

    # ✅ SEGURIDAD: Solo permitir rutas dentro del proyecto
    project_root = Path(__file__).parent.parent.parent.resolve()
    if not str(path_obj).startswith(str(project_root)):
        raise PermissionError(
            f"Ruta {param_name} debe estar dentro del directorio del proyecto"
        )

    return path_obj


def watch_folder_and_rebuild(
    data_dir: str, index_dir: str, poll_interval: int = 10, rebuild_hnsw: bool = True
):
    """
    Vigila carpeta y reconstruye índice con validaciones de seguridad.
    Previene command injection mediante sanitización de rutas.
    """
    # ✅ SEGURIDAD: Validar y sanitizar todas las rutas de entrada
    try:
        data_dir_path = _validate_path_security(data_dir, "data_dir")
        index_dir_path = _validate_path_security(index_dir, "index_dir")
    except (ValueError, PermissionError) as e:
        print(f"❌ Error de seguridad: {e}")
        return

    # Crear directorio si no existe
    index_dir_path.mkdir(parents=True, exist_ok=True)

    last_state = _snapshot(data_dir_path)
    print(f"✅ Vigilando la carpeta: {data_dir_path}")
    print(f"✅ Directorio de índices: {index_dir_path}")
    print(
        f"✅ Reconstrucción HNSW: {'habilitada' if rebuild_hnsw else 'deshabilitada'}"
    )

    while True:
        current_state = _snapshot(data_dir_path)
        if current_state != last_state:
            print("🔄 Cambio detectado en la carpeta. Actualizando índice...")
            try:
                # ✅ SEGURIDAD: Usar subprocess.run en lugar de call, con cwd restringido
                project_root = Path(__file__).parent.parent.parent.resolve()

                # Primero intento incremental
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/update_index_incremental.py",
                        "--data-dir",
                        str(data_dir_path),
                        "--index-dir",
                        str(index_dir_path),
                    ],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 2:
                    print(
                        "🔄 Reconstrucción completa requerida. Ejecutando build_corpus_index..."
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/build_corpus_index.py",
                            "--data-dir",
                            str(data_dir_path),
                            "--index-dir",
                            str(index_dir_path),
                        ],
                        cwd=project_root,
                        check=True,
                    )

                # Después de actualizar FAISS, reconstruir HNSW para consistencia
                if rebuild_hnsw:
                    print("🏗️ Reconstruyendo índice HNSW...")
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/build_hnsw_index.py",
                            "--index-dir",
                            str(index_dir_path),
                        ],
                        cwd=project_root,
                        check=True,
                    )

                last_state = current_state
                print("✅ Índices actualizados. Vigilando...\n")

            except subprocess.CalledProcessError as e:
                print(f"❌ Error ejecutando comando: {e}")
                # Continuar vigilando a pesar del error
            except Exception as e:
                print(f"❌ Error inesperado: {e}")
                # Continuar vigilando

        time.sleep(poll_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vigila una carpeta y reconstruye el índice global al detectar cambios"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Carpeta con datos (pdf, txt, md, jsonl)",
    )
    parser.add_argument(
        "--index_dir", type=str, default="index/", help="Carpeta de salida para índices"
    )
    parser.add_argument(
        "--interval", type=int, default=10, help="Intervalo de revisión en segundos"
    )
    parser.add_argument(
        "--no-hnsw-rebuild",
        action="store_true",
        help="Desabilitar reconstrucción automática de HNSW",
    )
    args = parser.parse_args()
    watch_folder_and_rebuild(
        args.data_dir,
        args.index_dir,
        args.interval,
        rebuild_hnsw=not args.no_hnsw_rebuild,
    )
