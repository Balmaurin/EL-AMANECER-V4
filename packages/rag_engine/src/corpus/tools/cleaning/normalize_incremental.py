"""
Pipeline incremental para RAG - solo procesa archivos nuevos/modificados
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Set

log = logging.getLogger("rag.incremental")


class IncrementalTracker:
    """Rastrea archivos procesados para evitar reprocesamiento"""

    def __init__(self, base: Path):
        self.base = base
        self.manifest_path = base / ".manifest.json"
        self.processed: Dict[str, dict] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, dict]:
        """Carga registro de archivos procesados"""
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning(f"Error cargando manifest: {e}")
        return {}

    def _save_manifest(self):
        """Guarda registro de archivos procesados"""
        self.manifest_path.write_text(
            json.dumps(self.processed, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _file_hash(self, filepath: Path) -> str:
        """Calcula hash del contenido del archivo"""
        h = hashlib.sha256()
        h.update(filepath.read_bytes())
        return h.hexdigest()

    def needs_processing(self, filepath: Path) -> bool:
        """Verifica si un archivo necesita ser procesado"""
        key = filepath.name

        # Archivo nuevo
        if key not in self.processed:
            return True

        # Archivo modificado (diferente hash)
        current_hash = self._file_hash(filepath)
        stored_hash = self.processed[key].get("hash")

        if current_hash != stored_hash:
            log.info(f"Archivo modificado detectado: {filepath.name}")
            return True

        return False

    def mark_processed(self, filepath: Path, doc_id: str):
        """Marca archivo como procesado"""
        self.processed[filepath.name] = {
            "hash": self._file_hash(filepath),
            "doc_id": doc_id,
            "timestamp": filepath.stat().st_mtime,
        }
        self._save_manifest()

    def get_removed_files(self, current_files: Set[str]) -> Set[str]:
        """Detecta archivos que fueron borrados de raw/"""
        stored_files = set(self.processed.keys())
        return stored_files - current_files

    def remove_file(self, filename: str):
        """Elimina archivo del tracking"""
        if filename in self.processed:
            doc_id = self.processed[filename]["doc_id"]
            del self.processed[filename]
            self._save_manifest()
            return doc_id
        return None


def normalize_corpus_incremental(base: Path, min_len: int = 200):
    """
    Versión incremental de normalize_corpus - solo procesa nuevos/modificados

    Returns:
        dict: Estadísticas de procesamiento
            - docs_new: Archivos nuevos procesados
            - docs_updated: Archivos modificados reprocesados
            - docs_removed: Archivos eliminados
            - docs_skipped: Archivos sin cambios
    """
    import yaml
    from datasketch import MinHashLSH

    from tools.cleaning.normalize import _process_file

    # Cargar config
    cfg = yaml.safe_load(open("config/universal.yaml", "r", encoding="utf-8"))
    ent = cfg.get("enterprise", {})
    min_len = int(ent.get("normalize", {}).get("min_chars", min_len))
    min_quality = float(ent.get("normalize", {}).get("min_quality", 0.5))
    dedup = ent.get("normalize", {}).get("dedup", {"enabled": False})

    raw = base / "raw"
    cleaned = base / "cleaned"
    chunks = base / "chunks"
    embeddings = base / "embeddings"
    index = base / "index"

    cleaned.mkdir(parents=True, exist_ok=True)

    # Inicializar tracker
    tracker = IncrementalTracker(base)

    # LSH para deduplicación
    lsh = None
    if dedup.get("enabled", False):
        jacc = float(dedup.get("jaccard_threshold", 0.9))
        n_perm = int(dedup.get("n_perm", 128))
        lsh = MinHashLSH(threshold=jacc, num_perm=n_perm)

    stats = {"docs_new": 0, "docs_updated": 0, "docs_removed": 0, "docs_skipped": 0}

    # Procesar archivos en raw/
    current_files = set()
    for f in raw.glob("*"):
        if not f.is_file():
            continue

        current_files.add(f.name)

        # [+] SOLO procesa si es nuevo o modificado
        if tracker.needs_processing(f):
            try:
                doc_id = _hash_file(f)

                # Limpiar outputs anteriores de este doc
                _clean_doc_outputs(base, doc_id)

                # Procesar
                if _process_file(f, cleaned, min_len, min_quality, lsh=lsh):
                    tracker.mark_processed(f, doc_id)

                    if f.name in tracker.processed:
                        stats["docs_updated"] += 1
                        log.info(f"[OK] Actualizado: {f.name}")
                    else:
                        stats["docs_new"] += 1
                        log.info(f"[OK] Nuevo: {f.name}")
            except Exception as e:
                log.error(f"Error procesando {f.name}: {e}")
        else:
            stats["docs_skipped"] += 1
            log.debug(f"⊘ Sin cambios: {f.name}")

    # Detectar archivos eliminados
    removed = tracker.get_removed_files(current_files)
    for filename in removed:
        doc_id = tracker.remove_file(filename)
        if doc_id:
            _clean_doc_outputs(base, doc_id)
            stats["docs_removed"] += 1
            log.info(f"✗ Eliminado: {filename}")

    return stats


def _hash_file(filepath: Path) -> str:
    """Genera hash SHA256 del archivo"""
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()


def _clean_doc_outputs(base: Path, doc_id: str):
    """Elimina outputs generados de un documento específico"""
    # Limpiar cleaned/
    cleaned = base / "cleaned"
    for f in cleaned.glob(f"*{doc_id}*"):
        f.unlink(missing_ok=True)

    # Limpiar chunks/
    chunks = base / "chunks"
    for f in chunks.glob(f"{doc_id}*"):
        f.unlink(missing_ok=True)

    # Nota: embeddings e índices se reconstruyen después
    log.debug(f"Limpiados outputs de doc_id={doc_id[:8]}...")


# ============= USO =============
if __name__ == "__main__":
    from pathlib import Path

    # Ejemplo de uso
    base = Path("corpus/universal/demo")
    stats = normalize_corpus_incremental(base)

    print(
        f"""
    [data] Pipeline Incremental - Resultados:
    
    [+] Nuevos:        {stats['docs_new']}
    [~] Actualizados:  {stats['docs_updated']}
    ✗  Eliminados:    {stats['docs_removed']}
    ⊘  Sin cambios:   {stats['docs_skipped']}
    """
    )
