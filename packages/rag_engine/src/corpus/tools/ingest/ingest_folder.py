import logging
import subprocess  # nosec B404
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from tools.common.paths import CONFIG_FILE

logger = logging.getLogger(__name__)

# Optional parsers
from pypdf import PdfReader

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore
try:
    import pytesseract  # type: ignore
    from PIL import Image  # pillow present in requirements
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

ALLOWED = {".pdf", ".txt", ".md"}


def _is_scanned_pdf(fp: Path, sample_pages: int = 2) -> bool:
    # Try pdfplumber heuristic
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(fp)) as pdf:
                pages = min(len(pdf.pages), sample_pages)
                for i in range(pages):
                    t = (pdf.pages[i].extract_text() or "").strip()
                    if t:
                        return False
                return True
        except Exception as e:
            logger.debug(f"pdfplumber scan check failed for {fp}: {e}")
    # Fallback pypdf
    try:
        reader = PdfReader(str(fp))
        t = (reader.pages[0].extract_text() or "").strip()
        return not bool(t)
    except Exception as e:
        logger.debug(f"pypdf scan check failed for {fp}: {e}")
        return True


def _ocrmypdf_text(fp: Path, langs: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        out_pdf = Path(td) / "ocr.pdf"
        cmd = [
            "ocrmypdf",
            "--skip-big",
            "--redo-ocr",
            "--force-ocr",
            "--language",
            langs,
            "--output-type",
            "pdf",
            str(fp),
            str(out_pdf),
        ]
        try:
            subprocess.check_call(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )  # nosec B603
            reader = PdfReader(str(out_pdf))
            return "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""


def _read_pdf(fp: Path, ocr_conf: dict) -> str:
    # Try native text first
    try:
        text = []
        reader = PdfReader(str(fp))
        for page in reader.pages:
            text.append(page.extract_text() or "")
        combined = "\n".join(text).strip()
        if combined:
            return combined
    except Exception as e:
        logger.debug(f"Native PDF extraction failed for {fp}: {e}")

    # OCR if enabled
    if not ocr_conf.get("enabled", False):
        return ""
    langs = ocr_conf.get("languages", "spa+eng")
    if ocr_conf.get("force_ocr", False):
        return _ocrmypdf_text(fp, langs)

    # Heuristic: only OCR if scanned
    if _is_scanned_pdf(fp):
        # Prefer ocrmypdf
        txt = _ocrmypdf_text(fp, langs)
        if txt:
            return txt
        # Fallback PyMuPDF + Tesseract if available
        if fitz is not None and pytesseract is not None:
            try:
                out = []
                with fitz.open(str(fp)) as doc:
                    pages = min(len(doc), int(ocr_conf.get("max_pages", 200)))
                    for i in range(pages):
                        pix = doc.load_page(i).get_pixmap(dpi=200)
                        img = Image.frombytes(
                            "RGB", [pix.width, pix.height], pix.samples
                        )
                        out.append(pytesseract.image_to_string(img, lang=langs))
                return "\n".join(out)
            except Exception as e:
                logger.debug(f"PyMuPDF+Tesseract OCR failed for {fp}: {e}")
                return ""
    return ""


def _read_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _read_file(fp: Path, ocr_conf: dict) -> str:
    suf = fp.suffix.lower()
    if suf == ".pdf":
        return _read_pdf(fp, ocr_conf)
    if suf in {".txt", ".md"}:
        return _read_text(fp)
    return ""


def ingest_folder(
    src: Path,
    dst_raw: Path,
    workers: int = 4,
    max_file_mb: int = 128,
    skip_hidden: bool = True,
):
    # Load simple OCR config from YAML (avoid strict pydantic coupling)
    try:
        cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
        ocr_conf = (cfg.get("ingest") or {}).get("ocr") or {}
    except Exception:
        ocr_conf = {}

    dst_raw.mkdir(parents=True, exist_ok=True)
    files = []
    for fp in src.rglob("*"):
        if not fp.is_file():
            continue
        if skip_hidden and any(p.startswith(".") for p in fp.relative_to(src).parts):
            continue
        if fp.suffix.lower() not in ALLOWED:
            continue
        try:
            if fp.stat().st_size > max_file_mb * 1024 * 1024:
                continue
        except Exception as e:
            logger.debug(f"Cannot stat file {fp}: {e}")
        files.append(fp)

    def process(fp: Path):
        text = _read_file(fp, ocr_conf)
        if not text.strip():
            return None
        out = dst_raw / fp.name
        out.write_text(text, encoding="utf-8")
        return {"file": str(fp), "bytes": len(text.encode("utf-8"))}

    results = []
    if workers <= 1 or len(files) < 2:
        for f in files:
            r = process(f)
            if r:
                results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process, f): f for f in files}
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    results.append(r)
    return {"files": len(files), "ingested": len(results)}
