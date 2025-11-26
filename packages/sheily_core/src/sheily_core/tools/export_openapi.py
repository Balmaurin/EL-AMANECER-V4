# Export OpenAPI spec from the FastAPI app
import json
import logging
import pathlib

from services.sheily_api.app import app

logger = logging.getLogger(__name__)


spec = app.openapi()
p = pathlib.Path(__file__).resolve().parents[1] / "openapi.json"
p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
logger.info("Wrote OpenAPI spec to", p)
