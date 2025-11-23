"""
Servidor usando hypercorn en lugar de uvicorn
"""

import asyncio

from hypercorn.asyncio import serve
from hypercorn.config import Config

from backend.main_api import app


async def main():
    config = Config()
    config.bind = ["0.0.0.0:8001"]

    print("=" * 60)
    print("SHEILY BACKEND - HYPERCORN")
    print("=" * 60)
    print("http://localhost:8001")
    print("http://localhost:8001/docs")
    print("")

    await serve(app, config)


if __name__ == "__main__":
    asyncio.run(main())
