"""``python -m web.backend``: serve the API and the built frontend."""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .config import AppConfig


def main() -> None:
    """Parse the command line and start uvicorn."""
    ap = argparse.ArgumentParser(prog="python -m web.backend")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    cfg = AppConfig.default()
    if not cfg.frontend_dist.is_dir():
        print(f"note: {cfg.frontend_dist} not found; only the API is served (build it with `npm run build`)")
    print(f"serving on http://{args.host}:{args.port}")
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
