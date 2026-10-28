"""FastAPI application factory, middleware, exception handlers, and route aggregation."""

import logging
import mimetypes
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from mei.api.routes.insights import router as insights_router
from mei.api.routes.jobs import router as jobs_router
from mei.api.routes.leaderboard import router as leaderboard_router
from mei.api.routes.retrain import router as retrain_router
from mei.api.routes.runs import router as runs_router
from mei.api.routes.scoring import router as scoring_router
from mei.api.routes.status import router as status_router
from mei.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application for MEI Platform."""
    app = FastAPI(
        title="Multimodal Student Engagement Intelligence (MEI) API",
        description=(
            "Research-grade analytical API serving real model artifacts, cross-validation "
            "leaderboards, longitudinal trajectory patterns, SHAP explainability insights, "
            "and student cohort scoring. Component ID: KND_CDAP_IT_05."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS Configuration for localhost university frontend
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Privacy-Conscious Request Logging Middleware (Rule G2, Privacy)
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        start_time = time.time()
        # Strictly avoid logging request payload bytes to protect student privacy
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        return response

    # 3. Clean Structured Error Handling (No Stack Traces to Client)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": request.url.path,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error_messages = [
            f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg')}"
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": True,
                "status_code": 422,
                "detail": "; ".join(error_messages),
                "errors": exc.errors(),
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled server exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": True,
                "status_code": 500,
                "detail": f"Internal server error: {exc}",
                "path": request.url.path,
            },
        )

    # 4. Mount Routers
    app.include_router(status_router)
    app.include_router(leaderboard_router)
    app.include_router(insights_router)
    app.include_router(scoring_router)
    app.include_router(runs_router)
    app.include_router(jobs_router)
    app.include_router(retrain_router)

    # 5. Serve Built Frontend in Production (Rule 10, CLI serve)
    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        assets_dir = frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/")
        async def serve_spa_root():
            index_file = frontend_dist / "index.html"
            if index_file.is_file():
                return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
            raise HTTPException(status_code=404, detail="Frontend index.html not found")

        @app.get("/{full_path:path}")
        async def serve_spa(request: Request, full_path: str):
            # API and documentation routes are handled by routers or 404
            if full_path.startswith("api/") or full_path in [
                "health",
                "docs",
                "redoc",
                "openapi.json",
            ]:
                raise HTTPException(
                    status_code=404, detail=f"API route not found: /{full_path}"
                )

            # Serve specific static file if present (e.g. favicon, logo)
            target_file = frontend_dist / full_path
            if target_file.is_file():
                media_type, _ = mimetypes.guess_type(str(target_file))
                return Response(
                    content=target_file.read_bytes(),
                    media_type=media_type or "application/octet-stream",
                )

            # SPA fallback to index.html for client-side routing
            index_file = frontend_dist / "index.html"
            if index_file.is_file():
                return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
            raise HTTPException(status_code=404, detail="Frontend index.html not found")

    return app
