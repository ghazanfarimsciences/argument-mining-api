from fastapi import FastAPI
from api.routes import chat, health

def create_app() -> FastAPI:
    app = FastAPI(
        title="Argument Mining API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # include our routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])

    return app

app = create_app()