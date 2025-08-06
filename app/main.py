from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat, health
import uvicorn

def create_app() -> FastAPI:
    app = FastAPI(
        title="Argument Mining API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # include our routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
