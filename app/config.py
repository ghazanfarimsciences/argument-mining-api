from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_API_URL: str      = "https://models.example.com/render"
    TIMEOUT_SECONDS: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
