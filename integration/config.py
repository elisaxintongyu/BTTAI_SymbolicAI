# config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0

    # YOLO / ONNX
    yolo_onnx_path: str = "model.onnx"

    # FastAPI
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
