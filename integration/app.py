# app.py
from fastapi import FastAPI
import uvicorn

try:
    from .models import AskRequest, PipelineResponse
    from .pipeline_service import PipelineService
    from .config import settings
except ImportError:  # pragma: no cover - supports running from integration/ directly
    from models import AskRequest, PipelineResponse
    from pipeline_service import PipelineService
    from config import settings


app = FastAPI(title="Neural Symbolic Monkeys Backend")

# Create a single pipeline instance (simple DI)
pipeline = PipelineService()


@app.post("/pipeline", response_model=PipelineResponse)
def run_pipeline(req: AskRequest) -> PipelineResponse:
    """
    Main endpoint that your Next.js backend calls.

    POST /pipeline
    {
      "image_url": "/uploads/xyz.png",
      "question": "How can the monkey reach the banana?"
    }
    """
    return pipeline.run(req)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
