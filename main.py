from fastapi import FastAPI
from routers.audio import router as audio_router

app = FastAPI(
    title="API de Detección de Tos",
    description="API RESTful para el análisis de audio y detección de tos usando YAMNet y GreenArcade",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(audio_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5005, reload=False)
