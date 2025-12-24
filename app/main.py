from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from contextlib import asynccontextmanager
from app.routes import router
from app.advanced_service import detector
from fastapi.middleware.cors import CORSMiddleware
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load_model()
    yield

app = FastAPI(
    title="Structured License Plate API",
    version="2.0",
    lifespan=lifespan
)

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

app.include_router(router)

# Serve Static Files (Frontend)
# Navigate up from: app/main.py -> app/ -> license_plate_detection_service/ -> IT22174444/ -> frontend
# Path(__file__).resolve().parent is 'app'
base_path = Path(__file__).resolve().parent.parent.parent
frontend_path = base_path / "frontend"

if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"error": f"Frontend directory not found at {frontend_path}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)