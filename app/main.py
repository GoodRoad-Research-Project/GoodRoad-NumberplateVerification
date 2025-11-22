from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.service import detector

@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load_model()
    yield

app = FastAPI(
    title="Structured License Plate API",
    version="2.0",
    lifespan=lifespan
)

app.include_router(router)

@app.get("/")
def root():
    return {"status": "online", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)